"""Ingest, split, encode, compare, explain, and evaluate biomedical data."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


SCHEMA_VERSION = 4
PACKAGE_NAME = "quantum-health"

# Missing-value sentinels recognized by load_csv_dataset() and encode_raw_row() alike,
# so a single new record is encoded identically to how training rows were encoded.
MISSING_VALUE_SENTINELS = {"na", "n/a", "nan", "none", "null", "missing", "-", "?"}


def package_version() -> str:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "editable-or-unknown"


def runtime_manifest() -> dict[str, str]:
    dependency_names = (
        "numpy",
        "scikit-learn",
        "qiskit",
        "qiskit-aer",
        "qiskit-machine-learning",
        "qiskit-algorithms",
        "qiskit-ibm-runtime",
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in dependency_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def dataset_fingerprint(dataset: "LoadedDataset") -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset.X, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(dataset.y, dtype=np.int64).tobytes())
    digest.update("\0".join(dataset.feature_names).encode("utf-8"))
    digest.update(dataset.positive_label.encode("utf-8"))
    for metadata in (
        dataset.groups,
        dataset.times,
        dataset.row_ids,
        dataset.sites,
        dataset.outcome_times,
    ):
        if metadata is not None:
            digest.update("\0".join(str(value) for value in metadata).encode("utf-8"))
    for name in sorted(dataset.subgroups):
        digest.update(name.encode("utf-8"))
        digest.update("\0".join(str(value) for value in dataset.subgroups[name]).encode("utf-8"))
    if dataset.task_profile is not None:
        digest.update(json.dumps(dataset.task_profile, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    positive_label: str
    negative_label: str
    provenance: dict[str, Any] = field(default_factory=dict)
    groups: np.ndarray | None = None
    times: np.ndarray | None = None
    row_ids: np.ndarray | None = None
    sites: np.ndarray | None = None
    outcome_times: np.ndarray | None = None
    subgroups: dict[str, np.ndarray] = field(default_factory=dict)
    task_profile: dict[str, Any] | None = None


@dataclass
class PreprocessingPipeline:
    """Fit-on-training-only transforms reusable for persisted inference."""

    feature_names: list[str]
    n_qubits: int
    reduction: str = "anova"
    imputer: SimpleImputer | None = None
    standardizer: StandardScaler | None = None
    selector: SelectKBest | None = None
    pca: PCA | None = None
    angle_scaler: MinMaxScaler | None = None
    selected_features: list[str] = field(default_factory=list)
    raw_reference: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreprocessingPipeline":
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.reduction not in {"anova", "pca"}:
            raise ValueError("reduction must be 'anova' or 'pca'")
        if X.shape[1] < self.n_qubits:
            raise ValueError(
                f"dataset has {X.shape[1]} features but {self.n_qubits} qubits were requested"
            )
        self.imputer = SimpleImputer(strategy="median")
        self.standardizer = StandardScaler()
        self.selector = None
        self.pca = None
        X_standard = self.standardizer.fit_transform(self.imputer.fit_transform(X))
        self.raw_reference = np.asarray(self.imputer.statistics_, dtype=float)
        component_count = min(self.n_qubits, X_standard.shape[1])
        if self.reduction == "pca":
            self.pca = PCA(n_components=component_count, svd_solver="full")
            X_selected = self.pca.fit_transform(X_standard)
            self.selected_features = [f"PC{i + 1}" for i in range(component_count)]
        else:
            self.selector = SelectKBest(f_classif, k=component_count)
            X_selected = self.selector.fit_transform(X_standard, y)
            self.selected_features = [
                self.feature_names[index] for index in self.selector.get_support(indices=True)
            ]
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.angle_scaler.fit(X_selected)
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not all(
            component is not None
            for component in (self.imputer, self.standardizer, self.angle_scaler)
        ):
            raise RuntimeError("preprocessing pipeline must be fitted before transform")
        assert self.imputer is not None
        assert self.standardizer is not None
        assert self.angle_scaler is not None
        X_standard = self.standardizer.transform(self.imputer.transform(X))
        if self.reduction == "pca":
            if self.pca is None:
                raise RuntimeError("PCA preprocessing pipeline is missing its fitted reducer")
            X_selected = self.pca.transform(X_standard)
        else:
            if self.selector is None:
                raise RuntimeError("ANOVA preprocessing pipeline is missing its fitted selector")
            X_selected = self.selector.transform(X_standard)
        return X_selected, self.angle_scaler.transform(X_selected)

    def explanation_features(self, limit: int = 16) -> list[str]:
        """Return a bounded raw-feature set for affordable local explanations."""

        if limit < 1:
            return []
        if self.reduction == "anova" and self.selector is not None:
            return [
                self.feature_names[index]
                for index in self.selector.get_support(indices=True)[:limit]
            ]
        if self.reduction == "pca" and self.pca is not None:
            importance = np.sum(np.abs(self.pca.components_), axis=0)
            indices = np.argsort(importance)[::-1][:limit]
            return [self.feature_names[index] for index in indices]
        return self.feature_names[:limit]


@dataclass(frozen=True)
class PreparedDataset:
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_validation_raw: np.ndarray | None
    X_train_classical: np.ndarray
    X_test_classical: np.ndarray
    X_train_quantum: np.ndarray
    X_test_quantum: np.ndarray
    X_validation_classical: np.ndarray | None
    X_validation_quantum: np.ndarray | None
    y_train: np.ndarray
    y_validation: np.ndarray | None
    y_test: np.ndarray
    selected_features: list[str]
    quantum_reference: np.ndarray
    classical_reference: np.ndarray
    preprocessor: PreprocessingPipeline
    test_row_ids: np.ndarray | None = None
    test_sites: np.ndarray | None = None
    test_subgroups: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantumContext:
    feature_map: Any
    kernel: Any
    sampler: Any
    pass_manager: Any
    backend_name: str
    backend_qubits: int | None
    probe: dict[str, Any]


def load_breast_cancer_dataset() -> LoadedDataset:
    """Load a public binary benchmark and make malignant class positive."""

    source = load_breast_cancer()
    return LoadedDataset(
        name="sklearn_breast_cancer_wisconsin_diagnostic",
        X=np.asarray(source.data, dtype=float),
        y=(np.asarray(source.target) == 0).astype(int),
        feature_names=[str(name) for name in source.feature_names],
        positive_label="malignant",
        negative_label="benign",
        provenance={
            "source": "sklearn.datasets.load_breast_cancer",
            "dataset": "Breast Cancer Wisconsin Diagnostic",
            "access": "public benchmark",
        },
        row_ids=np.asarray([f"row-{index + 1}" for index in range(len(source.target))], dtype=str),
    )


def load_csv_dataset(
    path: str | Path,
    target: str,
    positive_label: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    id_column: str | None = None,
    site_column: str | None = None,
    outcome_time_column: str | None = None,
    subgroup_columns: Iterable[str] | None = None,
    leakage_columns: Iterable[str] | None = None,
    task_profile: dict[str, Any] | None = None,
) -> LoadedDataset:
    """Load a numeric CSV without adding a dataframe dependency."""

    csv_path = Path(path)
    subgroup_names = tuple(str(name) for name in (subgroup_columns or ()))
    leakage_names = tuple(str(name) for name in (leakage_columns or ()))
    if len(set(subgroup_names)) != len(subgroup_names):
        raise ValueError("subgroup columns must be unique")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        if target not in reader.fieldnames:
            raise ValueError(f"target column {target!r} is not in the CSV header")
        if group_column and group_column not in reader.fieldnames:
            raise ValueError(f"group column {group_column!r} is not in the CSV header")
        if time_column and time_column not in reader.fieldnames:
            raise ValueError(f"time column {time_column!r} is not in the CSV header")
        for metadata_name, column_name in (
            ("id", id_column),
            ("site", site_column),
            ("outcome time", outcome_time_column),
        ):
            if column_name and column_name not in reader.fieldnames:
                raise ValueError(f"{metadata_name} column {column_name!r} is not in the CSV header")
        for subgroup_name in subgroup_names:
            if subgroup_name not in reader.fieldnames:
                raise ValueError(f"subgroup column {subgroup_name!r} is not in the CSV header")
        if group_column and group_column == target:
            raise ValueError("group column must differ from target")
        if time_column and time_column == target:
            raise ValueError("time column must differ from target")
        if group_column and time_column and group_column == time_column:
            raise ValueError("group and time columns must differ")
        if group_column and time_column:
            # Both are valid for early-detection data: the splitter will enforce
            # chronology while keeping a patient on one side of the boundary.
            pass
        excluded = {
            target,
            group_column,
            time_column,
            id_column,
            site_column,
            outcome_time_column,
            *subgroup_names,
            *leakage_names,
        }
        feature_names = [name for name in reader.fieldnames if name not in excluded]
        if not feature_names:
            raise ValueError("CSV must contain at least one feature column")
        raw_rows = list(reader)

        # Categorical columns (free text such as "Male"/"Female") are not
        # numeric and load_csv_dataset only ever accepted numeric columns.
        # Rather than reject every dataset with a text column, auto-detect
        # and one-hot encode them here so any condition's CSV adapter gets
        # this for free. A missing-value sentinel (e.g. a stray "N/A" in an
        # otherwise numeric column, such as this dataset's bmi field) is
        # treated as missing, not as evidence the column is categorical.
        categories: dict[str, list[str]] = {}
        for name in feature_names:
            seen: set[str] = set()
            is_categorical = False
            for row in raw_rows:
                raw = str(row.get(name, "")).strip()
                if not raw or raw.lower() in MISSING_VALUE_SENTINELS:
                    continue
                try:
                    float(raw)
                except ValueError:
                    is_categorical = True
                seen.add(raw)
            if is_categorical:
                categories[name] = sorted(seen)

        labels: list[str] = []
        values: list[list[float]] = []
        groups: list[str] = []
        times: list[str] = []
        row_ids: list[str] = []
        sites: list[str] = []
        outcome_times: list[str | None] = []
        subgroups: dict[str, list[str]] = {name: [] for name in subgroup_names}
        for row_number, row in enumerate(raw_rows, start=2):
            row_ids.append(str(row.get(id_column, "") or row_number - 1) if id_column else f"row-{row_number - 1}")
            label = str(row.get(target, "")).strip()
            if not label:
                raise ValueError(f"empty target at CSV row {row_number}")
            labels.append(label)
            if group_column:
                group = str(row.get(group_column, "")).strip()
                if not group:
                    raise ValueError(f"empty group at CSV row {row_number}")
                groups.append(group)
            if time_column:
                timestamp = str(row.get(time_column, "")).strip()
                if not timestamp:
                    raise ValueError(f"empty time value at CSV row {row_number}")
                times.append(timestamp)
            if site_column:
                site = str(row.get(site_column, "")).strip()
                if not site:
                    raise ValueError(f"empty site value at CSV row {row_number}")
                sites.append(site)
            if outcome_time_column:
                raw_outcome_time = str(row.get(outcome_time_column, "")).strip()
                outcome_times.append(raw_outcome_time or None)
            for subgroup_name in subgroup_names:
                subgroups[subgroup_name].append(str(row.get(subgroup_name, "")).strip() or "unknown")
            parsed_row: list[float] = []
            for name in feature_names:
                raw = str(row.get(name, "")).strip()
                if name in categories:
                    parsed_row.extend(
                        1.0 if raw == category else 0.0 for category in categories[name]
                    )
                    continue
                if not raw or raw.lower() in MISSING_VALUE_SENTINELS:
                    parsed_row.append(np.nan)
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"feature {name!r} at CSV row {row_number} is not numeric"
                    ) from exc
                parsed_row.append(value if np.isfinite(value) else np.nan)
            values.append(parsed_row)

        expanded_feature_names = [
            f"{name}={category}" if name in categories else name
            for name in feature_names
            for category in (categories.get(name) or [None])
        ]

    if len(values) < 4:
        raise ValueError("CSV must contain at least four data rows")
    unique_labels = sorted(set(labels))
    if len(unique_labels) != 2:
        raise ValueError("target column must contain exactly two distinct labels")

    if positive_label is None:
        preferred = ("1", "true", "yes", "positive", "disease", "malignant")
        positive = next(
            (candidate for candidate in preferred if candidate in {x.lower() for x in unique_labels}),
            unique_labels[-1],
        )
        positive = next(label for label in unique_labels if label.lower() == positive.lower())
    else:
        positive = str(positive_label).strip()
        if positive not in unique_labels:
            raise ValueError(
                f"positive label {positive!r} is not one of {unique_labels!r}"
            )
    negative = next(label for label in unique_labels if label != positive)

    return LoadedDataset(
        name=csv_path.name,
        X=np.asarray(values, dtype=float),
        y=np.asarray([int(label == positive) for label in labels], dtype=int),
        feature_names=expanded_feature_names,
        positive_label=positive,
        negative_label=negative,
        provenance={
            "source": "numeric CSV",
            "path": str(csv_path),
            "access": "caller-supplied",
            "group_column": group_column,
            "time_column": time_column,
            "id_column": id_column,
            "site_column": site_column,
            "outcome_time_column": outcome_time_column,
            "subgroup_columns": list(subgroup_names),
            "leakage_columns": list(leakage_names),
            "categorical_columns": {name: values for name, values in categories.items()},
        },
        groups=np.asarray(groups, dtype=str) if group_column else None,
        times=np.asarray(times, dtype=str) if time_column else None,
        row_ids=np.asarray(row_ids, dtype=str),
        sites=np.asarray(sites, dtype=str) if site_column else None,
        outcome_times=np.asarray(outcome_times, dtype=object) if outcome_time_column else None,
        subgroups={name: np.asarray(values, dtype=str) for name, values in subgroups.items()},
        task_profile=task_profile,
    )


def encode_raw_row(
    raw_row: dict[str, Any],
    feature_names: Iterable[str],
    categorical_columns: dict[str, list[str]],
) -> np.ndarray:
    """Encode one new record the same way `load_csv_dataset()` encoded training rows.

    `feature_names` is the artifact's post-encoding column list (as saved on
    `PreprocessingPipeline.feature_names`); a categorical source column's expanded
    columns are named `"<source column>=<category>"`. `categorical_columns` is the
    dict `load_csv_dataset()` recorded in `LoadedDataset.provenance["categorical_columns"]`
    for the training dataset. This lets a single case bundle row be scored by a saved
    artifact without needing the source CSV's categorical columns pre-expanded by hand.
    """

    encoded: list[float] = []
    for name in feature_names:
        source_column, sep, category = name.partition("=")
        if sep and source_column in categorical_columns:
            raw = str(raw_row.get(source_column, "")).strip()
            encoded.append(1.0 if raw == category else 0.0)
            continue
        raw = str(raw_row.get(name, "")).strip()
        if not raw or raw.lower() in MISSING_VALUE_SENTINELS:
            encoded.append(np.nan)
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"feature {name!r} is not numeric: {raw!r}") from exc
        encoded.append(value if np.isfinite(value) else np.nan)
    return np.asarray(encoded, dtype=float)


def load_profile_dataset(path: str | Path) -> LoadedDataset:
    """Load and validate a caller-supplied early-detection profile."""

    from .protocol import (
        load_early_detection_profile,
        resolve_profile_dataset,
        validate_early_detection_profile,
    )

    profile, profile_directory = load_early_detection_profile(path)
    dataset = load_csv_dataset(
        resolve_profile_dataset(profile, profile_directory),
        target=profile.target_column,
        positive_label=profile.positive_label,
        group_column=profile.group_column,
        time_column=profile.index_time_column,
        id_column=profile.id_column,
        site_column=profile.site_column,
        outcome_time_column=profile.outcome_time_column,
        subgroup_columns=profile.subgroup_columns,
        leakage_columns=profile.leakage_columns,
        task_profile=profile.as_dict(),
    )
    validate_early_detection_profile(profile, dataset)
    return replace(
        dataset,
        name=profile.name,
        provenance={**dataset.provenance, "profile": str(Path(path))},
    )


def load_high_dimensional_csv_dataset(
    path: str | Path,
    target: str,
    positive_label: str | None = None,
    modality: str = "gene_expression",
    **metadata_columns: Any,
) -> LoadedDataset:
    """Load a wide numeric matrix for a bounded PCA-to-qubit experiment."""

    if modality not in {"gene_expression", "imaging_features", "ehr_numeric"}:
        raise ValueError(
            "high-dimensional modality must be gene_expression, imaging_features, or ehr_numeric"
        )
    return load_csv_dataset(
        path,
        target,
        positive_label,
        task_profile={
            "task_type": "benchmark_matrix",
            "modality": modality,
            "reduction": "pca",
        },
        **metadata_columns,
    )


def load_prediction_csv(
    path: str | Path,
    feature_names: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Load feature-only CSV rows in the order required by a saved model."""

    csv_path = Path(path)
    expected = [str(name) for name in feature_names]
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        missing = [name for name in expected if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"prediction CSV is missing model features: {', '.join(missing)}"
            )
        values: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            parsed_row: list[float] = []
            for name in expected:
                raw = str(row.get(name, "")).strip()
                if not raw:
                    parsed_row.append(np.nan)
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"feature {name!r} at CSV row {row_number} is not numeric"
                    ) from exc
                parsed_row.append(value if np.isfinite(value) else np.nan)
            values.append(parsed_row)
    if not values:
        raise ValueError("prediction CSV must contain at least one data row")
    return np.asarray(values, dtype=float), expected


def _split_indices(
    y: np.ndarray,
    held_out_size: float,
    seed: int,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
    split_name: str = "held-out",
) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None and times is not None:
        if len(groups) != len(y) or len(times) != len(y):
            raise ValueError("group and time metadata must have one value per row")
        if len(np.unique(np.asarray(times, dtype=str))) <= 1:
            # ponytail: a constant timestamp has no temporal ordering; use the
            # existing grouped splitter until event-level times are available.
            return _split_indices(
                y,
                held_out_size,
                seed,
                groups=groups,
                split_name=split_name,
            )
        order = np.argsort(np.asarray(times, dtype=str), kind="stable")
        held_out_count = max(1, int(np.ceil(len(y) * held_out_size)))
        if held_out_count >= len(y):
            raise ValueError(f"{split_name} time split needs rows in both partitions")
        candidate_cutoffs = sorted(
            range(1, len(y)),
            key=lambda cutoff: abs((len(y) - cutoff) - held_out_count),
        )
        for cutoff in candidate_cutoffs:
            train_index = np.asarray(order[:cutoff])
            held_out_index = np.asarray(order[cutoff:])
            train_groups = set(np.asarray(groups)[train_index].tolist())
            held_out_groups = set(np.asarray(groups)[held_out_index].tolist())
            if (
                train_groups.isdisjoint(held_out_groups)
                and len(np.unique(y[train_index])) == 2
                and len(np.unique(y[held_out_index])) == 2
            ):
                return train_index, held_out_index
        raise ValueError(
            f"{split_name} chronological split could not preserve both classes and group boundaries"
        )
    if groups is not None:
        if len(groups) != len(y):
            raise ValueError("group metadata must have one value per row")
        from sklearn.model_selection import GroupShuffleSplit

        for attempt in range(20):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=held_out_size,
                random_state=seed + attempt,
            )
            train_index, held_out_index = next(
                splitter.split(np.zeros((len(y), 1)), y, groups)
            )
            if len(np.unique(y[train_index])) == 2 and len(np.unique(y[held_out_index])) == 2:
                return np.asarray(train_index), np.asarray(held_out_index)
        raise ValueError(
            f"{split_name} group split could not preserve both classes after 20 attempts"
        )
    if times is not None:
        if len(times) != len(y):
            raise ValueError("time metadata must have one value per row")
        order = np.argsort(np.asarray(times, dtype=str), kind="stable")
        held_out_count = max(1, int(np.ceil(len(y) * held_out_size)))
        if held_out_count >= len(y):
            raise ValueError(f"{split_name} time split needs rows in both partitions")
        train_index = order[:-held_out_count]
        held_out_index = order[-held_out_count:]
        if len(np.unique(y[train_index])) != 2 or len(np.unique(y[held_out_index])) != 2:
            raise ValueError(
                f"{split_name} chronological split could not preserve both classes"
            )
        return np.asarray(train_index), np.asarray(held_out_index)

    train_index, held_out_index = train_test_split(
        np.arange(len(y)),
        test_size=held_out_size,
        stratify=y,
        random_state=seed,
    )
    return np.asarray(train_index), np.asarray(held_out_index)


def _has_temporal_variation(times: np.ndarray | None) -> bool:
    return times is not None and len(np.unique(np.asarray(times, dtype=str))) > 1


def _cap_indices(
    y: np.ndarray,
    cap: int,
    seed: int,
    split_name: str,
    times: np.ndarray | None = None,
) -> np.ndarray:
    if cap <= 0 or len(y) <= cap:
        return np.arange(len(y))
    if cap < len(np.unique(y)):
        raise ValueError(f"--max-{split_name} must leave both classes represented")
    if times is not None:
        if split_name == "train":
            selected = np.arange(len(y) - cap, len(y))
        else:
            selected = np.arange(cap)
        if len(np.unique(y[selected])) != 2:
            raise ValueError(
                f"--max-{split_name} chronological cap must leave both classes represented"
            )
        return selected
    selected, _ = train_test_split(
        np.arange(len(y)),
        train_size=cap,
        stratify=y,
        random_state=seed,
    )
    return np.asarray(selected)


def prepare_dataset(
    dataset: LoadedDataset,
    n_qubits: int,
    test_size: float,
    seed: int,
    max_train: int,
    max_test: int,
    validation_size: float | None = None,
    reduction: str = "anova",
    holdout_site: str | None = None,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> PreparedDataset:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if validation_size is not None and not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if dataset.X.shape[1] < n_qubits:
        raise ValueError(
            f"dataset has {dataset.X.shape[1]} features but {n_qubits} qubits were requested"
        )

    if dataset.X.ndim != 2 or dataset.y.ndim != 1:
        raise ValueError("dataset features must be 2-D and targets must be 1-D")
    if len(dataset.feature_names) != dataset.X.shape[1]:
        raise ValueError("dataset feature_names must match the feature matrix columns")
    if len(dataset.X) != len(dataset.y):
        raise ValueError("dataset features and targets must have equal row counts")
    if set(np.unique(dataset.y)) != {0, 1}:
        raise ValueError("dataset targets must be encoded as both binary classes 0 and 1")
    if reduction not in {"anova", "pca"}:
        raise ValueError("reduction must be 'anova' or 'pca'")
    if holdout_site is not None and dataset.sites is None:
        raise ValueError("holdout_site requires site metadata")
    if split_indices is not None and holdout_site is not None:
        raise ValueError("split_indices and holdout_site cannot be combined")
    if split_indices is not None:
        train_index = np.asarray(split_indices[0], dtype=int)
        test_index = np.asarray(split_indices[1], dtype=int)
        if len(set(train_index.tolist()).intersection(test_index.tolist())):
            raise ValueError("explicit train and test indices must be disjoint")
        if any(index < 0 or index >= len(dataset.y) for index in np.concatenate((train_index, test_index))):
            raise ValueError("explicit split indices are outside the dataset")
    elif holdout_site is not None:
        assert dataset.sites is not None
        test_index = np.flatnonzero(dataset.sites == str(holdout_site))
        train_index = np.flatnonzero(dataset.sites != str(holdout_site))
        if not len(test_index) or not len(train_index):
            raise ValueError("holdout_site must leave rows in both partitions")
    else:
        train_index, test_index = _split_indices(
            dataset.y,
            test_size,
            seed,
            groups=dataset.groups,
            times=dataset.times,
            split_name="test",
        )
    if len(np.unique(dataset.y[train_index])) != 2 or len(np.unique(dataset.y[test_index])) != 2:
        raise ValueError("train and test partitions must each contain both classes")
    X_train, X_test = dataset.X[train_index], dataset.X[test_index]
    y_train, y_test = dataset.y[train_index], dataset.y[test_index]
    groups_train = dataset.groups[train_index] if dataset.groups is not None else None
    times_train = dataset.times[train_index] if dataset.times is not None else None
    train_cap_index = _cap_indices(y_train, max_train, seed + 1, "train", times_train)
    test_cap_index = _cap_indices(
        y_test,
        max_test,
        seed + 2,
        "test",
        dataset.times[test_index] if dataset.times is not None else None,
    )
    X_train, y_train = X_train[train_cap_index], y_train[train_cap_index]
    X_test, y_test = X_test[test_cap_index], y_test[test_cap_index]
    if groups_train is not None:
        groups_train = groups_train[train_cap_index]
    if times_train is not None:
        times_train = times_train[train_cap_index]

    X_fit, y_fit = X_train, y_train
    X_validation: np.ndarray | None = None
    y_validation: np.ndarray | None = None
    if validation_size is not None:
        try:
            fit_index, validation_index = _split_indices(
                y_train,
                validation_size,
                seed + 3,
                groups=groups_train,
                times=times_train,
                split_name="validation",
            )
            X_fit, X_validation = X_train[fit_index], X_train[validation_index]
            y_fit, y_validation = y_train[fit_index], y_train[validation_index]
        except ValueError as exc:
            raise ValueError(
                "validation split could not preserve both classes or metadata boundaries"
            ) from exc

    preprocessor = PreprocessingPipeline(
        feature_names=list(dataset.feature_names),
        n_qubits=n_qubits,
        reduction=reduction,
    ).fit(X_fit, y_fit)
    X_train_classical, X_train_quantum = preprocessor.transform(X_fit)
    X_test_classical, X_test_quantum = preprocessor.transform(X_test)
    if X_validation is not None:
        X_validation_classical, X_validation_quantum = preprocessor.transform(X_validation)
    else:
        X_validation_classical = None
        X_validation_quantum = None

    return PreparedDataset(
        X_train_raw=np.asarray(X_fit, dtype=float),
        X_test_raw=np.asarray(X_test, dtype=float),
        X_validation_raw=(
            np.asarray(X_validation, dtype=float) if X_validation is not None else None
        ),
        X_train_classical=X_train_classical,
        X_test_classical=X_test_classical,
        X_train_quantum=X_train_quantum,
        X_test_quantum=X_test_quantum,
        X_validation_classical=X_validation_classical,
        X_validation_quantum=X_validation_quantum,
        y_train=np.asarray(y_fit, dtype=int),
        y_validation=(
            np.asarray(y_validation, dtype=int) if y_validation is not None else None
        ),
        y_test=np.asarray(y_test, dtype=int),
        selected_features=preprocessor.selected_features,
        quantum_reference=np.median(X_train_quantum, axis=0),
        classical_reference=np.median(X_train_classical, axis=0),
        preprocessor=preprocessor,
        test_row_ids=(
            dataset.row_ids[test_index[test_cap_index]]
            if dataset.row_ids is not None
            else np.asarray([f"row-{index + 1}" for index in test_index[test_cap_index]], dtype=str)
        ),
        test_sites=(
            dataset.sites[test_index[test_cap_index]] if dataset.sites is not None else None
        ),
        test_subgroups={
            name: values[test_index[test_cap_index]]
            for name, values in dataset.subgroups.items()
        },
    )


def _fake_backend() -> Any:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    except ImportError as exc:
        raise RuntimeError(
            "fake/noisy execution needs the hardware extra: "
            "pip install -e '.[hardware]'"
        ) from exc
    return FakeSherbrooke()


def _two_qubit_count(circuit: Any) -> int:
    return sum(
        1
        for instruction in circuit.data
        if getattr(instruction.operation, "num_qubits", 0) == 2
    )


def _circuit_probe(circuit: Any, backend_name: str, backend_qubits: int | None) -> dict[str, Any]:
    return {
        "backend": backend_name,
        "backend_qubits": backend_qubits,
        "logical_qubits": int(circuit.num_qubits),
        "feature_map_depth": int(circuit.depth()),
        "two_qubit_gates": _two_qubit_count(circuit),
        "operations": {str(name): int(count) for name, count in circuit.count_ops().items()},
    }


def build_quantum_context(
    mode: str,
    n_qubits: int,
    shots: int,
    seed: int,
    aer_noise: str = "none",
    feature_map_reps: int = 1,
    feature_map_entanglement: str = "linear",
) -> QuantumContext:
    """Build one feature map and execution path for all requested QML models."""

    if mode not in {"statevector", "aer", "fake", "ibm"}:
        raise ValueError(f"unsupported backend mode: {mode}")
    if shots < 1:
        raise ValueError("shots must be positive")
    if feature_map_entanglement not in {"linear", "full", "circular"}:
        raise ValueError("feature_map_entanglement must be linear, full, or circular")

    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import (
        FidelityQuantumKernel,
        FidelityStatevectorKernel,
    )
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    feature_map = zz_feature_map(n_qubits, reps=feature_map_reps, entanglement=feature_map_entanglement)
    pass_manager = None

    if mode == "statevector":
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(default_shots=shots, seed=seed)
        kernel = FidelityStatevectorKernel(feature_map=feature_map)
        backend_name = "statevector"
        backend_qubits = None
        probe_circuit = feature_map
    elif mode == "aer":
        from qiskit_aer.primitives import SamplerV2

        options: dict[str, Any] = {}
        backend_name = "aer"
        backend_qubits = None
        if aer_noise == "fake":
            fake_backend = _fake_backend()
            from qiskit_aer.noise import NoiseModel

            options["backend_options"] = {
                "noise_model": NoiseModel.from_backend(fake_backend)
            }
            backend_name = f"aer-noise:{fake_backend.name}"
            backend_qubits = int(fake_backend.num_qubits)
        sampler = SamplerV2(default_shots=shots, seed=seed, options=options)
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=ComputeUncompute(sampler=sampler),
        )
        probe_circuit = feature_map
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService

        if mode == "fake":
            backend = _fake_backend()
        else:
            backend_name_from_env = os.environ.get("IBM_BACKEND", "").strip()
            if not backend_name_from_env:
                raise RuntimeError("IBM_BACKEND must name a configured IBM backend")
            service = QiskitRuntimeService()
            backend = service.backend(backend_name_from_env)

        pass_manager = generate_preset_pass_manager(
            optimization_level=1,
            backend=backend,
            seed_transpiler=seed,
        )
        sampler = SamplerV2(
            mode=backend,
            options={"default_shots": shots},
        )
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=ComputeUncompute(sampler=sampler, pass_manager=pass_manager),
        )
        backend_name = str(backend.name)
        backend_qubits = int(backend.num_qubits)
        probe_circuit = pass_manager.run(feature_map)

    return QuantumContext(
        feature_map=feature_map,
        kernel=kernel,
        sampler=sampler,
        pass_manager=pass_manager,
        backend_name=backend_name,
        backend_qubits=backend_qubits,
        probe=_circuit_probe(probe_circuit, backend_name, backend_qubits),
    )


def _scores(
    model: Any,
    X: np.ndarray,
    prefer_probability: bool = False,
) -> np.ndarray | None:
    if prefer_probability and hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(X), dtype=float)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return probabilities[:, 1]
        except AttributeError:
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        # PegasosQSVC defines its positive margin using the first sorted label.
        if getattr(model, "_label_pos", 1) != 1:
            scores = -scores
        return scores
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    return None


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None,
    probability_score: bool = False,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "evaluated_rows": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "roc_auc": None,
            "brier_score": None,
            "confusion_matrix_labels": ["negative", "positive"],
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    matrix = np.asarray([[tn, fp], [fn, tp]], dtype=int)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    negative_predictive_value = tn / (tn + fn) if tn + fn else None
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None and sensitivity is not None and precision + sensitivity
        else None
    )
    balanced_accuracy = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2
        else None
    )
    auc = None
    pr_auc = None
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            candidate_auc = float(roc_auc_score(y_true, score))
            if np.isfinite(candidate_auc):
                auc = candidate_auc
            candidate_pr_auc = float(average_precision_score(y_true, score))
            if np.isfinite(candidate_pr_auc):
                pr_auc = candidate_pr_auc
        except ValueError:
            auc = None
    brier = None
    if probability_score and score is not None:
        brier = float(brier_score_loss(y_true, score))
    calibration_error = None
    if probability_score and score is not None:
        probabilities = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
        bins = np.linspace(0.0, 1.0, 11)
        calibration_error = 0.0
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            members = (probabilities >= lower) & (
                probabilities < upper if upper < 1.0 else probabilities <= upper
            )
            if np.any(members):
                calibration_error += float(np.mean(members)) * abs(
                    float(np.mean(probabilities[members])) - float(np.mean(y_true[members]))
                )
    return {
        "evaluated_rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": negative_predictive_value,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "expected_calibration_error": calibration_error,
        "confusion_matrix_labels": ["negative", "positive"],
        "confusion_matrix": matrix.tolist(),
    }


def decision_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray | None,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Return net benefit values for a probability-based clinical policy."""

    if probabilities is None or len(y_true) == 0:
        return []
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("decision-curve labels and probabilities must have equal length")
    points = thresholds if thresholds is not None else np.arange(0.05, 1.0, 0.05)
    rows: list[dict[str, float]] = []
    n_rows = len(y_true)
    for threshold in points:
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            continue
        predicted = probabilities >= threshold
        true_positive = float(np.sum(predicted & (y_true == 1)))
        false_positive = float(np.sum(predicted & (y_true == 0)))
        net_benefit = true_positive / n_rows - false_positive / n_rows * threshold / (1 - threshold)
        prevalence = float(np.mean(y_true))
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "model_net_benefit": net_benefit,
                "treat_all_net_benefit": treat_all,
                "treat_none_net_benefit": 0.0,
            }
        )
    return rows


def calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray | None,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return reliability-plot points without requiring a plotting library."""

    if probabilities is None or len(y_true) == 0:
        return []
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    if len(y_true) != len(probabilities):
        raise ValueError("calibration labels and probabilities must have equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        members = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(members):
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(np.sum(members)),
                    "mean_predicted": float(np.mean(probabilities[members])),
                    "observed_frequency": float(np.mean(y_true[members])),
                }
            )
    return rows


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None,
    subgroups: dict[str, np.ndarray],
    probability_score: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate the held-out predictions separately for supplied subgroups."""

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for name, values in subgroups.items():
        values = np.asarray(values, dtype=str)
        if len(values) != len(y_true):
            raise ValueError(f"subgroup {name!r} does not match held-out rows")
        reports[name] = {}
        for value in np.unique(values):
            members = values == value
            reports[name][str(value)] = classification_metrics(
                np.asarray(y_true)[members],
                np.asarray(y_pred)[members],
                np.asarray(score)[members] if score is not None else None,
                probability_score=probability_score,
            )
    return reports


def select_threshold(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    policy: str = "default",
    target_sensitivity: float | None = None,
) -> dict[str, Any]:
    """Select an operating threshold using validation data only."""

    if policy not in {"default", "max_balanced_accuracy", "target_sensitivity"}:
        raise ValueError(f"unsupported threshold policy: {policy}")
    if policy == "target_sensitivity":
        if target_sensitivity is None or not 0 < target_sensitivity <= 1:
            raise ValueError("target_sensitivity must be between 0 and 1")
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y_validation) != len(probabilities):
        raise ValueError("validation labels and probabilities must have equal length")
    if len(y_validation) == 0 or len(np.unique(y_validation)) != 2:
        raise ValueError("threshold selection needs validation examples from both classes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("validation probabilities must be finite")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    if policy == "default":
        threshold = 0.5
        validation_metrics = classification_metrics(
            y_validation,
            (probabilities >= threshold).astype(int),
            probabilities,
            probability_score=True,
        )
        return {
            "policy": policy,
            "threshold": threshold,
            "validation_rows": int(len(y_validation)),
            "validation_metrics": validation_metrics,
        }

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    records: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = classification_metrics(
            y_validation,
            (probabilities >= threshold).astype(int),
            probabilities,
            probability_score=True,
        )
        records.append((float(threshold), metrics))

    if policy == "max_balanced_accuracy":
        threshold, validation_metrics = max(
            records,
            key=lambda record: (
                record[1]["balanced_accuracy"],
                record[1]["specificity"] if record[1]["specificity"] is not None else -1.0,
                -abs(record[0] - 0.5),
            ),
        )
    else:
        assert target_sensitivity is not None
        eligible = [
            record
            for record in records
            if record[1]["sensitivity"] is not None
            and record[1]["sensitivity"] >= target_sensitivity
        ]
        threshold, validation_metrics = max(
            eligible,
            key=lambda record: (
                record[1]["specificity"] if record[1]["specificity"] is not None else -1.0,
                record[1]["balanced_accuracy"],
                record[0],
            ),
        )

    return {
        "policy": policy,
        "threshold": threshold,
        "target_sensitivity": target_sensitivity,
        "validation_rows": int(len(y_validation)),
        "validation_metrics": validation_metrics,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None,
    probability_score: bool,
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    """Estimate held-out metric intervals without refitting the model."""

    if samples <= 0:
        return None
    if len(y_true) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        name: []
        for name in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "negative_predictive_value",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
        )
    }
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        metrics = classification_metrics(
            y_true[indices],
            y_pred[indices],
            score[indices] if score is not None else None,
            probability_score=probability_score,
        )
        for name, collected in values.items():
            value = metrics[name]
            if value is not None and np.isfinite(value):
                collected.append(float(value))
    intervals: dict[str, Any] = {}
    for name, collected in values.items():
        intervals[name] = {
            "lower": float(np.percentile(collected, 2.5)) if collected else None,
            "upper": float(np.percentile(collected, 97.5)) if collected else None,
            "n": len(collected),
        }
    return {
        "method": "nonparametric bootstrap",
        "confidence": 0.95,
        "requested_samples": samples,
        "metrics": intervals,
    }


def input_sensitivity(
    model: Any,
    X: np.ndarray,
    feature_names: Iterable[str],
    reference: np.ndarray,
    limit: int = 16,
) -> dict[str, Any]:
    """Measure score change when one encoded feature is replaced by its median."""

    sample = np.asarray(X[:limit], dtype=float)
    if len(sample) == 0:
        return {"status": "no_test_examples"}
    baseline = _scores(model, sample)
    if baseline is None:
        return {"status": "model_has_no_continuous_score"}

    impacts: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names):
        altered = sample.copy()
        altered[:, index] = reference[index]
        changed = baseline - _scores(model, altered)
        impacts.append(
            {
                "feature": str(feature_name),
                "mean_abs_score_delta": float(np.mean(np.abs(changed))),
                "mean_signed_score_delta": float(np.mean(changed)),
            }
        )
    impacts.sort(key=lambda item: item["mean_abs_score_delta"], reverse=True)
    return {
        "status": "ok",
        "method": "one-feature median replacement",
        "interpretation": "input sensitivity, not causal or clinical attribution",
        "n_examples": int(len(sample)),
        "features": impacts,
    }


def explain_raw_inputs(
    model: Any,
    preprocessor: PreprocessingPipeline,
    X_raw: np.ndarray,
    feature_space: str,
    row_ids: Iterable[str] | None = None,
    limit: int = 16,
) -> dict[str, Any]:
    """Explain rows in the original feature space using training references."""

    sample = np.asarray(X_raw[:limit], dtype=float)
    if len(sample) == 0:
        return {"status": "no_examples"}
    transformed_classical, transformed_quantum = preprocessor.transform(sample)
    model_input = transformed_classical if feature_space == "classical" else transformed_quantum
    baseline = _scores(model, model_input, prefer_probability=True)
    if baseline is None:
        return {"status": "model_has_no_continuous_score"}
    if preprocessor.raw_reference is None:
        return {"status": "preprocessor_has_no_training_reference"}

    raw_reference = np.asarray(preprocessor.raw_reference, dtype=float)
    feature_names = preprocessor.explanation_features(limit=16)
    feature_indices = [preprocessor.feature_names.index(name) for name in feature_names]
    row_id_list = list(row_ids) if row_ids is not None else []
    feature_impacts: list[list[float]] = [[] for _ in feature_names]
    row_reports: list[dict[str, Any]] = []
    for row_index, row in enumerate(sample):
        row_impacts: list[dict[str, Any]] = []
        for position, (feature_index, feature_name) in enumerate(
            zip(feature_indices, feature_names, strict=True)
        ):
            altered = row.copy()
            altered[feature_index] = raw_reference[feature_index]
            altered_classical, altered_quantum = preprocessor.transform(altered.reshape(1, -1))
            altered_input = (
                altered_classical if feature_space == "classical" else altered_quantum
            )
            altered_score = _scores(model, altered_input, prefer_probability=True)
            if altered_score is None:
                continue
            delta = float(baseline[row_index] - altered_score[0])
            feature_impacts[position].append(delta)
            row_impacts.append(
                {
                    "feature": feature_name,
                    "score_delta": delta,
                    "direction": "supports_positive" if delta > 0 else "supports_negative",
                }
            )
        row_impacts.sort(key=lambda item: abs(float(item["score_delta"])), reverse=True)
        row_reports.append(
            {
                "row_id": (
                    str(row_id_list[row_index])
                    if row_index < len(row_id_list)
                    else f"row-{row_index + 1}"
                ),
                "baseline_score": float(baseline[row_index]),
                "top_features": row_impacts[:5],
            }
        )

    summary: list[dict[str, Any]] = [
        {
            "feature": name,
            "mean_abs_score_delta": float(np.mean(np.abs(values))) if values else 0.0,
            "mean_signed_score_delta": float(np.mean(values)) if values else 0.0,
        }
        for name, values in zip(feature_names, feature_impacts, strict=True)
    ]
    summary.sort(key=lambda item: float(item["mean_abs_score_delta"]), reverse=True)
    return {
        "status": "ok",
        "method": "one-feature reference replacement in original feature space",
        "interpretation": "input sensitivity, not causal or clinical attribution",
        "reference": "training imputation median",
        "feature_space": feature_space,
        "n_examples": int(len(sample)),
        "features": summary,
        "rows": row_reports,
    }


def _normalise_models(models: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for name in models:
        name = name.strip().lower()
        if not name:
            continue
        if name == "classical":
            expanded.extend(["logistic_regression", "rbf_svc", "hist_gradient_boosting"])
        elif name in {
            "logistic_regression",
            "rbf_svc",
            "hist_gradient_boosting",
            "qsvc",
            "pegasos_qsvc",
            "vqc",
        }:
            expanded.append(name)
        else:
            raise ValueError(f"unknown model {name!r}")
    if not expanded:
        raise ValueError("at least one model must be requested")
    return list(dict.fromkeys(expanded))


@dataclass
class SavedModelArtifact:
    """Pickleable model bundle for local inference."""

    schema_version: int
    package_version: str
    model_name: str
    model: Any
    preprocessor: PreprocessingPipeline
    feature_space: str
    selected_features: list[str]
    threshold: float | None
    threshold_policy: str
    abstain_margin: float | None
    probability_score: bool
    calibration: dict[str, Any]
    dataset: dict[str, Any]
    execution: dict[str, Any]
    hardware_probe: dict[str, Any]
    software: dict[str, str]


@dataclass
class ValidationSigmoidCalibratedModel:
    """Calibrate a non-scikit estimator on a held-out validation split."""

    base_model: Any
    calibrator: LogisticRegression
    classes_: np.ndarray = field(default_factory=lambda: np.array([0, 1], dtype=int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = _scores(self.base_model, X, prefer_probability=True)
        if raw is None:
            raise ValueError("base model did not provide probabilities for calibration")
        calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack((1.0 - calibrated, calibrated))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def model_artifact_manifest(artifact: SavedModelArtifact) -> dict[str, Any]:
    """Return the non-binary model contract suitable for review or logging."""

    return {
        "schema_version": artifact.schema_version,
        "package_version": artifact.package_version,
        "model_name": artifact.model_name,
        "feature_space": artifact.feature_space,
        "input_features": list(artifact.preprocessor.feature_names),
        "qubits": artifact.preprocessor.n_qubits,
        "reduction": artifact.preprocessor.reduction,
        "selected_features": list(artifact.selected_features),
        "threshold": artifact.threshold,
        "threshold_policy": artifact.threshold_policy,
        "abstain_margin": artifact.abstain_margin,
        "probability_score": artifact.probability_score,
        "calibration": dict(artifact.calibration),
        "dataset": dict(artifact.dataset),
        "execution": dict(artifact.execution),
        "hardware_probe": dict(artifact.hardware_probe),
        "software": dict(artifact.software),
    }


def _build_model(
    model_name: str,
    context: QuantumContext | None,
    seed: int,
    pegasos_steps: int,
    vqc_maxiter: int,
    parameters: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    parameters = parameters or {}
    if model_name == "logistic_regression":
        return (
            LogisticRegression(
                C=float(parameters.get("C", 1.0)),
                max_iter=int(parameters.get("max_iter", 1000)),
                random_state=seed,
            ),
            "classical",
        )
    if model_name == "rbf_svc":
        return (
            SVC(
                C=float(parameters.get("C", 1.0)),
                gamma=parameters.get("gamma", "scale"),
                kernel="rbf",
            ),
            "classical",
        )
    if model_name == "hist_gradient_boosting":
        return (
            HistGradientBoostingClassifier(
                max_iter=int(parameters.get("max_iter", 200)),
                max_leaf_nodes=int(parameters.get("max_leaf_nodes", 31)),
                learning_rate=float(parameters.get("learning_rate", 0.1)),
                random_state=seed,
            ),
            "classical",
        )

    if context is None:
        raise RuntimeError(f"model {model_name} needs a quantum execution context")
    if model_name == "qsvc":
        from qiskit_machine_learning.algorithms import QSVC

        return QSVC(
            quantum_kernel=context.kernel,
            C=float(parameters.get("C", 1.0)),
        ), "quantum"
    if model_name == "pegasos_qsvc":
        from qiskit_machine_learning.algorithms import PegasosQSVC

        return (
            PegasosQSVC(
                quantum_kernel=context.kernel,
                C=float(parameters.get("C", 1.0)),
                num_steps=int(parameters.get("num_steps", pegasos_steps)),
                seed=seed,
            ),
            "quantum",
        )

    from qiskit.circuit.library import TwoLocal, efficient_su2, real_amplitudes
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
    from qiskit_machine_learning.algorithms import VQC

    num_qubits = context.feature_map.num_qubits
    maxiter = max(int(parameters.get("maxiter", vqc_maxiter)), 2 * num_qubits + 2)
    optimizer_name = str(parameters.get("optimizer", "cobyla")).lower()
    if optimizer_name not in {"cobyla", "spsa", "l_bfgs_b"}:
        raise ValueError("vqc optimizer must be cobyla, spsa, or l_bfgs_b")
    if optimizer_name == "spsa":
        optimizer = SPSA(maxiter=maxiter)
    elif optimizer_name == "l_bfgs_b":
        optimizer = L_BFGS_B(maxiter=maxiter)
    else:
        optimizer = COBYLA(maxiter=maxiter)

    ansatz_name = str(parameters.get("ansatz", "real_amplitudes")).lower()
    ansatz_reps = int(parameters.get("ansatz_reps", 1))
    ansatz_entanglement = str(parameters.get("ansatz_entanglement", "linear"))
    if ansatz_entanglement not in {"linear", "full", "circular"}:
        raise ValueError("vqc ansatz_entanglement must be linear, full, or circular")
    if ansatz_name == "efficient_su2":
        ansatz = efficient_su2(num_qubits, reps=ansatz_reps, entanglement=ansatz_entanglement)
    elif ansatz_name == "two_local":
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            entanglement=ansatz_entanglement,
            reps=ansatz_reps,
        )
    elif ansatz_name == "real_amplitudes":
        ansatz = real_amplitudes(num_qubits, reps=ansatz_reps, entanglement=ansatz_entanglement)
    else:
        raise ValueError("vqc ansatz must be real_amplitudes, efficient_su2, or two_local")

    return (
        VQC(
            num_qubits=num_qubits,
            feature_map=context.feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=context.sampler,
            pass_manager=context.pass_manager,
        ),
        "quantum",
    )


def _model_arrays(
    prepared: PreparedDataset,
    feature_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if feature_space == "classical":
        return (
            prepared.X_train_classical,
            prepared.X_test_classical,
            prepared.X_validation_classical,
            prepared.classical_reference,
        )
    return (
        prepared.X_train_quantum,
        prepared.X_test_quantum,
        prepared.X_validation_quantum,
        prepared.quantum_reference,
    )


def _prepare_run(
    dataset: LoadedDataset,
    model_names: list[str],
    backend: str,
    n_qubits: int,
    shots: int,
    test_size: float,
    seed: int,
    max_train: int,
    max_test: int,
    aer_noise: str,
    validation_size: float | None,
    reduction: str,
    holdout_site: str | None,
    split_indices: tuple[np.ndarray, np.ndarray] | None,
    feature_map_reps: int = 1,
    feature_map_entanglement: str = "linear",
) -> tuple[PreparedDataset, QuantumContext | None]:
    prepared = prepare_dataset(
        dataset,
        n_qubits=n_qubits,
        test_size=test_size,
        seed=seed,
        max_train=max_train,
        max_test=max_test,
        validation_size=validation_size,
        reduction=reduction,
        holdout_site=holdout_site,
        split_indices=split_indices,
    )
    quantum_models = {"qsvc", "pegasos_qsvc", "vqc"}.intersection(model_names)
    if not quantum_models:
        return prepared, None
    return prepared, build_quantum_context(
        mode=backend,
        n_qubits=n_qubits,
        shots=shots,
        seed=seed,
        aer_noise=aer_noise,
        feature_map_reps=feature_map_reps,
        feature_map_entanglement=feature_map_entanglement,
    )


def save_model_artifact(artifact: SavedModelArtifact, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, pickle.PicklingError, TypeError) as exc:
        raise RuntimeError(
            "this model/backend cannot be persisted as a local artifact"
        ) from exc
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest_path.write_text(
        json.dumps(model_artifact_manifest(artifact), indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_artifact(path: str | Path) -> SavedModelArtifact:
    model_path = Path(path)
    try:
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
    except (AttributeError, EOFError, ImportError, pickle.UnpicklingError) as exc:
        raise ValueError(
            "model artifact is invalid or was created by another environment"
        ) from exc
    if not isinstance(artifact, SavedModelArtifact):
        raise ValueError("model artifact was not created by quantum-health")
    if artifact.schema_version == 3:
        # Older local artifacts used ANOVA preprocessing and did not persist
        # the original-feature reference required by the richer explainer.
        artifact.preprocessor.reduction = "anova"
        artifact.preprocessor.pca = None
        artifact.preprocessor.raw_reference = getattr(
            artifact.preprocessor,
            "raw_reference",
            None,
        )
        artifact.schema_version = SCHEMA_VERSION
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported model artifact schema {artifact.schema_version}; expected {SCHEMA_VERSION}"
        )
    return artifact


def predict_with_model_artifact(
    artifact: SavedModelArtifact,
    X: np.ndarray,
    feature_names: Iterable[str],
    dataset_name: str = "prediction_csv",
    row_ids: Iterable[str] | None = None,
    explain: bool = False,
) -> dict[str, Any]:
    """Transform new rows and apply the saved model's operating policy."""

    expected = artifact.preprocessor.feature_names
    received = [str(name) for name in feature_names]
    if received != expected:
        raise ValueError(
            "prediction features must match training columns in the same order"
        )
    X_classical, X_quantum = artifact.preprocessor.transform(np.asarray(X, dtype=float))
    model_input = X_classical if artifact.feature_space == "classical" else X_quantum
    score = _scores(
        artifact.model,
        model_input,
        prefer_probability=artifact.probability_score,
    )
    if artifact.threshold is not None:
        if score is None:
            raise ValueError("saved threshold requires a continuous model score")
        predictions = (score >= artifact.threshold).astype(int)
    else:
        predictions = np.asarray(artifact.model.predict(model_input), dtype=int).reshape(-1)

    if artifact.abstain_margin is not None:
        if artifact.threshold is None or score is None:
            raise ValueError("saved abstention policy requires a probability threshold")
        abstained = np.abs(score - artifact.threshold) < artifact.abstain_margin
    else:
        abstained = np.zeros(len(predictions), dtype=bool)

    explanation = (
        explain_raw_inputs(
            artifact.model,
            artifact.preprocessor,
            np.asarray(X, dtype=float),
            artifact.feature_space,
            row_ids=row_ids,
        )
        if explain
        else {"status": "disabled"}
    )
    row_id_list = list(row_ids) if row_ids is not None else []

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version(),
        "dataset": dataset_name,
        "model_name": artifact.model_name,
        "feature_space": artifact.feature_space,
        "selected_features": list(artifact.selected_features),
        "input_features": list(expected),
        "threshold": artifact.threshold,
        "threshold_policy": artifact.threshold_policy,
        "calibration": dict(artifact.calibration),
        "hardware_probe": dict(artifact.hardware_probe),
        "score_type": "probability" if artifact.probability_score else "decision",
        "predictions": predictions.tolist(),
        "abstained": abstained.tolist(),
        "scores": score.tolist() if score is not None else None,
        "prediction_rows": [
            {
                "row_id": (
                    str(row_id_list[index])
                    if index < len(row_id_list)
                    else f"row-{index + 1}"
                ),
                "prediction": int(predictions[index]),
                "abstained": bool(abstained[index]),
                "score": float(score[index]) if score is not None else None,
            }
            for index in range(len(predictions))
        ],
        "explanation": explanation,
        "abstention": {
            "enabled": artifact.abstain_margin is not None,
            "margin": artifact.abstain_margin,
            "total_rows": int(len(predictions)),
            "abstained_rows": int(np.sum(abstained)),
            "evaluated_rows": int(np.sum(~abstained)),
            "coverage": float(np.mean(~abstained)) if len(predictions) else None,
        },
    }


def _fit_model(
    model_name: str,
    prepared: PreparedDataset,
    context: QuantumContext | None,
    seed: int,
    pegasos_steps: int,
    vqc_maxiter: int,
    calibrate: bool,
    model_parameters: dict[str, Any] | None = None,
) -> tuple[Any, str, np.ndarray, np.ndarray | None, np.ndarray, str]:
    model, feature_space = _build_model(
        model_name,
        context,
        seed,
        pegasos_steps,
        vqc_maxiter,
        parameters=model_parameters,
    )
    X_train, X_test, X_validation, reference = _model_arrays(prepared, feature_space)
    calibration_strategy = "none"
    if calibrate and hasattr(model, "get_params"):
        model = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=3,
        )
        calibration_strategy = "3-fold training cross-validation"
        model.fit(X_train, prepared.y_train)
    elif calibrate:
        if X_validation is None or prepared.y_validation is None:
            raise ValueError(
                f"{model_name} calibration needs a validation split because it is not a scikit-learn estimator"
            )
        if len(np.unique(prepared.y_validation)) != 2:
            raise ValueError(
                f"{model_name} calibration needs validation probabilities and both classes"
            )
        model.fit(X_train, prepared.y_train)
        validation_score = _scores(model, X_validation, prefer_probability=True)
        if validation_score is None:
            raise ValueError(f"{model_name} did not provide probabilities for calibration")
        calibrator = LogisticRegression(max_iter=1000, random_state=seed)
        calibrator.fit(validation_score.reshape(-1, 1), prepared.y_validation)
        model = ValidationSigmoidCalibratedModel(model, calibrator)
        calibration_strategy = "sigmoid fit on held-out validation split"
    else:
        model.fit(X_train, prepared.y_train)
    return model, feature_space, X_test, X_validation, reference, calibration_strategy


def run_experiment(
    dataset: LoadedDataset,
    models: Iterable[str] = ("classical", "qsvc"),
    backend: str = "statevector",
    n_qubits: int = 4,
    shots: int = 512,
    test_size: float = 0.2,
    seed: int = 7,
    max_train: int = 80,
    max_test: int = 40,
    aer_noise: str = "none",
    pegasos_steps: int = 50,
    vqc_maxiter: int = 25,
    calibrate: bool = False,
    allow_remote_calibration: bool = False,
    validation_size: float | None = None,
    threshold_policy: str = "default",
    target_sensitivity: float | None = None,
    abstain_margin: float | None = None,
    bootstrap_samples: int = 0,
    model_artifact_path: str | Path | None = None,
    explain: bool = True,
    allow_remote_explanations: bool = False,
    reduction: str = "anova",
    holdout_site: str | None = None,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    feature_map_reps: int = 1,
    feature_map_entanglement: str = "linear",
) -> dict[str, Any]:
    model_names = _normalise_models(models)
    if pegasos_steps < 1:
        raise ValueError("pegasos_steps must be positive")
    if threshold_policy not in {"default", "max_balanced_accuracy", "target_sensitivity"}:
        raise ValueError(f"unsupported threshold policy: {threshold_policy}")
    if threshold_policy == "default" and target_sensitivity is not None:
        raise ValueError("target_sensitivity needs threshold_policy=target_sensitivity")
    if abstain_margin is not None and not 0 <= abstain_margin < 1:
        raise ValueError("abstain_margin must be between 0 and 1")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    if reduction not in {"anova", "pca"}:
        raise ValueError("reduction must be 'anova' or 'pca'")
    if model_artifact_path is not None and len(model_names) != 1:
        raise ValueError("saving a model artifact requires exactly one model")
    if threshold_policy != "default" and validation_size is None:
        validation_size = 0.2
    effective_calibration = (
        calibrate or threshold_policy != "default" or abstain_margin is not None
    )
    if effective_calibration and "vqc" in model_names and validation_size is None:
        validation_size = 0.2
    if effective_calibration and backend == "ibm" and not allow_remote_calibration:
        raise ValueError(
            "remote calibration fits multiple hardware-backed models; "
            "pass allow_remote_calibration=True to enable it"
        )
    if n_qubits > dataset.X.shape[1]:
        raise ValueError(
            f"dataset has {dataset.X.shape[1]} features but {n_qubits} qubits were requested"
        )
    prepared, context = _prepare_run(
        dataset,
        model_names=model_names,
        backend=backend,
        n_qubits=n_qubits,
        shots=shots,
        test_size=test_size,
        seed=seed,
        max_train=max_train,
        max_test=max_test,
        aer_noise=aer_noise,
        validation_size=validation_size,
        reduction=reduction,
        holdout_site=holdout_site,
        split_indices=split_indices,
        feature_map_reps=feature_map_reps,
        feature_map_entanglement=feature_map_entanglement,
    )
    if effective_calibration:
        class_counts = np.bincount(prepared.y_train, minlength=2)
        if int(class_counts.min()) < 3:
            raise ValueError(
                "calibration needs at least three training examples from each class"
            )

    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version(),
        "software": runtime_manifest(),
        "dataset": {
            "name": dataset.name,
            "rows": int(len(dataset.y)),
            "features": int(dataset.X.shape[1]),
            "positive_label": dataset.positive_label,
            "negative_label": dataset.negative_label,
            "fingerprint": dataset_fingerprint(dataset),
            "provenance": dataset.provenance,
            "task_profile": dataset.task_profile,
            "sites": sorted(set(dataset.sites.tolist())) if dataset.sites is not None else [],
            "subgroups": sorted(dataset.subgroups),
        },
        "split": {
            "strategy": (
                "site_holdout"
                if holdout_site is not None
                else "group_and_chronological"
                if dataset.groups is not None and _has_temporal_variation(dataset.times)
                else "group"
                if dataset.groups is not None
                else "chronological"
                if dataset.times is not None
                else "stratified_random"
            ),
            "holdout_site": holdout_site,
            "train_rows": int(len(prepared.y_train)),
            "train_pool_rows": int(
                len(prepared.y_train)
                + (len(prepared.y_validation) if prepared.y_validation is not None else 0)
            ),
            "validation_rows": int(
                len(prepared.y_validation) if prepared.y_validation is not None else 0
            ),
            "test_rows": int(len(prepared.y_test)),
            "test_size": test_size,
            "validation_size": validation_size,
            "seed": seed,
        },
        "preprocessing": {
            "input_features": list(dataset.feature_names),
            "imputation": "training median",
            "standardization": "training mean and standard deviation",
            "feature_selection": (
                "ANOVA F-test fit on training split"
                if reduction == "anova"
                else "PCA fit on training split"
            ),
            "angle_range": [-float(np.pi / 2), float(np.pi / 2)],
            "selected_features": prepared.selected_features,
            "qubits": n_qubits,
            "fit_rows": int(len(prepared.y_train)),
        },
        "execution": {
            "backend_mode": backend,
            "shots": shots,
            "aer_noise": aer_noise,
            "requested_models": model_names,
            "pegasos_steps": pegasos_steps,
            "vqc_maxiter": vqc_maxiter,
            "threshold_policy": threshold_policy,
            "target_sensitivity": target_sensitivity,
            "abstain_margin": abstain_margin,
            "bootstrap_samples": bootstrap_samples,
            "reduction": reduction,
            "model_parameters": model_params or {},
            "calibration": {
                "requested": calibrate,
                "enabled": effective_calibration,
                "automatic": effective_calibration and not calibrate,
                "automatic_reason": (
                    "threshold or abstention policy requires probabilities"
                    if effective_calibration and not calibrate
                    else None
                ),
                "method": "sigmoid calibration" if effective_calibration else None,
                "folds": 3 if effective_calibration else None,
                "per_model": {},
            },
        },
        "hardware_probe": (
            context.probe if context is not None else {"status": "not_requested"}
        ),
        "models": {},
    }
    if context is not None:
        results["execution"]["resolved_backend"] = context.backend_name
        results["execution"]["backend_qubits"] = context.backend_qubits

    for model_name in model_names:
        started = time.perf_counter()
        (
            model,
            feature_space,
            X_test,
            X_validation,
            reference,
            calibration_strategy,
        ) = _fit_model(
            model_name,
            prepared,
            context,
            seed,
            pegasos_steps,
            vqc_maxiter,
            effective_calibration,
            model_parameters=(model_params or {}).get(model_name),
        )
        probability_score = effective_calibration or model_name in {
            "logistic_regression",
            "hist_gradient_boosting",
        }
        score = _scores(model, X_test, prefer_probability=probability_score)
        threshold_info: dict[str, Any]
        if threshold_policy == "default":
            threshold = 0.5 if probability_score and score is not None else None
            if threshold is None:
                y_pred_raw = np.asarray(model.predict(X_test), dtype=int).reshape(-1)
                threshold_info = {
                    "policy": "model_default",
                    "threshold": None,
                    "validation_rows": 0,
                }
            else:
                assert score is not None
                y_pred_raw = np.asarray(score >= threshold, dtype=int).reshape(-1)
                threshold_info = {
                    "policy": threshold_policy,
                    "threshold": threshold,
                    "validation_rows": 0,
                }
                if X_validation is not None and prepared.y_validation is not None:
                    validation_score = _scores(
                        model,
                        X_validation,
                        prefer_probability=True,
                    )
                    if validation_score is not None:
                        threshold_info = select_threshold(
                            prepared.y_validation,
                            validation_score,
                            policy="default",
                        )
        else:
            if X_validation is None or prepared.y_validation is None:
                raise ValueError("a validation split is required for threshold selection")
            validation_score = _scores(model, X_validation, prefer_probability=True)
            if validation_score is None:
                raise ValueError(
                    f"{model_name} did not provide probabilities for threshold selection"
                )
            threshold_info = select_threshold(
                prepared.y_validation,
                validation_score,
                policy=threshold_policy,
                target_sensitivity=target_sensitivity,
            )
            threshold = float(threshold_info["threshold"])
            if score is None:
                raise ValueError(f"{model_name} did not provide test probabilities")
            assert score is not None
            y_pred_raw = np.asarray(score >= threshold, dtype=int).reshape(-1)

        margin = abstain_margin
        if margin is not None:
            if threshold is None or score is None or not probability_score:
                raise ValueError("abstention requires probability scores and a numeric threshold")
            assert score is not None
            abstained = np.abs(score - threshold) < margin
        else:
            abstained = np.zeros(len(X_test), dtype=bool)
        covered = ~abstained
        evaluation_y = prepared.y_test[covered]
        evaluation_pred = y_pred_raw[covered]
        evaluation_score = score[covered] if score is not None else None
        metrics = classification_metrics(
            evaluation_y,
            evaluation_pred,
            evaluation_score,
            probability_score=probability_score,
        )
        model_explanation: dict[str, Any]
        if not explain:
            model_explanation = {"status": "disabled"}
        elif backend == "ibm" and not allow_remote_explanations:
            model_explanation = {
                "status": "skipped",
                "reason": "remote sensitivity would submit repeated hardware jobs",
            }
        else:
            model_explanation = explain_raw_inputs(
                model,
                prepared.preprocessor,
                prepared.X_test_raw,
                feature_space,
                row_ids=prepared.test_row_ids,
            )

        subgroup_report = subgroup_metrics(
            evaluation_y,
            evaluation_pred,
            evaluation_score,
            {
                name: values[covered]
                for name, values in prepared.test_subgroups.items()
            },
            probability_score=probability_score,
        )
        resource_report = {
            "training_rows": int(len(prepared.y_train)),
            "test_rows": int(len(prepared.y_test)),
            "feature_count": int(prepared.X_train_classical.shape[1]),
            "qubits": n_qubits if feature_space == "quantum" else None,
            "shots": shots if feature_space == "quantum" else None,
            "backend": context.backend_name if context is not None else None,
            "estimated_kernel_pairs": (
                int(
                    len(prepared.y_train) * (len(prepared.y_train) + 1) // 2
                    + len(prepared.y_train) * len(prepared.y_test)
                )
                if feature_space == "quantum"
                else None
            ),
            "circuit_probe": dict(context.probe) if context is not None else None,
        }
        prediction_rows = [
            {
                "row_id": str(prepared.test_row_ids[index])
                if prepared.test_row_ids is not None
                else f"row-{index + 1}",
                "prediction": int(y_pred_raw[index]),
                "abstained": bool(abstained[index]),
                "score": float(score[index]) if score is not None else None,
                "label": int(prepared.y_test[index]),
            }
            for index in range(len(y_pred_raw))
        ]

        results["models"][model_name] = {
            "feature_space": feature_space,
            "parameters": dict((model_params or {}).get(model_name, {})),
            "calibration_strategy": calibration_strategy,
            "metrics": metrics,
            "clinical_evaluation": {
                "calibration_curve": calibration_bins(
                    evaluation_y,
                    evaluation_score if probability_score else None,
                ),
                "decision_curve": decision_curve(
                    evaluation_y,
                    evaluation_score if probability_score else None,
                ),
                "subgroups": subgroup_report,
            },
            "resource": resource_report,
            "confidence_intervals": bootstrap_confidence_intervals(
                evaluation_y,
                evaluation_pred,
                evaluation_score,
                probability_score,
                bootstrap_samples,
                seed,
            ),
            "threshold": threshold_info,
            "abstention": {
                "enabled": abstain_margin is not None,
                "margin": abstain_margin,
                "rule": "abstain when absolute probability distance from threshold is below margin"
                if abstain_margin is not None
                else None,
                "total_rows": int(len(X_test)),
                "abstained_rows": int(np.sum(abstained)),
                "evaluated_rows": int(np.sum(covered)),
                "coverage": float(np.mean(covered)) if len(covered) else None,
            },
            "score_type": "probability" if probability_score else "decision",
            "predictions": y_pred_raw.tolist(),
            "abstained": abstained.tolist(),
            "scores": score.tolist() if score is not None else None,
            "prediction_rows": prediction_rows,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "explanation": model_explanation,
        }
        results["execution"]["calibration"]["per_model"][model_name] = calibration_strategy
        if model_artifact_path is not None:
            artifact = SavedModelArtifact(
                schema_version=SCHEMA_VERSION,
                package_version=results["package_version"],
                model_name=model_name,
                model=model,
                preprocessor=prepared.preprocessor,
                feature_space=feature_space,
                selected_features=list(prepared.selected_features),
                threshold=threshold,
                threshold_policy=str(threshold_info["policy"]),
                abstain_margin=abstain_margin,
                probability_score=probability_score,
                calibration=dict(results["execution"]["calibration"]),
                dataset=dict(results["dataset"]),
                execution=dict(results["execution"]),
                hardware_probe=dict(results["hardware_probe"]),
                software=dict(results["software"]),
            )
            save_model_artifact(artifact, model_artifact_path)
            results["model_artifact"] = {
                "path": str(model_artifact_path),
                "manifest_path": f"{model_artifact_path}.manifest.json",
                "model_name": model_name,
                "schema_version": SCHEMA_VERSION,
            }

    return results


def run_repeated_experiment(
    dataset: LoadedDataset,
    repeats: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run fixed-model experiments over consecutive, reproducible seeds."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    run_kwargs = dict(kwargs)
    if repeats > 1 and run_kwargs.get("model_artifact_path") is not None:
        raise ValueError("saving a model artifact is only supported for one run")
    run_kwargs["models"] = tuple(run_kwargs.get("models", ("classical", "qsvc")))
    first_seed = int(run_kwargs.get("seed", 7))
    runs: list[dict[str, Any]] = []
    for index in range(repeats):
        current = dict(run_kwargs)
        current["seed"] = first_seed + index
        if index:
            # Explanations are useful on one run; repeating hardware jobs for
            # perturbation explanations would obscure the model comparison.
            current["explain"] = False
        runs.append(run_experiment(dataset, **current))

    if repeats == 1:
        return runs[0]

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
    )
    summary: dict[str, Any] = {}
    for model_name in runs[0]["models"]:
        summary[model_name] = {}
        for metric_name in metric_names:
            values = [
                run["models"][model_name]["metrics"][metric_name]
                for run in runs
                if run["models"][model_name]["metrics"][metric_name] is not None
            ]
            summary[model_name][metric_name] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "n": len(values),
            }
        coverage_values = [
            run["models"][model_name]["abstention"]["coverage"]
            for run in runs
            if run["models"][model_name]["abstention"]["coverage"] is not None
        ]
        summary[model_name]["coverage"] = {
            "mean": float(np.mean(coverage_values)) if coverage_values else None,
            "std": float(np.std(coverage_values)) if coverage_values else None,
            "n": len(coverage_values),
        }

    result = dict(runs[0])
    result["repeated_evaluation"] = {
        "repeats": repeats,
        "seeds": [run["split"]["seed"] for run in runs],
        "metric_summary": summary,
        "runs": [
            {
                "seed": run["split"]["seed"],
                "models": {
                    name: {
                        "metrics": model["metrics"],
                        "elapsed_seconds": model["elapsed_seconds"],
                    }
                    for name, model in run["models"].items()
                },
            }
            for run in runs
        ],
    }
    return result


def write_results(results: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
