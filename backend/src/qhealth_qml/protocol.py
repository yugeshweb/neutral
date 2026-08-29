"""Data contracts for indication-specific early-detection studies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import LoadedDataset


@dataclass(frozen=True)
class EarlyDetectionProfile:
    """A small, explicit contract for turning rows into a prediction task."""

    name: str
    dataset_path: str
    target_column: str
    positive_label: str | None = None
    group_column: str | None = None
    index_time_column: str | None = None
    outcome_time_column: str | None = None
    site_column: str | None = None
    id_column: str | None = None
    horizon_days: int | None = None
    outcome_definition: str = ""
    leakage_columns: tuple[str, ...] = ()
    subgroup_columns: tuple[str, ...] = ()
    modality: str = "tabular"
    reduction: str = "pca"
    task_type: str = "early_detection"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_early_detection_profile(path: str | Path) -> tuple[EarlyDetectionProfile, Path]:
    """Read a JSON profile and return it with the profile directory."""

    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile is not valid JSON: {profile_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("early-detection profile must be a JSON object")

    missing = [name for name in ("name", "dataset_path", "target_column") if not raw.get(name)]
    if missing:
        raise ValueError(f"profile is missing required fields: {', '.join(missing)}")
    profile = EarlyDetectionProfile(
        name=str(raw["name"]),
        dataset_path=str(raw["dataset_path"]),
        target_column=str(raw["target_column"]),
        positive_label=(str(raw["positive_label"]) if raw.get("positive_label") is not None else None),
        group_column=(str(raw["group_column"]) if raw.get("group_column") else None),
        index_time_column=(
            str(raw["index_time_column"]) if raw.get("index_time_column") else None
        ),
        outcome_time_column=(
            str(raw["outcome_time_column"]) if raw.get("outcome_time_column") else None
        ),
        site_column=(str(raw["site_column"]) if raw.get("site_column") else None),
        id_column=(str(raw["id_column"]) if raw.get("id_column") else None),
        horizon_days=(int(raw["horizon_days"]) if raw.get("horizon_days") is not None else None),
        outcome_definition=str(raw.get("outcome_definition", "")),
        leakage_columns=tuple(str(value) for value in raw.get("leakage_columns", [])),
        subgroup_columns=tuple(str(value) for value in raw.get("subgroup_columns", [])),
        modality=str(raw.get("modality", "tabular")),
        reduction=str(raw.get("reduction", "pca")),
        task_type=str(raw.get("task_type", "early_detection")),
    )
    if profile.task_type != "early_detection":
        raise ValueError("profile task_type must be 'early_detection'")
    if profile.reduction not in {"anova", "pca"}:
        raise ValueError("profile reduction must be 'anova' or 'pca'")
    if profile.modality not in {"tabular", "gene_expression", "imaging_features", "ehr_numeric"}:
        raise ValueError(
            "profile modality must be tabular, gene_expression, imaging_features, or ehr_numeric"
        )
    if profile.horizon_days is not None and profile.horizon_days < 1:
        raise ValueError("profile horizon_days must be positive")
    if len(set(profile.subgroup_columns)) != len(profile.subgroup_columns):
        raise ValueError("profile subgroup_columns must be unique")
    return profile, profile_path.parent


def resolve_profile_dataset(profile: EarlyDetectionProfile, profile_directory: Path) -> Path:
    dataset_path = Path(profile.dataset_path)
    return dataset_path if dataset_path.is_absolute() else profile_directory / dataset_path


def _parse_time(value: Any, field: str, row_index: int) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty at dataset row {row_index}")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{field} at dataset row {row_index} must be ISO-8601 compatible"
        ) from exc


def validate_early_detection_profile(
    profile: EarlyDetectionProfile,
    dataset: "LoadedDataset",
) -> dict[str, Any]:
    """Validate the parts of an early-detection protocol visible in the rows."""

    errors: list[str] = []
    if not profile.group_column or dataset.groups is None:
        errors.append("group_column is required so repeated patient rows can be isolated")
    if not profile.index_time_column or dataset.times is None:
        errors.append("index_time_column is required for chronological evaluation")
    if not profile.outcome_time_column or dataset.outcome_times is None:
        errors.append("outcome_time_column is required to define the prediction horizon")
    if profile.horizon_days is None:
        errors.append("horizon_days is required")
    if not profile.outcome_definition.strip():
        errors.append("outcome_definition is required")
    leaked = sorted(set(profile.leakage_columns).intersection(dataset.feature_names))
    if leaked:
        errors.append(f"leakage_columns are present as model features: {', '.join(leaked)}")
    if profile.site_column and dataset.sites is None:
        errors.append("site_column is not present in the dataset")
    for name in profile.subgroup_columns:
        if name not in dataset.subgroups:
            errors.append(f"subgroup column {name!r} is not present in the dataset")

    temporal_checks: dict[str, int] = {}
    if not errors and profile.horizon_days is not None:
        assert dataset.times is not None
        assert dataset.outcome_times is not None
        within_horizon = 0
        outside_horizon = 0
        for row_index, (target, index_value, outcome_value) in enumerate(
            zip(dataset.y, dataset.times, dataset.outcome_times, strict=True),
            start=2,
        ):
            index_time = _parse_time(index_value, profile.index_time_column or "index time", row_index)
            outcome_text = "" if outcome_value is None else str(outcome_value).strip()
            if not outcome_text:
                if int(target) == 1:
                    errors.append(f"positive row {row_index} has no outcome time")
                continue
            outcome_time = _parse_time(
                outcome_text,
                profile.outcome_time_column or "outcome time",
                row_index,
            )
            delta_days = (outcome_time - index_time).total_seconds() / 86400
            if delta_days < 0:
                errors.append(f"outcome precedes index time at dataset row {row_index}")
            elif int(target) == 1 and delta_days <= profile.horizon_days:
                within_horizon += 1
            elif int(target) == 0 and delta_days <= profile.horizon_days:
                errors.append(
                    f"negative row {row_index} has an outcome inside the prediction horizon"
                )
            else:
                outside_horizon += 1
        temporal_checks = {
            "positive_outcomes_within_horizon": within_horizon,
            "outcomes_outside_horizon": outside_horizon,
        }

    if errors:
        raise ValueError("invalid early-detection profile: " + "; ".join(errors))
    return {
        "status": "valid",
        "task_type": profile.task_type,
        "horizon_days": profile.horizon_days,
        "outcome_definition": profile.outcome_definition,
        "checks": temporal_checks,
    }
