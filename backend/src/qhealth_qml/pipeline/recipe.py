"""`Recipe` - the single serialized object that makes training and
prediction the same pipeline (design.md §7, spec.md FR-002, FR-030,
FR-031, FR-032, FR-074, FR-102, FR-104-FR-109).

`align()` (§7.1) is the piece that removes the platform's D1 defect: it is
the ONE function that turns a raw field dict into a feature vector, called
per-row by `transform()` for a batch and directly for a single record.
Because both paths call the identical function, single-record and batch
output cannot diverge by construction (FR-009, FR-074) - that is the
mechanism behind T3.4 (train ≡ predict), not a convention enforced by
testing alone.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import clean
from . import representation as repr_module
from .select import SelectionResult, leakage_suspicion_check, select_features, select_features_stable
from .spec import SourceSpec
from .types import Batch, Issue, IssueCode, PreparedArrays, Sample

RECIPE_SCHEMA_VERSION = 1


class RecipeVersionError(Exception):
    """FR-108: the running code refuses to load an artifact whose schema
    version differs, naming both."""


class RecipePredatesRecipeFormatError(Exception):
    """FR-109: an artifact from before the recipe format existed produces
    this specific error rather than a generic parse failure, and never
    partially loads."""


class ManifestMismatchError(Exception):
    """FR-107: a sidecar manifest that disagrees with its payload rejects
    both, rather than trusting either one alone."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


@dataclass
class Recipe:
    schema_version: int
    spec: SourceSpec
    code_version: str
    fitted_at: str

    # --- encode stage (frozen from the training split) ---
    feature_names: list[str] = field(default_factory=list)
    categorical_vocab: dict[str, list[str]] = field(default_factory=dict)
    unseen_policy: dict[str, str] = field(default_factory=dict)
    quarantined_columns: dict[str, str] = field(default_factory=dict)
    dropped_columns: dict[str, str] = field(default_factory=dict)
    censor_ceilings: dict[str, float] = field(default_factory=dict)

    # --- transform stage (sklearn objects, fitted on train-minus-validation) ---
    imputer: Any = None
    standardizer: Any = None
    selector: Any = None
    pca: Any = None
    angle_scaler: Any = None
    selected_features: list[str] = field(default_factory=list)
    n_qubits: int = 6
    angle_scale: float = 1.0
    reduction: str = "anova"
    selection_frequency: dict[str, float] | None = None

    # --- guards ---
    train_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    cohort_geometry: dict[str, Any] = field(default_factory=lambda: {"d5_exposure": "not_applicable", "reason": "non-imaging modality"})

    # --- operating point (owned by serving; carried here so one object survives) ---
    score_offset: float = 0.0
    score_scale: float = 1.0
    threshold: float | None = None
    threshold_policy: str = ""

    fit_issues: list[Issue] = field(default_factory=list)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, batch: Batch, y: np.ndarray) -> "Recipe":
        """Fits every stage of this recipe on `batch` (already sliced by
        the caller to train-minus-validation, per FR-091 - this method
        does not split). `y` must align 1:1 with `batch.samples`."""

        if len(batch) != len(y):
            raise ValueError(f"batch has {len(batch)} samples but y has {len(y)} labels")
        if len(batch) == 0:
            raise ValueError("cannot fit a recipe on an empty batch")

        issues: list[Issue] = []
        samples = batch.samples

        if self.spec.modality == "tabular":
            self._fit_tabular_feature_names(samples)
        else:
            self._fit_representation_feature_names(samples, issues)

        # Encode every training row through align() - the SAME function
        # predict-time single records go through (FR-009, FR-074).
        rows = []
        for s in samples:
            vec, align_issues = self.align(s)
            rows.append(vec)
            issues.extend(align_issues)
        X_raw = np.vstack(rows)
        self._fill_missing_indicators(X_raw)

        # Per-feature training statistics (FR-084), before imputation.
        train_stats: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.feature_names):
            col = X_raw[:, j]
            finite = col[np.isfinite(col)]
            train_stats[name] = {
                "min": float(finite.min()) if finite.size else float("nan"),
                "max": float(finite.max()) if finite.size else float("nan"),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": float(finite.std()) if finite.size else 0.0,
                "missing_frac": float(np.isnan(col).mean()),
            }
        self.train_stats = train_stats

        # --- transform stage ---
        self.imputer = SimpleImputer(strategy="median")
        self.standardizer = StandardScaler()
        X_std = self.standardizer.fit_transform(self.imputer.fit_transform(X_raw))

        groups = np.array([s.subject_id or s.sample_id for s in samples])
        selection = self.spec.selection
        if selection.stability_enabled:
            result = select_features_stable(
                X_std, y, self.feature_names, groups,
                n_qubits=self.n_qubits, reduction=selection.reduction,
                folds=selection.stability_folds, min_selected_in=selection.stability_min_selected_in,
            )
        else:
            result = select_features(X_std, y, self.feature_names, n_qubits=self.n_qubits, reduction=selection.reduction)
        issues.extend(result.issues)

        self.selector = result.selector
        self.pca = result.pca
        self.selected_features = result.selected_features
        self.selection_frequency = result.selection_frequency
        self.reduction = selection.reduction
        self.angle_scale = selection.angle_scale

        X_selected = self.pca.transform(X_std) if self.pca is not None else X_std[:, result.selected_indices]
        self.angle_scaler = MinMaxScaler(feature_range=(-math.pi / 2 * self.angle_scale, math.pi / 2 * self.angle_scale))
        self.angle_scaler.fit(X_selected)

        # Leakage-suspicion sniff (FR-093) - flags, never auto-drops.
        issues.extend(leakage_suspicion_check(X_selected, y, result.selected_features))
        for leak_col in self.spec.leakage_columns:
            if leak_col in result.selected_features:
                raise ValueError(
                    f"declared leakage_column '{leak_col}' appears in the fitted selector's "
                    f"support - FR-092 violation. It must be excluded before fitting, not after."
                )

        self.fit_issues = issues
        self.fitted_at = _now_iso()
        return self

    def _fit_tabular_feature_names(self, samples: list[Sample]) -> None:
        """Builds `feature_names`, `categorical_vocab`, `unseen_policy`,
        `quarantined_columns`, `dropped_columns` and `censor_ceilings` from
        the training batch's raw field dicts - everything `align()` needs
        to encode a tabular row by name. Column-level cleaning (quarantine,
        drop, cardinality guard) is inherently tabular; the representation
        path (`_fit_representation_feature_names`) has no analogue for it
        because a deterministic extractor's feature set is declared by the
        spec/extractor, not discovered from column contents."""

        all_columns = sorted({name for s in samples for name in s.fields if not name.endswith(("__censored", "__missing"))})

        # Categorical vs numeric: decided from the harmonized dtype (a
        # string value means the tabular adapter could not parse it as a
        # number and did not treat it as a sentinel/censored bound).
        categorical_vocab: dict[str, list[str]] = {}
        for name in all_columns:
            values = [s.fields.get(name) for s in samples]
            has_string = any(isinstance(v, str) for v in values)
            if has_string:
                categories = sorted({str(v) for v in values if isinstance(v, str)})
                categorical_vocab[name] = categories

        numeric_columns = [c for c in all_columns if c not in categorical_vocab]

        # Quarantine: aggregate the per-record signals harmonize() already
        # emitted (FR-037, FR-038) rather than re-deriving them from raw text.
        implausible_counts: dict[str, int] = {}
        present_counts: dict[str, int] = {}
        corruption_flagged: set[str] = set()
        for s in samples:
            for iss in s.issues:
                if iss.code == IssueCode.IMPLAUSIBLE_VALUE and iss.field:
                    implausible_counts[iss.field] = implausible_counts.get(iss.field, 0) + 1
                if iss.code == "thousands_separator_signature_candidate" and iss.field:
                    corruption_flagged.add(iss.field)
            for name in numeric_columns:
                if not _is_missing(s.fields.get(name)):
                    present_counts[name] = present_counts.get(name, 0) + 1

        quarantined: dict[str, str] = {}
        for name in corruption_flagged:
            quarantined[name] = "thousands-separator corruption signature"
        threshold = self.spec.quarantine_threshold
        for name, n_implausible in implausible_counts.items():
            if name in quarantined or name not in numeric_columns:
                continue
            total = present_counts.get(name, 0)
            if total > 0 and (n_implausible / total) > threshold:
                quarantined[name] = f"{n_implausible}/{total} ({n_implausible / total:.1%}) values outside plausible_range, exceeding {threshold:.1%}"

        # Constant / all-missing dropping (FR-040).
        dropped: dict[str, str] = {}
        for name in numeric_columns:
            if name in quarantined:
                continue
            values = [s.fields.get(name) for s in samples]
            present = [v for v in values if not _is_missing(v)]
            if not present:
                dropped[name] = "all values missing"
            elif len(set(present)) <= 1:
                dropped[name] = "constant (no variance)"
        for name, categories in list(categorical_vocab.items()):
            if len(categories) <= 1:
                dropped[name] = "constant (single category)"
                del categorical_vocab[name]

        # High-cardinality guard (FR-044).
        declared_transforms = {c for c, spec in self.spec.columns.items() if spec.transform}
        offenders = clean.high_cardinality_columns(
            categorical_vocab, len(samples), self.spec.max_categories, declared_transforms
        )
        if offenders:
            names = ", ".join(f"{k} ({v} categories)" for k, v in offenders.items())
            raise ValueError(
                f"column(s) exceed the cardinality guard with no declared transform: {names}. "
                f"Declare a `transform` for each in the specification, or drop the column."
            )

        # Build the canonical, ORDERED feature_names list.
        unseen_policy: dict[str, str] = {}
        censor_ceilings: dict[str, float] = {}
        feature_names: list[str] = []
        for name in numeric_columns:
            if name in quarantined or name in dropped:
                continue
            feature_names.append(name)
            has_censor_companion = any(f"{name}__censored" in s.fields for s in samples)
            if has_censor_companion:
                feature_names.append(f"{name}__censored")
                col_spec = self.spec.columns.get(name)
                if col_spec and col_spec.value_ceiling is not None:
                    censor_ceilings[name] = col_spec.value_ceiling
            if self.spec.missing_indicator:
                feature_names.append(f"{name}__missing")
        for name, categories in categorical_vocab.items():
            if name in dropped:
                continue
            policy = self.spec.columns.get(name).unseen_policy if self.spec.columns.get(name) else "indicator"
            unseen_policy[name] = policy
            for cat in categories:
                feature_names.append(f"{name}={cat}")
            if policy == "indicator":
                feature_names.append(f"{name}__unseen")

        if not feature_names:
            raise ValueError("no feature columns survived cleaning - every column was quarantined, dropped or excluded")

        self.feature_names = feature_names
        self.categorical_vocab = categorical_vocab
        self.unseen_policy = unseen_policy
        self.quarantined_columns = quarantined
        self.dropped_columns = dropped
        self.censor_ceilings = censor_ceilings

    def _fit_representation_feature_names(self, samples: list[Sample], issues: list[Issue]) -> None:
        """The non-tabular analogue of `_fit_tabular_feature_names`: calls
        the spec-declared deterministic extractor once per training sample
        to learn `feature_names` (design.md §9.2.7). A well-behaved
        deterministic extractor is a pure function of the DECLARED config
        (channels/sequences), not the data, so every sample must agree on
        the same name list - a disagreement means the extractor is not
        actually a pure function of its declared config, which would
        silently break train==predict identity if allowed through."""

        names_per_sample: list[list[str]] = []
        for s in samples:
            _, names = repr_module.extract(s, self.spec)
            names_per_sample.append(names)

        if not names_per_sample:
            raise ValueError("no samples to derive representation feature names from")
        first = names_per_sample[0]
        for other in names_per_sample[1:]:
            if other != first:
                raise ValueError(
                    "representation extractor returned different feature names for different "
                    "training samples - a deterministic extractor must be a pure function of "
                    "the declared config, not the data (design.md §9.2.7)"
                )
        if not first:
            raise ValueError("representation extractor returned zero features")

        self.feature_names = list(first)
        self.categorical_vocab = {}
        self.unseen_policy = {}
        self.quarantined_columns = {}
        self.dropped_columns = {}
        self.censor_ceilings = {}

    # ------------------------------------------------------------------
    # align - the single-record entry point (design.md §7.1)
    # ------------------------------------------------------------------

    def align(self, sample: Sample) -> tuple[np.ndarray, list[Issue]]:
        """Given ONE harmonized `Sample`, returns `(X_raw_row, issues)`.
        Branches once on modality, then defers to a modality-specific
        private method - both still funnel every train AND predict row
        through this single public entry point (FR-009, FR-074), which is
        what keeps single-record and batch, train and predict, from being
        able to diverge by construction."""

        if self.spec.modality == "tabular":
            return self._align_tabular(sample.fields)
        return self._align_representation(sample)

    def _align_tabular(self, fields: dict[str, Any]) -> tuple[np.ndarray, list[Issue]]:
        """Given a raw field dict from ONE record, returns
        `(X_raw_row, issues)`. Never reorders by position, never renames,
        never guesses - every column is addressed by name against
        `self.feature_names` (FR-021). A `required_field_missing` issue is
        emitted at most once per source column, however many dummy columns
        it expands into."""

        values = np.full(len(self.feature_names), np.nan, dtype=float)
        required = set(self.spec.required_fields)
        missing_required: dict[str, Issue] = {}

        for i, name in enumerate(self.feature_names):
            if "=" in name:
                col, _, category = name.partition("=")
                if col in self.categorical_vocab:
                    raw = fields.get(col)
                    if _is_missing(raw):
                        if col in required and col not in missing_required:
                            missing_required[col] = Issue(
                                IssueCode.REQUIRED_FIELD_MISSING, "reject",
                                f"Required field '{col}' is blank and will not be imputed.", field=col,
                            )
                        values[i] = 0.0
                    else:
                        values[i] = 1.0 if str(raw) == category else 0.0
                    continue
            if name.endswith("__unseen"):
                col = name[: -len("__unseen")]
                raw = fields.get(col)
                if _is_missing(raw):
                    values[i] = 0.0
                else:
                    values[i] = 0.0 if str(raw) in self.categorical_vocab.get(col, []) else 1.0
                continue
            if name.endswith("__censored"):
                raw = fields.get(name)
                values[i] = 0.0 if _is_missing(raw) else float(raw)
                continue
            if name.endswith("__missing"):
                values[i] = np.nan  # filled by the caller after the full row is known (see fit())
                continue

            # plain numeric column
            raw = fields.get(name)
            if _is_missing(raw):
                if name in required and name not in missing_required:
                    missing_required[name] = Issue(
                        IssueCode.REQUIRED_FIELD_MISSING, "reject",
                        f"Required field '{name}' is blank and will not be imputed.", field=name,
                    )
                continue  # stays NaN -> imputed at transform time (FR-031); NEVER zero-filled (FR-032)
            try:
                values[i] = float(raw)
            except (TypeError, ValueError):
                continue

        return values, list(missing_required.values())

    def _align_representation(self, sample: Sample) -> tuple[np.ndarray, list[Issue]]:
        """The non-tabular analogue of `_align_tabular`: calls the
        spec-declared extractor for this one sample, then reorders/
        validates its output by NAME against the frozen `feature_names`
        (never by position - FR-021 applies here exactly as it does for
        tabular columns)."""

        vector, names = repr_module.extract(sample, self.spec)
        by_name = dict(zip(names, vector))

        if not self.feature_names:
            # Called before fit() has set feature_names (e.g. from inside
            # _fit_representation_feature_names' own vector-computing call
            # in a future caller) - return as extracted, unordered.
            return np.asarray(vector, dtype=float), []

        values = np.full(len(self.feature_names), np.nan, dtype=float)
        missing: list[Issue] = []
        for i, name in enumerate(self.feature_names):
            if name in by_name:
                values[i] = by_name[name]
            else:
                missing.append(
                    Issue(
                        IssueCode.REQUIRED_FIELD_MISSING, "reject",
                        f"representation extractor did not produce declared feature '{name}' for this record.",
                        field=name,
                    )
                )
        return values, missing

    def _fill_missing_indicators(self, X_raw: np.ndarray) -> None:
        """`align()` leaves every `__missing` column as NaN (it has no way
        to know, per-name, which base column it companions without
        re-deriving `feature_names`' structure) - filled here, in-place,
        the ONE place both `fit()` and `transform()` call, so the two
        can't compute this differently (the exact divergence class FR-009
        exists to prevent)."""

        for j, name in enumerate(self.feature_names):
            if name.endswith("__missing"):
                base_idx = self.feature_names.index(name[: -len("__missing")])
                X_raw[:, j] = np.isnan(X_raw[:, base_idx]).astype(float)

    # ------------------------------------------------------------------
    # transform / fit_transform
    # ------------------------------------------------------------------

    def transform(self, batch: Batch) -> tuple[PreparedArrays, list[Issue]]:
        """`align()` applied per record and stacked (FR-009) - this is
        why single-record and batch output cannot diverge: they are the
        same function, called once per row either way."""

        rows = []
        all_issues: list[Issue] = []
        row_ids, subject_ids, sites = [], [], []
        for s in batch.samples:
            vec, issues = self.align(s)
            rows.append(vec)
            all_issues.extend(issues)
            row_ids.append(s.sample_id)
            subject_ids.append(s.subject_id or "")
            sites.append(s.site or "")

        X_raw = np.vstack(rows) if rows else np.zeros((0, len(self.feature_names)))
        self._fill_missing_indicators(X_raw)
        X_std = self.standardizer.transform(self.imputer.transform(X_raw))
        if self.pca is not None:
            X_selected = self.pca.transform(X_std)
        else:
            idx = [self.feature_names.index(f) for f in self.selected_features]
            X_selected = X_std[:, idx]
        X_quantum = self.angle_scaler.transform(X_selected)

        y = np.array([s.label if s.label is not None else -1 for s in batch.samples], dtype=int) if any(s.label is not None for s in batch.samples) else None

        arrays = PreparedArrays(
            X_raw=X_raw, X_classical=X_selected, X_quantum=X_quantum, y=y,
            row_ids=np.array(row_ids), subject_ids=np.array(subject_ids) if subject_ids else None,
            sites=np.array(sites) if sites else None, subgroups={},
            feature_names=list(self.feature_names), selected_features=list(self.selected_features),
        )
        return arrays, all_issues

    def fit_transform(self, batch: Batch, y: np.ndarray) -> tuple[PreparedArrays, list[Issue]]:
        self.fit(batch, y)
        return self.transform(batch)

    # ------------------------------------------------------------------
    # OOD guard (FR-084, FR-085)
    # ------------------------------------------------------------------

    def ood_report(self, X_raw: np.ndarray, *, ceiling: float = 0.2) -> list[Issue]:
        if X_raw.size == 0 or not self.train_stats:
            return []
        n_features = X_raw.shape[1]
        novel = np.zeros(X_raw.shape[0], dtype=bool)
        for j, name in enumerate(self.feature_names[:n_features]):
            stats = self.train_stats.get(name)
            if not stats or math.isnan(stats["std"]):
                continue
            lo = stats["min"] - 3 * stats["std"]
            hi = stats["max"] + 3 * stats["std"]
            col = X_raw[:, j]
            novel |= (col < lo) | (col > hi)
        novelty_fraction = float(novel.mean()) if X_raw.shape[0] else 0.0
        if novelty_fraction > ceiling:
            return [
                Issue(
                    IssueCode.INPUT_OUT_OF_DISTRIBUTION, "warn",
                    f"{novelty_fraction:.0%} of rows have at least one feature outside the "
                    f"training range +/- 3 std - a flag, not a rejection; this does not detect "
                    f"subtle domain shift.",
                    detail={"novelty_fraction": novelty_fraction},
                )
            ]
        return []

    # ------------------------------------------------------------------
    # fingerprint, save, load
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(
            {
                "spec": _spec_digest(self.spec),
                "feature_names": self.feature_names,
                "selected_features": self.selected_features,
                "n_qubits": self.n_qubits,
                "reduction": self.reduction,
                "angle_scale": self.angle_scale,
                "code_version": self.code_version,
            },
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "fitted_at": self.fitted_at,
            "spec_name": self.spec.name,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "n_qubits": self.n_qubits,
            "reduction": self.reduction,
            "angle_scale": self.angle_scale,
            "categorical_vocab_sizes": {k: len(v) for k, v in self.categorical_vocab.items()},
            "quarantined_columns": self.quarantined_columns,
            "dropped_columns": self.dropped_columns,
            "selection_frequency": self.selection_frequency,
            "cohort_geometry": self.cohort_geometry,
            "train_stats": self.train_stats,
            "fingerprint": self.fingerprint(),
        }

    def save(self, path: str | Path) -> Path:
        recipe_path = Path(path)
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        with recipe_path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        manifest_path = recipe_path.with_suffix(recipe_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(self.manifest(), indent=2, default=str), encoding="utf-8")
        return recipe_path

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        recipe_path = Path(path)
        manifest_path = recipe_path.with_suffix(recipe_path.suffix + ".manifest.json")

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_version = manifest.get("schema_version")
            if manifest_version != RECIPE_SCHEMA_VERSION:
                raise RecipeVersionError(
                    f"recipe manifest declares schema_version={manifest_version}, running code "
                    f"expects {RECIPE_SCHEMA_VERSION}"
                )
        # FR-106: an absent manifest falls through to the post-deserialization
        # check below rather than being treated as valid on its own.

        if not recipe_path.exists():
            raise FileNotFoundError(f"recipe not found: {recipe_path}")
        with recipe_path.open("rb") as fh:
            try:
                obj = pickle.load(fh)
            except Exception as exc:
                raise RecipePredatesRecipeFormatError(
                    f"could not deserialize {recipe_path} as a Recipe - it may predate the "
                    f"recipe format entirely and requires refitting."
                ) from exc

        if not isinstance(obj, Recipe):
            raise RecipePredatesRecipeFormatError(
                f"{recipe_path} does not contain a Recipe object - it predates the recipe "
                f"format and requires refitting; it will not be partially loaded."
            )
        if obj.schema_version != RECIPE_SCHEMA_VERSION:
            raise RecipeVersionError(
                f"recipe at {recipe_path} has schema_version={obj.schema_version}, running "
                f"code expects {RECIPE_SCHEMA_VERSION}"
            )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != obj.fingerprint():
                raise ManifestMismatchError(
                    f"sidecar manifest for {recipe_path} disagrees with its payload "
                    f"(fingerprint mismatch) - rejecting both rather than trusting either."
                )
        return obj


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec_digest(spec: SourceSpec) -> str:
    import hashlib

    payload = json.dumps(spec.as_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
