"""`SourceSpec` - the declared, versioned, human-authored contract for one
data source (design.md §6, spec.md FR-002, FR-019, FR-020, FR-132).

Extends today's `protocol.EarlyDetectionProfile` rather than replacing it:
every field that profile already has keeps its name and meaning, so the
seven profiles already committed under `backend/profiles/` continue to
load unmodified (T003's done-condition) - `SourceSpec.load()` accepts them
directly. Every field this schema adds beyond that is optional with a
documented default, gated by `spec_version`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

SPEC_VERSION = 2

Modality = Literal["tabular", "ecg", "eeg", "gait", "ct", "mr", "angio", "genomic", "image2d", "imaging"]
"""`image2d` is NOT part of the team-head spec (spec.md/design.md name
only the other eight) - it was added to this build to close a real gap:
plain PNG/JPEG image datasets (no DICOM/NIfTI header, no scanner
metadata - e.g. Kaggle-style `yes/`/`no/` folders of MRI photos) have no
adapter anywhere in the formal spec. See `adapters/image_2d.py` and
`PIPELINE_STATUS.md` for the full caveat.

`imaging` is a third addition: a real, committed profile
(`profiles/p1_stroke_isles_imaging.json`, from origin/main) predates
spec_version 2's granular ct/mr/angio split and uses the older, generic
`protocol.EarlyDetectionProfile`-era term. Rather than editing a
teammate's real profile content to fit this package's taxonomy, `imaging`
is accepted as a valid but unspecific modality - `dicom_series`'s
header-modality-mismatch check (FR-... modality checked, not assumed)
simply has nothing to compare against for it and is skipped, which is
graceful degradation, not a silent wrong answer."""
TemporalFraming = Literal["prediction", "detection", "characterisation", "screening"]


class SpecValidationError(ValueError):
    """Raised by `SourceSpec.validate()`. Always names the offending field
    (FR-019) - never a bare `ValueError` a caller has to parse text to act
    on."""

    def __init__(self, message: str, *, field_name: str | None = None):
        super().__init__(message)
        self.field_name = field_name


@dataclass(frozen=True)
class ColumnSpec:
    """One column's declared handling. Every field optional; an
    undeclared column gets the specification's defaults."""

    units: str | None = None
    convert_to: str | None = None
    plausible_range: tuple[float, float] | None = None
    value_ceiling: float | None = None
    censor_tokens: tuple[str, ...] = (">", "<", "≥", "≤")
    on_implausible: Literal["quarantine", "clip", "nan", "error"] = "quarantine"
    transform: str | None = None  # e.g. "hour_of_day" - required for high-cardinality strings
    type_override: Literal["categorical", "numeric"] | None = None
    unseen_policy: Literal["indicator", "zeros", "reject"] = "indicator"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ColumnSpec":
        plausible = raw.get("plausible_range")
        return cls(
            units=raw.get("units"),
            convert_to=raw.get("convert_to"),
            plausible_range=tuple(plausible) if plausible else None,  # type: ignore[arg-type]
            value_ceiling=raw.get("value_ceiling"),
            censor_tokens=tuple(raw.get("censor_tokens", (">", "<", "≥", "≤"))),
            on_implausible=raw.get("on_implausible", "quarantine"),
            transform=raw.get("transform"),
            type_override=raw.get("type"),
            unseen_policy=raw.get("unseen_policy", "indicator"),
        )


@dataclass(frozen=True)
class SplitSpec:
    strategy: Literal[
        "grouped", "chronological", "grouped_chronological", "site_holdout", "predeclared_folds"
    ] = "grouped"
    test_size: float = 0.25
    validation_size: float = 0.2
    seed: int = 7
    holdout_site: str | None = None
    folds_column: str | None = None
    predeclared: dict[str, list[Any]] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SplitSpec":
        return cls(
            strategy=raw.get("strategy", "grouped"),
            test_size=raw.get("test_size", 0.25),
            validation_size=raw.get("validation_size", 0.2),
            seed=raw.get("seed", 7),
            holdout_site=raw.get("holdout_site"),
            folds_column=raw.get("folds_column"),
            predeclared=raw.get("predeclared"),
        )


@dataclass(frozen=True)
class SelectionSpec:
    reduction: Literal["anova", "pca", "mutual_info"] = "anova"
    stability_enabled: bool = False
    stability_folds: int = 5
    stability_min_selected_in: int = 4
    angle_scale: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SelectionSpec":
        stability = raw.get("stability", {}) or {}
        return cls(
            reduction=raw.get("reduction", "anova"),
            stability_enabled=stability.get("enabled", False),
            stability_folds=stability.get("folds", 5),
            stability_min_selected_in=stability.get("min_selected_in", 4),
            angle_scale=raw.get("angle_scale", 1.0),
        )


@dataclass(frozen=True)
class SourceLocation:
    root: str = ""
    pattern: str = ""
    archive_member: str | None = None
    url: str | None = None

    def resolved_pattern(self, *, fallback_dir: Path | None = None) -> str:
        """Env-var expanded, joined with root. Never an absolute host path
        in a committed spec (FR-132) - only `${VAR}`-rooted or repo-relative.

        `fallback_dir` is provenance, not a declared field: when `root` is
        empty and `pattern` is relative, a v1-migrated profile's dataset
        path resolves against the directory the spec itself was loaded
        from, matching how `protocol.resolve_profile_dataset` already
        behaves for the seven existing profiles."""

        root = os.path.expandvars(self.root) if self.root else ""
        if root:
            return str(Path(root) / self.pattern)
        if fallback_dir is not None and self.pattern and not Path(self.pattern).is_absolute():
            return str((fallback_dir / self.pattern).resolve())
        return self.pattern


@dataclass
class SourceSpec:
    """See design.md §6.1 for the full annotated schema this mirrors."""

    spec_version: int
    name: str
    modality: Modality
    target_column: str
    positive_label: str | None = None
    id_column: str | None = None
    group_column: str | None = None
    index_time_column: str | None = None
    outcome_time_column: str | None = None
    site_column: str | None = None
    horizon_days: int | None = None
    outcome_definition: str = ""
    temporal_framing: TemporalFraming = "detection"
    leakage_columns: tuple[str, ...] = ()
    subgroup_columns: tuple[str, ...] = ()
    adapter: str | None = None
    source: SourceLocation = field(default_factory=SourceLocation)
    unknown_column_policy: Literal["ignore", "error"] = "ignore"
    missing_sentinels: tuple[str, ...] = (
        "na", "n/a", "nan", "none", "null", "missing", "-", "?", "unknown", ".",
    )
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    quarantine_threshold: float = 0.05
    max_categories: int = 50
    missing_indicator: bool = False
    min_required_coverage: float = 1.0
    required_fields: tuple[str, ...] = ()
    label_exclusions: dict[str, bool] = field(
        default_factory=lambda: {"censored_before_horizon": True, "competing_risk": True, "qc_failed": True}
    )
    split: SplitSpec = field(default_factory=SplitSpec)
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    reduction: str = "anova"  # kept for EarlyDetectionProfile compatibility; mirrors selection.reduction
    n_qubits: int = 6

    # Raw dict retained for round-tripping fields this dataclass doesn't
    # model yet (imaging/signal blocks) without losing them.
    _raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _loaded_from_dir: Path | None = field(default=None, repr=False, compare=False)
    """Provenance only, set by `load()` - the directory the spec file
    itself came from, used as the base for resolving a relative
    `source.pattern` at read time when `source.root` is empty. Never
    validated by Rule 9 (it isn't a declared field), never serialized into
    a committed spec file."""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source_path: Path | None = None) -> "SourceSpec":
        raw = dict(raw)  # never mutate the caller's dict
        spec_version = int(raw.get("spec_version", 1))

        # Migration from spec_version 1 (protocol.EarlyDetectionProfile,
        # today's backend/profiles/*.json) to 2. Explicit and tested, per
        # design.md §6.3 rule 1 - not a silent field rename.
        if spec_version == 1:
            raw = _migrate_v1_to_v2(raw)
            spec_version = 2

        modality = raw.get("modality", "tabular")
        source_raw = raw.get("source")
        if source_raw is None:
            # v1 profiles declare a flat, relative `dataset_path` instead
            # (e.g. "../data/p1_stroke_clinical/..."). Kept as `pattern`
            # with an EMPTY root - never the loading file's absolute
            # directory (FR-132: no absolute host path in a spec, even one
            # synthesised internally). `Pipeline.read()` resolves a
            # relative pattern against the loaded-from directory at read
            # time instead, via `_loaded_from_dir`, which is provenance,
            # not a declared/validated field.
            source_raw = {"root": "", "pattern": raw.get("dataset_path", "")}
        source_loc = SourceLocation(
            root=source_raw.get("root", ""),
            pattern=source_raw.get("pattern", ""),
            archive_member=source_raw.get("archive_member"),
            url=source_raw.get("url"),
        )

        columns_raw = raw.get("columns", {}) or {}
        columns = {name: ColumnSpec.from_dict(spec) for name, spec in columns_raw.items()}

        selection_raw = raw.get("selection", {}) or {}
        reduction = raw.get("reduction") or selection_raw.get("reduction", "anova")
        if "reduction" not in selection_raw:
            selection_raw = {**selection_raw, "reduction": reduction}

        return cls(
            spec_version=spec_version,
            name=raw["name"],
            modality=modality,
            target_column=raw["target_column"],
            positive_label=(str(raw["positive_label"]) if raw.get("positive_label") is not None else None),
            id_column=raw.get("id_column"),
            group_column=raw.get("group_column"),
            index_time_column=raw.get("index_time_column"),
            outcome_time_column=raw.get("outcome_time_column"),
            site_column=raw.get("site_column"),
            horizon_days=(int(raw["horizon_days"]) if raw.get("horizon_days") is not None else None),
            outcome_definition=str(raw.get("outcome_definition", "")),
            temporal_framing=raw.get("temporal_framing", "detection"),
            leakage_columns=tuple(raw.get("leakage_columns", ())),
            subgroup_columns=tuple(raw.get("subgroup_columns", ())),
            adapter=raw.get("adapter"),
            source=source_loc,
            unknown_column_policy=raw.get("unknown_column_policy", "ignore"),
            missing_sentinels=tuple(
                s.lower() for s in raw.get(
                    "missing_sentinels",
                    ("na", "n/a", "nan", "none", "null", "missing", "-", "?", "unknown", "."),
                )
            ),
            columns=columns,
            quarantine_threshold=raw.get("quarantine_threshold", 0.05),
            max_categories=raw.get("max_categories", 50),
            missing_indicator=raw.get("missing_indicator", False),
            min_required_coverage=raw.get("min_required_coverage", 1.0),
            required_fields=tuple(raw.get("required_fields", ())),
            label_exclusions=raw.get(
                "label_exclusions",
                {"censored_before_horizon": True, "competing_risk": True, "qc_failed": True},
            ),
            split=SplitSpec.from_dict(raw.get("split", {}) or {}),
            selection=SelectionSpec.from_dict(selection_raw),
            reduction=reduction,
            n_qubits=raw.get("n_qubits", 6),
            _raw=raw,
            _loaded_from_dir=source_path.parent if source_path else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourceSpec":
        spec_path = Path(path)
        try:
            raw = json.loads(spec_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SpecValidationError(f"specification file not found: {spec_path}") from exc
        except json.JSONDecodeError as exc:
            raise SpecValidationError(f"specification is not valid JSON: {spec_path}") from exc
        if not isinstance(raw, dict):
            raise SpecValidationError("specification must be a JSON object")
        spec = cls.from_dict(raw, source_path=spec_path)
        spec.validate()
        return spec

    # ------------------------------------------------------------------
    # Validation - design.md §6.3, nine rules, checked before any data read
    # ------------------------------------------------------------------

    def validate(self) -> None:
        # Rule 1: spec_version known.
        if self.spec_version not in (1, 2):
            raise SpecValidationError(
                f"unknown spec_version {self.spec_version!r} (known: 1, 2)", field_name="spec_version"
            )

        # Rule 2: modality recognised (adapter existence checked by the
        # registry at dispatch time, not here, to avoid a spec.py <->
        # registry.py import cycle).
        known_modalities = {"tabular", "ecg", "eeg", "gait", "ct", "mr", "angio", "genomic", "image2d", "imaging"}
        if self.modality not in known_modalities:
            raise SpecValidationError(
                f"unknown modality {self.modality!r} (known: {sorted(known_modalities)})",
                field_name="modality",
            )

        # Rule 3: required-by-task fields for temporal_framing == "prediction".
        if self.temporal_framing == "prediction":
            missing = [
                name
                for name, value in (
                    ("horizon_days", self.horizon_days),
                    ("index_time_column", self.index_time_column),
                    ("outcome_time_column", self.outcome_time_column),
                    ("group_column", self.group_column),
                )
                if value is None
            ] + (["outcome_definition"] if not self.outcome_definition.strip() else [])
            if missing:
                raise SpecValidationError(
                    f"temporal_framing='prediction' requires {missing}, all missing/blank",
                    field_name=missing[0],
                )

        # Rule 4: leakage_columns disjoint from feature columns (required_fields
        # when declared; otherwise checked again at read time against the
        # actual header, per FR-026 - this is the load-time half of FR-020).
        if self.required_fields:
            overlap = set(self.leakage_columns) & set(self.required_fields)
            if overlap:
                raise SpecValidationError(
                    f"leakage_columns intersects required_fields: {sorted(overlap)}",
                    field_name="leakage_columns",
                )

        # Rule 5: target_column distinct from every administrative column.
        # id_column and group_column deliberately MAY coincide - §6.1's own
        # worked example uses `"id_column": "patient_id", "group_column":
        # "patient_id"` for the common one-row-per-patient case, so that
        # pair (and similarly id/index_time, id/site) is not an error here;
        # only the target column leaking into an administrative role is.
        for role, col in (
            ("id_column", self.id_column),
            ("group_column", self.group_column),
            ("index_time_column", self.index_time_column),
            ("site_column", self.site_column),
        ):
            if col is not None and col == self.target_column:
                raise SpecValidationError(
                    f"{role} must not be the same column as target_column ({col!r})",
                    field_name=role,
                )

        # Rule 6: imaging (not modelled by this dataclass beyond `_raw`
        # passthrough in Phase 0-1 tabular scope; validated when the
        # imaging block is present regardless).
        imaging = self._raw.get("imaging")
        if imaging:
            required_seq = set(imaging.get("required_sequences", ()))
            sequences = set(imaging.get("sequences", ()))
            if not required_seq <= sequences:
                raise SpecValidationError(
                    f"required_sequences {sorted(required_seq - sequences)} not in declared sequences",
                    field_name="imaging.required_sequences",
                )
            if self.modality == "ct" and not imaging.get("ct_windows"):
                raise SpecValidationError("modality 'ct' requires imaging.ct_windows", field_name="imaging.ct_windows")

        # Rule 7: signal.
        signal = self._raw.get("signal")
        if signal:
            if not signal.get("channels"):
                raise SpecValidationError("signal.channels must be non-empty", field_name="signal.channels")
            if signal.get("target_samples", 1) <= 0:
                raise SpecValidationError("signal.target_samples must be positive", field_name="signal.target_samples")
            if signal.get("resample_to_hz", 1) <= 0:
                raise SpecValidationError("signal.resample_to_hz must be positive", field_name="signal.resample_to_hz")

        # Rule 8: a column with cardinality that will exceed max_categories
        # needs a declared transform - checked properly at read time
        # (cardinality isn't knowable from the spec alone); here we only
        # catch the unconditional case: a column explicitly typed
        # categorical with no transform and no room to be exempted.
        # (Full enforcement lives in clean.py at fit time, per design.md §9.2.3.)

        # Rule 9: no absolute host path unless env-expanded.
        root = self.source.root
        if root and not root.startswith("${") and Path(root).is_absolute():
            raise SpecValidationError(
                f"source.root {root!r} is an absolute host path; use ${{QHEALTH_DATA_ROOT}}-style "
                "expansion or a repo-relative path",
                field_name="source.root",
            )

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}


_CENSOR_PATTERN = re.compile(r"^\s*[><≥≤]")


def _migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """`protocol.EarlyDetectionProfile` (spec_version absent or 1) -> the
    v2 shape. Every v1 field keeps its exact name; nothing here renames a
    field a v1 profile already uses (T003's done-condition depends on it).
    """

    migrated = dict(raw)
    migrated.setdefault("temporal_framing", "prediction" if raw.get("horizon_days") else "detection")
    migrated.setdefault("modality", "tabular")
    return migrated
