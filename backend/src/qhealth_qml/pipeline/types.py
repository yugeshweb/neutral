"""Canonical types for the unified ingestion & preprocessing pipeline.

Implements FR-005, FR-006, FR-094 from spec.md, and the type definitions in
design.md §5. This is Phase 0 (T002): pure data, no behaviour. Every
dataclass here is frozen (immutable) exactly where design.md says it is -
that immutability is not stylistic, it is what makes "a stateless stage
reads only the sample it's transforming" checkable rather than a promise.

Source of truth for shape: design.md §5. Where this file and that document
disagree, the document governs (spec.md's own stated authority rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Iterator, Literal

import numpy as np

Severity = Literal["info", "warn", "reject", "error"]
Modality = Literal["tabular", "ecg", "eeg", "gait", "ct", "mr", "angio", "genomic"]
ArrayKind = Literal["signal", "volume"]  # signal: (C,T)   volume: (C,D,H,W)

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
LABEL_EXCLUDE = -1
"""A real value, not a sentinel - censored / competing-risk / QC-failed
records carry this label explicitly (FR-094) rather than being silently
counted as negative, which is the single most common way a prognosis
benchmark quietly inflates (design.md §11.4)."""


@dataclass(frozen=True)
class Issue:
    """One machine-readable thing that happened. Never a bare string.

    `code` MUST be one of the stable catalogue in `issue_codes.py` (design.md
    §10) - tests assert on `code`, never on `message` text (FR-079).
    """

    code: str
    severity: Severity
    message: str  # exactly one human sentence
    field: str | None = None  # column / lead / sequence it concerns
    detail: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    """Where bytes come from. Resolving this is the adapter's job (FR-010),
    not the caller's - the caller never says which format or modality."""

    kind: Literal["path", "dir", "bytes", "stream", "zip_member", "url"]
    locator: str  # path, URL, or "archive.zip::member/path"
    payload: bytes | None = None  # only for kind == "bytes"
    media_hint: str | None = None  # optional MIME/format hint; never trusted alone


@dataclass(frozen=True)
class RawRecord:
    """Adapter-internal: decoded bytes before the declared spec is applied.
    Never crosses the adapter boundary - `harmonize()` consumes it and
    produces a `Sample`; nothing downstream of that sees a `RawRecord`."""

    source: Source
    payload: Any  # dict[str,str] | np.ndarray | pydicom dataset | ...
    meta: dict[str, Any] = dc_field(default_factory=dict)  # sampling rate, modality, zooms, slope/intercept


@dataclass(frozen=True)
class Sample:
    """The canonical mid-point. Identical shape whether it came from a CSV
    row, a FHIR bundle, a WFDB record, an EDF file, or a DICOM series -
    that uniformity is what lets every downstream stage (QC, encode,
    transform) be written once, not once per modality (FR-005)."""

    sample_id: str
    subject_id: str | None  # grouped splits key on THIS, never sample_id
    index_time: str | None  # ISO-8601
    outcome_time: str | None  # ISO-8601
    site: str | None
    label: int | None  # None at predict time; LABEL_EXCLUDE is a valid value
    fields: dict[str, Any]  # tabular/EHR, source units already applied, sentinels -> NaN
    arrays: dict[str, np.ndarray]  # {"signal": (C,T)} | {"volume": (C,D,H,W)}
    subgroups: dict[str, str]  # fairness slices
    provenance: dict[str, Any]  # files read, digests, transforms applied
    issues: list[Issue] = dc_field(default_factory=list)

    @property
    def scored(self) -> bool:
        """Excluded rows never reach a model (FR-095)."""
        return self.label != LABEL_EXCLUDE


@dataclass(frozen=True)
class Batch:
    """n >= 1. A single record is the n == 1 case, not a different type
    (FR-006) - there is deliberately no separate single-record code path."""

    samples: list[Sample]
    spec: Any = None  # SourceSpec; typed loosely here to avoid a circular import
    provenance: dict[str, Any] = dc_field(default_factory=dict)  # dataset-level: source digests, counts, timings
    issues: list[Issue] = dc_field(default_factory=list)  # dataset-level: quarantined columns, etc.

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)


@dataclass(frozen=True)
class QCVerdict:
    status: Literal["accept", "accept_with_flags", "reject"]
    issues: list[Issue] = dc_field(default_factory=list)

    @property
    def scorable(self) -> bool:
        return self.status != "reject"


@dataclass(frozen=True)
class PreparedArrays:
    """What the engine consumes. Mirrors `experiment.PreparedDataset`'s
    fields on purpose so a future Phase 4 rewire needs no shape change."""

    X_raw: np.ndarray  # (n, n_features) post-encode, pre-transform
    X_classical: np.ndarray  # (n, k) standardized + selected/reduced
    X_quantum: np.ndarray  # (n, k) angle-scaled to [-pi/2*s, +pi/2*s]
    y: np.ndarray | None  # (n,) int, EXCLUDE rows already dropped
    row_ids: np.ndarray
    subject_ids: np.ndarray | None
    sites: np.ndarray | None
    subgroups: dict[str, np.ndarray]
    feature_names: list[str]  # post-encode names, in order
    selected_features: list[str]  # post-selection names, in order


@dataclass(frozen=True)
class Ledger:
    """Everything that was decided, in a form a reviewer can audit and a
    test can diff (FR-125). `n_in == n_scored + n_excluded + n_rejected`
    always holds - that identity is asserted in tests, not just documented."""

    n_in: int
    n_scored: int
    n_excluded: int
    n_rejected: int
    excluded_by_reason: dict[str, int] = dc_field(default_factory=dict)
    rejected_by_code: dict[str, int] = dc_field(default_factory=dict)
    quarantined_columns: dict[str, str] = dc_field(default_factory=dict)
    dropped_columns: dict[str, str] = dc_field(default_factory=dict)
    missingness: dict[str, float] = dc_field(default_factory=dict)
    fingerprints: dict[str, str] = dc_field(default_factory=dict)
    timings_ms: dict[str, float] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.n_scored + self.n_excluded + self.n_rejected
        if total != self.n_in:
            raise ValueError(
                f"Ledger invariant violated: n_in={self.n_in} != "
                f"n_scored({self.n_scored}) + n_excluded({self.n_excluded}) + "
                f"n_rejected({self.n_rejected}) = {total}"
            )


@dataclass(frozen=True)
class FitResult:
    recipe: Any  # Recipe; typed loosely to avoid a circular import
    train: PreparedArrays
    validation: PreparedArrays | None
    test: PreparedArrays
    ledger: Ledger


class IssueCode:
    """The stable catalogue from design.md §10, referenced by name rather
    than string literal everywhere in this package (T005 done-condition).
    Adding a code: add it here, add its row to the table in design.md §10,
    never invent one inline at a call site."""

    SOURCE_UNREADABLE = "source_unreadable"
    FORMAT_AMBIGUOUS = "format_ambiguous"
    HEADER_MISSING = "header_missing"
    COLUMN_MISSING = "column_missing"
    COLUMN_DUPLICATED = "column_duplicated"
    COLUMN_BLANK_HEADER = "column_blank_header"
    RAGGED_ROW = "ragged_row"
    TARGET_NOT_BINARY = "target_not_binary"
    SENTINEL_TO_NAN = "sentinel_to_nan"
    SENTINEL_CLAMPED = "sentinel_clamped"
    IMPLAUSIBLE_VALUE = "implausible_value"
    COLUMN_QUARANTINED = "column_quarantined"
    COLUMN_CONSTANT = "column_constant"
    COLUMN_ALL_MISSING = "column_all_missing"
    UNIT_CONVERTED = "unit_converted"
    PROBABLE_UNIT_MISMATCH = "probable_unit_mismatch"
    HIGH_CARDINALITY = "high_cardinality"
    UNSEEN_CATEGORY = "unseen_category"
    REQUIRED_FIELD_MISSING = "required_field_missing"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    POPULATION_FILTER_UNMET = "population_filter_unmet"
    QUALITY_CONSTRAINT_UNMET = "quality_constraint_unmet"
    LABEL_EXCLUDED_CENSORED = "label_excluded_censored"
    LABEL_EXCLUDED_COMPETING_RISK = "label_excluded_competing_risk"
    CHANNEL_MISSING = "channel_missing"
    SAMPLING_RATE_MISMATCH = "sampling_rate_mismatch"
    FLATLINE_CHANNEL = "flatline_channel"
    SATURATION = "saturation"
    NAN_RUN = "nan_run"
    SEQUENCE_MISSING = "sequence_missing"
    MODALITY_MISMATCH = "modality_mismatch"
    ORIENTATION_UNKNOWN = "orientation_unknown"
    SPACING_MISSING = "spacing_missing"
    GRID_MISMATCH = "grid_mismatch"
    LOW_FOREGROUND = "low_foreground"
    NON_FINITE_INPUT = "non_finite_input"
    INPUT_OUT_OF_DISTRIBUTION = "input_out_of_distribution"
    RECIPE_VERSION_MISMATCH = "recipe_version_mismatch"
    CACHE_INVALIDATED = "cache_invalidated"
    LEAKAGE_SUSPECTED = "leakage_suspected"
    DUPLICATE_SAMPLE_ID = "duplicate_sample_id"


ISSUE_CATALOGUE: dict[str, Severity] = {
    IssueCode.SOURCE_UNREADABLE: "error",
    IssueCode.FORMAT_AMBIGUOUS: "error",
    IssueCode.HEADER_MISSING: "error",
    IssueCode.COLUMN_MISSING: "error",
    IssueCode.COLUMN_DUPLICATED: "warn",
    IssueCode.COLUMN_BLANK_HEADER: "warn",
    IssueCode.RAGGED_ROW: "warn",
    IssueCode.TARGET_NOT_BINARY: "error",
    IssueCode.SENTINEL_TO_NAN: "info",
    IssueCode.SENTINEL_CLAMPED: "warn",
    IssueCode.IMPLAUSIBLE_VALUE: "info",
    IssueCode.COLUMN_QUARANTINED: "warn",
    IssueCode.COLUMN_CONSTANT: "info",
    IssueCode.COLUMN_ALL_MISSING: "info",
    IssueCode.UNIT_CONVERTED: "info",
    IssueCode.PROBABLE_UNIT_MISMATCH: "warn",
    IssueCode.HIGH_CARDINALITY: "error",
    IssueCode.UNSEEN_CATEGORY: "warn",
    IssueCode.REQUIRED_FIELD_MISSING: "reject",
    IssueCode.INSUFFICIENT_COVERAGE: "reject",
    IssueCode.POPULATION_FILTER_UNMET: "reject",
    IssueCode.QUALITY_CONSTRAINT_UNMET: "reject",
    IssueCode.LABEL_EXCLUDED_CENSORED: "info",
    IssueCode.LABEL_EXCLUDED_COMPETING_RISK: "info",
    IssueCode.CHANNEL_MISSING: "reject",
    IssueCode.SAMPLING_RATE_MISMATCH: "reject",
    IssueCode.FLATLINE_CHANNEL: "warn",
    IssueCode.SATURATION: "warn",
    IssueCode.NAN_RUN: "reject",
    IssueCode.SEQUENCE_MISSING: "reject",
    IssueCode.MODALITY_MISMATCH: "reject",
    IssueCode.ORIENTATION_UNKNOWN: "warn",
    IssueCode.SPACING_MISSING: "warn",
    IssueCode.GRID_MISMATCH: "reject",
    IssueCode.LOW_FOREGROUND: "reject",
    IssueCode.NON_FINITE_INPUT: "reject",
    IssueCode.INPUT_OUT_OF_DISTRIBUTION: "warn",
    IssueCode.RECIPE_VERSION_MISMATCH: "error",
    IssueCode.CACHE_INVALIDATED: "info",
    IssueCode.LEAKAGE_SUSPECTED: "warn",
    IssueCode.DUPLICATE_SAMPLE_ID: "warn",
}
"""design.md §10, verbatim. `test_types.py` asserts this has no duplicate
codes and that every `IssueCode.*` constant appears here exactly once."""


@dataclass(frozen=True)
class RunResult:
    """`verdicts` has one entry per INPUT sample, in input order, including
    rejected ones. `arrays` contains only the scorable rows - re-associate
    through `arrays.row_ids`. A rejected sample never appears in `arrays`;
    that is the mechanism behind FR-082, not a convention callers must
    remember."""

    arrays: PreparedArrays
    verdicts: list[QCVerdict]
    ledger: Ledger
