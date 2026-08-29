# P0 Platform Architecture

**Feature**: `001-neurological-conditions`
**Design version**: 1.0.0
**Date**: 2026-08-29
**Prerequisite records**: `../research/P0-platform-reuse-record.md`, `../research/P1-stroke-reuse-record.md`
**Implements**: FR-001 … FR-033 (platform half), Key Entities, SC-001 … SC-013

This document is implementation-complete: an engineer should be able to build P0 from it without
making further design decisions. It specifies the schema, the module layout, the routing algorithm,
the safety invariant, the exact registry files to author, and the full new/changed file list.

**Governing constraint**: P0 wraps `qhealth_qml.experiment` / `study` / `protocol`. It re-implements
none of it. Every executor call site is named explicitly in section 4.

---

## 1. Conventions

**Language mapping.** Python dataclasses live in `backend/src/qhealth_qml/platform/schema.py`.
TypeScript interfaces live in `src/lib/platform.ts`. **Field names are identical in both
languages** (`snake_case` on both sides — the TS side deliberately keeps snake_case so JSON crosses
the wire untransformed and no mapping layer exists).

| Python | TypeScript |
|---|---|
| `str` | `string` |
| `int`, `float` | `number` |
| `bool` | `boolean` |
| `list[T]` | `T[]` |
| `dict[str, T]` | `Record<string, T>` |
| `T \| None` | `T \| null` |
| `Literal[...]` | union of string literals |
| `dict[str, Any]` | `Record<string, unknown>` |

Optional-in-Python means `= None` with a `| None` type; the TS field is then `field: T \| null`
(present, nullable) — **not** `field?: T`. Absent keys are never used; the serializer always emits
every field. This makes "the model did not report calibration" (`null`) distinguishable from a
schema mismatch.

**Timestamps** are ISO-8601 UTC strings (`str` / `string`) everywhere. No `datetime` crosses the
boundary.

**Every dataclass** gets `to_dict()` (recursive, enum→str) and a module-level
`<Name>_from_dict(raw: dict) -> <Name>` that validates enums and required keys and raises
`SchemaError` with the field path. No validation library (P0 reuse record §2).

---

## 2. Enumerations

These are the normative status sets. Nothing may invent a status outside them.

```python
# FR-009 — per-model status for a case. Exactly these nine values.
ModelStatus = Literal[
    "compatible", "incompatible", "not available", "insufficient data",
    "ready", "running", "completed", "abstained", "failed",
]

# FR-017 — per-finding status. Exactly these six values.
FindingStatus = Literal[
    "not evaluated", "not available", "insufficient data",
    "negative", "positive", "abstained",
]
```

The spaces in the literals are intentional: they are the spec's exact strings and are used verbatim
in both the API payload and the UI, so there is no translation table to drift.

```python
Modality = Literal[
    "structured_clinical", "imaging", "signal", "biomarker",
    "genomic", "document", "derived_features",
]                                                   # FR-004

AssetFormat = Literal["csv", "json", "fhir", "dicom", "nifti", "edf", "npz"]   # FR-006

TaskType = Literal[
    "binary_classification", "multilabel_classification", "segmentation",
    "event_detection", "regression", "progression_risk",
]                                                   # FR-015

ReadinessTier   = Literal["high_reference", "research_only"]                   # FR-002
ReferenceTier   = Literal["high", "moderate", "low"]                           # Product Decision (1)
Availability    = Literal["available", "not available"]                        # FR-003
Lifecycle       = Literal["experimental", "operational_reference", "deprecated"]  # FR-040
ScoreType       = Literal[
    "calibrated_probability", "probability", "decision_function",
    "regression_value", "mask", "intervals", "none",
]
CalibrationStatus = Literal["calibrated", "uncalibrated", "not_assessed", "failed"]
UncertaintyKind   = Literal["calibrated_probability", "decision_margin", "bootstrap_ci", "none"]
ExplanationStatus = Literal["available", "unavailable", "not_applicable", "surrogate"]  # FR-020
AssetValidationStatus = Literal["accepted", "rejected", "missing", "quality_failed"]    # FR-008
RunStatus       = Literal["ready", "running", "completed", "failed"]
RunMode         = Literal["research", "demo"]                                  # FR-031
ReuseDecision   = Literal["reused", "adapted", "reproduced", "newly_authored"] # FR-037
RealGainDecision = Literal["passed", "failed", "not_assessed"]                 # FR-039
EvidenceKind    = Literal[
    "feature_contribution", "image_region", "signal_interval",
    "biomarker", "quality_flag", "reference_link",
]
Severity        = Literal["info", "warning", "error"]
```

TypeScript mirror (`src/lib/platform.ts`), same names, same members:

```ts
export type ModelStatus =
  | 'compatible' | 'incompatible' | 'not available' | 'insufficient data'
  | 'ready' | 'running' | 'completed' | 'abstained' | 'failed'

export type FindingStatus =
  | 'not evaluated' | 'not available' | 'insufficient data'
  | 'negative' | 'positive' | 'abstained'

export type Modality =
  | 'structured_clinical' | 'imaging' | 'signal' | 'biomarker'
  | 'genomic' | 'document' | 'derived_features'

export type AssetFormat = 'csv' | 'json' | 'fhir' | 'dicom' | 'nifti' | 'edf' | 'npz'
export type TaskType =
  | 'binary_classification' | 'multilabel_classification' | 'segmentation'
  | 'event_detection' | 'regression' | 'progression_risk'
export type ReadinessTier = 'high_reference' | 'research_only'
export type ReferenceTier = 'high' | 'moderate' | 'low'
export type Availability = 'available' | 'not available'
export type Lifecycle = 'experimental' | 'operational_reference' | 'deprecated'
export type ScoreType =
  | 'calibrated_probability' | 'probability' | 'decision_function'
  | 'regression_value' | 'mask' | 'intervals' | 'none'
export type CalibrationStatus = 'calibrated' | 'uncalibrated' | 'not_assessed' | 'failed'
export type UncertaintyKind = 'calibrated_probability' | 'decision_margin' | 'bootstrap_ci' | 'none'
export type ExplanationStatus = 'available' | 'unavailable' | 'not_applicable' | 'surrogate'
export type AssetValidationStatus = 'accepted' | 'rejected' | 'missing' | 'quality_failed'
export type RunStatus = 'ready' | 'running' | 'completed' | 'failed'
export type RunMode = 'research' | 'demo'
export type ReuseDecision = 'reused' | 'adapted' | 'reproduced' | 'newly_authored'
export type RealGainDecision = 'passed' | 'failed' | 'not_assessed'
export type EvidenceKind =
  | 'feature_contribution' | 'image_region' | 'signal_interval'
  | 'biomarker' | 'quality_flag' | 'reference_link'
export type Severity = 'info' | 'warning' | 'error'
```

---

## 3. Canonical schema

Eight Key Entities plus fourteen supporting structs. `PLATFORM_SCHEMA_VERSION = 1` is a module
constant emitted on every top-level payload.

### 3.1 Supporting structs

```python
@dataclass(frozen=True)
class Reference:
    label: str
    url: str
    citation: str | None = None

@dataclass(frozen=True)
class ValidationIssue:
    code: str                       # machine-readable, e.g. "unit_mismatch"
    severity: Severity
    message: str                    # MUST NOT contain a source value (FR-030)
    field: str | None = None        # canonical field name only
    asset_id: str | None = None

@dataclass(frozen=True)
class FieldMapping:                 # FR-007
    canonical_field: str
    source_field: str
    source_value: str | None
    code_system: str | None
    unit: str | None
    timestamp: str | None
    transform: str                  # "identity" | "one_hot" | "binary_map" | "rescale" | ...
    note: str | None = None

@dataclass(frozen=True)
class Visit:
    visit_id: str
    timestamp: str | None
    note: str | None = None

@dataclass(frozen=True)
class OptionalModalitySpec:
    modality: Modality
    trained_for_absence: bool       # Edge case: "not trained for missingness" -> must not run

@dataclass(frozen=True)
class InputContract:
    required_modalities: list[Modality]
    optional_modalities: list[OptionalModalitySpec]
    required_fields: list[str]      # canonical field names that must be present and valid
    min_rows: int                   # 1 for single-case inference
    population: str                 # free text, shown in the catalog (FR-003)
    population_filters: dict[str, str]   # canonical_field -> predicate, e.g. {"age": ">=18"}
    quality_constraints: dict[str, str]  # canonical_field / modality -> constraint text (FR-008)

@dataclass(frozen=True)
class PreprocessingSpec:            # FR-013 (classical half)
    imputation: str                 # "median"
    scaling: str                    # "standard"
    reduction: str                  # "anova" | "pca" | "none"
    n_components: int | None
    angle_scaling: str | None       # "minmax" for quantum feature spaces
    fitted_on: str                  # "train_partition_only" (FR-022)

@dataclass(frozen=True)
class QuantumSpec:                  # FR-013 (quantum half); None for classical-only models
    framework: str                  # "qiskit"
    encoding: str                   # "ZZFeatureMap"
    ansatz: str | None              # "real_amplitudes" | None
    circuit_version: str
    backend_mode: str               # "statevector" | "aer" | "fake" | "ibm"
    n_qubits: int
    shots: int | None

@dataclass(frozen=True)
class ArtifactRef:
    kind: str                       # "saved_model_artifact" | "container_image" | "none"
    path: str | None                # MUST resolve under the configured runtime dir (P0 record §4)
    manifest_path: str | None
    sha256: str | None
    schema_version: int | None      # SavedModelArtifact.schema_version

@dataclass(frozen=True)
class CalibrationSpec:
    method: str                     # "none" | "validation_sigmoid"
    status: CalibrationStatus
    note: str | None = None

@dataclass(frozen=True)
class ExplainabilitySpec:
    method: str                     # "input_sensitivity" | "none" | "planned:<name>"
    scope: str                      # "global" | "per_row" | "region" | "none"
    surrogate: bool                 # true => Finding.explanation_status == "surrogate" (FR-020)

@dataclass(frozen=True)
class SafetySpec:
    allows_negative_finding: bool   # false => this model may never emit "negative"
    abstain_margin: float | None
    requires_full_coverage: bool    # true => any missing required modality -> insufficient data
    disclaimer: str                 # FR-018
    limitations: list[str]

@dataclass(frozen=True)
class ReuseManifestEntry:           # FR-035
    component: str
    decision: ReuseDecision
    source_url: str | None
    release_or_commit: str | None
    paper_citation: str | None
    license: str | None
    license_verified: bool
    weight_source: str | None
    weight_sha256: str | None
    preprocessing_assumptions: str | None
    io_contract: str | None
    local_modifications: str | None

@dataclass(frozen=True)
class ModelCard:                    # FR-027
    intended_use: str
    excluded_use: str
    training_population: str
    evaluation_population: str
    label_policy: str
    data_sources: list[Reference]
    data_licenses: list[str]
    limitations: list[str]
    calibration_status: CalibrationStatus
    explanation_method: str
    maintainer: str
    reuse_manifest: list[ReuseManifestEntry]
    research_record: str            # repo-relative path to the FR-034 record

@dataclass(frozen=True)
class Uncertainty:
    kind: UncertaintyKind
    value: float | None
    lower: float | None
    upper: float | None
    calibration_status: CalibrationStatus

@dataclass(frozen=True)
class CoverageReport:
    required_present: int
    required_total: int
    optional_present: int
    optional_total: int
    coverage_ratio: float           # required_present / max(required_total, 1)
    missing: list[str]
    quality_failed: list[str]

@dataclass(frozen=True)
class LeakageCheck:                 # SC-008
    name: str                       # "subject_identity" | "repeated_acquisition" |
                                    # "site_grouping" | "training_only_preprocessing"
    status: str                     # "passed" | "failed" | "not_applicable"
    detail: str

@dataclass(frozen=True)
class StageEvent:
    at: str
    stage: str
    model_id: str | None
    level: Severity
    message: str

@dataclass(frozen=True)
class ResourceUsage:
    wall_seconds: float
    training_rows: int | None
    test_rows: int | None
    feature_count: int | None
    qubits: int | None
    shots: int | None
    backend: str | None
    estimated_kernel_pairs: int | None
```

```ts
export interface Reference { label: string; url: string; citation: string | null }

export interface ValidationIssue {
  code: string; severity: Severity; message: string
  field: string | null; asset_id: string | null
}

export interface FieldMapping {
  canonical_field: string; source_field: string; source_value: string | null
  code_system: string | null; unit: string | null; timestamp: string | null
  transform: string; note: string | null
}

export interface Visit { visit_id: string; timestamp: string | null; note: string | null }

export interface OptionalModalitySpec { modality: Modality; trained_for_absence: boolean }

export interface InputContract {
  required_modalities: Modality[]
  optional_modalities: OptionalModalitySpec[]
  required_fields: string[]
  min_rows: number
  population: string
  population_filters: Record<string, string>
  quality_constraints: Record<string, string>
}

export interface PreprocessingSpec {
  imputation: string; scaling: string; reduction: string
  n_components: number | null; angle_scaling: string | null; fitted_on: string
}

export interface QuantumSpec {
  framework: string; encoding: string; ansatz: string | null; circuit_version: string
  backend_mode: string; n_qubits: number; shots: number | null
}

export interface ArtifactRef {
  kind: string; path: string | null; manifest_path: string | null
  sha256: string | null; schema_version: number | null
}

export interface CalibrationSpec { method: string; status: CalibrationStatus; note: string | null }
export interface ExplainabilitySpec { method: string; scope: string; surrogate: boolean }

export interface SafetySpec {
  allows_negative_finding: boolean; abstain_margin: number | null
  requires_full_coverage: boolean; disclaimer: string; limitations: string[]
}

export interface ReuseManifestEntry {
  component: string; decision: ReuseDecision; source_url: string | null
  release_or_commit: string | null; paper_citation: string | null
  license: string | null; license_verified: boolean
  weight_source: string | null; weight_sha256: string | null
  preprocessing_assumptions: string | null; io_contract: string | null
  local_modifications: string | null
}

export interface ModelCard {
  intended_use: string; excluded_use: string
  training_population: string; evaluation_population: string; label_policy: string
  data_sources: Reference[]; data_licenses: string[]; limitations: string[]
  calibration_status: CalibrationStatus; explanation_method: string; maintainer: string
  reuse_manifest: ReuseManifestEntry[]; research_record: string
}

export interface Uncertainty {
  kind: UncertaintyKind; value: number | null
  lower: number | null; upper: number | null; calibration_status: CalibrationStatus
}

export interface CoverageReport {
  required_present: number; required_total: number
  optional_present: number; optional_total: number
  coverage_ratio: number; missing: string[]; quality_failed: string[]
}

export interface LeakageCheck { name: string; status: string; detail: string }
export interface StageEvent {
  at: string; stage: string; model_id: string | null; level: Severity; message: string
}
export interface ResourceUsage {
  wall_seconds: number; training_rows: number | null; test_rows: number | null
  feature_count: number | null; qubits: number | null; shots: number | null
  backend: string | null; estimated_kernel_pairs: number | null
}
```

### 3.2 ConditionDefinition (FR-003, FR-032, FR-033)

```python
@dataclass(frozen=True)
class ConditionDefinition:
    condition_id: str               # "neuro.stroke.ischemic.acute"
    name: str
    domain: str                     # "neurological"
    priority: str                   # "P1".."P6"
    task_type: TaskType
    readiness_tier: ReadinessTier
    reference_label_tier: ReferenceTier
    target_definition: str
    population: str
    required_modalities: list[Modality]
    optional_modalities: list[Modality]
    expected_output: str
    reference_datasets: list[Reference]
    evidence_links: list[Reference]
    limitations: list[str]
```

```ts
export interface ConditionDefinition {
  condition_id: string; name: string; domain: string; priority: string
  task_type: TaskType; readiness_tier: ReadinessTier; reference_label_tier: ReferenceTier
  target_definition: string; population: string
  required_modalities: Modality[]; optional_modalities: Modality[]
  expected_output: string
  reference_datasets: Reference[]; evidence_links: Reference[]; limitations: string[]
}
```

### 3.3 ModelDefinition (FR-013, FR-014, FR-026, FR-027, FR-037, FR-040)

```python
@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    condition_id: str
    version: str                    # semver
    display_name: str
    task_type: TaskType
    availability: Availability
    lifecycle: Lifecycle
    executor: str                   # "tabular_qml" | "imaging_segmentation" | "none"
    input_contract: InputContract
    dataset_profile_path: str | None   # repo-relative EarlyDetectionProfile JSON
    temporal_validation: str        # "validated" | "not applicable — cross-sectional" | "n/a"
    preprocessing: PreprocessingSpec
    quantum: QuantumSpec | None
    classical_baseline_model_id: str | None    # FR-014; None for a classical model
    artifact: ArtifactRef
    calibration: CalibrationSpec
    explainability: ExplainabilitySpec
    output_score_type: ScoreType
    evaluation_record_ids: list[str]           # MUST be non-empty when availability=="available"
    safety: SafetySpec
    model_card: ModelCard
```

```ts
export interface ModelDefinition {
  model_id: string; condition_id: string; version: string; display_name: string
  task_type: TaskType; availability: Availability; lifecycle: Lifecycle; executor: string
  input_contract: InputContract
  dataset_profile_path: string | null
  temporal_validation: string
  preprocessing: PreprocessingSpec
  quantum: QuantumSpec | null
  classical_baseline_model_id: string | null
  artifact: ArtifactRef
  calibration: CalibrationSpec
  explainability: ExplainabilitySpec
  output_score_type: ScoreType
  evaluation_record_ids: string[]
  safety: SafetySpec
  model_card: ModelCard
}
```

### 3.4 DataBundle (FR-004, FR-005, FR-030)

```python
@dataclass(frozen=True)
class DataBundle:
    bundle_id: str
    case_id: str                    # pseudonymous; "local-<uuid4>" when the source had none
    case_id_source: str             # "supplied" | "assigned_local"   (Edge case: no identifier)
    created_at: str
    source: str                     # "upload" | "fixture" | "demo"
    synthetic: bool                 # FR-031
    visits: list[Visit]
    assets: list[ModalityAsset]
    validation: list[ValidationIssue]
    content_hashes: dict[str, str]  # asset_id -> sha256
    provenance: dict[str, str]      # adapter name, adapter version, source system, ...
```

```ts
export interface DataBundle {
  bundle_id: string; case_id: string; case_id_source: string; created_at: string
  source: string; synthetic: boolean
  visits: Visit[]; assets: ModalityAsset[]; validation: ValidationIssue[]
  content_hashes: Record<string, string>; provenance: Record<string, string>
}
```

### 3.5 ModalityAsset (FR-006, FR-007, FR-008)

```python
@dataclass(frozen=True)
class ModalityAsset:
    asset_id: str
    bundle_id: str
    modality: Modality
    format: AssetFormat
    role: str                       # "clinical_table" | "ncct" | "ctp" | "eeg" | ...
    visit_id: str | None
    uri: str                        # local path or "memory:<asset_id>"; never a remote URL in V1
    content_hash: str               # sha256
    byte_size: int
    rows: int | None
    dimensions: list[int] | None    # e.g. [512, 512, 32]
    units: dict[str, str]           # canonical_field -> unit
    acquired_at: str | None
    validation_status: AssetValidationStatus
    validation_issues: list[ValidationIssue]
    field_mappings: list[FieldMapping]
    derived_from: list[str]         # asset_ids this asset was derived from
```

```ts
export interface ModalityAsset {
  asset_id: string; bundle_id: string; modality: Modality; format: AssetFormat; role: string
  visit_id: string | null; uri: string; content_hash: string; byte_size: number
  rows: number | null; dimensions: number[] | null; units: Record<string, string>
  acquired_at: string | null
  validation_status: AssetValidationStatus; validation_issues: ValidationIssue[]
  field_mappings: FieldMapping[]; derived_from: string[]
}
```

### 3.6 RoutingDecision + AssessmentRun (FR-009, FR-010, FR-012, FR-028)

```python
@dataclass(frozen=True)
class RoutingDecision:
    model_id: str
    model_version: str
    condition_id: str
    status: ModelStatus
    reason: str                     # required, non-empty, for every non-"ready" status
    satisfied_modalities: list[Modality]
    missing_required: list[Modality]
    missing_optional: list[Modality]
    unmet_constraints: list[str]

@dataclass(frozen=True)
class AssessmentRun:
    run_id: str
    bundle_id: str
    domain: str                     # "neurological"
    started_at: str
    completed_at: str | None
    status: RunStatus
    mode: RunMode
    synthetic: bool
    schema_version: int
    routing: list[RoutingDecision]
    findings: list[Finding]
    stage_events: list[StageEvent]
    resource: ResourceUsage
    fingerprints: dict[str, str]    # FR-028: model, dataset, preprocessing, package,
                                    # backend, input
    disclaimer: str
    errors: list[str]
```

```ts
export interface RoutingDecision {
  model_id: string; model_version: string; condition_id: string
  status: ModelStatus; reason: string
  satisfied_modalities: Modality[]; missing_required: Modality[]
  missing_optional: Modality[]; unmet_constraints: string[]
}

export interface AssessmentRun {
  run_id: string; bundle_id: string; domain: string
  started_at: string; completed_at: string | null
  status: RunStatus; mode: RunMode; synthetic: boolean; schema_version: number
  routing: RoutingDecision[]; findings: Finding[]; stage_events: StageEvent[]
  resource: ResourceUsage; fingerprints: Record<string, string>
  disclaimer: string; errors: string[]
}
```

### 3.7 Finding (FR-016, FR-017, FR-018, FR-020)

```python
@dataclass(frozen=True)
class Finding:
    finding_id: str
    run_id: str
    model_id: str
    model_version: str
    condition_id: str
    condition_name: str
    task_type: TaskType
    status: FindingStatus
    reason: str                     # required and non-empty for every non-positive/negative status
    score: float | None
    score_type: ScoreType
    output: dict[str, Any]          # task-shaped payload; {} when there is no output
    threshold: float | None
    threshold_policy: str | None
    uncertainty: Uncertainty
    abstained: bool
    reference_label_tier: ReferenceTier
    input_coverage: CoverageReport
    modalities_used: list[Modality]
    evidence: list[EvidenceItem]
    explanation_status: ExplanationStatus
    limitations: list[str]
    disclaimer: str
    synthetic: bool
```

`output` payload shape by `task_type` (the only permitted keys):

| task_type | `output` keys |
|---|---|
| `binary_classification` | `{"label": int, "positive_label": str}` |
| `multilabel_classification` | `{"labels": dict[str, float]}` |
| `segmentation` | `{"regions": list[dict], "volume_ml": float \| None, "mask_uri": str \| None}` |
| `event_detection` | `{"intervals": list[{"start_seconds": float, "end_seconds": float, "score": float}]}` |
| `regression` / `progression_risk` | `{"value": float, "unit": str \| None}` |

```ts
export interface Finding {
  finding_id: string; run_id: string; model_id: string; model_version: string
  condition_id: string; condition_name: string; task_type: TaskType
  status: FindingStatus; reason: string
  score: number | null; score_type: ScoreType; output: Record<string, unknown>
  threshold: number | null; threshold_policy: string | null
  uncertainty: Uncertainty; abstained: boolean
  reference_label_tier: ReferenceTier
  input_coverage: CoverageReport; modalities_used: Modality[]
  evidence: EvidenceItem[]; explanation_status: ExplanationStatus
  limitations: string[]; disclaimer: string; synthetic: boolean
}
```

### 3.8 EvidenceItem (FR-019)

```python
@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    finding_id: str
    kind: EvidenceKind
    label: str
    value: float | None
    unit: str | None
    region: dict[str, float] | None      # {"x","y","w","h"} or {"x","y","z","r"}, normalised 0..1
    interval: dict[str, float] | None    # {"start_seconds","end_seconds"}
    source_asset_id: str | None
    confidence: float | None
    note: str | None
```

```ts
export interface EvidenceItem {
  evidence_id: string; finding_id: string; kind: EvidenceKind; label: string
  value: number | null; unit: string | null
  region: Record<string, number> | null; interval: Record<string, number> | null
  source_asset_id: string | null; confidence: number | null; note: string | null
}
```

### 3.9 EvaluationRecord (FR-023, FR-024, FR-025, FR-036, SC-008)

```python
@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    model_id: str
    model_version: str
    condition_id: str
    created_at: str
    dataset_profile: dict[str, Any]      # EarlyDetectionProfile.as_dict()
    dataset_fingerprint: str             # experiment.dataset_fingerprint()
    split_strategy: str                  # e.g. "stratified_random"
    split_summary: dict[str, int]        # train_rows, validation_rows, test_rows, repeats
    preprocessing: PreprocessingSpec
    leakage_checks: list[LeakageCheck]
    metrics: dict[str, float | None]
    segmentation_metrics: dict[str, float | None]   # FR-024; {} when not a segmentation task
    event_metrics: dict[str, float | None]          # FR-024; {} when not an event task
    calibration: dict[str, Any]
    abstention: dict[str, Any]
    resource: ResourceUsage
    confidence_intervals: dict[str, dict[str, float]]
    baseline_model_id: str | None
    baseline_metrics: dict[str, float | None]
    paired_comparison: dict[str, Any]    # study.run_nested_evaluation() delta block
    real_gain_decision: RealGainDecision
    real_gain_reason: str
    software: dict[str, str]             # experiment.runtime_manifest()
    source_result_path: str              # raw run_repeated_experiment JSON on disk
```

```ts
export interface EvaluationRecord {
  evaluation_id: string; model_id: string; model_version: string; condition_id: string
  created_at: string
  dataset_profile: Record<string, unknown>; dataset_fingerprint: string
  split_strategy: string; split_summary: Record<string, number>
  preprocessing: PreprocessingSpec
  leakage_checks: LeakageCheck[]
  metrics: Record<string, number | null>
  segmentation_metrics: Record<string, number | null>
  event_metrics: Record<string, number | null>
  calibration: Record<string, unknown>; abstention: Record<string, unknown>
  resource: ResourceUsage
  confidence_intervals: Record<string, Record<string, number>>
  baseline_model_id: string | null; baseline_metrics: Record<string, number | null>
  paired_comparison: Record<string, unknown>
  real_gain_decision: RealGainDecision; real_gain_reason: string
  software: Record<string, string>; source_result_path: string
}
```

---

## 4. Module layout and the wrap-not-duplicate contract

New package: `backend/src/qhealth_qml/platform/`. Nothing under it defines a model, a metric, a
split, or a preprocessing step; every one of those is a call into the existing engine.

### `platform/schema.py`
Every dataclass and enum in section 3, plus `to_dict()` / `*_from_dict()` and `SchemaError`.
Calls into the engine: **none** (pure data). Imports `experiment.SCHEMA_VERSION` only to embed it.

### `platform/registry.py`
Loads and validates JSON under `platform/registry_data/`. Public API:

```python
load_registry(root: Path | None = None) -> Registry
Registry.conditions() -> list[ConditionDefinition]
Registry.models(condition_id: str | None = None) -> list[ModelDefinition]
Registry.model(model_id: str) -> ModelDefinition
Registry.catalog(domain: str = "neurological") -> list[CatalogEntry]   # FR-002, SC-001
Registry.evaluation_records(model_id: str) -> list[EvaluationRecord]
```

`CatalogEntry` is a view struct = `{condition: ConditionDefinition, models: list[ModelDefinition],
availability: Availability}` where `availability` is `"available"` iff at least one of its models
is. Registry validation rules, enforced at load and unit-tested:

1. Every `ModelDefinition.condition_id` resolves.
2. `availability == "available"` ⇒ `artifact.path is not None` **and** `evaluation_record_ids` is
   non-empty. This is what makes a fabricated "working model" impossible to register.
3. `quantum is not None` ⇒ `classical_baseline_model_id` resolves to a registered model (FR-014).
4. `lifecycle == "operational_reference"` ⇒ some referenced `EvaluationRecord` has
   `real_gain_decision == "passed"` **or** the model has no `quantum` spec (FR-039/FR-040: a QML
   model that failed the gate cannot be the operational reference).
5. `model_card.research_record` points at an existing file (FR-034, SC-014).
6. `model_card.reuse_manifest` is non-empty (SC-015).
7. `artifact.path`, when set, resolves inside the configured runtime directory (pickle-safety, P0
   record §4).

Engine calls: none. It reads `EarlyDetectionProfile` via `protocol.load_early_detection_profile()`
purely to validate that `dataset_profile_path` parses.

### `platform/adapters.py`
Source-format → canonical. Implements FR-006 (formats it actually supports today) and FR-007.

```python
adapt_csv(path, role, modality="structured_clinical", bundle_id=...) -> ModalityAsset
adapt_json(path, ...) -> ModalityAsset
register_adapter(format: AssetFormat, fn) -> None
```

`adapt_csv` records one `FieldMapping` per column (`transform="identity"` for pass-through,
`"binary_map"` / `"one_hot"` when a declared mapping table was applied) and computes the sha256.
DICOM/NIfTI/EDF adapters are **not** implemented in P0 — `register_adapter` is the seam they slot
into later (P1 reuse record: no model consumes them yet). Requesting an unregistered format
produces a `rejected` asset with code `unsupported_format`, never a crash (US-2 scenario 3).

Engine calls: none.

### `platform/case_bundle.py`
Builds and validates a `DataBundle`. Public API:

```python
build_bundle(spec: dict, *, synthetic: bool = False) -> DataBundle
validate_bundle(bundle: DataBundle, registry: Registry) -> DataBundle   # returns a new bundle
```

`validate_bundle` performs FR-008 checks per asset — identifier presence, declared modality vs
detected format, sha256 recomputation, schema/column presence, unit table, dimension sanity,
timestamp parseability and ordering — and sets `validation_status` to `accepted` / `rejected` /
`quality_failed`. It assigns `case_id = "local-<uuid4>"` with `case_id_source = "assigned_local"`
when no identifier is supplied. It never mutates or overwrites a source value; conflicts become
`ValidationIssue`s (Edge case: duplicate studies / conflicting units / impossible timestamps).

Engine calls: none. (Deliberate: bundle validation must not depend on the training engine.)

### `platform/routing.py`
The algorithm in section 5. Public API:

```python
route(bundle: DataBundle, registry: Registry, domain: str = "neurological")
    -> list[RoutingDecision]
```

Engine calls: none.

### `platform/execution.py`
The only module that touches the engine. Public API:

```python
run_assessment(bundle, registry, *, mode="research", runtime_dir: Path) -> AssessmentRun
benchmark_model(model_id, registry, *, repeats=10, **overrides) -> EvaluationRecord
```

Executor dispatch on `ModelDefinition.executor`:

| executor | behaviour |
|---|---|
| `"tabular_qml"` | inference and benchmarking as described below |
| `"imaging_segmentation"` | **not implemented in P0.** Raises `NotImplementedError`, which is unreachable in practice because registry rule 2 forces such a model to `availability: "not available"`, and routing short-circuits it before dispatch. |
| `"none"` | never dispatched; reserved for catalog-only entries |

**`tabular_qml` inference path** (per-case `Finding`), exact call sequence:

1. `experiment.load_model_artifact(model.artifact.path)` → `SavedModelArtifact`.
2. Read the bundle's `structured_clinical` asset; project its columns onto
   `artifact.preprocessor.feature_names` **in that exact order** (the engine raises if the order
   differs). A missing or extra column is a contract violation → status `insufficient data`,
   never an improvised fill.
3. `experiment.predict_with_model_artifact(artifact, X, feature_names, dataset_name=case_id,
   row_ids=[case_id], explain=True)`.
4. Translate the returned dict into a `Finding` **via `safety.finalize_finding()` only**
   (section 6). `explanation.global_top_features` / `per_row[0].top_features` become
   `EvidenceItem(kind="feature_contribution")`.

**`tabular_qml` benchmark path** (`EvaluationRecord`), exact call sequence:

1. `protocol.load_early_detection_profile(model.dataset_profile_path)` → `(profile, profile_dir)`.
2. `protocol.resolve_profile_dataset(profile, profile_dir)` → CSV path.
3. **`experiment.load_csv_dataset(csv_path, target=profile.target_column,
   positive_label=profile.positive_label, group_column=profile.group_column,
   time_column=profile.index_time_column, id_column=profile.id_column,
   site_column=profile.site_column, outcome_time_column=profile.outcome_time_column,
   subgroup_columns=profile.subgroup_columns, leakage_columns=profile.leakage_columns,
   task_profile=profile.as_dict())`.**
   **Not** `experiment.load_profile_dataset()` — it invokes
   `protocol.validate_early_detection_profile()`, which hard-requires index/outcome timestamps and
   `horizon_days`. For a cross-sectional profile the executor skips that validator and records
   `model.temporal_validation` in `EvaluationRecord.dataset_profile["temporal_validation"]`.
   When `model.temporal_validation == "validated"`, call `experiment.load_profile_dataset()`
   instead and let the validator run. This branch is decided by the registry field, not by
   inspecting the data.
4. `experiment.run_repeated_experiment(dataset, repeats=..., models=[...], backend=...,
   n_qubits=..., shots=..., test_size=..., validation_size=..., max_train=..., max_test=...,
   threshold_policy=..., target_sensitivity=..., abstain_margin=..., calibrate=...,
   bootstrap_samples=..., reduction=profile.reduction, seed=...)`.
5. For the paired delta and parameter grids: `study.run_nested_evaluation(dataset, ...)`.
6. To mint the deployable artifact, one additional single-repeat call with
   `model_artifact_path=<runtime_dir>/<model_id>.joblib` (the engine forbids
   `repeats > 1` with `model_artifact_path`).
7. Map the result dict into `EvaluationRecord`: `metrics` ← `result["models"][name]["metrics"]`,
   `calibration` ← `clinical_evaluation.calibration_curve` + engine calibration block,
   `abstention` ← `["abstention"]`, `resource` ← `["resource"]`,
   `confidence_intervals` ← `["confidence_intervals"]`, `split_summary` ← `result["split"]`,
   `dataset_fingerprint` ← `result["dataset"]["fingerprint"]`,
   `software` ← `experiment.runtime_manifest()`.
8. Build `leakage_checks` from the split block: `subject_identity` (passed when
   `group_column` is set **or** `id_column` values are unique), `repeated_acquisition`,
   `site_grouping` (`not_applicable` with a stated reason when the dataset has no site column —
   SC-008 requires the check to appear, not to pass), `training_only_preprocessing`
   (always `passed`, evidenced by `PreprocessingSpec.fitted_on`).
9. `real_gain_decision` is computed from `paired_comparison`: `passed` only when the delta CI on
   the declared primary metric excludes zero in the candidate's favour **and** no listed guard
   metric regresses beyond its declared tolerance; otherwise `failed`. `not_assessed` when no
   paired comparison was run. The reason string is always populated.

Every model runs inside its own `try/except`; a raised exception becomes routing status `failed`
plus a `Finding` with status `not evaluated` and the redacted reason, and **does not abort the
loop** (FR-012, SC-009).

### `platform/safety.py`
The single choke point for producing a clinically meaningful finding, plus PHI redaction.

```python
DISCLAIMER: str   # FR-018 text, one constant, used everywhere
finalize_finding(*, model, decision, coverage, raw, run_id, synthetic) -> Finding
not_evaluated(model, decision, run_id, reason, synthetic) -> Finding
redact(message: str, *, allowed_fields: set[str]) -> str      # FR-030, SC-011
fingerprints(bundle, model, artifact) -> dict[str, str]       # FR-028
```

Engine calls: none.

### `platform/registry_data/`
`platform.json`, `conditions/*.json`, `models/*.json`, `evaluations/*.json`. Data only; adding a
model is a JSON file plus a profile, with no code change (FR-026, SC-012).

### Changes to `dashboard.py` (no new server — P0 reuse record §1)
Four routes added to the existing `DashboardHandler`, using its existing `_send_json` /
`_request_json` / `train_lock`:

| Route | Purpose |
|---|---|
| `GET /api/catalog` | `Registry.catalog()` → the FR-002 / SC-001 catalog. No data required. |
| `POST /api/bundle` | Build + validate a `DataBundle` from an upload spec; returns the bundle with per-asset validation. |
| `POST /api/assess` | `execution.run_assessment()` → `AssessmentRun`. Guarded by `train_lock`. |
| `GET /api/assessment?run_id=` | Fetch a completed `AssessmentRun` for the export view. |

---

## 5. Routing algorithm (FR-009, FR-010, FR-011)

`route(bundle, registry, domain)` iterates every registered `ModelDefinition` whose condition is in
`domain` and emits exactly one `RoutingDecision`. Order of tests is normative — the first match
wins, and every branch sets a non-empty `reason`.

```
present        = {a.modality for a in bundle.assets if a.validation_status == "accepted"}
quality_failed = {a.modality for a in bundle.assets if a.validation_status == "quality_failed"}
rejected       = {a.modality for a in bundle.assets if a.validation_status == "rejected"}

for model in registry.models(domain=domain):
    c = model.input_contract

    # 1. No artifact -> the model cannot run, and it is not the case's fault.
    if model.availability == "not available" or model.artifact.path is None:
        -> "not available"; reason = "no registered model artifact for <model_id> v<version>"

    # 2. Required modality absent entirely.
    missing_required = [m for m in c.required_modalities if m not in present]
    if missing_required:
        # 2a. It was supplied but failed QC / was rejected -> the data exists but is unusable.
        if any(m in quality_failed or m in rejected for m in missing_required):
            -> "insufficient data"; reason = "<modality> failed validation: <issue codes>"
        # 2b. It was never supplied.
        else:
            -> "incompatible"; reason = "requires <modality list>; not present in this case"

    # 3. Optional modality absent AND the model was not trained for its absence.
    #    (Edge case: "must not run with improvised imputation".)
    untrained_gaps = [o.modality for o in c.optional_modalities
                      if o.modality not in present and not o.trained_for_absence]
    if untrained_gaps:
        -> "insufficient data";
           reason = "<modality> is absent and this model was not evaluated for its absence"

    # 4. Full-coverage models reject any required-modality QC failure even if a
    #    substitute asset of the same modality was accepted.
    if model.safety.requires_full_coverage and (set(c.required_modalities) & quality_failed):
        -> "insufficient data"; reason = "<modality> failed quality control"

    # 5. Field-level and population contract.
    unmet = check_fields(bundle, c.required_fields, c.min_rows) \
          + check_population(bundle, c.population_filters) \
          + check_quality(bundle, c.quality_constraints)
    if unmet:
        -> "insufficient data"; reason = "; ".join(unmet)      # canonical field names only

    # 6. Contract satisfied.
    -> "ready"; reason = "input contract satisfied"
```

`compatible` vs `ready`: `compatible` is the answer to "could this model ever run on a case shaped
like this" and is what `POST /api/bundle` returns for a preview; `ready` is the answer to "will this
run now" and is what `POST /api/assess` acts on. `route()` emits `ready`; a `preview=True` flag
downgrades a `ready` to `compatible` and leaves every other status unchanged.

`running`, `completed`, `abstained`, and `failed` are **execution** statuses. `execution.py`
transitions a `ready` decision through `running` to exactly one terminal value:

```
ready -> running -> completed      (a score was produced and not abstained)
                 -> abstained      (abstain_margin triggered, or calibration check failed)
                 -> failed         (executor raised)
```

`route()` never emits them, and no status outside the FR-009 nine is ever written.

**Status → finding mapping (the FR-011 / SC-003 safety table).** This is the only permitted mapping:

| RoutingDecision.status | Finding.status | Score present? |
|---|---|---|
| `not available` | `not available` | no |
| `incompatible` | `not evaluated` | no |
| `insufficient data` | `insufficient data` | no |
| `failed` | `not evaluated` | no |
| `abstained` | `abstained` | score may be present, no label |
| `completed`, score ≥ threshold | `positive` | yes |
| `completed`, score < threshold | `negative` | yes |

There is **no path from a missing, rejected, or quality-failed input to `negative`.** When the
routing list contains no `ready` decision, `AssessmentRun.status` is `completed` with zero
positive/negative findings and `disclaimer` carrying the "no compatible assessment" text — never a
reassuring result (US-3 scenario 4, SC-003).

---

## 6. The safety invariant

`safety.finalize_finding()` is the **only** function in the codebase permitted to construct a
`Finding` whose status is `positive` or `negative`. It asserts, and raises `SafetyError` on failure:

1. `decision.status == "completed"`.
2. `raw["score"] is not None` and `model.artifact.path is not None`.
3. `coverage.required_present == coverage.required_total` and `coverage.quality_failed == []`.
4. `model.safety.allows_negative_finding is True` when the computed status is `negative`
   (a screening-only model can be configured to emit `positive` / `abstained` / `insufficient data`
   only).
5. `not raw["abstained"]` — an abstention returns status `abstained` before reaching the
   positive/negative branch.
6. `finding.disclaimer == safety.DISCLAIMER` and `finding.reference_label_tier` equals the
   condition's tier — so a low-reference model cannot be rendered without its tier.
7. `finding.synthetic == run.synthetic`; when `run.mode == "demo"`, `synthetic` is forced `True`
   and the disclaimer is prefixed with the FR-031 synthetic marker.

Every other status is produced by `safety.not_evaluated()`, which cannot set a score. A unit test
greps the codebase to assert `Finding(` is constructed nowhere outside `safety.py`.

**PHI redaction (FR-030, SC-011).** `safety.redact()` is applied to every `ValidationIssue.message`,
every `StageEvent.message`, every `AssessmentRun.errors` entry, and every exception message that
crosses into a log or an HTTP body. It permits canonical field names, asset ids, model ids, and
numeric counts, and replaces anything else that looks like a value with `<redacted>`. Engine
exceptions such as `feature 'bmi' at CSV row 41 is not numeric` are rewritten to
`feature 'bmi' at row <redacted> failed numeric validation`.

---

## 7. Registry entries to author now

### 7.1 `registry_data/platform.json` — the P0 entry (no disease model)

```json
{
  "schema_version": 1,
  "registry_version": "0.1.0",
  "domain": "neurological",
  "disease_models": [],
  "disclaimer": "Research prediction, not a medical diagnosis. This platform does not provide diagnosis, treatment recommendations, or triage decisions.",
  "no_compatible_assessment_text": "No registered model could be run on the data supplied. This is not a negative result and does not indicate the absence of any neurological condition.",
  "synthetic_marker": "SYNTHETIC / DEMO OUTPUT — not model inference and not clinical evidence.",
  "research_records": [
    "specs/001-neurological-conditions/research/P0-platform-reuse-record.md",
    "specs/001-neurological-conditions/research/P1-stroke-reuse-record.md"
  ]
}
```

P0's acceptance is precisely that this registry loads, the catalog renders, and an assessment over
any bundle produces zero positive/negative findings because `disease_models` is empty.

### 7.2 `registry_data/conditions/neuro.stroke.ischemic.acute.json`

```json
{
  "condition_id": "neuro.stroke.ischemic.acute",
  "name": "Acute ischemic stroke",
  "domain": "neurological",
  "priority": "P1",
  "task_type": "segmentation",
  "readiness_tier": "high_reference",
  "reference_label_tier": "high",
  "target_definition": "Acute ischemic lesion / infarct core at presentation, referenced against expert lesion annotation and/or follow-up diffusion MRI.",
  "population": "Adults presenting with suspected acute ischemic stroke.",
  "required_modalities": ["imaging"],
  "optional_modalities": ["structured_clinical"],
  "expected_output": "Case-level risk, lesion mask/regions, and an infarct-core volume estimate.",
  "reference_datasets": [
    {"label": "ISLES'24", "url": "https://pubs.rsna.org/doi/full/10.1148/ryai.250603", "citation": "ISLES'24 multimodal stroke lesion segmentation benchmark, Radiology: Artificial Intelligence, 2025"}
  ],
  "evidence_links": [
    {"label": "DeepISLES", "url": "https://github.com/ezequieldlrosa/DeepIsles", "citation": null},
    {"label": "DeepISLES paper", "url": "https://www.nature.com/articles/s41467-025-62373-x", "citation": "Nature Communications, 2025"}
  ],
  "limitations": [
    "Acute recognition task, not a diagnosis.",
    "A follow-up-DWI reference standard measures final infarct rather than the acute core at presentation.",
    "No imaging model artifact is registered on this platform yet; see the model entries."
  ]
}
```

### 7.3 `backend/profiles/p1_stroke_clinical.json` — exact content to write

**Update (post-design, applied directly to the engine, not via a one-off adapter script):**
`experiment.load_csv_dataset()` has been extended to auto-detect and one-hot-encode text/categorical
columns, and to treat common missing-value sentinels (`N/A`, `NaN`, `None`, `null`, `-`, `?`,
case-insensitive) as missing rather than a parse error. This closes the gap generally for every
future condition's CSV, not just this one, per the reuse-first-but-extend-fully principle — see
"Engine fixes applied" in the P0 reuse record. Consequently there is **no separate
`prepare.py` adapter and no derived CSV**: `dataset_path` points straight at the raw Kaggle file, and
the loader emits one feature column per numeric field plus `"<column>=<value>"` one-hot columns for
each categorical field (encoded feature names use `=`, e.g. `gender=Female`, `work_type=Private`).
A stray `explain_raw_inputs()` index bug (it indexed a length-16-capped list by the *absolute* raw
feature index instead of position) was also fixed, since any dataset with more than 16 raw columns
hit it — see the same section.

`dataset_path` is relative to the profile file, resolved by `protocol.resolve_profile_dataset()`.

```json
{
  "name": "p1_stroke_clinical",
  "dataset_path": "../data/p1_stroke_clinical/healthcare-dataset-stroke-data.csv",
  "target_column": "stroke",
  "positive_label": "1",
  "id_column": "id",
  "group_column": null,
  "index_time_column": null,
  "outcome_time_column": null,
  "site_column": null,
  "horizon_days": null,
  "outcome_definition": "Recorded stroke status in the source table. The source does not document whether this is a prevalent history or an incident event, and no index or outcome time is available, so no prediction horizon can be defined. Treat as an associative risk-factor model, not early detection.",
  "leakage_columns": [],
  "subgroup_columns": [],
  "modality": "tabular",
  "reduction": "anova",
  "task_type": "early_detection"
}
```

Notes an implementer needs:

- `task_type` must be the literal `"early_detection"`; `protocol.load_early_detection_profile()`
  rejects anything else. It is a loader constraint, not a claim about the task — the honest
  description lives in `outcome_definition` and in `ModelDefinition.temporal_validation`.
- `reduction: "anova"` is deliberate: `SelectKBest` keeps four *named* source features, so
  `explain_raw_inputs()` produces interpretable evidence. PCA would not.
- `group_column` is null because there is one row per patient and all 5110 `id` values are unique;
  the leakage check records this fact rather than claiming grouping was applied.
- `subgroup_columns` is empty so all 17 encoded columns stay as features. Subgroup metrics are
  obtained from a separate CLI run with `--subgroup-column`, not from this profile.
- The feature columns the loader now produces from the raw file directly: `age`, `hypertension`,
  `heart_disease`, `avg_glucose_level`, `bmi` (numeric; `bmi`'s `N/A` rows become `NaN`, imputed by
  the existing median imputer — no engine change needed there) plus one-hot columns
  `gender=Female`/`gender=Male`/`gender=Other`, `ever_married=No`/`ever_married=Yes`,
  `Residence_type=Rural`/`Residence_type=Urban`, `work_type=Govt_job`/`work_type=Never_worked`/
  `work_type=Private`/`work_type=Self-employed`/`work_type=children`,
  `smoking_status=Unknown`/`smoking_status=formerly smoked`/`smoking_status=never smoked`/
  `smoking_status=smokes` (21 columns total). `id` and `stroke` are excluded by the loader.
  `gender=Other` is kept as its own column (n=1) rather than folded into missing — the model card's
  "not validated for gender recorded as 'Other'" limitation covers it instead of silently dropping it.

### 7.4 `registry_data/models/stroke-clinical-risk-tabular.json` — P1 arm B, runs now

```json
{
  "model_id": "stroke-clinical-risk-tabular",
  "condition_id": "neuro.stroke.ischemic.acute",
  "version": "0.1.0",
  "display_name": "Stroke clinical risk head (tabular, research)",
  "task_type": "binary_classification",
  "availability": "available",
  "lifecycle": "experimental",
  "executor": "tabular_qml",
  "input_contract": {
    "required_modalities": ["structured_clinical"],
    "optional_modalities": [],
    "required_fields": ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
                        "gender", "ever_married", "Residence_type", "work_type", "smoking_status"],
    "min_rows": 1,
    "population": "Adults with recorded cardiovascular risk factors. Not validated for anyone under 18 and not validated for gender recorded as 'Other' (n=1 in the source cohort).",
    "population_filters": {"age": ">=18"},
    "quality_constraints": {
      "age": "0 <= age <= 120",
      "bmi": "10 <= bmi <= 100 when present",
      "avg_glucose_level": "40 <= avg_glucose_level <= 400"
    }
  },
  "dataset_profile_path": "backend/profiles/p1_stroke_clinical.json",
  "temporal_validation": "not applicable — cross-sectional cohort, no index or outcome time",
  "preprocessing": {
    "imputation": "median", "scaling": "standard", "reduction": "anova",
    "n_components": 4, "angle_scaling": "minmax", "fitted_on": "train_partition_only"
  },
  "quantum": {
    "framework": "qiskit", "encoding": "ZZFeatureMap", "ansatz": null,
    "circuit_version": "qiskit-2.5/zzfeaturemap-reps2", "backend_mode": "statevector",
    "n_qubits": 4, "shots": 512
  },
  "classical_baseline_model_id": "stroke-clinical-risk-tabular-classical",
  "artifact": {
    "kind": "saved_model_artifact",
    "path": "runtime/models/stroke-clinical-risk-tabular.joblib",
    "manifest_path": "runtime/models/stroke-clinical-risk-tabular.manifest.json",
    "sha256": null, "schema_version": 4
  },
  "calibration": {"method": "validation_sigmoid", "status": "not_assessed", "note": "Set to 'calibrated' or 'failed' only from an EvaluationRecord."},
  "explainability": {"method": "input_sensitivity", "scope": "per_row", "surrogate": false},
  "output_score_type": "calibrated_probability",
  "evaluation_record_ids": ["eval-stroke-clinical-risk-tabular-0.1.0"],
  "safety": {
    "allows_negative_finding": true,
    "abstain_margin": 0.05,
    "requires_full_coverage": true,
    "disclaimer": "Research risk estimate, not a diagnosis and not a stroke detection result.",
    "limitations": [
      "Reference-label confidence is LOW. The source table has no datasheet, no named institution, and no documented collection protocol.",
      "The source does not state whether the stroke label is prevalent or incident, so reverse causation cannot be excluded. This is an association model, not early detection.",
      "30.2% of smoking_status is 'Unknown' and that missingness is likely non-random.",
      "Prevalence is 4.87%; accuracy is not a meaningful metric for this model.",
      "This model does NOT detect an acute ischemic lesion and is not a substitute for imaging.",
      "No site or scanner column exists, so site-held-out external validation was not possible."
    ]
  },
  "model_card": {
    "intended_use": "Research benchmarking of a hybrid quantum-classical classifier against tuned classical baselines on structured cardiovascular risk factors.",
    "excluded_use": "Any clinical use. Acute stroke detection or triage. Any use as evidence that a patient does not have a stroke.",
    "training_population": "5110 rows, one per patient, from the Kaggle stroke-prediction table. 249 positive (4.87%).",
    "evaluation_population": "Stratified random holdout of the same table. No external cohort.",
    "label_policy": "Binary 'stroke' column as supplied, positive_label='1'. Temporal relationship to the recorded risk factors is undocumented.",
    "data_sources": [
      {"label": "Kaggle fedesoriano/stroke-prediction-dataset", "url": "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset", "citation": null}
    ],
    "data_licenses": ["kaggle:copyright-authors — UNVERIFIED, redistribution blocked pending manual review"],
    "limitations": ["See safety.limitations."],
    "calibration_status": "not_assessed",
    "explanation_method": "input_sensitivity over the 4 ANOVA-selected raw features",
    "maintainer": "tech@centai.in",
    "reuse_manifest": [
      {"component": "training/evaluation engine", "decision": "adapted", "source_url": null, "release_or_commit": "quantum-health 0.1.0 (in-repo)", "paper_citation": null, "license": "in-repo, license field not yet declared", "license_verified": false, "weight_source": null, "weight_sha256": null, "preprocessing_assumptions": "numeric CSV columns, or text columns auto one-hot encoded; common missing-value sentinels treated as NaN", "io_contract": "LoadedDataset -> run_repeated_experiment result dict", "local_modifications": "load_csv_dataset() extended to auto-detect and one-hot encode categorical text columns and to recognize N/A-style missing sentinels (generalized capability, not stroke-specific); explain_raw_inputs() index-out-of-range bug fixed for datasets with >16 raw feature columns"},
      {"component": "source dataset", "decision": "reused", "source_url": "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset", "release_or_commit": "downloaded 2026-08-29, 316971 bytes", "paper_citation": null, "license": "copyright-authors", "license_verified": false, "weight_source": null, "weight_sha256": null, "preprocessing_assumptions": "one row per patient; id is a non-informative surrogate key", "io_contract": "12-column CSV", "local_modifications": "none to source values"}
    ],
    "research_record": "specs/001-neurological-conditions/research/P1-stroke-reuse-record.md"
  }
}
```

A sibling file `registry_data/models/stroke-clinical-risk-tabular-classical.json` is identical
except: `model_id` suffixed `-classical`, `quantum: null`, `classical_baseline_model_id: null`,
`explainability.method: "input_sensitivity"`, and `lifecycle: "operational_reference"`. Registry
rule 3 requires it to exist (FR-014), and registry rule 4 lets **only** it be the operational
reference until the QML entry's `EvaluationRecord.real_gain_decision` is `passed`.

### 7.5 `registry_data/models/stroke-lesion-core-imaging.json` — arm A, `not available`

```json
{
  "model_id": "stroke-lesion-core-imaging",
  "condition_id": "neuro.stroke.ischemic.acute",
  "version": "0.0.0",
  "display_name": "Acute ischemic lesion / infarct-core segmentation (planned)",
  "task_type": "segmentation",
  "availability": "not available",
  "lifecycle": "experimental",
  "executor": "imaging_segmentation",
  "input_contract": {
    "required_modalities": ["imaging"],
    "optional_modalities": [{"modality": "structured_clinical", "trained_for_absence": true}],
    "required_fields": [],
    "min_rows": 1,
    "population": "Adults presenting with suspected acute ischemic stroke, imaged with NCCT/CTA/CTP.",
    "population_filters": {},
    "quality_constraints": {
      "imaging": "Skull-stripped, co-registered volumes at the ISLES'24 spacing contract."
    }
  },
  "dataset_profile_path": null,
  "temporal_validation": "n/a",
  "preprocessing": {
    "imputation": "none", "scaling": "none", "reduction": "none",
    "n_components": null, "angle_scaling": null, "fitted_on": "n/a"
  },
  "quantum": null,
  "classical_baseline_model_id": null,
  "artifact": {"kind": "none", "path": null, "manifest_path": null, "sha256": null, "schema_version": null},
  "calibration": {"method": "none", "status": "not_assessed", "note": "No artifact."},
  "explainability": {"method": "planned:lesion_overlay", "scope": "region", "surrogate": false},
  "output_score_type": "mask",
  "evaluation_record_ids": [],
  "safety": {
    "allows_negative_finding": false,
    "abstain_margin": null,
    "requires_full_coverage": true,
    "disclaimer": "No model artifact is registered. This entry exists to show the coverage gap and must never produce a finding.",
    "limitations": [
      "DEFERRED. Blocked on ISLES'24 access (grand-challenge.org registration and DUA, a manual human step) and on disk: 5.9 GB free as of 2026-08-29.",
      "Planned reused implementation: DeepISLES (github.com/ezequieldlrosa/DeepIsles) as the reference segmentation ensemble, nnU-Net (github.com/MIC-DKFZ/nnUNet) as the retraining fallback, MONAI (github.com/Project-MONAI/MONAI) for volume I/O, transforms, and Dice/IoU metrics.",
      "DeepISLES code license and model-weight license are separate and both unverified; weights may inherit ISLES data terms.",
      "Segmentation stays classical. Any quantum component would sit only on a compact fused image/clinical representation."
    ]
  },
  "model_card": {
    "intended_use": "Planned: acute ischemic lesion segmentation and infarct-core volume estimation from presentation CT/CTA/CTP.",
    "excluded_use": "Everything, currently. No artifact exists.",
    "training_population": "Planned: ISLES'24 multi-centre cohort.",
    "evaluation_population": "Planned: ISLES'24 held-out centre plus an external cohort.",
    "label_policy": "Expert lesion annotation and/or follow-up diffusion MRI.",
    "data_sources": [
      {"label": "ISLES'24", "url": "https://pubs.rsna.org/doi/full/10.1148/ryai.250603", "citation": "Radiology: Artificial Intelligence, 2025"}
    ],
    "data_licenses": ["ISLES'24 DUA — not accepted, access not obtained"],
    "limitations": ["See safety.limitations."],
    "calibration_status": "not_assessed",
    "explanation_method": "planned: lesion mask overlay + per-region volume",
    "maintainer": "tech@centai.in",
    "reuse_manifest": [
      {"component": "segmentation network", "decision": "reused", "source_url": "https://github.com/ezequieldlrosa/DeepIsles", "release_or_commit": null, "paper_citation": "DeepISLES, Nature Communications 2025, doi:10.1038/s41467-025-62373-x", "license": "permissive (unverified); weight license separate and unverified", "license_verified": false, "weight_source": "DeepISLES release artifacts", "weight_sha256": null, "preprocessing_assumptions": "ISLES'24 modality set, spacing, and skull-stripping contract", "io_contract": "co-registered CT/CTA/CTP volumes -> lesion mask", "local_modifications": "none — not yet integrated"},
      {"component": "retraining fallback", "decision": "reproduced", "source_url": "https://github.com/MIC-DKFZ/nnUNet", "release_or_commit": null, "paper_citation": "Isensee et al., Nature Methods 2021", "license": "Apache-2.0 (unverified)", "license_verified": false, "weight_source": null, "weight_sha256": null, "preprocessing_assumptions": "self-configured from a dataset fingerprint", "io_contract": "NIfTI volumes -> segmentation", "local_modifications": "none — not yet integrated"},
      {"component": "volume I/O, transforms, Dice/IoU", "decision": "reused", "source_url": "https://github.com/Project-MONAI/MONAI", "release_or_commit": null, "paper_citation": null, "license": "Apache-2.0 (unverified)", "license_verified": false, "weight_source": null, "weight_sha256": null, "preprocessing_assumptions": "NIfTI/DICOM with valid affine", "io_contract": "LoadImage/Spacing/DiceMetric", "local_modifications": "none — not yet integrated"}
    ],
    "research_record": "specs/001-neurological-conditions/research/P1-stroke-reuse-record.md"
  }
}
```

This entry is what makes US-1 scenario 3 and SC-004 true without
fabricating a model: routing always returns `not available`, and `safety.finalize_finding()` can
never be reached for it.

### 7.6 SC-004 completion stubs

SC-004 requires **four** high-reference model contracts at launch. Three more condition files and
three more model files, all `availability: "not available"`, following 7.2/7.5 verbatim:

| Condition file | Model file | Cited planned reuse |
|---|---|---|
| `neuro.ich.acute.json` (P2, CT multilabel) | `ich-subtype-ct.json` | RSNA ICH challenge, MONAI |
| `neuro.seizure.eeg.json` (P4, event detection) | `seizure-onset-eeg.json` | CHB-MIT, MNE-Python, Braindecode/EEGNet |
| `neuro.glioma.mpmri.json` (P3, segmentation) | `glioma-characterization-mpmri.json` | BraTS, TCGA-GBM, nnU-Net, MONAI |

Two research-only condition files with no model entry satisfy FR-033/SC-005:
`neuro.alzheimers.progression.json` and `neuro.parkinsons.prodromal.json`, both
`readiness_tier: "research_only"`, `reference_label_tier: "moderate"`, and an
`expected_output` phrased as progression/risk, never diagnosis. These are JSON only — no code.

---

## 8. Files to create and change

### New backend files

| Path | Purpose |
|---|---|
| `backend/src/qhealth_qml/platform/__init__.py` | Package marker; re-exports `load_registry`, `route`, `run_assessment`. |
| `backend/src/qhealth_qml/platform/schema.py` | All entities, enums, `to_dict`/`from_dict`, `SchemaError`, `PLATFORM_SCHEMA_VERSION`. |
| `backend/src/qhealth_qml/platform/registry.py` | Load/validate registry JSON; catalog and model lookup; the seven registry rules. |
| `backend/src/qhealth_qml/platform/adapters.py` | Source-format → `ModalityAsset` with `FieldMapping` provenance; CSV/JSON today, extension seam for DICOM/NIfTI/EDF. |
| `backend/src/qhealth_qml/platform/case_bundle.py` | Build and validate a `DataBundle`; identifier assignment; hashing; FR-008 checks. |
| `backend/src/qhealth_qml/platform/routing.py` | `route()` — the section 5 algorithm. |
| `backend/src/qhealth_qml/platform/execution.py` | `run_assessment()` / `benchmark_model()`; the only module calling `experiment` and `study`. |
| `backend/src/qhealth_qml/platform/safety.py` | `finalize_finding()`, `not_evaluated()`, `redact()`, `fingerprints()`, `DISCLAIMER`. |
| `backend/profiles/p1_stroke_clinical.json` | The P1 arm-B `EarlyDetectionProfile` (section 7.3). |
| `backend/src/qhealth_qml/platform/registry_data/platform.json` | P0 platform record (section 7.1). |
| `backend/src/qhealth_qml/platform/registry_data/conditions/*.json` | 6 condition files (1 stroke + 3 SC-004 + 2 research-only). |
| `backend/src/qhealth_qml/platform/registry_data/models/*.json` | 5 model files (tabular QML, its classical pair, imaging stroke, ICH, seizure, glioma stubs). |
| `backend/tests/test_platform.py` | The P0-1 … P0-7 acceptance tests from the P0 reuse record §5. |
| `backend/tests/fixtures/bundles/*.json` | Fixture bundles: clinical-only, clinical+imaging, quality-failed imaging, unsupported format. |

### Changed backend files

| Path | Change |
|---|---|
| `backend/src/qhealth_qml/dashboard.py` | Add `GET /api/catalog`, `POST /api/bundle`, `POST /api/assess`, `GET /api/assessment` to `DashboardHandler`; reuse `_send_json`, `_request_json`, `train_lock`, CORS. No new server. |
| `backend/pyproject.toml` | Add `qhealth_qml.platform.registry_data` JSON to `[tool.setuptools.package-data]`. Also **add a `license` field** (P0 reuse record §8 risk 1). |
| `backend/.gitignore` (new if absent) | Exclude `data/p1_stroke_clinical/*.csv` and `runtime/` (P1 record §4 license constraint). |

### New frontend files

| Path | Purpose |
|---|---|
| `src/lib/platform.ts` | TypeScript mirror of section 3 plus `fetchCatalog()`, `postBundle()`, `postAssess()`, `fetchAssessment()` built on `qmlApi.ts`'s exported `request<T>()`. |
| `src/components/ConditionCatalog.tsx` | FR-002/SC-001: readiness-grouped catalog; per-condition detail with task, required modalities, output, reference dataset, readiness tier, limitations, availability badge. |
| `src/components/FindingCard.tsx` | One `Finding` rendered with its FR-016 fields; status chip styled per the six `FindingStatus` values; "explanation unavailable" when `explanation_status != "available"`. |
| `src/components/AssessmentPanel.tsx` | The `AssessmentRun` view: routing table (all nine `ModelStatus` values with reasons) above the findings list; failures never hide completed findings. |

### Changed frontend files

| Path | Change |
|---|---|
| `src/lib/qmlApi.ts` | Export the existing `request<T>()` helper so `platform.ts` reuses the fetch/error layer. No type removed or renamed. |
| `src/lib/pipeline/types.ts` | Add `modelId?: string` to `PipelineEvent` so per-model stages are attributable. |
| `src/lib/pipeline/runner.ts` | Add `createAssessmentRunner(bundleId)` beside `createApiRunner` / `createMockRunner`; it polls `/api/assess` and emits per-model `PipelineEvent`s. |
| `src/hooks/usePipeline.ts` | Accept an optional runner so the assessment flow reuses the existing state machine. |
| `src/components/LaunchScreen.tsx` | Add the FR-001 "Neurological conditions" domain entry that opens `ConditionCatalog`. |
| `src/lib/export.ts` | FR-029/SC-010: export the full `AssessmentRun` — every finding, not-evaluated state, failure, provenance, model version, disclaimer, and `synthetic` flag. |
| `src/lib/findings.ts` | Mark `deriveFindings()` demo-only and force `synthetic: true` on anything it produces (FR-031). Do not delete — it backs the existing demo mode. |

Totals: 8 new Python modules, 1 profile JSON, 12 registry JSON files, 2 test files, 3 changed
backend files, 4 new TS files, 7 changed TS files. No new runtime dependency in either language.

---

## 9. Build order and acceptance

1. `schema.py` + `registry.py` + the JSON in section 7 → `GET /api/catalog` renders
   (SC-001, SC-004, SC-005).
2. `adapters.py` + `case_bundle.py` + fixtures → `POST /api/bundle` reports accepted / rejected /
   missing per asset (US-2).
3. `routing.py` + `safety.py` → tests P0-2, P0-3 pass with zero fabricated negatives
   (SC-002, SC-003).
4. `execution.py` tabular path → test P0-1 (byte-identical to a direct engine call) and P0-4
   (fault isolation) pass (SC-009).
5. `benchmark_model()` → produce `eval-stroke-clinical-risk-tabular-0.1.0`, flip the QML entry's
   `calibration.status` and `real_gain_decision` from the measured result, and only then may
   `availability` stay `"available"` (registry rule 2).
6. Frontend catalog → assessment → export (SC-006, SC-010).

**P1 is not startable before step 5 exists**, because registry rule 2 refuses to load an
`available` model with no `EvaluationRecord` — the build order in the spec is enforced by the
registry validator, not by convention.
