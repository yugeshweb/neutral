import { request } from './qmlApi'

// ============================================================================
// Enumerations (Section 2)
// ============================================================================

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

// ============================================================================
// Supporting structs (Section 3.1)
// ============================================================================

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

// ============================================================================
// Key Entities (Sections 3.2-3.9)
// ============================================================================

export interface ConditionDefinition {
  condition_id: string; name: string; domain: string; priority: string
  task_type: TaskType; readiness_tier: ReadinessTier; reference_label_tier: ReferenceTier
  target_definition: string; population: string
  required_modalities: Modality[]; optional_modalities: Modality[]
  expected_output: string
  reference_datasets: Reference[]; evidence_links: Reference[]; limitations: string[]
}

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

export interface DataBundle {
  bundle_id: string; case_id: string; case_id_source: string; created_at: string
  source: string; synthetic: boolean
  visits: Visit[]; assets: ModalityAsset[]; validation: ValidationIssue[]
  content_hashes: Record<string, string>; provenance: Record<string, string>
}

export interface ModalityAsset {
  asset_id: string; bundle_id: string; modality: Modality; format: AssetFormat; role: string
  visit_id: string | null; uri: string; content_hash: string; byte_size: number
  rows: number | null; dimensions: number[] | null; units: Record<string, string>
  acquired_at: string | null
  validation_status: AssetValidationStatus; validation_issues: ValidationIssue[]
  field_mappings: FieldMapping[]; derived_from: string[]
}

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

export interface EhrValidationCheck {
  name: string
  status: 'passed' | 'warning'
  detail: string
}

export interface EhrValidationResult {
  status: 'passed'
  endpoint: string
  source_format: string
  dataset: {
    name: string
    rows: number
    features: number
    feature_names: string[]
    positive_label: string
    negative_label: string
    label_counts: Record<string, number>
    missing_cells: number
    fingerprint: string
  }
  checks: EhrValidationCheck[]
  bundle: DataBundle
  routing: RoutingDecision[]
  disclaimer: string
}

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

export interface EvidenceItem {
  evidence_id: string; finding_id: string; kind: EvidenceKind; label: string
  value: number | null; unit: string | null
  region: Record<string, number> | null; interval: Record<string, number> | null
  source_asset_id: string | null; confidence: number | null; note: string | null
}

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

// ============================================================================
// Catalog Entry (from section 4, backend view struct)
// ============================================================================

export interface CatalogEntry {
  condition: ConditionDefinition
  models: ModelDefinition[]
  availability: Availability
}

// ============================================================================
// API Fetch Functions (section 4 routes)
// ============================================================================

// `request()` already prepends qmlApi's own API_ROOT (VITE_QML_API_URL) — this
// module targets the same local dashboard server, so paths here must be relative,
// not prefixed again, or the two base URLs would concatenate.

export async function fetchCatalog(): Promise<CatalogEntry[]> {
  return request<CatalogEntry[]>('/api/catalog')
}

export async function postBundle(spec: unknown): Promise<DataBundle> {
  return request<DataBundle>('/api/bundle', {
    method: 'POST',
    body: JSON.stringify(spec),
  })
}

export async function postAssess(bundleId: string): Promise<AssessmentRun> {
  return request<AssessmentRun>('/api/assess', {
    method: 'POST',
    body: JSON.stringify({ bundle_id: bundleId }),
  })
}

export async function postEhrValidation(input: {
  csv_text: string
  dataset_name: string
  source_format: string
  target: string
  positive_label?: string
  id_column?: string
  group_column?: string
  time_column?: string
  site_column?: string
  outcome_time_column?: string
}): Promise<EhrValidationResult> {
  return request<EhrValidationResult>('/api/validation/ehr', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export async function fetchAssessment(runId: string): Promise<AssessmentRun> {
  return request<AssessmentRun>(`/api/assessment?run_id=${encodeURIComponent(runId)}`)
}
