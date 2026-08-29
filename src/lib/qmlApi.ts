export type QmlMetricKey =
  | 'accuracy'
  | 'balanced_accuracy'
  | 'sensitivity'
  | 'specificity'
  | 'precision'
  | 'negative_predictive_value'
  | 'f1'
  | 'roc_auc'
  | 'pr_auc'
  | 'brier_score'
  | 'expected_calibration_error'

export type QmlMetrics = Partial<Record<QmlMetricKey, number | null>> & {
  evaluated_rows?: number
  confusion_matrix_labels?: string[]
  confusion_matrix?: number[][]
}

export type QmlResource = {
  training_rows?: number
  test_rows?: number
  feature_count?: number
  qubits?: number | null
  shots?: number | null
  backend?: string | null
  estimated_kernel_pairs?: number | null
  circuit_probe?: Record<string, unknown> | null
}

export type QmlExplanation = {
  status?: string
  global_top_features?: Array<{
    feature?: string
    feature_name?: string
    mean_absolute_sensitivity?: number
    sensitivity?: number
  }>
  per_row?: Array<{
    row_id?: string
    top_features?: Array<{
      feature?: string
      feature_name?: string
      sensitivity?: number
      change?: number
    }>
  }>
}

export type QmlModelResult = {
  feature_space?: string
  parameters?: Record<string, unknown>
  metrics?: QmlMetrics
  clinical_evaluation?: {
    calibration_curve?: unknown[]
    decision_curve?: unknown[]
    subgroups?: Record<string, unknown>
  }
  resource?: QmlResource
  threshold?: {
    policy?: string
    threshold?: number | null
    validation_rows?: number
  }
  abstention?: {
    enabled?: boolean
    margin?: number | null
    total_rows?: number
    abstained_rows?: number
    evaluated_rows?: number
    coverage?: number | null
  }
  score_type?: string
  predictions?: number[]
  scores?: number[] | null
  prediction_rows?: Array<{
    row_id?: string
    prediction?: number
    abstained?: boolean
    score?: number | null
    label?: number
  }>
  elapsed_seconds?: number
  explanation?: QmlExplanation
}

export type QmlResult = {
  schema_version?: number
  package_version?: string
  dataset?: {
    name?: string
    rows?: number
    features?: number
    positive_label?: string
    negative_label?: string
    fingerprint?: string
    provenance?: Record<string, unknown>
    task_profile?: Record<string, unknown> | null
  }
  split?: {
    strategy?: string
    train_rows?: number
    validation_rows?: number
    test_rows?: number
    test_size?: number
    seed?: number
  }
  preprocessing?: {
    input_features?: string[]
    selected_features?: string[]
    qubits?: number
    reduction?: string
    feature_selection?: string
  }
  execution?: {
    backend_mode?: string
    resolved_backend?: string
    shots?: number
    requested_models?: string[]
    reduction?: string
  }
  hardware_probe?: Record<string, unknown>
  models?: Record<string, QmlModelResult>
  model_artifact?: {
    path?: string
    manifest_path?: string
    ehr_contract_path?: string
    model_name?: string
    selected_for_inference?: string
  }
  study?: Record<string, unknown>
}

export type QmlModelInfo = {
  available: boolean
  model_name?: string
  feature_names?: string[]
  selected_features?: string[]
  feature_space?: string
  threshold?: number | null
  threshold_policy?: string
  abstain_margin?: number | null
  hardware_probe?: Record<string, unknown>
  dataset?: QmlResult['dataset']
  ehr_contract_available?: boolean
}

export type QmlPrediction = {
  schema_version?: number
  package_version?: string
  dataset?: string
  model_name?: string
  feature_space?: string
  selected_features?: string[]
  input_features?: string[]
  threshold?: number | null
  threshold_policy?: string
  score_type?: string
  predictions?: number[]
  abstained?: boolean[]
  scores?: number[] | null
  prediction_rows?: Array<{
    row_id?: string
    prediction?: number
    abstained?: boolean
    score?: number | null
  }>
  explanation?: QmlExplanation
  abstention?: QmlModelResult['abstention']
  ehr?: {
    patient_id?: string
    index_time?: string
    feature_view?: Record<string, unknown>
    validation?: Record<string, unknown>
  }
}

export type TrainRequest = {
  source: 'wdbc' | 'csv' | 'ehr'
  dataset_name?: string
  csv_text?: string
  target?: string
  positive_label?: string
  group_column?: string
  time_column?: string
  id_column?: string
  site_column?: string
  subgroup_columns?: string[]
  leakage_columns?: string[]
  outcome_time_column?: string
  ehr_input?: unknown
  ehr_mapping?: Record<string, unknown>
  ehr_feature_specs?: unknown
  ehr_label_field?: string
  ehr_index_time_field?: string
  ehr_labels?: Array<string | number | boolean>
  ehr_index_times?: string[]
  models?: string[]
  selection_repeats?: number
  selection_metric?: QmlMetricKey
  backend?: 'statevector' | 'aer' | 'fake' | 'ibm'
  n_qubits?: number
  shots?: number
  max_train?: number
  max_test?: number
  seed?: number
  reduction?: 'anova' | 'pca'
}

/**
 * Where the Python backend lives.
 *
 * Defaults to the port `qhealth-dashboard` binds, so the catalog and EHR
 * validation screens work with no configuration once the backend is running.
 * Override with VITE_QML_API_URL when it is hosted elsewhere; set it to an
 * empty string to route through a dev-server proxy on the same origin.
 */
const DEFAULT_API_ROOT = 'http://127.0.0.1:8765'

const configured = import.meta.env.VITE_QML_API_URL
const API_ROOT = (configured === undefined ? DEFAULT_API_ROOT : configured).replace(/\/$/, '')

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  const payload = (await response.json().catch(() => ({}))) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error ?? `QML API returned ${response.status}`)
  }
  return payload
}

export function fetchResult() {
  return request<QmlResult>('/api/result')
}

export function fetchModel() {
  return request<QmlModelInfo>('/api/model')
}

export function fetchSamples() {
  return request<{ rows: Array<{ row_id?: string; features?: Record<string, number | null> }> }>(
    '/api/samples',
  )
}

export function trainModel(body: TrainRequest, signal?: AbortSignal) {
  return request<QmlResult>('/api/train', {
    method: 'POST',
    body: JSON.stringify(body),
    signal,
  })
}

export function validateEhr(input: unknown, mapping: Record<string, unknown>) {
  return request<Record<string, unknown>>('/api/ehr/validate', {
    method: 'POST',
    body: JSON.stringify({ ehr_input: input, ehr_mapping: mapping }),
  })
}

export function trainEhrCohort(body: Omit<TrainRequest, 'source'>) {
  return trainModel({ ...body, source: 'ehr' })
}

export function predictWithModel(
  features: Record<string, number | null>,
  rowId = 'dashboard-row',
) {
  return request<QmlPrediction>('/api/predict', {
    method: 'POST',
    body: JSON.stringify({ features, row_id: rowId, explain: true }),
  })
}

export function predictEhrCase(
  input: unknown,
  patientId: string | undefined,
  indexTime: string,
  rowId = 'ehr-case',
) {
  return request<QmlPrediction>('/api/predict', {
    method: 'POST',
    body: JSON.stringify({
      ehr_input: input,
      patient_id: patientId,
      index_time: indexTime,
      row_id: rowId,
      explain: true,
    }),
  })
}

export function modelEntries(result: QmlResult | null | undefined) {
  return Object.entries(result?.models ?? {})
}

export function metricValue(model: QmlModelResult | undefined, key: QmlMetricKey) {
  const value = model?.metrics?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function modelLabel(name: string) {
  return name
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function formatMetric(value: number | null | undefined, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a'
}
