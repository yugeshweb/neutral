/**
 * Explainability & Clinical Decision Support Service
 *
 * This service handles clinical model explainability, feature attribution,
 * and Region of Interest (ROI) radiological findings.
 *
 * It is built to consume the standardized API response contract defined in
 * /mock/mock.json. When an actual backend model service is running, swapping
 * `API_EXPLAIN_URL` or passing a live prediction payload seamlessly switches
 * from mock demonstration data to genuine model inferences without any UI changes.
 */

export type FeatureDirection = 'malignant' | 'benign' | 'neutral'
export type SeverityLevel = 'low' | 'moderate' | 'high'

export interface ClinicalFeatureAttribution {
  feature_id: string
  label: string
  unit: string
  value: number
  population_typical: number
  contribution: number
  direction: FeatureDirection
  clinical_significance: string
}

export interface RoiCoordinates {
  x: number
  y: number
  r: number
  bbox?: {
    x_min: number
    y_min: number
    x_max: number
    y_max: number
  }
}

export interface RoiFinding {
  id: string
  label: string
  severity: SeverityLevel
  confidence: number
  coordinates: RoiCoordinates
  significance: string
  pathological_mechanism: string
  differential_diagnoses: string[]
  measurable_metrics: Record<string, string>
  recommended_action: string
  notes?: string[]
}

export interface GradCamSaliency {
  target_layer: string
  target_class: string
  peak_activation: number
  colormap?: 'jet' | 'turbo' | 'viridis' | 'inferno'
  interpolation?: 'bilinear' | 'bicubic'
  resolution?: [number, number]
  matrix?: number[][]
}

export interface PredictionHead {
  probability: number
  label: string
  decision_margin: number
  uncertainty_interval?: [number, number]
}

export interface PredictionNarrative {
  verdict_summary: string
  primary_drivers: string
  counter_evidence: string
  clinical_confidence: string
  caveats: string[]
}

export interface ConditionExplainability {
  condition_id: string
  condition_name: string
  modality: string
  task_type: string
  model: {
    id: string
    name: string
    quantum_backend?: string
    classical_baseline?: string
    calibration_status?: string
    threshold: number
    gradcam?: GradCamSaliency
  }
  prediction: {
    probability: number
    label: string
    confidence_margin: number
    quantum_head: PredictionHead
    classical_head: PredictionHead
    consensus: string
  }
  narrative: PredictionNarrative
  feature_attributions: ClinicalFeatureAttribution[]
  roi_findings: RoiFinding[]
}

export interface ExplainabilityPayload {
  $schema?: string
  version: string
  endpoint: string
  source: 'mock' | 'live_model'
  timestamp: string
  metadata: {
    synthetic: boolean
    disclaimer: string
    regulatory_tier: string
    engine: string
  }
  conditions: Record<string, ConditionExplainability>
}

// In-memory cache for fetched mock data
let cachedPayload: ExplainabilityPayload | null = null

/**
 * Fetch explainability data from /mock/mock.json or the configured API endpoint.
 *
 * When an actual model API is connected via VITE_QML_API_URL, this can be
 * switched to POST /api/predict/explain.
 */
export async function fetchExplainability(
  conditionId = 'breast-cancer',
  options?: { signal?: AbortSignal; forceMock?: boolean }
): Promise<ConditionExplainability> {
  const configuredApi = import.meta.env.VITE_QML_API_URL

  // If a live backend is configured and mock is not forced, query live API
  if (configuredApi && !options?.forceMock) {
    try {
      const res = await fetch(`${configuredApi.replace(/\/$/, '')}/api/predict/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ condition_id: conditionId }),
        signal: options?.signal,
      })
      if (res.ok) {
        const live = (await res.json()) as ExplainabilityPayload
        if (live.conditions && live.conditions[conditionId]) {
          return live.conditions[conditionId]
        }
      }
    } catch {
      // Fall through to local /mock/mock.json
    }
  }

  // Use cached payload if available
  if (cachedPayload && cachedPayload.conditions[conditionId]) {
    return cachedPayload.conditions[conditionId]
  }

  try {
    const res = await fetch('/mock/mock.json', { signal: options?.signal })
    if (!res.ok) {
      throw new Error(`Failed to load mock explainability: HTTP ${res.status}`)
    }
    const data = (await res.json()) as ExplainabilityPayload
    cachedPayload = data
    return data.conditions[conditionId] ?? data.conditions['breast-cancer']
  } catch (err) {
    console.warn('[Explainability] Could not fetch /mock/mock.json, falling back to static schema:', err)
    return getFallbackExplainability(conditionId)
  }
}

/**
 * Synchronous / fallback schema provider to ensure zero crash risk
 */
export function getFallbackExplainability(conditionId = 'breast-cancer'): ConditionExplainability {
  return {
    condition_id: conditionId,
    condition_name: conditionId === 'brain-seizure' ? 'Brain Lesion' : 'Breast Carcinoma',
    modality: 'digital_imaging_and_tabular',
    task_type: 'binary_classification',
    model: {
      id: 'hqml-vqc-v2',
      name: 'Hybrid Quantum-Classical VQC',
      threshold: 0.5,
    },
    prediction: {
      probability: 0.88,
      label: 'malignant',
      confidence_margin: 0.76,
      quantum_head: {
        probability: 0.88,
        label: 'malignant',
        decision_margin: 0.76,
      },
      classical_head: {
        probability: 0.84,
        label: 'malignant',
        decision_margin: 0.68,
      },
      consensus: 'concordant_high_suspicion',
    },
    narrative: {
      verdict_summary: 'Investigational model shows elevated probability for targeted condition.',
      primary_drivers: 'Driven predominantly by boundary irregularities and atypical morphometrics.',
      counter_evidence: 'Internal texture parameters remain within normal standard deviation limits.',
      clinical_confidence: 'Decision margin exceeds review threshold. Requires histopathological verification.',
      caveats: [
        'Decision support view only. Non-diagnostic.',
        'Requires certified clinical radiologist / pathologist review.',
      ],
    },
    feature_attributions: [],
    roi_findings: [],
  }
}
