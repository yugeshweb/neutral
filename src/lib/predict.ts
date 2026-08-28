/**
 * Mock inference for the prediction view.
 *
 * Nothing here is a trained model. The "prediction" is a fixed logistic
 * function over the eight selected features, with hand-chosen weights that
 * merely reproduce the direction of the textbook WDBC relationships (larger,
 * more irregular nuclei read as malignant). It is deterministic, so the same
 * inputs always yield the same output, and it is obviously synthetic on
 * inspection.
 *
 * TODO: when a real served model exists, replace `predict` with a call that
 * returns the same `Prediction` shape. The panel reads only this interface.
 */

export type FeatureSpec = {
  id: string
  label: string
  /** clinical unit, shown after the value */
  unit: string
  min: number
  max: number
  step: number
  /** population mean, used as the reset value */
  typical: number
  /** direction and strength of the mock contribution */
  weight: number
}

/**
 * The eight features the pipeline's selection stage retains. Ranges are the
 * observed spread in the Wisconsin Breast Cancer study.
 */
export const FEATURES: FeatureSpec[] = [
  { id: 'radius_mean', label: 'Radius (mean)', unit: 'mm', min: 6, max: 29, step: 0.1, typical: 14.1, weight: 1.15 },
  { id: 'texture_mean', label: 'Texture (mean)', unit: 'sd', min: 9, max: 40, step: 0.1, typical: 19.3, weight: 0.62 },
  { id: 'perimeter_worst', label: 'Perimeter (worst)', unit: 'mm', min: 50, max: 252, step: 0.5, typical: 107.3, weight: 1.42 },
  { id: 'area_worst', label: 'Area (worst)', unit: 'mm²', min: 185, max: 4254, step: 5, typical: 880.6, weight: 1.08 },
  { id: 'smoothness_worst', label: 'Smoothness (worst)', unit: '', min: 0.07, max: 0.23, step: 0.001, typical: 0.132, weight: 0.55 },
  { id: 'concavity_mean', label: 'Concavity (mean)', unit: '', min: 0, max: 0.43, step: 0.001, typical: 0.089, weight: 0.94 },
  { id: 'concave_points_mean', label: 'Concave points (mean)', unit: '', min: 0, max: 0.21, step: 0.001, typical: 0.049, weight: 1.61 },
  { id: 'symmetry_worst', label: 'Symmetry (worst)', unit: '', min: 0.16, max: 0.67, step: 0.001, typical: 0.29, weight: 0.48 },
]

export type Attribution = {
  id: string
  label: string
  /** signed contribution to the logit, malignant-positive */
  contribution: number
}

export type Prediction = {
  /** malignant probability, 0..1 */
  probability: number
  label: 'benign' | 'malignant'
  /** margin from the decision threshold, 0..1 */
  confidence: number
  /** per-feature contributions, largest magnitude first */
  attributions: Attribution[]
}

export type FeatureValues = Record<string, number>

export const DEFAULT_VALUES: FeatureValues = Object.fromEntries(
  FEATURES.map((f) => [f.id, f.typical]),
)

/** Two stored cases, so the demo can show both sides of the boundary. */
export const PRESETS: { id: string; label: string; note: string; values: FeatureValues }[] = [
  {
    id: 'benign',
    label: 'Benign-leaning case',
    note: 'small, regular nuclei',
    values: {
      radius_mean: 11.4,
      texture_mean: 15.8,
      perimeter_worst: 78.2,
      area_worst: 452,
      smoothness_worst: 0.114,
      concavity_mean: 0.031,
      concave_points_mean: 0.018,
      symmetry_worst: 0.242,
    },
  },
  {
    id: 'malignant',
    label: 'Malignant-leaning case',
    note: 'large, irregular nuclei',
    values: {
      radius_mean: 20.6,
      texture_mean: 26.4,
      perimeter_worst: 168.5,
      area_worst: 2020,
      smoothness_worst: 0.176,
      concavity_mean: 0.244,
      concave_points_mean: 0.142,
      symmetry_worst: 0.412,
    },
  },
]

/** Scale a raw feature to roughly [-1, 1] about its population mean. */
function standardise(spec: FeatureSpec, value: number) {
  const half = (spec.max - spec.min) / 2
  return half === 0 ? 0 : (value - spec.typical) / half
}

const BIAS = -0.35

/**
 * Deterministic logistic score over the eight features. `variant` shifts the
 * result slightly so the quantum and classical readouts differ the way the
 * benchmark table claims they do - the quantum head is a touch more decisive.
 */
export function predict(values: FeatureValues, variant: 'classical' | 'quantum'): Prediction {
  const gain = variant === 'quantum' ? 2.35 : 2.05

  const attributions: Attribution[] = FEATURES.map((spec) => ({
    id: spec.id,
    label: spec.label,
    contribution: standardise(spec, values[spec.id] ?? spec.typical) * spec.weight * gain,
  }))

  const logit = attributions.reduce((sum, a) => sum + a.contribution, 0) + BIAS
  const probability = 1 / (1 + Math.exp(-logit))

  return {
    probability,
    label: probability >= 0.5 ? 'malignant' : 'benign',
    confidence: Math.abs(probability - 0.5) * 2,
    attributions: [...attributions].sort(
      (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
    ),
  }
}
