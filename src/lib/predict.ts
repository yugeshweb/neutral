/**
 * Inference and Feature Attribution for the prediction view.
 *
 * Connects clinical morphological features with signed logit attributions
 * and pathological explainability.
 *
 * TODO: when a live model API is served, swap `predict` with a call to
 * POST /api/predict/explain or the service contract in `lib/explainability.ts`.
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
  /** direction and strength of the contribution */
  weight: number
  /** clinical / pathological meaning of this morphometric feature */
  description?: string
}

/**
 * The eight features the pipeline selection stage retains. Ranges are the
 * observed spread in the Wisconsin Diagnostic Breast Cancer (WDBC) cohort.
 */
export const FEATURES: FeatureSpec[] = [
  {
    id: 'radius_mean',
    label: 'Radius (mean)',
    unit: 'μm',
    min: 6,
    max: 29,
    step: 0.1,
    typical: 14.1,
    weight: 1.15,
    description: 'Mean distance from center to nuclear boundary. Increased radius indicates nucleomegaly and accelerated cell cycling.',
  },
  {
    id: 'texture_mean',
    label: 'Texture (mean)',
    unit: 'std',
    min: 9,
    max: 40,
    step: 0.1,
    typical: 19.3,
    weight: 0.62,
    description: 'Standard deviation of gray-scale pixel intensities inside chromatin. Reflects coarse chromatin clumping and hyperchromasia.',
  },
  {
    id: 'perimeter_worst',
    label: 'Perimeter (worst)',
    unit: 'μm',
    min: 50,
    max: 252,
    step: 0.5,
    typical: 107.3,
    weight: 1.42,
    description: 'Largest nuclear perimeter among cell clusters. Markedly elevated values signal prominent pleomorphic giant cells.',
  },
  {
    id: 'area_worst',
    label: 'Area (worst)',
    unit: 'μm²',
    min: 185,
    max: 4254,
    step: 5,
    typical: 880.6,
    weight: 1.08,
    description: 'Peak nuclear cross-sectional area. Severe outliers represent nuclear aneuploidy and malignant dedifferentiation.',
  },
  {
    id: 'smoothness_worst',
    label: 'Smoothness (worst)',
    unit: '',
    min: 0.07,
    max: 0.23,
    step: 0.001,
    typical: 0.132,
    weight: 0.55,
    description: 'Local radial variance along the perimeter. Irregular borders reflect disrupted nuclear envelope lamina.',
  },
  {
    id: 'concavity_mean',
    label: 'Concavity (mean)',
    unit: '',
    min: 0,
    max: 0.43,
    step: 0.001,
    typical: 0.089,
    weight: 0.94,
    description: 'Severity of inward contour notches. Deep notches signify nuclear envelope folding and cellular atypia.',
  },
  {
    id: 'concave_points_mean',
    label: 'Concave points (mean)',
    unit: '',
    min: 0,
    max: 0.21,
    step: 0.001,
    typical: 0.049,
    weight: 1.61,
    description: 'Frequency of discrete membrane indentations. Strongest single morphological differentiator of malignancy.',
  },
  {
    id: 'symmetry_worst',
    label: 'Symmetry (worst)',
    unit: '',
    min: 0.16,
    max: 0.67,
    step: 0.001,
    typical: 0.29,
    weight: 0.48,
    description: 'Asymmetry across perpendicular nuclear axes. High asymmetry is a hallmark of uncoordinated malignant mitosis.',
  },
]

export type Attribution = {
  id: string
  label: string
  /** signed contribution to the logit, malignant-positive */
  contribution: number
  description?: string
  value?: number
  typical?: number
  unit?: string
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

/** Presets matching clinical reference presentations */
export const PRESETS: { id: string; label: string; note: string; values: FeatureValues }[] = [
  {
    id: 'benign',
    label: 'Benign-leaning case',
    note: 'small, regular, uniform nuclei',
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
    note: 'large, irregular, pleomorphic nuclei',
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
  {
    id: 'borderline',
    label: 'Borderline induration',
    note: 'moderate size with focal nuclear notches',
    values: {
      radius_mean: 14.8,
      texture_mean: 20.1,
      perimeter_worst: 98.4,
      area_worst: 695,
      smoothness_worst: 0.138,
      concavity_mean: 0.142,
      concave_points_mean: 0.078,
      symmetry_worst: 0.312,
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
 * Deterministic logistic score over the eight features.
 * Attributions measure signed logit displacement relative to the population mean baseline.
 */
export function predict(values: FeatureValues, variant: 'classical' | 'quantum'): Prediction {
  const gain = variant === 'quantum' ? 2.35 : 2.05

  const attributions: Attribution[] = FEATURES.map((spec) => {
    const rawVal = values[spec.id] ?? spec.typical
    return {
      id: spec.id,
      label: spec.label,
      contribution: standardise(spec, rawVal) * spec.weight * gain,
      description: spec.description,
      value: rawVal,
      typical: spec.typical,
      unit: spec.unit,
    }
  })

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
