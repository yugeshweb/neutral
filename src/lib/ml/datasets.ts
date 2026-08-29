import { makeRng } from '../quantum/statevector'
import type { Dataset } from './pipeline'

/**
 * Built-in benchmark datasets.
 *
 * These are generated from the published summary statistics of the real
 * studies - per-class means, standard deviations and the correlation structure
 * between related measurements - not copied records. That keeps the repo free
 * of redistributed patient data while giving the pipeline something with
 * realistic scale, skew and class overlap to work on.
 *
 * They are labelled as synthetic wherever they appear. A user who wants the
 * genuine article uploads the CSV, and every adapter path already handles it.
 */

export type DatasetMeta = {
  id: string
  name: string
  source: string
  rows: number
  positiveLabel: string
  negativeLabel: string
  /** clinical description per feature, for the explainability mapping */
  featureDescriptions: Record<string, string>
  synthetic: boolean
}

type FeatureSpec = {
  name: string
  description: string
  /** mean for the negative and positive class */
  negMean: number
  posMean: number
  negSd: number
  posSd: number
  /** log-normal features are skewed like real measurements */
  skewed?: boolean
  /** index of a feature this one is derived from, creating collinearity */
  derivedFrom?: number
  derivedScale?: number
}

/** Box-Muller, so the generated columns are genuinely Gaussian. */
function gauss(rng: () => number): number {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function build(
  specs: FeatureSpec[],
  nRows: number,
  positiveRate: number,
  seed: number,
): { X: number[][]; y: number[] } {
  const rng = makeRng(seed)
  const X: number[][] = []
  const y: number[] = []

  for (let i = 0; i < nRows; i++) {
    const positive = rng() < positiveRate ? 1 : 0
    const row: number[] = []

    for (let j = 0; j < specs.length; j++) {
      const s = specs[j]

      if (s.derivedFrom !== undefined) {
        // Collinear by construction: the collinearity-drop step has real work.
        const base = row[s.derivedFrom]
        row.push(base * (s.derivedScale ?? 1) + gauss(rng) * s.negSd * 0.12)
        continue
      }

      const mean = positive ? s.posMean : s.negMean
      const sd = positive ? s.posSd : s.negSd
      let v = mean + gauss(rng) * sd

      if (s.skewed) {
        // Right-skew, as physical measurements usually are.
        v = mean * Math.exp((v - mean) / Math.max(mean, 1e-6) - 0.05)
      }
      row.push(Math.max(0, v))
    }

    X.push(row)
    y.push(positive)
  }
  return { X, y }
}

// ---- Breast Cancer Wisconsin ----------------------------------------------

const WDBC_SPECS: FeatureSpec[] = [
  { name: 'radius_mean', description: 'Mean distance from centre to points on the nucleus perimeter', negMean: 12.15, posMean: 17.46, negSd: 1.78, posSd: 3.2 },
  { name: 'texture_mean', description: 'Standard deviation of grey-scale values in the nucleus', negMean: 17.91, posMean: 21.6, negSd: 4.0, posSd: 3.78 },
  { name: 'perimeter_mean', description: 'Nucleus perimeter length', negMean: 78.08, posMean: 115.4, negSd: 11.8, posSd: 21.9, derivedFrom: 0, derivedScale: 6.45 },
  { name: 'area_mean', description: 'Nucleus cross-sectional area', negMean: 462.8, posMean: 978.4, negSd: 134.3, posSd: 368, skewed: true },
  { name: 'smoothness_mean', description: 'Local variation in radius lengths', negMean: 0.0925, posMean: 0.1029, negSd: 0.0134, posSd: 0.0126 },
  { name: 'compactness_mean', description: 'Perimeter squared over area, minus one', negMean: 0.08, posMean: 0.145, negSd: 0.034, posSd: 0.054 },
  { name: 'concavity_mean', description: 'Severity of concave portions of the contour', negMean: 0.046, posMean: 0.161, negSd: 0.043, posSd: 0.075 },
  { name: 'concave_points_mean', description: 'Number of concave portions of the contour', negMean: 0.0257, posMean: 0.0880, negSd: 0.0159, posSd: 0.0344 },
  { name: 'symmetry_mean', description: 'Symmetry of the nucleus about its axis', negMean: 0.174, posMean: 0.193, negSd: 0.0248, posSd: 0.0276 },
  { name: 'fractal_dimension_mean', description: 'Coastline approximation of the boundary complexity', negMean: 0.0629, posMean: 0.0627, negSd: 0.0067, posSd: 0.0076 },
  { name: 'radius_worst', description: 'Largest mean radius across the three worst measurements', negMean: 13.38, posMean: 21.13, negSd: 1.98, posSd: 4.28 },
  { name: 'texture_worst', description: 'Worst-case grey-scale standard deviation', negMean: 23.52, posMean: 29.32, negSd: 5.49, posSd: 5.44 },
  { name: 'perimeter_worst', description: 'Worst-case nucleus perimeter', negMean: 87.0, posMean: 141.4, negSd: 13.5, posSd: 29.5, derivedFrom: 10, derivedScale: 6.67 },
  { name: 'area_worst', description: 'Worst-case nucleus area', negMean: 558.9, posMean: 1422.3, negSd: 163.6, posSd: 597.9, skewed: true },
  { name: 'smoothness_worst', description: 'Worst-case local radius variation', negMean: 0.125, posMean: 0.145, negSd: 0.020, posSd: 0.022 },
  { name: 'concavity_worst', description: 'Worst-case contour concavity', negMean: 0.166, posMean: 0.451, negSd: 0.140, posSd: 0.182 },
  { name: 'concave_points_worst', description: 'Worst-case count of concave contour portions', negMean: 0.0744, posMean: 0.1822, negSd: 0.0359, posSd: 0.0461 },
  { name: 'symmetry_worst', description: 'Worst-case nucleus symmetry', negMean: 0.270, posMean: 0.323, negSd: 0.0416, posSd: 0.0747 },
]

// ---- Heart Disease UCI -----------------------------------------------------

const HEART_SPECS: FeatureSpec[] = [
  { name: 'age', description: 'Age in years', negMean: 52.6, posMean: 56.6, negSd: 9.6, posSd: 7.9 },
  { name: 'resting_bp', description: 'Resting systolic blood pressure (mm Hg)', negMean: 129.2, posMean: 134.4, negSd: 16.2, posSd: 18.7 },
  { name: 'cholesterol', description: 'Serum cholesterol (mg/dL)', negMean: 242.6, posMean: 251.5, negSd: 53.5, posSd: 49.4, skewed: true },
  { name: 'max_heart_rate', description: 'Maximum heart rate achieved during exercise', negMean: 158.4, posMean: 139.1, negSd: 19.2, posSd: 22.6 },
  { name: 'st_depression', description: 'ST depression induced by exercise relative to rest', negMean: 0.58, posMean: 1.57, negSd: 0.78, posSd: 1.30 },
  { name: 'num_vessels', description: 'Major vessels coloured by fluoroscopy (0 to 3)', negMean: 0.27, posMean: 1.13, negSd: 0.64, posSd: 1.02 },
  { name: 'chest_pain_type', description: 'Chest pain category, higher is more atypical', negMean: 2.79, posMean: 3.56, negSd: 0.92, posSd: 0.72 },
  { name: 'exercise_angina', description: 'Exercise-induced angina present', negMean: 0.14, posMean: 0.55, negSd: 0.35, posSd: 0.50 },
  { name: 'st_slope', description: 'Slope of the peak exercise ST segment', negMean: 1.41, posMean: 1.83, negSd: 0.59, posSd: 0.55 },
  { name: 'fasting_blood_sugar', description: 'Fasting blood sugar above 120 mg/dL', negMean: 0.14, posMean: 0.16, negSd: 0.35, posSd: 0.37 },
]

// ---- Parkinsons ------------------------------------------------------------

const PARKINSONS_SPECS: FeatureSpec[] = [
  { name: 'mdvp_fo', description: 'Average vocal fundamental frequency (Hz)', negMean: 181.9, posMean: 145.2, negSd: 34.4, posSd: 40.4 },
  { name: 'mdvp_fhi', description: 'Maximum vocal fundamental frequency (Hz)', negMean: 223.6, posMean: 188.4, negSd: 71.0, posSd: 92.5, skewed: true },
  { name: 'mdvp_flo', description: 'Minimum vocal fundamental frequency (Hz)', negMean: 145.2, posMean: 106.9, negSd: 43.5, posSd: 44.7 },
  { name: 'jitter_percent', description: 'Cycle-to-cycle variation in fundamental frequency', negMean: 0.0039, posMean: 0.0069, negSd: 0.0018, posSd: 0.0053 },
  { name: 'jitter_abs', description: 'Absolute jitter in microseconds', negMean: 0.0000239, posMean: 0.0000504, negSd: 0.0000112, posSd: 0.0000372 },
  { name: 'shimmer', description: 'Cycle-to-cycle variation in amplitude', negMean: 0.0176, posMean: 0.0336, negSd: 0.0076, posSd: 0.0194 },
  { name: 'shimmer_db', description: 'Amplitude variation in decibels', negMean: 0.162, posMean: 0.321, negSd: 0.070, posSd: 0.187, derivedFrom: 5, derivedScale: 9.2 },
  { name: 'nhr', description: 'Noise-to-harmonics ratio in the voice signal', negMean: 0.0115, posMean: 0.0292, negSd: 0.0100, posSd: 0.0402, skewed: true },
  { name: 'hnr', description: 'Harmonics-to-noise ratio (dB)', negMean: 24.68, posMean: 20.97, negSd: 4.29, posSd: 4.46 },
  { name: 'rpde', description: 'Recurrence period density entropy, a nonlinear measure', negMean: 0.443, posMean: 0.517, negSd: 0.096, posSd: 0.100 },
  { name: 'dfa', description: 'Detrended fluctuation analysis, signal fractal scaling', negMean: 0.696, posMean: 0.725, negSd: 0.058, posSd: 0.053 },
  { name: 'spread1', description: 'Nonlinear measure of fundamental frequency variation', negMean: -6.76, posMean: -5.33, negSd: 1.17, posSd: 1.09 },
  { name: 'ppe', description: 'Pitch period entropy', negMean: 0.123, posMean: 0.233, negSd: 0.054, posSd: 0.084 },
]

export const DATASET_META: DatasetMeta[] = [
  {
    id: 'breast-cancer',
    name: 'Breast Cancer Wisconsin',
    source: 'UCI ML Repository / WDBC',
    rows: 569,
    positiveLabel: 'malignant',
    negativeLabel: 'benign',
    featureDescriptions: Object.fromEntries(WDBC_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
  {
    id: 'heart-disease',
    name: 'Heart Disease UCI',
    source: 'UCI ML Repository / Cleveland',
    rows: 303,
    positiveLabel: 'disease present',
    negativeLabel: 'no disease',
    featureDescriptions: Object.fromEntries(HEART_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
  {
    id: 'parkinsons',
    name: "Parkinson's Voice",
    source: 'UCI ML Repository / Oxford',
    rows: 195,
    positiveLabel: "Parkinson's",
    negativeLabel: 'healthy',
    featureDescriptions: Object.fromEntries(PARKINSONS_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
]

const SPECS: Record<string, { specs: FeatureSpec[]; rows: number; rate: number }> = {
  'breast-cancer': { specs: WDBC_SPECS, rows: 569, rate: 0.373 },
  'heart-disease': { specs: HEART_SPECS, rows: 303, rate: 0.459 },
  parkinsons: { specs: PARKINSONS_SPECS, rows: 195, rate: 0.754 },
}

const cache = new Map<string, Dataset>()

export function loadDataset(id: string): Dataset {
  const hit = cache.get(id)
  if (hit) return hit

  const entry = SPECS[id]
  const meta = DATASET_META.find((m) => m.id === id)
  if (!entry || !meta) throw new Error(`unknown dataset: ${id}`)

  const { X, y } = build(entry.specs, entry.rows, entry.rate, 1234)
  const ds: Dataset = {
    id,
    name: meta.name,
    featureNames: entry.specs.map((s) => s.name),
    X,
    y,
    positiveLabel: meta.positiveLabel,
    negativeLabel: meta.negativeLabel,
  }
  cache.set(id, ds)
  return ds
}

export function datasetMeta(id: string): DatasetMeta | undefined {
  return DATASET_META.find((m) => m.id === id)
}

/** Clinical description for a feature, for the explainability screen. */
export function describeFeature(datasetId: string, feature: string): string | null {
  return datasetMeta(datasetId)?.featureDescriptions[feature] ?? null
}
