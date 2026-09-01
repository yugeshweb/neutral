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

// ---- Brain Seizure EEG -----------------------------------------------------

const SEIZURE_SPECS: FeatureSpec[] = [
  { name: 'delta_power', description: 'Normalized power in 0.5–4 Hz delta oscillation band', negMean: 0.32, posMean: 0.14, negSd: 0.08, posSd: 0.05 },
  { name: 'theta_power', description: 'Normalized power in 4–8 Hz theta band', negMean: 0.26, posMean: 0.16, negSd: 0.06, posSd: 0.04 },
  { name: 'alpha_power', description: 'Normalized power in 8–13 Hz alpha band', negMean: 0.28, posMean: 0.10, negSd: 0.07, posSd: 0.03 },
  { name: 'beta_power', description: 'Normalized power in 13–30 Hz beta band', negMean: 0.11, posMean: 0.31, negSd: 0.03, posSd: 0.08, skewed: true },
  { name: 'gamma_power', description: 'Normalized power in 30–80 Hz gamma paroxysm band', negMean: 0.03, posMean: 0.29, negSd: 0.015, posSd: 0.09, skewed: true },
  { name: 'spectral_entropy', description: 'Fourier spectral distribution entropy', negMean: 0.82, posMean: 0.52, negSd: 0.06, posSd: 0.09 },
  { name: 'hjorth_mobility', description: 'Mean frequency estimation ratio', negMean: 0.74, posMean: 1.88, negSd: 0.18, posSd: 0.35 },
  { name: 'hjorth_complexity', description: 'Frequency spread / sinusoidal deviation', negMean: 1.12, posMean: 2.38, negSd: 0.22, posSd: 0.45 },
  { name: 'sample_entropy', description: 'Non-linear regularity metric', negMean: 1.28, posMean: 0.58, negSd: 0.25, posSd: 0.15 },
  { name: 'line_length', description: 'Total trajectory variation (sharp spikes)', negMean: 58.0, posMean: 235.0, negSd: 14.0, posSd: 62.0, skewed: true },
]

export const DATASET_META: DatasetMeta[] = [
  {
    id: 'breast-cancer',
    name: 'Breast Cancer Wisconsin (WDBC)',
    source: 'UCI Machine Learning / Clinical FNA',
    rows: 569,
    positiveLabel: 'Malignant',
    negativeLabel: 'Benign',
    featureDescriptions: Object.fromEntries(WDBC_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
  {
    id: 'brain-seizure',
    name: 'Brain Seizure EEG Dynamics',
    source: 'Bonn University Neurophysiology',
    rows: 500,
    positiveLabel: 'Seizure Detected',
    negativeLabel: 'Normal EEG Baseline',
    featureDescriptions: Object.fromEntries(SEIZURE_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
  {
    id: 'heart-disease',
    name: 'Heart Disease & Myocardial Infarction',
    source: 'Cleveland Clinic Foundation / UCI',
    rows: 303,
    positiveLabel: 'High Risk (Stenosis/CAD)',
    negativeLabel: 'Normal / Low Risk',
    featureDescriptions: Object.fromEntries(HEART_SPECS.map((s) => [s.name, s.description])),
    synthetic: true,
  },
]

const SPECS: Record<string, { specs: FeatureSpec[]; rows: number; rate: number }> = {
  'breast-cancer': { specs: WDBC_SPECS, rows: 569, rate: 0.373 },
  'brain-seizure': { specs: SEIZURE_SPECS, rows: 500, rate: 0.400 },
  'heart-disease': { specs: HEART_SPECS, rows: 303, rate: 0.459 },
}

const cache = new Map<string, Dataset>()

/** The one uploaded dataset in play this session, if any. */
export const CUSTOM_DATASET_ID = 'custom'
let customDataset: Dataset | null = null
let customMeta: DatasetMeta | null = null

/**
 * Registers an uploaded file as a trainable dataset, under the fixed id
 * `CUSTOM_DATASET_ID`. Deliberately kept out of `DATASET_META` - that list
 * drives the preset picker cards, and a file that has not been uploaded yet
 * must not appear there.
 */
export function registerCustomDataset(dataset: Omit<Dataset, 'id'>, source: string): Dataset {
  const ds: Dataset = { ...dataset, id: CUSTOM_DATASET_ID }
  customDataset = ds
  customMeta = {
    id: CUSTOM_DATASET_ID,
    name: dataset.name,
    source,
    rows: dataset.X.length,
    positiveLabel: dataset.positiveLabel,
    negativeLabel: dataset.negativeLabel,
    featureDescriptions: {},
    synthetic: false,
  }
  cache.delete(CUSTOM_DATASET_ID)
  return ds
}

export function hasCustomDataset(): boolean {
  return customDataset !== null
}

export function loadDataset(id: string): Dataset {
  const hit = cache.get(id)
  if (hit) return hit

  if (id === CUSTOM_DATASET_ID) {
    if (!customDataset) throw new Error('no uploaded dataset has been registered yet')
    cache.set(id, customDataset)
    return customDataset
  }

  const entry = SPECS[id] ?? SPECS['breast-cancer']
  const meta = DATASET_META.find((m) => m.id === id) ?? DATASET_META[0]

  const { X, y } = build(entry.specs, entry.rows, entry.rate, 1234)
  const ds: Dataset = {
    id: meta.id,
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
  if (id === CUSTOM_DATASET_ID) return customMeta ?? undefined
  return DATASET_META.find((m) => m.id === id)
}

/** Clinical description for a feature, for the explainability screen. */
export function describeFeature(datasetId: string, feature: string): string | null {
  return datasetMeta(datasetId)?.featureDescriptions[feature] ?? null
}

