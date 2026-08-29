import { Vqc } from '../quantum/vqc'
import type { PcaResult } from './features'
import type { Matrix } from './stats'

/**
 * Explainability.
 *
 * Two problems are solved here. The first is ordinary: attribute a prediction
 * to the inputs the model actually saw. The second is the one that matters
 * clinically - when the model was trained on principal components, "component
 * 3 contributed 0.4" tells a doctor nothing. Attributions must be pushed back
 * through the PCA rotation onto the original measurements.
 */

export type Attribution = {
  name: string
  /** signed contribution toward the positive class */
  value: number
  /** the input value that produced it, in original units where known */
  raw?: number
  /** clinical meaning, when the dataset supplies one */
  description?: string | null
}

/**
 * Permutation importance: shuffle one feature and measure how much the model's
 * performance degrades. Model-agnostic, so it works identically for the VQC
 * and every classical baseline.
 */
export function permutationImportance(
  predict: (X: Matrix) => number[],
  X: Matrix,
  y: number[],
  featureNames: string[],
  rng: () => number,
  repeats = 3,
): Attribution[] {
  const baseline = logLoss(y, predict(X))
  const d = X[0]?.length ?? 0
  const out: Attribution[] = []

  for (let j = 0; j < d; j++) {
    let total = 0
    for (let r = 0; r < repeats; r++) {
      const shuffled = X.map((row) => [...row])
      // Fisher-Yates on column j only.
      for (let i = shuffled.length - 1; i > 0; i--) {
        const k = Math.floor(rng() * (i + 1))
        const tmp = shuffled[i][j]
        shuffled[i][j] = shuffled[k][j]
        shuffled[k][j] = tmp
      }
      total += logLoss(y, predict(shuffled)) - baseline
    }
    out.push({ name: featureNames[j] ?? `f${j}`, value: total / repeats })
  }

  return out.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
}

function logLoss(y: number[], p: number[]): number {
  let s = 0
  for (let i = 0; i < y.length; i++) {
    const q = Math.min(Math.max(p[i], 1e-7), 1 - 1e-7)
    s += -(y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q))
  }
  return s / Math.max(1, y.length)
}

/**
 * Per-sample attribution by occlusion: replace one feature with the dataset
 * mean and measure how far the prediction moves. The sign says which way that
 * feature pushed this particular patient.
 *
 * Occlusion rather than SHAP because it needs one extra forward pass per
 * feature rather than exponentially many coalitions, and on a quantum circuit
 * every forward pass is expensive.
 */
export function localAttribution(
  predictOne: (x: number[]) => number,
  x: number[],
  baseline: number[],
  featureNames: string[],
): Attribution[] {
  const base = predictOne(x)
  const out: Attribution[] = []

  for (let j = 0; j < x.length; j++) {
    const occluded = [...x]
    occluded[j] = baseline[j]
    // Positive means: having this value rather than the average pushed the
    // prediction toward the positive class.
    out.push({
      name: featureNames[j] ?? `f${j}`,
      value: base - predictOne(occluded),
      raw: x[j],
    })
  }

  return out.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
}

/**
 * Maps component-space attributions back onto the original clinical features.
 *
 * This is the hard part. After PCA each component is a weighted mixture of
 * every original measurement:
 *
 *   PC_c = sum_j  loading[c][j] * (x_j - mean_j) / sd_j
 *
 * so a component's attribution distributes over the original features in
 * proportion to its loadings. Summing across components gives each real
 * measurement's total contribution:
 *
 *   attribution(x_j) = sum_c  attribution(PC_c) * loading[c][j]
 *
 * The sign is preserved, so "larger tumour radius pushed this toward
 * malignant" survives the round trip - which is the only form a clinician can
 * act on.
 */
export function mapComponentsToFeatures(
  componentAttributions: Attribution[],
  pcaResult: PcaResult,
  featureNames: string[],
  originalValues?: number[],
  describe?: (name: string) => string | null,
): Attribution[] {
  const d = featureNames.length
  const totals = new Array(d).fill(0)

  componentAttributions.forEach((attr, c) => {
    const loadings = pcaResult.loadings[c]
    if (!loadings) return
    for (let j = 0; j < d; j++) {
      totals[j] += attr.value * loadings[j]
    }
  })

  return featureNames
    .map((name, j) => ({
      name,
      value: totals[j],
      raw: originalValues?.[j],
      description: describe?.(name) ?? null,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
}

/**
 * How much each original feature contributes to a component, for the
 * "what is PC1 made of" readout.
 */
export function componentComposition(
  pcaResult: PcaResult,
  component: number,
  featureNames: string[],
  topN = 5,
): { name: string; loading: number }[] {
  const loadings = pcaResult.loadings[component]
  if (!loadings) return []
  return featureNames
    .map((name, j) => ({ name, loading: loadings[j] }))
    .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
    .slice(0, topN)
}

/** Convenience wrapper: attribution for one VQC prediction. */
export function explainVqcPrediction(
  model: Vqc,
  x: number[],
  baseline: number[],
  featureNames: string[],
): Attribution[] {
  return localAttribution((v) => model.predictOne(v), x, baseline, featureNames)
}

/** Column means of the training matrix, the neutral reference for occlusion. */
export function columnMeans(X: Matrix): number[] {
  const d = X[0]?.length ?? 0
  const out = new Array(d).fill(0)
  for (const row of X) for (let j = 0; j < d; j++) out[j] += row[j]
  return out.map((v) => v / Math.max(1, X.length))
}

export type RiskBand = 'low' | 'moderate' | 'high'

export function riskBand(probability: number): RiskBand {
  if (probability < 0.33) return 'low'
  if (probability < 0.66) return 'moderate'
  return 'high'
}
