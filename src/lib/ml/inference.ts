import { Vqc } from '../quantum/vqc'
import type { TrainedPipelineArtifact } from '../diseaseRegistry'
import { localAttribution, type Attribution } from './explain'
import { applyScaler } from './stats'

/**
 * Scoring a single case with the model that was actually fitted.
 *
 * The Predict screen used to compute a hand-weighted sigmoid over the raw
 * slider values, which meant training and inference shared nothing: retraining
 * changed the reported metrics but not a single prediction. This module closes
 * that gap by replaying the saved transform in the same order the pipeline
 * applied it, then running the persisted circuit parameters.
 *
 * Order matters and mirrors `runPipeline` exactly:
 *   impute -> scale (train-fitted) -> select or project -> circuit
 * Any deviation silently produces a number that looks plausible and is wrong.
 */

export type CaseScore = {
  /** positive-class probability from the trained VQC */
  probability: number
  /** the model-space vector the circuit actually saw */
  vector: number[]
  /** names aligned to `vector`, for attribution display */
  vectorNames: string[]
  attributions: Attribution[]
}

/** True when the artifact carries everything inference needs. */
export function isReplayable(a: TrainedPipelineArtifact | null): a is TrainedPipelineArtifact {
  return Boolean(
    a &&
      a.quantumWeights &&
      a.quantumConfig &&
      a.baselineVector &&
      (a.pca || a.keptIndices),
  )
}

/**
 * Applies the saved preprocessing chain to one raw feature row.
 * `raw` must be ordered by the artifact's `featureNames`.
 */
export function projectCase(artifact: TrainedPipelineArtifact, raw: number[]): number[] {
  const fill = artifact.imputeValues ?? []
  const imputed = raw.map((v, j) => (Number.isFinite(v) ? v : (fill[j] ?? 0)))

  // Scaler was fitted on the training fold only; reuse it, never refit.
  const scaled = applyScaler([imputed], artifact.scaler)[0]

  if (artifact.pca) {
    const { mean, scale, loadings } = artifact.pca
    // Must match pcaTransform exactly: standardise with the stored mean AND
    // scale before projecting. Centring alone silently distorts every
    // component whenever the columns have unequal spread.
    const z = scaled.map((v, j) => (v - (mean[j] ?? 0)) / (scale[j] || 1))
    return loadings.map((comp) => comp.reduce((s, w, j) => s + w * (z[j] ?? 0), 0))
  }

  return (artifact.keptIndices ?? []).map((j) => scaled[j] ?? 0)
}

/** Rebuilds the trained circuit from its persisted parameters. */
export function restoreVqc(artifact: TrainedPipelineArtifact): Vqc {
  return new Vqc(artifact.quantumConfig!, Float64Array.from(artifact.quantumWeights!))
}

/**
 * Full single-case inference: probability plus per-feature attribution.
 *
 * Attribution is occlusion against the training-set mean - replace one input
 * with its average value and measure how far the prediction moves. It is the
 * same reference the explain module uses elsewhere, so the numbers on this
 * screen are comparable with the ones computed during training.
 */
export function scoreCase(artifact: TrainedPipelineArtifact, raw: number[]): CaseScore {
  const model = restoreVqc(artifact)
  const vector = projectCase(artifact, raw)
  const baseline = artifact.baselineVector ?? new Array(vector.length).fill(0)

  const vectorNames = artifact.pca
    ? vector.map((_, i) => `PC${i + 1}`)
    : (artifact.keptIndices ?? []).map((j) => artifact.featureNames[j] ?? `f${j}`)

  const attributions = localAttribution(
    (v) => model.predictOne(v),
    vector,
    baseline,
    vectorNames,
  )

  return {
    probability: model.predictOne(vector),
    vector,
    vectorNames,
    attributions,
  }
}

export type BatchRow = {
  /** 1-based row number in the source file, for reporting */
  line: number
  probability: number
  positive: boolean
}

export type BatchResult = {
  rows: BatchRow[]
  /** rows skipped because no column matched a feature the model needs */
  skipped: number
  /** feature names in the file that the model actually uses */
  matched: string[]
  /** features the model expects that the file does not supply */
  missing: string[]
  positiveCount: number
}

/**
 * Scores every row of an uploaded table with the trained model.
 *
 * Columns are matched to the artifact's features by name, so column order in
 * the file does not matter and extra columns are ignored. A feature the model
 * needs but the file omits falls back to that feature's stored training-set
 * fill value, exactly as single-case scoring does - and the caller is told
 * which ones those were, because a prediction resting mostly on fill values
 * is not worth much and the user should be able to see that.
 */
export function scoreBatch(
  artifact: TrainedPipelineArtifact,
  headers: string[],
  rows: string[][],
  threshold = 0.5,
): BatchResult {
  const model = restoreVqc(artifact)

  // Map each of the model's features to a column index in the file.
  // Case-insensitive: a predict-time upload (PDF/HL7/FHIR extraction, a
  // hand-typed CSV header) naming the same measurement "MMSE" where training
  // saw "mmse" is the same column, not a missing one - matching on the exact
  // byte string here is the kind of standardizer/model disagreement that
  // fails silently rather than loudly, so this widens the match rather than
  // narrowing it. An exact-case match still wins first if both exist.
  const columnOf = new Map<string, number>()
  const columnOfCi = new Map<string, number>()
  headers.forEach((h, i) => {
    const trimmed = h.trim()
    columnOf.set(trimmed, i)
    if (!columnOfCi.has(trimmed.toLowerCase())) columnOfCi.set(trimmed.toLowerCase(), i)
  })
  const resolveColumn = (name: string): number | undefined =>
    columnOf.get(name) ?? columnOfCi.get(name.toLowerCase())

  const matched: string[] = []
  const missing: string[] = []
  for (const name of artifact.featureNames) {
    if (resolveColumn(name) !== undefined) matched.push(name)
    else missing.push(name)
  }

  const out: BatchRow[] = []
  let skipped = 0
  let positiveCount = 0

  rows.forEach((cells, index) => {
    // A row with nothing the model recognises is reported, not silently scored
    // off fill values alone.
    let usable = 0
    const raw = artifact.featureNames.map((name) => {
      const col = resolveColumn(name)
      if (col === undefined) return Number.NaN
      const v = Number(cells[col])
      if (Number.isFinite(v)) {
        usable++
        return v
      }
      return Number.NaN
    })

    if (usable === 0) {
      skipped++
      return
    }

    const probability = model.predictOne(projectCase(artifact, raw))
    if (!Number.isFinite(probability)) {
      skipped++
      return
    }

    const positive = probability >= threshold
    if (positive) positiveCount++
    out.push({ line: index + 2, probability, positive })
  })

  return { rows: out, skipped, matched, missing, positiveCount }
}
