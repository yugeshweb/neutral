import { makeRng } from '../quantum/statevector'
import { Vqc, trainVqc, type EpochRecord, type VqcConfig } from '../quantum/vqc'
import { makeBaseline, type BaselineKind } from './baselines'
import { pca, pcaTransform, rankFeatures, type PcaResult, type SelectionMethod } from './features'
import { evaluate, mcnemar, rocCurve, type Metrics } from './metrics'
import {
  applyScaler,
  boxStats,
  classCounts,
  fitScaler,
  imputeMissing,
  oversample,
  stratifiedFolds,
  stratifiedSplit,
  type BalanceStrategy,
  type BoxStats,
  type ImputeStrategy,
  type Matrix,
  type Scaler,
} from './stats'

/**
 * The end-to-end run: preprocessing, feature selection, training both lanes,
 * and evaluating them on an identical split.
 *
 * The whole configuration is one serialisable object, which is what makes a
 * run reproducible and exportable.
 */

export type RunConfig = {
  datasetId: string
  targetColumn: string
  testFraction: number
  seed: number
  impute: ImputeStrategy
  scaler: Scaler['kind']
  balance: BalanceStrategy
  selection: SelectionMethod
  /** how many features to keep - this IS the qubit count */
  nFeatures: number
  vqc: VqcConfig
  baselines: BaselineKind[]
  epochs: number
  learningRate: number
  batchSize: number
  cvFolds: number
}

export const DEFAULT_RUN: RunConfig = {
  datasetId: 'breast-cancer',
  targetColumn: 'diagnosis',
  testFraction: 0.25,
  seed: 42,
  impute: 'median',
  scaler: 'standard',
  balance: 'none',
  selection: 'mutual-info',
  nFeatures: 6,
  vqc: {
    qubits: 6,
    layers: 2,
    featureMap: 'angle',
    ansatz: 'strongly-entangling',
    backend: 'ideal',
    shots: 0,
    seed: 42,
  },
  baselines: ['logistic', 'random-forest', 'svm'],
  epochs: 24,
  learningRate: 0.2,
  batchSize: 20,
  cvFolds: 5,
}

/** The qubit count is derived, never set independently. */
export function qubitsFor(cfg: RunConfig): number {
  return cfg.nFeatures
}

export type ModelResult = {
  id: string
  label: string
  kind: 'quantum' | 'classical'
  metrics: Metrics
  /** raw probabilities on the test set, for ROC and McNemar */
  scores: number[]
  predictions: number[]
  trainMs: number
  inferenceMs: number
  cv?: BoxStats
  cvFolds?: number[]
}

export type RunResult = {
  config: RunConfig
  featureNames: string[]
  /** names of the features actually kept */
  keptFeatures: string[]
  ranking: { name: string; score: number; kept: boolean }[]
  pcaResult: PcaResult | null
  trainSize: number
  testSize: number
  classBalance: { label: number; count: number }[]
  yTest: number[]
  models: ModelResult[]
  convergence: EpochRecord[]
  /** quantum vs the best classical model */
  verdict: {
    winner: string
    loser: string
    metric: string
    delta: number
    pValue: number
    significant: boolean
    /** true when a classical model beat the quantum one */
    classicalWon: boolean
  } | null
  startedAt: number
  elapsedMs: number
}

export type RunProgress =
  | { phase: 'preprocess'; message: string }
  | { phase: 'features'; message: string }
  | { phase: 'quantum'; epoch: EpochRecord; total: number }
  | { phase: 'classical'; model: string; message: string }
  | { phase: 'evaluate'; message: string }
  | { phase: 'done'; result: RunResult }

export type Dataset = {
  id: string
  name: string
  featureNames: string[]
  X: Matrix
  y: number[]
  /** what the positive class means clinically */
  positiveLabel: string
  negativeLabel: string
}

/**
 * Runs the whole pipeline as a generator so the UI can render progress as it
 * happens. Yields after each meaningful step; the final yield carries the
 * complete result.
 */
export function* runPipeline(
  data: Dataset,
  cfg: RunConfig,
): Generator<RunProgress, void, void> {
  const startedAt = Date.now()
  const rng = makeRng(cfg.seed)

  // ---- preprocessing ------------------------------------------------------
  yield { phase: 'preprocess', message: `loaded ${data.X.length} rows x ${data.featureNames.length} features` }

  const imputed = imputeMissing(data.X, data.y, cfg.impute)
  if (imputed.rowsDropped > 0) {
    yield { phase: 'preprocess', message: `dropped ${imputed.rowsDropped} rows with missing values` }
  } else {
    const total = imputed.filled.reduce((a, b) => a + b, 0)
    yield {
      phase: 'preprocess',
      message: total > 0 ? `imputed ${total} cells (${cfg.impute})` : 'no missing values found',
    }
  }

  const split = stratifiedSplit(imputed.y, cfg.testFraction, rng)
  yield {
    phase: 'preprocess',
    message: `stratified split ${split.trainIdx.length} train / ${split.testIdx.length} test, seed ${cfg.seed}`,
  }

  let Xtr = split.trainIdx.map((i) => imputed.X[i])
  let ytr = split.trainIdx.map((i) => imputed.y[i])
  const Xte = split.testIdx.map((i) => imputed.X[i])
  const yte = split.testIdx.map((i) => imputed.y[i])

  if (cfg.balance === 'oversample') {
    const os = oversample(Xtr, ytr, rng)
    Xtr = os.X
    ytr = os.y
    yield { phase: 'preprocess', message: `oversampled minority class, +${os.added} rows` }
  }

  // Scaler fitted on train ONLY - this is what keeps the benchmark honest.
  const scaler = fitScaler(Xtr, cfg.scaler)
  const Str = applyScaler(Xtr, scaler)
  const Ste = applyScaler(Xte, scaler)
  yield {
    phase: 'preprocess',
    message: `${cfg.scaler} scaling fitted on train fold only, applied to both`,
  }

  // ---- feature selection --------------------------------------------------
  const ranked = rankFeatures(Str, ytr, data.featureNames, cfg.selection)
  const keep = ranked.slice(0, cfg.nFeatures)
  const keptIdx = keep.map((r) => r.index)

  let pcaResult: PcaResult | null = null
  let Ftr: Matrix
  let Fte: Matrix

  if (cfg.selection === 'pca') {
    // PCA replaces features with components rather than subsetting them.
    pcaResult = pca(Str, cfg.nFeatures, cfg.seed)
    Ftr = pcaTransform(Str, pcaResult)
    Fte = pcaTransform(Ste, pcaResult)
    const retained = pcaResult.cumulative[pcaResult.cumulative.length - 1] ?? 0
    yield {
      phase: 'features',
      message: `PCA to ${cfg.nFeatures} components, ${(retained * 100).toFixed(1)}% variance retained`,
    }
  } else {
    Ftr = Str.map((r) => keptIdx.map((j) => r[j]))
    Fte = Ste.map((r) => keptIdx.map((j) => r[j]))
    yield {
      phase: 'features',
      message: `${cfg.selection}: kept ${cfg.nFeatures} of ${data.featureNames.length} -> ${cfg.nFeatures} qubits`,
    }
  }

  const models: ModelResult[] = []

  // ---- quantum lane -------------------------------------------------------
  const vqcCfg: VqcConfig = { ...cfg.vqc, qubits: cfg.nFeatures, seed: cfg.seed }
  const model = new Vqc(vqcCfg)
  const convergence: EpochRecord[] = []

  const qStart = performance.now()
  for (const rec of trainVqc(model, Ftr, ytr, {
    epochs: cfg.epochs,
    learningRate: cfg.learningRate,
    batchSize: cfg.batchSize,
    seed: cfg.seed,
  })) {
    convergence.push(rec)
    yield { phase: 'quantum', epoch: rec, total: cfg.epochs }
  }
  const qTrainMs = performance.now() - qStart

  const qInferStart = performance.now()
  const qScores = model.predict(Fte)
  const qInferMs = performance.now() - qInferStart

  models.push({
    id: 'vqc',
    label: `VQC (${cfg.nFeatures} qubits, ${cfg.vqc.layers} layers)`,
    kind: 'quantum',
    metrics: evaluate(yte, qScores),
    scores: qScores,
    predictions: qScores.map((s) => (s >= 0.5 ? 1 : 0)),
    trainMs: qTrainMs,
    inferenceMs: qInferMs,
  })

  // ---- classical lane -----------------------------------------------------
  for (const kind of cfg.baselines) {
    yield { phase: 'classical', model: kind, message: `fitting ${kind}` }

    const clf = makeBaseline(kind, cfg.seed)
    const cStart = performance.now()
    clf.fit(Ftr, ytr)
    const cTrainMs = performance.now() - cStart

    const cInferStart = performance.now()
    const scores = clf.predictProba(Fte)
    const cInferMs = performance.now() - cInferStart

    // Cross-validation spread: the single number hides generalisation.
    const folds = stratifiedFolds(imputed.y, cfg.cvFolds, makeRng(cfg.seed))
    const foldAcc: number[] = []
    for (let f = 0; f < folds.length; f++) {
      const teIdx = folds[f]
      const trIdx = folds.filter((_, i) => i !== f).flat()
      const fs = fitScaler(trIdx.map((i) => imputed.X[i]), cfg.scaler)
      const ftr = applyScaler(trIdx.map((i) => imputed.X[i]), fs).map((r) =>
        keptIdx.map((j) => r[j]),
      )
      const fte = applyScaler(teIdx.map((i) => imputed.X[i]), fs).map((r) =>
        keptIdx.map((j) => r[j]),
      )
      const fclf = makeBaseline(kind, cfg.seed)
      fclf.fit(ftr, trIdx.map((i) => imputed.y[i]))
      foldAcc.push(
        evaluate(teIdx.map((i) => imputed.y[i]), fclf.predictProba(fte)).accuracy,
      )
    }

    models.push({
      id: kind,
      label: kind,
      kind: 'classical',
      metrics: evaluate(yte, scores),
      scores,
      predictions: scores.map((s) => (s >= 0.5 ? 1 : 0)),
      trainMs: cTrainMs,
      inferenceMs: cInferMs,
      cv: boxStats(foldAcc),
      cvFolds: foldAcc,
    })
  }

  yield { phase: 'evaluate', message: 'comparing models on the holdout split' }

  // ---- verdict ------------------------------------------------------------
  const quantum = models.find((m) => m.kind === 'quantum')!
  const classical = models.filter((m) => m.kind === 'classical')
  const bestClassical = classical.length
    ? classical.reduce((a, b) => (b.metrics.accuracy > a.metrics.accuracy ? b : a))
    : null

  let verdict: RunResult['verdict'] = null
  if (bestClassical) {
    const test = mcnemar(yte, bestClassical.predictions, quantum.predictions)
    const delta = quantum.metrics.accuracy - bestClassical.metrics.accuracy
    const classicalWon = delta < 0

    verdict = {
      winner: classicalWon ? bestClassical.label : quantum.label,
      loser: classicalWon ? quantum.label : bestClassical.label,
      metric: 'accuracy',
      delta: Math.abs(delta),
      pValue: test.pValue,
      significant: test.significant,
      classicalWon,
    }
  }

  const counts = classCounts(imputed.y)

  yield {
    phase: 'done',
    result: {
      config: cfg,
      featureNames: data.featureNames,
      keptFeatures:
        cfg.selection === 'pca'
          ? Array.from({ length: cfg.nFeatures }, (_, i) => `PC${i + 1}`)
          : keep.map((k) => k.name),
      ranking: ranked.map((r) => ({
        name: r.name,
        score: r.score,
        kept: keptIdx.includes(r.index),
      })),
      pcaResult,
      trainSize: Str.length,
      testSize: Ste.length,
      classBalance: [...counts].map(([label, count]) => ({ label, count })),
      yTest: yte,
      models,
      convergence,
      verdict,
      startedAt,
      elapsedMs: Date.now() - startedAt,
    },
  }
}

export { rocCurve }
