import type { PipelineNodeSpec } from './types'

/**
 * Topology of the hybrid quantum-classical pipeline.
 * Durations are tuned so the two lanes visibly overlap: the quantum lane is
 * deliberately slower and has more stages than the classical lane.
 */
export const PIPELINE_NODES: PipelineNodeSpec[] = [
  {
    id: 'ingest',
    label: 'Data Ingestion',
    subtitle: 'Wisconsin Breast Cancer / 569 rows',
    lane: 'shared',
    deps: [],
    duration: 2200,
    config: {
      source: 'sklearn.datasets',
      split: 'stratified 70 / 15 / 15',
      seed: '42',
    },
    metrics: { rows: 569, columns: 30, classes: 2, missing: 0 },
    script: [
      { at: 0.05, level: 'info', message: 'opening dataset handle wdbc.data' },
      { at: 0.35, level: 'info', message: 'parsed 569 records, 30 numeric columns' },
      { at: 0.6, level: 'info', message: 'class balance  malignant=212  benign=357' },
      { at: 0.85, level: 'info', message: 'stratified split 398 / 85 / 86' },
      { at: 1, level: 'success', message: 'ingestion complete, checksum ok' },
    ],
  },
  {
    id: 'clean',
    label: 'Cleaning & Scaling',
    subtitle: 'impute, standardise',
    lane: 'shared',
    deps: ['ingest'],
    duration: 2600,
    config: {
      imputer: 'median',
      scaler: 'StandardScaler',
      outliers: 'clip at 4 sigma',
    },
    metrics: { imputed: 0, clipped: 7, mean: '0.000', std: '1.000' },
    script: [
      { at: 0.1, level: 'info', message: 'scanning for null values' },
      { at: 0.3, level: 'info', message: 'no missing cells detected, imputer bypassed' },
      { at: 0.55, level: 'warn', message: '7 values beyond 4 sigma clipped in area_worst' },
      { at: 0.8, level: 'info', message: 'fitting StandardScaler on train fold only' },
      { at: 1, level: 'success', message: 'features standardised, mean 0.000 std 1.000' },
    ],
  },
  {
    id: 'features',
    label: 'Feature Selection',
    subtitle: '30 -> 8 features',
    lane: 'shared',
    deps: ['clean'],
    duration: 3000,
    config: {
      method: 'mutual information + RFE',
      target: '8 features',
      collinearity: 'drop |r| > 0.92',
    },
    metrics: { selected: 8, dropped: 22, top: 'perimeter_worst' },
    script: [
      { at: 0.12, level: 'info', message: 'computing mutual information scores' },
      { at: 0.38, level: 'info', message: 'dropped 11 collinear features, |r| > 0.92' },
      { at: 0.66, level: 'info', message: 'recursive elimination pass 3 of 5' },
      { at: 0.88, level: 'info', message: 'retained perimeter_worst, concave_points_mean, +6' },
      { at: 1, level: 'success', message: 'feature matrix reduced 30 -> 8' },
    ],
  },

  // ---- classical lane -----------------------------------------------------
  {
    id: 'baseline',
    label: 'Baseline Training',
    subtitle: 'XGBoost + RF',
    lane: 'classical',
    deps: ['features'],
    duration: 5200,
    config: {
      models: 'XGBoost, RandomForest',
      folds: '5-fold stratified CV',
      estimators: '400',
    },
    metrics: { cv_mean: 0.961, cv_std: 0.014, fit_s: 4.8 },
    script: [
      { at: 0.08, level: 'info', message: 'fitting RandomForest n_estimators=400' },
      { at: 0.26, level: 'info', message: 'fold 1/5  acc 0.953' },
      { at: 0.44, level: 'info', message: 'fold 3/5  acc 0.968' },
      { at: 0.62, level: 'info', message: 'fold 5/5  acc 0.959' },
      { at: 0.78, level: 'info', message: 'fitting XGBoost lr=0.05 depth=4' },
      { at: 0.94, level: 'info', message: 'early stop at round 213' },
      { at: 1, level: 'success', message: 'baseline CV accuracy 0.961 +/- 0.014' },
    ],
  },
  {
    id: 'classical-infer',
    label: 'Classical Inference',
    subtitle: 'holdout predictions',
    lane: 'classical',
    deps: ['baseline'],
    duration: 2400,
    config: {
      batch: '86 samples',
      threshold: '0.50',
      device: 'cpu',
    },
    metrics: { latency_ms: 3.1, throughput: '27k/s', predicted: 86 },
    script: [
      { at: 0.2, level: 'info', message: 'loading fitted ensemble from run cache' },
      { at: 0.55, level: 'info', message: 'scoring 86 holdout samples' },
      { at: 0.82, level: 'info', message: 'mean inference latency 3.1 ms' },
      { at: 1, level: 'success', message: 'classical predictions written' },
    ],
  },

  // ---- quantum lane -------------------------------------------------------
  {
    id: 'encode',
    label: 'Angle Encoding',
    subtitle: '8 features -> 8 qubits',
    lane: 'quantum',
    deps: ['features'],
    duration: 3200,
    config: {
      map: 'RY angle embedding',
      qubits: '8',
      normalise: '[0, pi]',
    },
    metrics: { qubits: 8, depth: 1, gates: 8 },
    script: [
      { at: 0.15, level: 'info', message: 'allocating 8-qubit register' },
      { at: 0.45, level: 'info', message: 'scaling features into [0, pi]' },
      { at: 0.75, level: 'info', message: 'building RY embedding layer' },
      { at: 1, level: 'success', message: 'state preparation circuit ready, depth 1' },
    ],
  },
  {
    id: 'vqc',
    label: 'Variational Circuit',
    subtitle: '4 layers / 96 params',
    lane: 'quantum',
    deps: ['encode'],
    duration: 8600,
    config: {
      ansatz: 'StronglyEntanglingLayers',
      layers: '4',
      optimiser: 'Adam lr=0.02',
      shots: '1024',
    },
    metrics: { params: 96, epochs: 40, final_loss: 0.187 },
    script: [
      { at: 0.05, level: 'info', message: 'ansatz StronglyEntanglingLayers(4, 8)' },
      { at: 0.14, level: 'info', message: 'epoch 04/40  loss 0.612  grad_norm 0.44' },
      { at: 0.3, level: 'info', message: 'epoch 12/40  loss 0.398  grad_norm 0.21' },
      { at: 0.46, level: 'warn', message: 'gradient variance rising, barren plateau watch' },
      { at: 0.62, level: 'info', message: 'epoch 24/40  loss 0.271  grad_norm 0.12' },
      { at: 0.8, level: 'info', message: 'epoch 33/40  loss 0.213  grad_norm 0.08' },
      { at: 0.93, level: 'info', message: 'epoch 40/40  loss 0.187  grad_norm 0.06' },
      { at: 1, level: 'success', message: 'variational parameters converged' },
    ],
  },
  {
    id: 'measure',
    label: 'Measurement',
    subtitle: 'expectation values',
    lane: 'quantum',
    deps: ['vqc'],
    duration: 3400,
    config: {
      observable: 'PauliZ on wire 0',
      shots: '1024',
      mitigation: 'readout matrix inverse',
    },
    metrics: { shots: 1024, mean_z: -0.418, stderr: 0.031 },
    script: [
      { at: 0.18, level: 'info', message: 'sampling 1024 shots per input' },
      { at: 0.44, level: 'info', message: 'applying readout error mitigation' },
      { at: 0.72, level: 'info', message: 'expval Z0 = -0.418 +/- 0.031' },
      { at: 1, level: 'success', message: 'expectation values collapsed to logits' },
    ],
  },

  // ---- converge -----------------------------------------------------------
  {
    id: 'benchmark',
    label: 'Benchmark & Compare',
    subtitle: 'classical vs quantum',
    lane: 'shared',
    deps: ['classical-infer', 'measure'],
    duration: 3600,
    config: {
      metrics: 'accuracy, ROC-AUC, sensitivity, specificity',
      test: 'McNemar, alpha = 0.05',
      holdout: '86 samples',
    },
    metrics: { delta_acc: '+0.012', p_value: 0.21, verdict: 'not significant' },
    script: [
      { at: 0.16, level: 'info', message: 'aligning prediction vectors on holdout index' },
      { at: 0.4, level: 'info', message: 'classical  acc 0.961  auc 0.982' },
      { at: 0.64, level: 'info', message: 'quantum    acc 0.973  auc 0.988' },
      { at: 0.84, level: 'warn', message: 'McNemar p=0.21, delta not statistically significant' },
      { at: 1, level: 'success', message: 'benchmark table assembled' },
    ],
  },
  {
    id: 'explain',
    label: 'Explainability',
    subtitle: 'SHAP',
    lane: 'shared',
    deps: ['benchmark'],
    duration: 3000,
    config: {
      explainer: 'KernelSHAP',
      background: '100 samples',
      nsamples: '512',
    },
    metrics: { top_feature: 'perimeter_worst', shap_max: 0.214, coverage: '8/8' },
    script: [
      { at: 0.2, level: 'info', message: 'building KernelSHAP background set' },
      { at: 0.48, level: 'info', message: 'attributing 86 predictions' },
      { at: 0.74, level: 'info', message: 'top driver perimeter_worst  |phi| 0.214' },
      { at: 1, level: 'success', message: 'attribution map generated for both models' },
    ],
  },
  {
    id: 'results',
    label: 'Results',
    subtitle: 'run summary',
    lane: 'shared',
    deps: ['explain'],
    duration: 1800,
    config: {
      export: 'run_report.json',
      artefacts: 'metrics, plots, attributions',
      retention: '30 days',
    },
    metrics: { artefacts: 6, size_kb: 412, status: 'sealed' },
    script: [
      { at: 0.3, level: 'info', message: 'collecting artefacts from 10 stages' },
      { at: 0.7, level: 'info', message: 'writing run_report.json (412 KB)' },
      { at: 1, level: 'success', message: 'run sealed, report available' },
    ],
  },
]

export const NODE_BY_ID = new Map(PIPELINE_NODES.map((n) => [n.id, n]))

/** Final comparison payload surfaced by the results panel. All values mocked. */
export const BENCHMARK_RESULTS = {
  classical: { accuracy: 0.961, rocAuc: 0.982, sensitivity: 0.943, specificity: 0.972 },
  quantum: { accuracy: 0.973, rocAuc: 0.988, sensitivity: 0.962, specificity: 0.979 },
}
