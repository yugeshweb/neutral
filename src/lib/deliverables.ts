/**
 * The problem statement's expected deliverables, mapped to where each one is
 * satisfied in this platform.
 *
 * This exists so the mapping is auditable rather than asserted: every row names
 * the section that carries it and the module that implements it, and `status`
 * distinguishes what genuinely works from what is presently mocked. Nothing
 * here should claim more than the code does.
 */

export type DeliveryStatus = 'live' | 'mocked' | 'partial'

export type Deliverable = {
  id: string
  /** the PS requirement, in its own words where possible */
  requirement: string
  /** which of the three sections carries it */
  section: 'Train' | 'Predict' | 'Compare' | 'Platform'
  /** how this platform satisfies it */
  delivered: string
  /** implementing module, repo-relative */
  module: string
  status: DeliveryStatus
}

export const STATUS_LABEL: Record<DeliveryStatus, string> = {
  live: 'implemented',
  partial: 'partial',
  mocked: 'mocked',
}

export const STATUS_COLOR: Record<DeliveryStatus, string> = {
  live: '#5FA88C',
  partial: '#C08A3E',
  mocked: '#8A8F98',
}

export const DELIVERABLES: Deliverable[] = [
  {
    id: 'ingestion',
    requirement: 'Data ingestion and handling pipelines',
    section: 'Train',
    delivered:
      'An adapter layer over five hospital source formats, all converging on one numeric matrix. CSV, FHIR R4 and HL7 v2 are fully implemented; DICOM and VCF are declared and throw a stated NotImplementedError.',
    module: 'src/lib/ingest/',
    status: 'partial',
  },
  {
    id: 'interoperability',
    requirement: 'Ingest real hospital data, not just CSV exports',
    section: 'Train',
    delivered:
      'FHIR R4 Bundle adapter walks the entry list, groups by Patient reference, reads Observation values by LOINC code and derives the label from Condition ICD-10 codes. Verified on a 60-patient, 602-entry sample bundle. An HL7 v2 adapter splits a pipe-delimited feed into ORU messages, groups by PID-3, and reads OBX-3/OBX-5 as LOINC-coded observations - the older interface still most widely deployed in hospitals.',
    module: 'src/lib/ingest/fhir.ts, src/lib/ingest/hl7.ts',
    status: 'live',
  },
  {
    id: 'preprocessing',
    requirement: 'Data pre-processing module',
    section: 'Train',
    delivered:
      'Drop / mean / median imputation and standard or min-max scaling, all computed on the real matrix. The scaler is fitted on the training fold only, with before-and-after distribution charts showing the effect.',
    module: 'src/lib/ml/stats.ts',
    status: 'live',
  },
  {
    id: 'feature-selection',
    requirement: 'Feature selection module',
    section: 'Train',
    delivered:
      'Four real methods - PCA, mutual information, ANOVA F-test and recursive elimination - with a ranked importance chart, a collinearity drop above |r| 0.92, and the retained count bound live to the qubit count.',
    module: 'src/lib/ml/features.ts',
    status: 'live',
  },
  {
    id: 'architecture',
    requirement: 'Hybrid quantum-classical architecture',
    section: 'Train',
    delivered:
      'Classical preprocessing and feature selection feed a quantum circuit, with classical baselines trained on the identical matrix. The whole run is one generator, stepped from the UI so progress renders as it computes.',
    module: 'src/lib/ml/pipeline.ts',
    status: 'live',
  },
  {
    id: 'quantum-model',
    requirement: 'Quantum-enhanced classification model',
    section: 'Train',
    delivered:
      'A real statevector simulator (2^n amplitudes, genuine gate matrices) running a VQC with three feature maps, three ansatze and three backends. Trained by Adam on exact parameter-shift gradients, verified against finite differences to 1e-11.',
    module: 'src/lib/quantum/',
    status: 'live',
  },
  {
    id: 'training',
    requirement: 'Hybrid model training workflow',
    section: 'Train',
    delivered:
      'One button trains the quantum and classical lanes on an identical split. Live convergence chart of the cost function per epoch, elapsed timer, progress, stop control and a scrolling log.',
    module: 'src/hooks/useRun.ts',
    status: 'live',
  },
  {
    id: 'inference',
    requirement: 'Prediction and inference workflow',
    section: 'Predict',
    delivered:
      'Per-patient inference against the trained circuit, returning a calibrated probability, a low/moderate/high risk band and the true label for comparison.',
    module: 'src/components/steps/ExplainStep.tsx',
    status: 'live',
  },
  {
    id: 'explainability',
    requirement: 'Model explainability features',
    section: 'Predict',
    delivered:
      'Permutation importance globally and occlusion attribution per patient. After PCA, component attributions are pushed back through the rotation onto the original clinical measurements - verified to recover 8/8 known-discriminative WDBC features.',
    module: 'src/lib/ml/explain.ts',
    status: 'live',
  },
  {
    id: 'metrics',
    requirement: 'Improve accuracy, sensitivity and specificity vs baselines',
    section: 'Compare',
    delivered:
      'Accuracy, sensitivity, specificity, precision, F1 and ROC-AUC computed from real predictions for every model, with confusion matrices and overlaid ROC curves.',
    module: 'src/lib/ml/metrics.ts',
    status: 'live',
  },
  {
    id: 'benchmark',
    requirement: 'Benchmark on accuracy, efficiency and generalization',
    section: 'Compare',
    delivered:
      'Five classical baselines trained on the identical split, measured training and inference time, k-fold cross-validation spread as a box plot, and an exact-binomial McNemar test. A classical win is displayed exactly as cleanly as a quantum one.',
    module: 'src/lib/ml/baselines.ts',
    status: 'live',
  },
  {
    id: 'evaluation',
    requirement: 'Performance evaluation and reporting',
    section: 'Compare',
    delivered:
      'Every metric on the results screen is computed in-browser from the run configuration, with backend, shot count, seed and elapsed time stated alongside. JSON and CSV export carry the run provenance.',
    module: 'src/lib/export.ts',
    status: 'partial',
  },
  {
    id: 'hardware',
    requirement: 'Compatible with near-term hardware and simulators',
    section: 'Platform',
    delivered:
      'Circuit width is capped at 10 qubits with depth and gate count reported live. Finite-shot sampling and a depolarising noise model reproduce device behaviour; the hardware-efficient ansatz matches real linear coupling maps.',
    module: 'src/lib/quantum/vqc.ts',
    status: 'partial',
  },
  {
    id: 'scalable',
    requirement: 'Scalable and interpretable platform',
    section: 'Platform',
    delivered:
      'The entire run is one serialisable RunConfig, so a result is reproducible from its configuration. A seven-step pipeline stepper carries the user through, with the qubit count visible on every screen after feature selection.',
    module: 'src/lib/ml/pipeline.ts',
    status: 'live',
  },
  {
    id: 'docs',
    requirement: 'Comprehensive documentation',
    section: 'Platform',
    delivered:
      'README covering stack, pipeline topology, the runner contract and its swap point, upload handling, export format and accessibility; this table maps every requirement to its module.',
    module: 'README.md',
    status: 'live',
  },
]

/** Counts by status, for the summary line above the table. */
export function deliveryTally() {
  return DELIVERABLES.reduce(
    (acc, d) => {
      acc[d.status] += 1
      return acc
    },
    { live: 0, partial: 0, mocked: 0 } as Record<DeliveryStatus, number>,
  )
}
