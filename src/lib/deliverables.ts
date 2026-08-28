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
      'An adapter layer over five hospital source formats, all converging on one numeric matrix. CSV and FHIR R4 are fully implemented; HL7 v2, DICOM and VCF are declared and throw a stated NotImplementedError.',
    module: 'src/lib/ingest/',
    status: 'partial',
  },
  {
    id: 'interoperability',
    requirement: 'Ingest real hospital data, not just CSV exports',
    section: 'Train',
    delivered:
      'FHIR R4 Bundle adapter walks the entry list, groups by Patient reference, reads Observation values by LOINC code and derives the label from Condition ICD-10 codes. Verified on a 60-patient, 602-entry sample bundle.',
    module: 'src/lib/ingest/fhir.ts',
    status: 'live',
  },
  {
    id: 'preprocessing',
    requirement: 'Data pre-processing module',
    section: 'Train',
    delivered:
      'Cleaning and scaling stage: median imputation, StandardScaler fitted on the training fold only, 4-sigma outlier clipping. Surfaced as an inspectable pipeline node.',
    module: 'src/lib/pipeline/graph.ts',
    status: 'mocked',
  },
  {
    id: 'feature-selection',
    requirement: 'Feature selection module',
    section: 'Train',
    delivered:
      'Mutual information scoring plus recursive elimination, reducing 30 features to 8, with collinear pairs above |r| 0.92 dropped.',
    module: 'src/lib/pipeline/graph.ts',
    status: 'mocked',
  },
  {
    id: 'architecture',
    requirement: 'Hybrid quantum-classical architecture',
    section: 'Train',
    delivered:
      'Directed graph that forks after feature selection into a classical lane and a quantum lane advancing on the same tick, then converges at benchmarking. Dependencies gate execution.',
    module: 'src/lib/pipeline/runner.ts',
    status: 'live',
  },
  {
    id: 'quantum-model',
    requirement: 'Quantum-enhanced classification model',
    section: 'Train',
    delivered:
      'Variational quantum classifier: RY angle encoding onto 8 qubits, StronglyEntanglingLayers ansatz over 4 layers and 96 parameters, PauliZ expectation readout with readout-error mitigation.',
    module: 'src/lib/pipeline/graph.ts',
    status: 'mocked',
  },
  {
    id: 'training',
    requirement: 'Hybrid model training workflow',
    section: 'Train',
    delivered:
      'Run controls with start, stop and reset; per-stage progress, live event log, and terminal states that propagate so downstream stages report as blocked on failure.',
    module: 'src/hooks/usePipeline.ts',
    status: 'live',
  },
  {
    id: 'inference',
    requirement: 'Prediction and inference workflow',
    section: 'Predict',
    delivered:
      'Single-case scoring over the 8 retained features, returning malignant probability, predicted label and decision margin from both the quantum and classical heads.',
    module: 'src/lib/predict.ts',
    status: 'mocked',
  },
  {
    id: 'explainability',
    requirement: 'Model explainability features',
    section: 'Predict',
    delivered:
      'Per-feature signed attribution to the malignant logit, ranked by magnitude and rendered as diverging bars; a SHAP stage in the pipeline reports global attribution.',
    module: 'src/components/PredictView.tsx',
    status: 'partial',
  },
  {
    id: 'metrics',
    requirement: 'Improve accuracy, sensitivity and specificity vs baselines',
    section: 'Compare',
    delivered:
      'Accuracy, ROC-AUC, sensitivity and specificity reported for both models with signed deltas, plus 2x2 confusion matrices isolating exactly which cases differ.',
    module: 'src/components/CompareView.tsx',
    status: 'mocked',
  },
  {
    id: 'benchmark',
    requirement: 'Benchmark on accuracy, efficiency and generalization',
    section: 'Compare',
    delivered:
      'Paired metric bars on a shared scale, a computational-cost table covering training time, inference latency and parameter count, and a McNemar significance test on the paired predictions.',
    module: 'src/components/CompareView.tsx',
    status: 'mocked',
  },
  {
    id: 'evaluation',
    requirement: 'Performance evaluation and reporting',
    section: 'Compare',
    delivered:
      'Run report exportable as JSON (stage records, metrics, full log) or the comparison table as CSV. Both carry a synthetic flag and disclaimer that survive outside the UI.',
    module: 'src/lib/export.ts',
    status: 'live',
  },
  {
    id: 'hardware',
    requirement: 'Compatible with near-term hardware and simulators',
    section: 'Platform',
    delivered:
      'An 8-qubit, depth-4 circuit sized for NISQ devices and simulators; shot count and readout mitigation are exposed as stage configuration rather than hard-coded.',
    module: 'src/lib/pipeline/graph.ts',
    status: 'partial',
  },
  {
    id: 'scalable',
    requirement: 'Scalable and interpretable platform',
    section: 'Platform',
    delivered:
      'One PipelineRunner interface behind the whole UI; a documented createSSERunner swap point moves the platform onto a real backend without any component changing.',
    module: 'src/lib/pipeline/runner.ts',
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
