import type { ReactElement } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'
import { VecBars, VecCircuit, VecTrace } from './vectors'

const QUANTUM = LANE_COLOR.quantum
const CLASSICAL = LANE_COLOR.classical

type Tile = {
  id: 'train' | 'predict' | 'benchmark'
  step: string
  title: string
  blurb: string
  accent: string
  Art: (p: { size?: number }) => ReactElement
}

const TILES: Tile[] = [
  {
    id: 'train',
    step: '01',
    title: 'Train',
    blurb: 'Fit a quantum and a classical model on the same split.',
    accent: QUANTUM,
    Art: VecCircuit,
  },
  {
    id: 'predict',
    step: '02',
    title: 'Predict',
    blurb: 'Score new records with the model training saved.',
    accent: CLASSICAL,
    Art: VecTrace,
  },
  {
    id: 'benchmark',
    step: '03',
    title: 'Benchmark',
    blurb: 'Read the gap between the two on the same metrics.',
    accent: QUANTUM,
    Art: VecBars,
  },
]

type RegistryRow = {
  modelId: string
  name: string
  modality: string
  framing: 'detection' | 'characterisation' | 'screening' | 'prediction'
  framingLabel: string
  performance: string
  status: 'best' | 'chance' | 'disabled' | 'deployable'
  statusLabel: string
  statusColor: string
}

const MODEL_REGISTRY_ROWS: RegistryRow[] = [
  {
    modelId: 'stroke-core-volume-mri',
    name: 'Stroke Ischemic Core Volume',
    modality: 'MRI: DWI + ADC + FLAIR',
    framing: 'characterisation',
    framingLabel: 'Property of Identified Finding',
    performance: 'BA 0.7765, AUC 0.879',
    status: 'best',
    statusLabel: 'Best available',
    statusColor: '#5FA88C',
  },
  {
    modelId: 'parkinsons-gait-signal',
    name: "Parkinson's Dynamic Gait",
    modality: '18-ch force-plate @ 100 Hz',
    framing: 'detection',
    framingLabel: 'Present Finding',
    performance: 'BA 0.7980 (subject-grouped)',
    status: 'best',
    statusLabel: 'Best available',
    statusColor: '#5FA88C',
  },
  {
    modelId: 'heart-disease',
    name: 'Heart Disease & Ischemia Risk',
    modality: '12-Lead ECG / Hemodynamics',
    framing: 'detection',
    framingLabel: 'Present Finding',
    performance: 'Classical BA 0.852, VQC BA 0.869',
    status: 'deployable',
    statusLabel: 'Deployable Active',
    statusColor: '#7C67FE',
  },
  {
    modelId: 'breast-cancer',
    name: 'Breast Carcinoma FNA',
    modality: 'Nuclear FNA Tabular',
    framing: 'detection',
    framingLabel: 'Present Finding',
    performance: 'Acc 97.7%, AUROC 0.992',
    status: 'deployable',
    statusLabel: 'Deployable Active',
    statusColor: '#7C67FE',
  },
  {
    modelId: 'alzheimers-oasis-tabular',
    name: "Alzheimer's Dementia Association",
    modality: 'OASIS-1 Tabular + Volumetric',
    framing: 'screening',
    framingLabel: 'Population Screening',
    performance: 'Classical BA 0.823, QSVC BA 0.564',
    status: 'chance',
    statusLabel: 'Screening reference',
    statusColor: '#60A5FA',
  },
  {
    modelId: 'glioma-mgmt-mpmri',
    name: 'Glioma MGMT Methylation',
    modality: 'MRI: T1/T1c/T2/FLAIR',
    framing: 'characterisation',
    framingLabel: 'Property of Identified Finding',
    performance: 'BA 0.533 (N=47 underpowered)',
    status: 'chance',
    statusLabel: 'At chance — research only',
    statusColor: '#D97706',
  },
  {
    modelId: 'ich-intraventricular-ct',
    name: 'Intraventricular Haemorrhage',
    modality: 'Head CT, 3 clinical windows',
    framing: 'detection',
    framingLabel: 'Present Finding',
    performance: 'BA ~0.54–0.58',
    status: 'chance',
    statusLabel: 'At chance — research only',
    statusColor: '#D97706',
  },
  {
    modelId: 'seizure-preictal-eeg',
    name: 'Preictal Seizure Prediction',
    modality: '14 Scalp EEG band-powers',
    framing: 'prediction',
    framingLabel: '⚠ Early Warning (lead time)',
    performance: 'LOPO BA 0.505 ± 0.257',
    status: 'disabled',
    statusLabel: 'Disabled — patient safety',
    statusColor: '#EF4444',
  },
]

export function HomeScreen({
  onOpen,
}: {
  onOpen: (id: 'train' | 'predict' | 'benchmark') => void
}) {
  return (
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen space-y-6">
        {/* Header with 7-Disease Badge */}
        <div className="border-b border-white/5 pb-4 text-center">
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-[11px] text-ink-dim">
            <span className="h-2 w-2 rounded-full bg-lane-quantum animate-pulse" />
            <span className="font-semibold text-ink">7 Registered Disease Domains</span>
            <span className="text-ink-faint">|</span>
            <span>5 Deployable Bundles</span>
            <span className="text-ink-faint">·</span>
            <span>1 Safety-Gated</span>
            <span className="text-ink-faint">·</span>
            <span>2 At Chance</span>
          </div>

          <h1 className="text-[19px] font-medium text-ink">
            Hybrid Quantum-Classical Disease Detection Platform
          </h1>
          <p className="mx-auto mt-1 max-w-[66ch] text-[13px] leading-relaxed text-ink-dim">
            Unified orchestration across 7 clinical disease domains. Deep classical encoders for spatial & signal compression with parameterized quantum Hilbert feature maps.
          </p>
        </div>

        {/* 3 Main Action Tiles */}
        <div className="mx-auto grid w-full max-w-[840px] grid-cols-1 gap-5 sm:grid-cols-3">
          {TILES.map(({ id, step, title, blurb, accent, Art }) => (
            <a
              key={id}
              href={`/${id}`}
              onClick={(e) => {
                if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) {
                  return
                }
                e.preventDefault()
                onOpen(id)
              }}
              className="tile group flex cursor-pointer flex-col p-3 no-underline"
              style={{ ['--tile-accent' as string]: accent }}
            >
              <div className="tile-art aspect-square w-full">
                <Art />
                <span className="engraved absolute left-2.5 top-2 font-mono text-[11px]">
                  {step}
                </span>
              </div>

              <div className="px-1 pb-1 pt-3">
                <h2 className="text-[15.5px] font-medium text-ink">{title}</h2>
                <p className="mt-1 text-[12.5px] leading-relaxed text-ink-dim">
                  {blurb}
                </p>
              </div>
            </a>
          ))}
        </div>

        {/* Model Registry Status Table */}
        <div className="mx-auto max-w-[840px] rounded-panel p-4" style={{ background: '#17181B', border: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="mb-3 flex items-baseline justify-between border-b border-white/5 pb-2">
            <div>
              <h2 className="text-[14px] font-medium text-ink">Clinical Model Registry & Status Breakdown</h2>
              <p className="font-mono text-[11px] text-ink-faint">
                7 registered disease domains, validation status, and temporal framing
              </p>
            </div>
            <span className="font-mono text-[11px] text-ink-faint">
              Joblib InferenceBundles
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-[11.5px]">
              <thead>
                <tr className="border-b border-white/5 text-ink-faint">
                  <th className="pb-2 font-medium">Condition / Model</th>
                  <th className="pb-2 font-medium">Modality</th>
                  <th className="pb-2 font-medium">Temporal Framing</th>
                  <th className="pb-2 font-medium">Held-Out Metric</th>
                  <th className="pb-2 text-right font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {MODEL_REGISTRY_ROWS.map((row) => (
                  <tr key={row.modelId} className="transition-colors hover:bg-white/[0.02]">
                    <td className="py-2.5 pr-2">
                      <div className="font-sans text-[12.5px] font-medium text-ink">{row.name}</div>
                      <div className="text-[10px] text-ink-faint">{row.modelId}</div>
                    </td>
                    <td className="py-2.5 pr-2 text-ink-dim">{row.modality}</td>
                    <td className="py-2.5 pr-2 text-ink-faint">
                      <span className="inline-block rounded-[4px] bg-white/5 px-1.5 py-0.5 text-[10px]">
                        {row.framingLabel}
                      </span>
                    </td>
                    <td className="py-2.5 pr-2 text-ink-dim">{row.performance}</td>
                    <td className="py-2.5 text-right">
                      <span
                        className="inline-flex items-center gap-1 rounded-[4px] px-2 py-0.5 text-[10.5px] font-medium"
                        style={{
                          color: row.statusColor,
                          background: alpha(row.statusColor, 0.12),
                          border: `1px solid ${alpha(row.statusColor, 0.3)}`,
                        }}
                      >
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: row.statusColor }} />
                        {row.statusLabel}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
