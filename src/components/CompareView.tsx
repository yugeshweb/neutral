import { BENCHMARK_RESULTS } from '../lib/pipeline/graph'
import { buildMetricsCsv, downloadText } from '../lib/export'
import { LANE_COLOR, alpha } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconDownload } from './icons'
import { PushButton } from './PushButton'

const METRICS = [
  {
    key: 'accuracy',
    label: 'Accuracy',
    note: 'correct predictions over the 86-sample holdout',
  },
  {
    key: 'rocAuc',
    label: 'ROC-AUC',
    note: 'ranking quality, independent of threshold',
  },
  {
    key: 'sensitivity',
    label: 'Sensitivity',
    note: 'malignant cases correctly caught - the cost of a miss is highest here',
  },
  {
    key: 'specificity',
    label: 'Specificity',
    note: 'benign cases correctly cleared, i.e. false-alarm rate',
  },
] as const

/** Cost side of the comparison - the quantum lane is not free. */
const EFFICIENCY = [
  { label: 'Training wall-clock', classical: '4.8 s', quantum: '31.2 s', favours: 'classical' },
  { label: 'Inference latency', classical: '3.1 ms', quantum: '46.5 ms', favours: 'classical' },
  { label: 'Model parameters', classical: '~48k', quantum: '96', favours: 'quantum' },
  { label: 'Feature dimensionality', classical: '8', quantum: '8 qubits', favours: 'neutral' },
] as const

const CONFUSION = {
  classical: { tp: 33, fn: 2, tn: 50, fp: 1 },
  quantum: { tp: 34, fn: 1, tn: 50, fp: 1 },
} as const

/** What each lane actually is, for readers who did not run the pipeline. */
const ARCHITECTURES = [
  {
    name: 'Classical baseline',
    color: LANE_COLOR.classical,
    body: 'A conventional gradient-boosted ensemble, chosen because it is the strongest thing a practitioner would reach for on tabular biomedical data. It is the bar the hybrid model has to clear.',
    spec: [
      ['models', 'XGBoost + RandomForest'],
      ['estimators', '400, depth 4'],
      ['validation', '5-fold stratified CV'],
      ['hardware', 'CPU'],
    ] as [string, string][],
  },
  {
    name: 'Hybrid quantum',
    color: LANE_COLOR.quantum,
    body: 'Classical pre-processing feeds a variational quantum circuit. The 8 selected features are angle-encoded onto 8 qubits, entangled through 4 trainable layers, and read out as a PauliZ expectation value.',
    spec: [
      ['encoding', 'RY angle embedding'],
      ['ansatz', 'StronglyEntanglingLayers'],
      ['parameters', '96, Adam lr=0.02'],
      ['hardware', 'simulator, 1024 shots'],
    ] as [string, string][],
  },
]

/** Conditions held constant across both lanes. */
const SETUP = [
  { label: 'dataset', value: 'WDBC, 569 rows' },
  { label: 'split', value: '70 / 15 / 15' },
  { label: 'features', value: '30 to 8' },
  { label: 'seed', value: '42' },
]

/** The caveats that decide whether the headline number means anything. */
const READING = [
  {
    point: 'Sensitivity matters more than accuracy here.',
    body: 'In early detection a missed malignancy costs far more than a false alarm, so the sensitivity row deserves more weight than the headline accuracy figure.',
  },
  {
    point: 'The holdout is small.',
    body: '86 samples means a single reclassified case moves accuracy by roughly 0.012 - which is exactly the size of the reported gap. Confidence intervals on these metrics are wide.',
  },
  {
    point: 'Simulator timings are not hardware timings.',
    body: 'The quantum lane runs on a state-vector simulator. Its cost reflects shot sampling on a classical CPU, and would change substantially on real quantum hardware, including queue and calibration overhead.',
  },
  {
    point: 'One run is not a benchmark.',
    body: 'These figures come from a single seed. A defensible claim of quantum advantage needs repeated runs across seeds and datasets, reported with variance.',
  },
]

/** Paired horizontal bars, one metric, both models. */
function MetricRow({
  label,
  note,
  classical,
  quantum,
}: {
  label: string
  note: string
  classical: number
  quantum: number
}) {
  const delta = quantum - classical
  const leader = delta > 0 ? 'quantum' : delta < 0 ? 'classical' : 'tie'

  return (
    <div
      className="py-3.5"
      style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[12px] font-medium text-ink">{label}</div>
          <div className="mt-0.5 font-mono text-[9.5px] text-ink-faint">{note}</div>
        </div>
        <span
          className="shrink-0 rounded-[5px] px-1.5 py-[2px] font-mono text-[9.5px] tabular-nums"
          style={{
            color: leader === 'tie' ? '#6A6C72' : '#5FA88C',
            background: leader === 'tie' ? 'transparent' : alpha('#5FA88C', 0.1),
          }}
        >
          {delta >= 0 ? '+' : ''}
          {delta.toFixed(3)}
        </span>
      </div>

      {/* one bar per model, sharing a 0..1 scale */}
      {(
        [
          { name: 'classical', value: classical, color: LANE_COLOR.classical },
          { name: 'quantum', value: quantum, color: LANE_COLOR.quantum },
        ] as const
      ).map((m) => (
        <div key={m.name} className="mt-1.5 flex items-center gap-2.5">
          <span className="w-[54px] shrink-0 font-mono text-[9.5px] text-ink-faint">
            {m.name}
          </span>
          <div className="h-[5px] flex-1 overflow-hidden rounded-full panel-well">
            <div
              className="h-full rounded-full transition-[width] duration-500 ease-out"
              style={{ width: `${m.value * 100}%`, background: alpha(m.color, 0.78) }}
            />
          </div>
          <span className="w-[44px] shrink-0 text-right font-mono text-[11px] tabular-nums text-ink">
            {m.value.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  )
}

/** 2x2 confusion matrix for one model. */
function Confusion({
  title,
  color,
  data,
}: {
  title: string
  color: string
  data: { tp: number; fn: number; tn: number; fp: number }
}) {
  const cells = [
    { label: 'true positive', value: data.tp, good: true },
    { label: 'false negative', value: data.fn, good: false },
    { label: 'false positive', value: data.fp, good: false },
    { label: 'true negative', value: data.tn, good: true },
  ]

  return (
    <div className="flex-1">
      <div className="mb-2 flex items-center gap-2">
        <span className="h-[6px] w-[6px] rounded-full" style={{ background: color }} />
        <span className="font-mono text-[10px] text-ink-dim">{title}</span>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {cells.map((c) => (
          <div
            key={c.label}
            className="rounded-[7px] px-2.5 py-2"
            style={{
              background: '#0D0E10',
              border: '1px solid rgba(255,255,255,0.05)',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            <div
              className="font-mono text-[16px] tabular-nums"
              style={{ color: c.good ? '#E8E9EB' : '#A3543D' }}
            >
              {c.value}
            </div>
            <div className="mt-0.5 font-mono text-[9px] text-ink-faint">{c.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

type Props = {
  trained: boolean
}

export function CompareView({ trained }: Props) {
  const { classical, quantum } = BENCHMARK_RESULTS

  const handleCsv = () => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    downloadText(`netural-metrics-${stamp}.csv`, 'text/csv', buildMetricsCsv())
  }

  return (
    <div className="console-scroll h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[1120px] px-6 py-6">
        <div className="mb-5 flex items-start gap-3">
          <div className="flex-1">
            <h1 className="text-[16px] font-medium tracking-[-0.01em] text-ink">
              Model comparison
            </h1>
            <p className="mt-1 font-mono text-[10px] text-ink-faint">
              hybrid quantum vs classical baseline / 86-sample holdout /{' '}
              {trained ? 'this session' : 'bundled reference run'}
            </p>
          </div>
          <DemoChip />
          <PushButton
            label="Metrics CSV"
            icon={<IconDownload className="h-3.5 w-3.5" />}
            onClick={handleCsv}
            tone="primary"
            accent={LANE_COLOR.quantum}
          />
        </div>

        {/* verdict banner - states the honest conclusion up front */}
        <div
          className="mb-4 flex items-start gap-3 rounded-panel px-4 py-3"
          style={{
            background: '#17181B',
            border: '1px solid rgba(255,255,255,0.06)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
          }}
        >
          <span
            className="mt-[5px] h-[7px] w-[7px] shrink-0 rounded-full"
            style={{ background: '#C08A3E' }}
          />
          <div>
            <div className="text-[12.5px] font-medium text-ink">
              Quantum leads on every metric, but not significantly
            </div>
            <p className="mt-1 font-mono text-[10px] leading-relaxed text-ink-faint">
              The hybrid model is ahead by +0.012 accuracy and +0.019 sensitivity. A
              McNemar test on the paired holdout predictions gives p = 0.21, above the
              0.05 threshold - on a sample this size the gap is not distinguishable from
              noise. Treat it as a promising direction, not a demonstrated advantage.
            </p>
          </div>
        </div>

        <div className="flex gap-4">
          {/* headline metrics */}
          <section
            className="min-w-0 flex-1 rounded-panel px-4 py-2"
            style={{
              background: '#17181B',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow:
                'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
            }}
            aria-label="Detection metrics"
          >
            <div className="flex items-baseline justify-between py-2.5">
              <h2 className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
                detection quality
              </h2>
              <span className="flex gap-3 font-mono text-[9px]">
                <span style={{ color: LANE_COLOR.classical }}>classical</span>
                <span style={{ color: LANE_COLOR.quantum }}>quantum</span>
              </span>
            </div>

            {METRICS.map((m, i) => (
              <div
                key={m.key}
                style={i === METRICS.length - 1 ? { borderBottom: 'none' } : undefined}
              >
                <MetricRow
                  label={m.label}
                  note={m.note}
                  classical={classical[m.key]}
                  quantum={quantum[m.key]}
                />
              </div>
            ))}
          </section>

          {/* cost side */}
          <section
            className="w-[320px] shrink-0 rounded-panel p-4"
            style={{
              background: '#17181B',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow:
                'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
            }}
            aria-label="Computational efficiency"
          >
            <h2 className="mb-3 font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
              computational cost
            </h2>

            <div className="space-y-3">
              {EFFICIENCY.map((row) => (
                <div key={row.label}>
                  <div className="mb-1 font-mono text-[10px] text-ink-dim">{row.label}</div>
                  <div className="flex gap-1.5">
                    {(['classical', 'quantum'] as const).map((side) => (
                      <div
                        key={side}
                        className="flex-1 rounded-[6px] px-2 py-1.5"
                        style={{
                          background: '#0D0E10',
                          border: `1px solid ${
                            row.favours === side
                              ? alpha('#5FA88C', 0.28)
                              : 'rgba(255,255,255,0.05)'
                          }`,
                          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                        }}
                      >
                        <div className="font-mono text-[9px] text-ink-faint">{side}</div>
                        <div
                          className="mt-0.5 font-mono text-[11px] tabular-nums"
                          style={{ color: row.favours === side ? '#5FA88C' : '#9A9CA1' }}
                        >
                          {row[side]}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <p
              className="mt-3.5 pt-3 font-mono text-[9px] leading-relaxed text-ink-faint/80"
              style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
            >
              The quantum lane runs on a simulator. Its latency reflects shot sampling, not
              hardware time, and would change substantially on a real device.
            </p>
          </section>
        </div>

        {/* confusion matrices */}
        <section
          className="mt-4 rounded-panel p-4"
          style={{
            background: '#17181B',
            border: '1px solid rgba(255,255,255,0.06)',
            boxShadow:
              'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
          }}
          aria-label="Confusion matrices"
        >
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
              error breakdown
            </h2>
            <span className="font-mono text-[9.5px] text-ink-faint">86 holdout samples</span>
          </div>

          <div className="flex gap-5">
            <Confusion
              title="Classical baseline"
              color={LANE_COLOR.classical}
              data={CONFUSION.classical}
            />
            <div className="w-px shrink-0" style={{ background: 'rgba(255,255,255,0.06)' }} />
            <Confusion
              title="Hybrid quantum"
              color={LANE_COLOR.quantum}
              data={CONFUSION.quantum}
            />
          </div>

          <p
            className="mt-3.5 pt-3 font-mono text-[9px] leading-relaxed text-ink-faint/80"
            style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
          >
            The whole difference between the two models is a single case: one malignant
            sample the classical baseline misses and the hybrid catches. That is what the
            McNemar test is weighing, and one discordant pair cannot reach significance.
          </p>
        </section>

        {/* about: what was compared, how, and how to read it */}
        <section
          className="mt-4 rounded-panel p-4"
          style={{
            background: '#17181B',
            border: '1px solid rgba(255,255,255,0.06)',
            boxShadow:
              'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
          }}
          aria-label="About this comparison"
        >
          <div className="mb-3.5 flex items-baseline justify-between">
            <h2 className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
              about this comparison
            </h2>
            <span className="font-mono text-[9.5px] text-ink-faint">methodology</span>
          </div>

          {/* the two architectures, side by side */}
          <div className="flex gap-4">
            {ARCHITECTURES.map((arch) => (
              <div key={arch.name} className="flex-1">
                <div className="mb-2 flex items-center gap-2">
                  <span
                    className="h-[6px] w-[6px] rounded-full"
                    style={{ background: arch.color }}
                  />
                  <span className="text-[12px] font-medium text-ink">{arch.name}</span>
                </div>
                <p className="mb-2.5 text-[11px] leading-relaxed text-ink-dim">
                  {arch.body}
                </p>
                <dl className="space-y-1">
                  {arch.spec.map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-3">
                      <dt className="font-mono text-[9.5px] text-ink-faint">{k}</dt>
                      <dd className="truncate font-mono text-[9.5px] text-ink-dim" title={v}>
                        {v}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>

          {/* shared experimental setup */}
          <div
            className="mt-4 pt-3.5"
            style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
          >
            <h3 className="mb-2.5 font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
              experimental setup
            </h3>
            <div className="grid grid-cols-4 gap-2">
              {SETUP.map((s) => (
                <div
                  key={s.label}
                  className="rounded-[7px] px-2.5 py-2"
                  style={{
                    background: '#0D0E10',
                    border: '1px solid rgba(255,255,255,0.05)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                  }}
                >
                  <div className="font-mono text-[9px] text-ink-faint">{s.label}</div>
                  <div className="mt-0.5 font-mono text-[10.5px] text-ink-dim">
                    {s.value}
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-2.5 font-mono text-[9px] leading-relaxed text-ink-faint/80">
              Both models see the identical 8-feature matrix, the same stratified split and
              the same random seed. The scaler and feature selector are fitted on the
              training fold only, so nothing from the holdout leaks into either lane.
            </p>
          </div>

          {/* how to read the result honestly */}
          <div
            className="mt-4 pt-3.5"
            style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
          >
            <h3 className="mb-2.5 font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
              how to read this
            </h3>
            <ul className="space-y-2">
              {READING.map((r) => (
                <li key={r.point} className="flex gap-2.5">
                  <span
                    className="mt-[6px] h-[3px] w-[3px] shrink-0 rounded-full"
                    style={{ background: LANE_COLOR.shared }}
                  />
                  <p className="text-[11px] leading-relaxed text-ink-dim">
                    <span className="text-ink">{r.point}</span> {r.body}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <p className="mt-5 max-w-[620px] font-mono text-[9px] leading-relaxed text-ink-faint/70">
          All figures on this page are placeholder values produced by a mock runner. No
          model was trained and no dataset was evaluated. Do not cite these figures as
          results.
        </p>
      </div>
    </div>
  )
}
