import { BASELINE_LABEL, type BaselineKind } from '../../lib/ml/baselines'
import { rocCurve } from '../../lib/ml/metrics'
import type { RunResult } from '../../lib/ml/pipeline'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { BoxPlot, RocChart } from '../charts'
import { DeliveryTable } from '../DeliveryTable'
import { IconPlay } from '../icons'
import { PushButton } from '../PushButton'
import { QubitBadge } from '../QubitBadge'
import { LiveChip, Panel, SectionLabel } from '../ui'

type Props = {
  result: RunResult | null
  /** a run is already in flight - offer "go watch it" instead of starting a second one */
  running?: boolean
  /** jumps to Train and starts a run with the current config - offered when there is nothing to show yet */
  onStartTraining?: () => void
}

const METRIC_COLUMNS = [
  { key: 'accuracy', label: 'Acc' },
  { key: 'sensitivity', label: 'Sens' },
  { key: 'specificity', label: 'Spec' },
  { key: 'precision', label: 'Prec' },
  { key: 'f1', label: 'F1' },
  { key: 'rocAuc', label: 'AUC' },
] as const

function label(id: string) {
  return BASELINE_LABEL[id as BaselineKind] ?? id
}

function ConfusionGrid({
  c,
  color,
  title,
  positiveLabel,
  negativeLabel,
}: {
  c: { tp: number; fn: number; fp: number; tn: number }
  color: string
  title: string
  positiveLabel: string
  negativeLabel: string
}) {
  const cells = [
    { v: c.tp, l: 'true positive', good: true },
    { v: c.fn, l: 'false negative', good: false, critical: true },
    { v: c.fp, l: 'false positive', good: false },
    { v: c.tn, l: 'true negative', good: true },
  ]

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="h-[6px] w-[6px] rounded-full" style={{ background: color }} />
        <span className="min-w-0 truncate font-mono text-[10px] text-ink-dim">{title}</span>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {cells.map((cell) => (
          <div
            key={cell.l}
            className="rounded-[7px] px-2 py-1.5"
            style={{
              background: '#0D0E10',
              border: `1px solid ${cell.critical && cell.v > 0 ? alpha('#A3543D', 0.35) : 'rgba(255,255,255,0.05)'}`,
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            <div
              className="font-mono text-[15px] tabular-nums"
              style={{ color: cell.good ? '#E8E9EB' : '#A3543D' }}
            >
              {cell.v}
            </div>
            <div className="font-mono text-[8px] text-ink-faint">{cell.l}</div>
          </div>
        ))}
      </div>
      <p className="mt-1.5 font-mono text-[8px] leading-relaxed text-ink-faint/70">
        positive = {positiveLabel}, negative = {negativeLabel}
      </p>
    </div>
  )
}

export function ResultsStep({ result, running, onStartTraining }: Props) {
  if (!result) {
    return (
      <Panel>
        <div className="grid place-items-center gap-3 py-14 text-center">
          <p className="font-mono text-[11px] text-ink-faint">
            {running
              ? 'Training is running now - come back here once it finishes.'
              : 'No results yet - train the model first, then come back here to compare it against the classical baselines.'}
          </p>
          {onStartTraining && (
            <PushButton
              label={running ? 'Go to Train' : 'Train now'}
              icon={<IconPlay className="h-3.5 w-3.5" />}
              onClick={onStartTraining}
              tone="primary"
              accent="#5FA88C"
            />
          )}
        </div>
      </Panel>
    )
  }

  const classical = result.models.filter((m) => m.kind === 'classical')
  const verdict = result.verdict

  const colorFor = (id: string, kind: string) =>
    kind === 'quantum' ? LANE_COLOR.quantum : id === 'logistic' ? LANE_COLOR.classical : '#8A8F98'

  const best = result.models.reduce((a, b) =>
    b.metrics.accuracy > a.metrics.accuracy ? b : a,
  )

  return (
    <div className="space-y-4">
      {/* verdict - reports a classical win exactly as cleanly */}
      {verdict && (
        <Panel>
          <div className="flex items-start gap-3">
            <span
              className="mt-[6px] h-[8px] w-[8px] shrink-0 rounded-full"
              style={{
                background: verdict.significant ? '#5FA88C' : LANE_COLOR.classical,
              }}
            />
            <div className="flex-1">
              <div className="flex items-baseline gap-2">
                <h2 className="text-[14px] font-medium text-ink">
                  {label(verdict.winner)} wins by {verdict.delta.toFixed(4)} accuracy
                </h2>
                <LiveChip label="measured" />
              </div>

              <p className="mt-1.5 font-mono text-[10px] leading-relaxed text-ink-faint">
                {verdict.classicalWon ? (
                  <>
                    The classical baseline beat the quantum model on this run. That is a real
                    result and it is reported exactly as a quantum win would be — an honest
                    benchmark is worth more than a flattering one.
                  </>
                ) : (
                  <>The quantum model finished ahead of the best classical baseline.</>
                )}{' '}
                McNemar&apos;s test on the paired predictions gives p ={' '}
                <span className="text-ink-dim">{verdict.pValue.toFixed(4)}</span>, which is{' '}
                {verdict.significant ? (
                  <span style={{ color: '#5FA88C' }}>below 0.05 — the gap is significant</span>
                ) : (
                  <span style={{ color: LANE_COLOR.classical }}>
                    above 0.05 — the gap is not distinguishable from noise
                  </span>
                )}{' '}
                on {result.testSize} held-out samples.
              </p>
            </div>

            <div className="shrink-0 text-right">
              <QubitBadge qubits={result.config.nFeatures} compact />
              <div className="mt-1.5 font-mono text-[8.5px] text-ink-faint">
                {result.config.vqc.backend} backend
              </div>
              <div className="font-mono text-[8.5px] text-ink-faint">
                {result.config.vqc.shots === 0
                  ? 'exact expectation'
                  : `${result.config.vqc.shots} shots`}
              </div>
            </div>
          </div>
        </Panel>
      )}

      {/* comparison table */}
      <Panel>
        <div className="mb-3 flex items-baseline justify-between">
          <SectionLabel>comparison — all models on the same holdout</SectionLabel>
          <span className="font-mono text-[9px] text-ink-faint">
            {result.testSize} test samples / seed {result.config.seed}
          </span>
        </div>

        <div className="console-scroll overflow-x-auto">
          <table className="w-full min-w-[720px] border-collapse text-left">
            <thead>
              <tr>
                <th
                  className="pb-2 pr-3 font-mono text-[8.5px] text-ink-faint"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                >
                  model
                </th>
                {METRIC_COLUMNS.map((m) => (
                  <th
                    key={m.key}
                    className="pb-2 pr-3 text-right font-mono text-[8.5px] text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                  >
                    {m.label}
                  </th>
                ))}
                <th
                  className="pb-2 pr-3 text-right font-mono text-[8.5px] text-ink-faint"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                >
                  train
                </th>
                <th
                  className="pb-2 text-right font-mono text-[8.5px] text-ink-faint"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                >
                  inference
                </th>
              </tr>
            </thead>
            <tbody>
              {result.models.map((m) => {
                const isBest = m.id === best.id
                return (
                  <tr
                    key={m.id}
                    style={{
                      background: isBest ? alpha('#5FA88C', 0.05) : 'transparent',
                    }}
                  >
                    <td
                      className="py-2 pr-3"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className="h-[6px] w-[6px] shrink-0 rounded-full"
                          style={{ background: colorFor(m.id, m.kind) }}
                        />
                        <span className="font-mono text-[10px] text-ink">{label(m.id)}</span>
                        {m.kind === 'quantum' && (
                          <span
                            className="rounded-[3px] px-1 py-[1px] font-mono text-[8px]"
                            style={{
                              color: LANE_COLOR.quantum,
                              background: alpha(LANE_COLOR.quantum, 0.12),
                            }}
                          >
                            quantum
                          </span>
                        )}
                      </span>
                    </td>
                    {METRIC_COLUMNS.map((col) => {
                      const v = m.metrics[col.key]
                      const isMax =
                        v === Math.max(...result.models.map((x) => x.metrics[col.key]))
                      return (
                        <td
                          key={col.key}
                          className="py-2 pr-3 text-right font-mono text-[10px] tabular-nums"
                          style={{
                            borderBottom: '1px solid rgba(255,255,255,0.04)',
                            color: isMax ? '#5FA88C' : '#9A9CA1',
                          }}
                        >
                          {v.toFixed(3)}
                        </td>
                      )
                    })}
                    <td
                      className="py-2 pr-3 text-right font-mono text-[9.5px] tabular-nums text-ink-faint"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                    >
                      {m.trainMs < 1000 ? `${m.trainMs.toFixed(0)}ms` : `${(m.trainMs / 1000).toFixed(1)}s`}
                    </td>
                    <td
                      className="py-2 text-right font-mono text-[9.5px] tabular-nums text-ink-faint"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                    >
                      {m.inferenceMs.toFixed(1)}ms
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid grid-cols-[260px_1fr] gap-4">
        {/* ROC */}
        <Panel>
          <SectionLabel>ROC curves</SectionLabel>
          <p className="mb-2 mt-1 font-mono text-[9px] text-ink-faint">
            all models, one axis
          </p>
          <RocChart
            curves={result.models.map((m) => ({
              label: label(m.id),
              color: colorFor(m.id, m.kind),
              points: rocCurve(result.yTest, m.scores),
              auc: m.metrics.rocAuc,
            }))}
          />
        </Panel>

        {/* CV spread */}
        <Panel>
          <div className="mb-2 flex items-baseline justify-between">
            <div>
              <SectionLabel>cross-validation spread</SectionLabel>
              <p className="mt-1 font-mono text-[9px] text-ink-faint">
                {result.config.cvFolds}-fold accuracy — the range, not one number
              </p>
            </div>
            <LiveChip label="measured" />
          </div>

          {classical.some((m) => m.cv) ? (
            <>
              <BoxPlot
                height={150}
                series={classical
                  .filter((m) => m.cv)
                  .map((m) => ({
                    label: label(m.id),
                    color: colorFor(m.id, m.kind),
                    stats: m.cv!,
                    folds: m.cvFolds,
                  }))}
              />
              <p className="mt-2 font-mono text-[9px] leading-relaxed text-ink-faint/85">
                A single accuracy figure proves nothing. The box shows the interquartile range
                across folds and the dots are the individual fold scores — a wide box means the
                result depends heavily on which patients landed in which fold.
              </p>
            </>
          ) : (
            <p className="py-8 text-center font-mono text-[10px] text-ink-faint">
              select a classical baseline to see cross-validation spread
            </p>
          )}
        </Panel>
      </div>

      {/* confusion matrices */}
      <Panel>
        <div className="mb-3 flex items-baseline justify-between">
          <SectionLabel>error breakdown</SectionLabel>
          <span className="font-mono text-[9px] text-ink-faint">
            false negatives are the costly errors in early detection
          </span>
        </div>

        <div className="grid gap-5" style={{ gridTemplateColumns: `repeat(${Math.min(result.models.length, 4)}, 1fr)` }}>
          {result.models.map((m) => (
            <ConfusionGrid
              key={m.id}
              c={m.metrics.confusion}
              color={colorFor(m.id, m.kind)}
              title={label(m.id)}
              positiveLabel="disease"
              negativeLabel="healthy"
            />
          ))}
        </div>
      </Panel>

      {/* provenance */}
      <Panel>
        <SectionLabel>run provenance</SectionLabel>
        <div className="mt-2.5 grid grid-cols-4 gap-2">
          {[
            { k: 'backend', v: result.config.vqc.backend },
            {
              k: 'shots',
              v: result.config.vqc.shots === 0 ? 'exact (analytic)' : String(result.config.vqc.shots),
            },
            { k: 'qubits', v: `${result.config.nFeatures}` },
            { k: 'ansatz', v: result.config.vqc.ansatz },
            { k: 'feature map', v: result.config.vqc.featureMap },
            { k: 'selection', v: result.config.selection },
            { k: 'seed', v: String(result.config.seed) },
            { k: 'elapsed', v: `${(result.elapsedMs / 1000).toFixed(1)}s` },
          ].map((s) => (
            <div
              key={s.k}
              className="rounded-[7px] px-2.5 py-2"
              style={{
                background: '#0D0E10',
                border: '1px solid rgba(255,255,255,0.05)',
                boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
              }}
            >
              <div className="font-mono text-[8.5px] text-ink-faint">{s.k}</div>
              <div className="mt-0.5 font-mono text-[10px] text-ink-dim">{s.v}</div>
            </div>
          ))}
        </div>
        <p className="mt-2.5 font-mono text-[9px] leading-relaxed text-ink-faint/80">
          Every figure on this page was computed by this browser from the configuration above.
          The quantum lane ran on a statevector simulator — no real quantum hardware was used.
        </p>
      </Panel>

      <DeliveryTable />
    </div>
  )
}
