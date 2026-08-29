import { useMemo } from 'react'
import { loadDataset } from '../../lib/ml/datasets'
import type { RunConfig } from '../../lib/ml/pipeline'
import {
  applyScaler,
  classCounts,
  fitScaler,
  imputeMissing,
  stratifiedSplit,
} from '../../lib/ml/stats'
import { makeRng } from '../../lib/quantum/statevector'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { Tooltip } from '../Tooltip'
import { Field, LiveChip, Panel, SectionLabel, Segmented } from '../ui'

/** Histogram of one column, drawn before and after scaling. */
function Histogram({
  values,
  color,
  bins = 22,
  height = 62,
}: {
  values: number[]
  color: string
  bins?: number
  height?: number
}) {
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const counts = new Array(bins).fill(0)
  for (const v of values) {
    let b = Math.floor(((v - lo) / (hi - lo || 1)) * bins)
    if (b >= bins) b = bins - 1
    if (b < 0) b = 0
    counts[b]++
  }
  const max = Math.max(...counts, 1)

  return (
    <div>
      <div className="flex items-end gap-[1px]" style={{ height }}>
        {counts.map((c, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-[1px]"
            style={{
              height: `${(c / max) * 100}%`,
              background: alpha(color, 0.55),
              minHeight: c > 0 ? 1 : 0,
            }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[8px] text-ink-faint">
        <span>{lo.toFixed(2)}</span>
        <span>{hi.toFixed(2)}</span>
      </div>
    </div>
  )
}

type Props = {
  config: RunConfig
  patch: (p: Partial<RunConfig>) => void
  locked: boolean
}

export function PreprocessStep({ config, patch, locked }: Props) {
  const data = useMemo(() => loadDataset(config.datasetId), [config.datasetId])

  // Recompute the actual effect of the current settings, live.
  const preview = useMemo(() => {
    const rng = makeRng(config.seed)
    const imputed = imputeMissing(data.X, data.y, config.impute)
    const split = stratifiedSplit(imputed.y, config.testFraction, rng)
    const Xtr = split.trainIdx.map((i) => imputed.X[i])
    const scaler = fitScaler(Xtr, config.scaler)
    const scaled = applyScaler(Xtr, scaler)

    // Show the widest-range column, where scaling matters most.
    let widest = 0
    let widestRange = 0
    for (let j = 0; j < data.featureNames.length; j++) {
      const col = Xtr.map((r) => r[j])
      const range = Math.max(...col) - Math.min(...col)
      if (range > widestRange) {
        widestRange = range
        widest = j
      }
    }

    return {
      before: Xtr.map((r) => r[widest]),
      after: scaled.map((r) => r[widest]),
      column: data.featureNames[widest],
      filled: imputed.filled.reduce((a, b) => a + b, 0),
      dropped: imputed.rowsDropped,
      counts: classCounts(imputed.y),
      trainSize: Xtr.length,
    }
  }, [data, config.impute, config.scaler, config.seed, config.testFraction])

  const angleMin = Math.min(...preview.after)
  const angleMax = Math.max(...preview.after)

  return (
    <div className="grid grid-cols-[320px_1fr] gap-4">
      <div className="space-y-4">
        <Panel>
          <SectionLabel>missing values</SectionLabel>
          <div className="mt-2.5">
            <Segmented
              ariaLabel="Missing value strategy"
              disabled={locked}
              value={config.impute}
              onChange={(v) => patch({ impute: v })}
              options={[
                { value: 'drop', label: 'drop', title: 'Remove any row with a missing cell' },
                { value: 'mean', label: 'mean', title: 'Fill with the column mean' },
                { value: 'median', label: 'median', title: 'Fill with the column median, robust to outliers' },
              ]}
            />
          </div>
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-ink-faint/85">
            {preview.dropped > 0
              ? `${preview.dropped} rows removed`
              : preview.filled > 0
                ? `${preview.filled} cells filled`
                : 'no missing values in this dataset'}
          </p>
        </Panel>

        <Panel>
          <div className="flex items-baseline justify-between">
            <SectionLabel>scaling</SectionLabel>
            <span
              className="rounded-[4px] px-1.5 py-[1px] font-mono text-[8.5px]"
              style={{ color: LANE_COLOR.classical, background: alpha(LANE_COLOR.classical, 0.1) }}
            >
              required
            </span>
          </div>
          <div className="mt-2.5">
            <Segmented
              ariaLabel="Scaler"
              disabled={locked}
              value={config.scaler}
              onChange={(v) => patch({ scaler: v })}
              options={[
                { value: 'standard', label: 'standard', title: 'Zero mean, unit variance' },
                { value: 'minmax', label: 'min-max', title: 'Rescale to a fixed range' },
              ]}
            />
          </div>

          <div
            className="mt-2.5 rounded-[7px] p-2.5"
            style={{
              background: alpha(LANE_COLOR.quantum, 0.06),
              border: `1px solid ${alpha(LANE_COLOR.quantum, 0.2)}`,
            }}
          >
            <p className="font-mono text-[9px] leading-relaxed text-ink-dim">
              Scaling is not optional here. Quantum encoding turns each value into a{' '}
              <Tooltip
                term="Rotation angle"
                body="Each feature is written into a qubit as an angle of rotation. Angles wrap at 2π, so two very different raw values can land on the same state unless every feature is first squeezed into one fixed range."
              >
                <span
                  className="underline decoration-dotted underline-offset-2"
                  style={{ color: LANE_COLOR.quantum }}
                >
                  rotation angle
                </span>
              </Tooltip>
              , so values must sit in a fixed range or they alias onto each other.
            </p>
          </div>
        </Panel>

        <Panel>
          <SectionLabel>class imbalance</SectionLabel>
          <div className="mt-2.5">
            <Segmented
              ariaLabel="Imbalance strategy"
              disabled={locked}
              value={config.balance}
              onChange={(v) => patch({ balance: v })}
              options={[
                { value: 'none', label: 'none' },
                { value: 'oversample', label: 'oversample', title: 'Duplicate minority samples until balanced' },
                { value: 'class-weight', label: 'weight', title: 'Weight the loss by inverse class frequency' },
              ]}
            />
          </div>
          <div className="mt-2.5 space-y-1">
            {[...preview.counts].map(([label, count]) => (
              <div key={label} className="flex items-center justify-between font-mono text-[9.5px]">
                <span className="text-ink-faint">class {label}</span>
                <span className="tabular-nums text-ink-dim">{count}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel>
        <div className="mb-3 flex items-baseline justify-between">
          <div>
            <SectionLabel>distribution: before and after</SectionLabel>
            <p className="mt-1 font-mono text-[9px] text-ink-faint">
              column <span className="text-ink-dim">{preview.column}</span>, the widest range in
              this dataset
            </p>
          </div>
          <LiveChip />
        </div>

        <div className="grid grid-cols-2 gap-5">
          <Field label="before scaling">
            <Histogram values={preview.before} color={LANE_COLOR.classical} />
            <p className="mt-1.5 font-mono text-[9px] text-ink-faint">
              raw clinical units, unbounded
            </p>
          </Field>

          <Field label="after scaling">
            <Histogram values={preview.after} color={LANE_COLOR.quantum} />
            <p className="mt-1.5 font-mono text-[9px] text-ink-faint">
              range {angleMin.toFixed(2)} to {angleMax.toFixed(2)} — safe to encode
            </p>
          </Field>
        </div>

        <div
          className="mt-4 flex items-start gap-2.5 rounded-[8px] p-3"
          style={{
            background: '#0D0E10',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
          }}
        >
          <span
            className="mt-[5px] h-[6px] w-[6px] shrink-0 rounded-full"
            style={{ background: '#5FA88C' }}
          />
          <p className="font-mono text-[9.5px] leading-relaxed text-ink-faint">
            The scaler is fitted on the{' '}
            <span className="text-ink-dim">{preview.trainSize} training rows only</span>, then
            applied to the test fold. Fitting on the full dataset would leak test statistics into
            training and inflate every score that follows.
          </p>
        </div>
      </Panel>
    </div>
  )
}
