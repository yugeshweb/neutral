import { useEffect, useRef } from 'react'
import { BENCHMARK_RESULTS } from '../lib/pipeline/graph'
import { LANE_COLOR, alpha } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconClose } from './icons'

const METRICS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'rocAuc', label: 'ROC-AUC' },
  { key: 'sensitivity', label: 'Sensitivity' },
  { key: 'specificity', label: 'Specificity' },
] as const

type Props = { open: boolean; onClose: () => void }

/** Horizontal meter in a recessed channel. Value is 0..1. */
function Meter({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-[4px] w-full overflow-hidden rounded-full panel-well">
      <div
        className="h-full rounded-full transition-[width] duration-500 ease-out"
        style={{ width: `${value * 100}%`, background: alpha(color, 0.75) }}
      />
    </div>
  )
}

function Column({
  title,
  subtitle,
  color,
  data,
  other,
}: {
  title: string
  subtitle: string
  color: string
  data: Record<string, number>
  other: Record<string, number>
}) {
  return (
    <div className="flex-1">
      <div className="mb-4 flex items-center gap-2">
        <span className="h-[7px] w-[7px] rounded-full" style={{ background: color }} />
        <div>
          <div className="text-[12.5px] font-medium leading-tight text-ink">{title}</div>
          <div className="mt-0.5 font-mono text-[9.5px] text-ink-faint">{subtitle}</div>
        </div>
      </div>

      <div className="space-y-3.5">
        {METRICS.map((m) => {
          const v = data[m.key]
          const delta = v - other[m.key]
          const leads = delta > 0
          return (
            <div key={m.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-2">
                <span className="font-mono text-[10px] text-ink-faint">{m.label}</span>
                <span className="flex items-baseline gap-1.5">
                  <span className="font-mono text-[13px] tabular-nums text-ink">
                    {v.toFixed(3)}
                  </span>
                  <span
                    className="font-mono text-[9.5px] tabular-nums"
                    style={{ color: leads ? '#5FA88C' : '#6A6C72' }}
                  >
                    {leads ? '+' : ''}
                    {delta.toFixed(3)}
                  </span>
                </span>
              </div>
              <Meter value={v} color={color} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function ResultsPanel({ open, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    ref.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const { classical, quantum } = BENCHMARK_RESULTS

  return (
    <div className="absolute inset-0 z-40 grid place-items-center p-6">
      <button
        type="button"
        aria-label="Dismiss results"
        onClick={onClose}
        className="absolute inset-0 cursor-default"
        style={{ background: 'rgba(6,6,8,0.72)' }}
      />

      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-label="Benchmark comparison"
        className="relative w-full max-w-[620px] rounded-panel outline-none"
        style={{
          background: '#17181B',
          border: '1px solid rgba(255,255,255,0.07)',
          boxShadow:
            'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 24px 50px rgba(0,0,0,0.7)',
        }}
      >
        <div
          className="flex items-start gap-3 px-5 py-4"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
        >
          <div className="flex-1">
            <h2 className="text-[13.5px] font-medium text-ink">Benchmark & Compare</h2>
            <p className="mt-1 font-mono text-[10px] text-ink-faint">
              holdout 86 samples / McNemar p=0.21
            </p>
          </div>
          <DemoChip />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close results"
            className="grid h-7 w-7 cursor-pointer place-items-center rounded-[7px] text-ink-faint transition-colors duration-150 hover:text-ink"
            style={{ background: '#111214', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <IconClose className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="flex gap-8 px-5 py-5">
          <Column
            title="Classical baseline"
            subtitle="XGBoost + RandomForest"
            color={LANE_COLOR.classical}
            data={classical}
            other={quantum}
          />
          <div className="w-px shrink-0" style={{ background: 'rgba(255,255,255,0.06)' }} />
          <Column
            title="Hybrid quantum"
            subtitle="8-qubit VQC / 4 layers"
            color={LANE_COLOR.quantum}
            data={quantum}
            other={classical}
          />
        </div>

        <div
          className="px-5 py-3"
          style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
        >
          <p className="font-mono text-[10px] leading-relaxed text-ink-faint">
            All figures are placeholder values generated by the mock runner. No model was
            trained and no dataset was evaluated.
          </p>
        </div>
      </div>
    </div>
  )
}
