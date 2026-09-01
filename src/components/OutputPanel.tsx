import { useState } from 'react'
import type { RunPhase } from '../hooks/usePipeline'
import { BENCHMARK_RESULTS } from '../lib/pipeline/graph'
import { LANE_COLOR } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconDownload } from './icons'
import { PushButton } from './PushButton'

type Format = 'json' | 'csv'

type Props = {
  phase: RunPhase
  stagesDone: number
  stagesTotal: number
  onDownload: (format: Format) => void
  onOpenComparison: () => void
}

const METRICS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'rocAuc', label: 'ROC-AUC' },
  { key: 'sensitivity', label: 'Sensitivity' },
  { key: 'specificity', label: 'Specificity' },
] as const

export function OutputPanel({
  phase,
  stagesDone,
  stagesTotal,
  onDownload,
  onOpenComparison,
}: Props) {
  const [format, setFormat] = useState<Format>('json')

  // A report is only meaningful once at least one stage has produced metrics.
  const ready = stagesDone > 0
  const partial = ready && phase !== 'complete'

  return (
    <section
      className="rounded-panel p-3"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
      aria-label="Output data"
    >
      <div className="mb-2.5 flex items-baseline justify-between">
        <h2 className="font-mono text-[11.5px] font-medium tracking-[0.02em] text-ink-faint">
          output data
        </h2>
        <span className="font-mono text-[11.5px] text-ink-faint">
          {stagesDone}/{stagesTotal} stages
        </span>
      </div>

      {!ready ? (
        <div
          className="rounded-[9px] px-3 py-5 text-center"
          style={{
            background: '#0D0E10',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
          }}
        >
          <p className="font-mono text-[12px] text-ink-faint">no output yet</p>
          <p className="mt-1 font-mono text-[11px] text-ink-faint/70">
            run the pipeline to generate a report
          </p>
        </div>
      ) : (
        <>
          <div className="mb-2.5">
            <DemoChip />
          </div>

          {/* headline comparison, mirrors the results panel */}
          <div
            className="rounded-[9px] p-2.5"
            style={{
              background: '#0D0E10',
              border: '1px solid rgba(255,255,255,0.05)',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            <div className="mb-2 flex justify-between font-mono text-[11px] tracking-[0.02em] text-ink-faint">
              <span>metric</span>
              <span className="flex gap-3">
                <span style={{ color: LANE_COLOR.classical }}>cls</span>
                <span style={{ color: LANE_COLOR.quantum }}>qnt</span>
              </span>
            </div>
            {METRICS.map((m) => (
              <div key={m.key} className="flex justify-between py-[3px]">
                <span className="font-mono text-[12px] text-ink-dim">{m.label}</span>
                <span className="flex gap-3 font-mono text-[12px] tabular-nums">
                  <span className="w-[38px] text-right text-ink-dim">
                    {BENCHMARK_RESULTS.classical[m.key].toFixed(3)}
                  </span>
                  <span className="w-[38px] text-right text-ink">
                    {BENCHMARK_RESULTS.quantum[m.key].toFixed(3)}
                  </span>
                </span>
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={onOpenComparison}
            className="mt-2 w-full cursor-pointer rounded-[7px] py-1.5 font-mono text-[11.5px] text-ink-faint transition-colors duration-150 hover:text-ink"
            style={{ background: 'rgba(255,255,255,0.03)' }}
          >
            open full comparison
          </button>

          {partial && (
            <div className="mt-2 font-mono text-[11px]" style={{ color: '#C08A3E' }}>
              run {phase} - report will be partial
            </div>
          )}
        </>
      )}

      {/* format selector + download */}
      <div className="mt-3 flex items-center gap-2">
        <div
          className="flex overflow-hidden rounded-[7px]"
          style={{
            background: '#0D0E10',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
          }}
          role="group"
          aria-label="Download format"
        >
          {(['json', 'csv'] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFormat(f)}
              aria-pressed={format === f}
              className="cursor-pointer px-2.5 py-1.5 font-mono text-[11.5px] transition-colors duration-150"
              style={{
                background: format === f ? 'rgba(255,255,255,0.06)' : 'transparent',
                color: format === f ? '#E8E9EB' : '#6A6C72',
              }}
            >
              {f}
            </button>
          ))}
        </div>

        <div className="flex-1">
          <PushButton
            label="Download"
            icon={<IconDownload className="h-3.5 w-3.5" />}
            onClick={() => onDownload(format)}
            disabled={!ready}
            tone="primary"
            accent={LANE_COLOR.quantum}
          />
        </div>
      </div>

      <p className="mt-2.5 font-mono text-[11px] leading-relaxed text-ink-faint/70">
        Report is flagged synthetic and carries a disclaimer field.
      </p>
    </section>
  )
}
