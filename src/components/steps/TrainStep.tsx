import { useEffect, useRef } from 'react'
import type { LogLine, RunPhase } from '../../hooks/useRun'
import type { RunConfig } from '../../lib/ml/pipeline'
import { circuitStats, type EpochRecord } from '../../lib/quantum/vqc'
import { formatElapsed, LANE_COLOR, alpha } from '../../lib/theme'
import { ConvergenceChart } from '../charts'
import { IconPlay, IconReset, IconStop } from '../icons'
import { PushButton } from '../PushButton'
import { QubitBadge } from '../QubitBadge'
import { LiveChip, Panel, SectionLabel } from '../ui'

type Props = {
  config: RunConfig
  phase: RunPhase
  logs: LogLine[]
  convergence: EpochRecord[]
  progress: number
  elapsed: number
  onStart: () => void
  onStop: () => void
  onReset: () => void
  onLoadDemo: () => void
}

const PHASE_COLOR: Record<RunPhase, string> = {
  idle: '#3A3C42',
  running: LANE_COLOR.classical,
  complete: '#5FA88C',
  stopped: '#4A4136',
}

export function TrainStep({
  config,
  phase,
  logs,
  convergence,
  progress,
  elapsed,
  onStart,
  onStop,
  onReset,
  onLoadDemo,
}: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const stats = circuitStats({ ...config.vqc, qubits: config.nFeatures })

  // Follow the log as it grows.
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logs.length, convergence.length])

  const last = convergence[convergence.length - 1]
  const running = phase === 'running'

  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className="h-[7px] w-[7px] rounded-full transition-colors duration-200"
              style={{
                background: PHASE_COLOR[phase],
                boxShadow: phase === 'idle' ? 'none' : `0 0 5px ${alpha(PHASE_COLOR[phase], 0.9)}`,
              }}
            />
            <span className="font-mono text-[13px] text-ink-dim">{phase}</span>
          </div>

          <span className="font-mono text-[14px] tabular-nums text-ink-dim">
            {formatElapsed(elapsed)}
          </span>

          <QubitBadge
            qubits={config.nFeatures}
            depth={stats.depth}
            gates={stats.gates}
            params={stats.params}
          />

          <div className="flex-1" />

          {running ? (
            <PushButton
              label="Stop"
              icon={<IconStop className="h-3.5 w-3.5" />}
              onClick={onStop}
              tone="primary"
              accent={LANE_COLOR.classical}
            />
          ) : (
            <PushButton
              label={phase === 'complete' ? 'Run again' : 'Run both lanes'}
              icon={<IconPlay className="h-3.5 w-3.5" />}
              onClick={phase === 'complete' ? () => { onReset(); setTimeout(onStart, 0) } : onStart}
              tone="primary"
              accent="#5FA88C"
            />
          )}
          <PushButton
            label="Reset"
            icon={<IconReset className="h-3.5 w-3.5" />}
            onClick={onReset}
            disabled={phase === 'idle'}
          />
          <PushButton label="Load demo run" onClick={onLoadDemo} disabled={running} />
        </div>

        {/* progress */}
        <div className="mt-3 h-[5px] w-full overflow-hidden rounded-full panel-well">
          <div
            className="h-full rounded-full transition-[width] duration-200"
            style={{
              width: `${progress * 100}%`,
              background: alpha(phase === 'complete' ? '#5FA88C' : LANE_COLOR.quantum, 0.8),
            }}
          />
        </div>
        <div className="mt-1.5 flex justify-between font-mono text-[11px] text-ink-faint">
          <span>
            {convergence.length > 0
              ? `quantum epoch ${convergence.length}/${config.epochs}`
              : 'one button runs quantum and classical together'}
          </span>
          <span className="tabular-nums">{Math.round(progress * 100)}%</span>
        </div>
      </Panel>

      <div className="grid grid-cols-[1fr_360px] gap-4">
        <Panel>
          <div className="mb-3 flex items-baseline justify-between">
            <div>
              <SectionLabel>convergence</SectionLabel>
              <p className="mt-1 font-mono text-[11px] text-ink-faint">
                cost function per iteration, computed live
              </p>
            </div>
            <LiveChip />
          </div>

          <ConvergenceChart points={convergence} height={190} />

          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5">
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-faint">
              <span
                className="h-[2px] w-[12px] rounded-full"
                style={{ background: LANE_COLOR.classical }}
              />
              loss (binary cross-entropy)
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-faint">
              <span
                className="h-[2px] w-[12px] rounded-full"
                style={{ background: LANE_COLOR.quantum }}
              />
              training accuracy
            </span>
          </div>

          {last && (
            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                { k: 'loss', v: last.loss.toFixed(4) },
                { k: 'train accuracy', v: last.trainAccuracy.toFixed(3) },
                { k: 'gradient norm', v: last.gradNorm.toFixed(4) },
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
                  <div className="font-mono text-[11px] text-ink-faint">{s.k}</div>
                  <div className="mt-0.5 font-mono text-[15.5px] tabular-nums text-ink">{s.v}</div>
                </div>
              ))}
            </div>
          )}

          {last && last.gradNorm < 0.01 && (
            <p
              className="mt-2.5 font-mono text-[11px] leading-relaxed"
              style={{ color: LANE_COLOR.classical }}
            >
              Gradient norm is very small: the optimiser may be on a barren plateau. Try fewer
              qubits or fewer layers.
            </p>
          )}
        </Panel>

        <Panel>
          <div className="mb-2.5 flex items-baseline justify-between">
            <SectionLabel>log</SectionLabel>
            <span className="font-mono text-[11px] text-ink-faint">
              {logs.length + convergence.length} entries
            </span>
          </div>

          <div
            ref={logRef}
            className="console-scroll h-[300px] overflow-y-auto rounded-[7px] p-2.5"
            style={{
              background: '#0D0E10',
              border: '1px solid rgba(255,255,255,0.05)',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            {logs.length === 0 && convergence.length === 0 ? (
              <p className="font-mono text-[11.5px] text-ink-faint/60">
                no entries, press run
              </p>
            ) : (
              <>
                {logs.map((l) => (
                  <div key={l.id} className="flex gap-2 py-[1px] font-mono text-[11px] leading-[1.6]">
                    <span className="shrink-0 text-ink-faint/50">[{l.phase}]</span>
                    <span className="min-w-0 flex-1 text-ink-dim">{l.message}</span>
                  </div>
                ))}
                {convergence.map((c) => (
                  <div
                    key={`e${c.epoch}`}
                    className="flex gap-2 py-[1px] font-mono text-[11px] leading-[1.6]"
                  >
                    <span className="shrink-0 text-ink-faint/50">[quantum]</span>
                    <span className="min-w-0 flex-1 text-ink-faint">
                      epoch {String(c.epoch).padStart(2, '0')}/{config.epochs} loss{' '}
                      {c.loss.toFixed(4)} acc {c.trainAccuracy.toFixed(3)}
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}
