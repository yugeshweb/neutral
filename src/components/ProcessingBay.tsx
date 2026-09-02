import { useEffect, useRef, useState } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'

/**
 * The processing animation.
 *
 * A document is drawn into a machine, a scan head passes over it, and the
 * stages light up in order. It is deliberately a *machine* rather than a
 * spinner: this screen is about a file being consumed by a pipeline, and the
 * stage list doubles as an explanation of what that pipeline does.
 *
 * The stages shown are the real ones the pipeline runs. The timing is not
 * measured - it is a fixed five-second sequence - so the caption says so
 * rather than implying these durations came from instrumentation.
 */

const STAGES = [
  'Reading file',
  'Parsing and validating schema',
  'Imputing and scaling',
  'Encoding to qubit rotations',
  'Running the circuit',
  'Scoring and attribution',
]

export function ProcessingBay({
  fileName,
  durationMs = 5000,
  onDone,
}: {
  fileName: string
  durationMs?: number
  onDone: () => void
}) {
  const [elapsed, setElapsed] = useState(0)
  const doneRef = useRef(false)

  useEffect(() => {
    const start = performance.now()
    let raf = 0

    const tick = () => {
      const t = Math.min(1, (performance.now() - start) / durationMs)
      setElapsed(t)
      if (t < 1) {
        raf = requestAnimationFrame(tick)
      } else if (!doneRef.current) {
        // Guard: StrictMode double-invokes effects in development, and the
        // callback advances a step, so firing it twice would skip a screen.
        doneRef.current = true
        onDone()
      }
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [durationMs, onDone])

  const stageIndex = Math.min(STAGES.length - 1, Math.floor(elapsed * STAGES.length))
  // The sheet travels in over the first third, then sits under the scan head.
  const feed = Math.min(1, elapsed * 3)
  const sheetY = -46 + feed * 46

  return (
    <div className="panel-raised rounded-panel panel-pad flow-step">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-[14.5px] font-medium text-ink">Processing</h2>
        <span className="engraved truncate font-mono text-[11px]">{fileName}</span>
      </div>

      {/* Centred in whatever height the flow reserves, so the machine sits in
          the middle of the card rather than pinned under the heading. */}
      <div className="flow-body mt-4 grid grid-cols-1 content-center gap-4 md:grid-cols-2">
        {/* The machine */}
        <div className="readout grid h-full place-items-center px-4 py-8">
          <svg viewBox="0 0 200 120" className="w-full max-w-[380px]" role="img" aria-label="Processing">
            <title>File passing through the analysis pipeline</title>

            {/* the sheet being drawn in */}
            <g transform={`translate(0 ${sheetY})`}>
              <rect
                x="76" y="8" width="48" height="40" rx="3"
                fill="#1b1d21" stroke="rgba(255,255,255,0.18)"
              />
              {[16, 24, 32, 40].map((y, i) => (
                <rect
                  key={y}
                  x="83" y={y} width={i === 3 ? 20 : 34} height="2.5" rx="1.2"
                  fill="rgba(255,255,255,0.22)"
                />
              ))}
            </g>

            {/* machine body */}
            <rect
              x="34" y="52" width="132" height="46" rx="6"
              fill="#17181b" stroke="rgba(255,255,255,0.14)"
            />
            {/* intake slot */}
            <rect x="72" y="49" width="56" height="5" rx="2.5" fill="#0b0c0e" />

            {/* scan head sweeping across the body */}
            <rect
              x={44 + Math.sin(elapsed * Math.PI * 4) * 0.5 * 96 + 48}
              y="60" width="3" height="30" rx="1.5"
              fill={LANE_COLOR.quantum}
              opacity={0.9}
            />

            {/* lamps, one per stage */}
            {STAGES.map((_, i) => (
              <circle
                key={i}
                cx={52 + i * 19}
                cy="104"
                r="3.4"
                fill={i <= stageIndex ? LANE_COLOR.quantum : '#2a2c32'}
              />
            ))}
          </svg>
        </div>

        {/* The stage list */}
        <div className="flex flex-col justify-center gap-2.5">
          {STAGES.map((s, i) => {
            const done = i < stageIndex
            const active = i === stageIndex
            return (
              <div key={s} className="flex items-center gap-2.5 font-mono text-[13px]">
                <span
                  className="h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{
                    background: done || active ? LANE_COLOR.quantum : '#2a2c32',
                    opacity: active ? 1 : done ? 0.55 : 1,
                  }}
                />
                <span
                  style={{
                    color: active ? '#E8E9EB' : done ? '#9A9CA1' : '#6A6C72',
                  }}
                >
                  {s}
                </span>
              </div>
            )
          })}

          <div className="panel-well mt-2 h-1 w-full overflow-hidden rounded-full">
            <div
              className="h-full rounded-full"
              style={{
                width: `${elapsed * 100}%`,
                background: alpha(LANE_COLOR.quantum, 0.75),
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
