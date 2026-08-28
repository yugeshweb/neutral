import { LANE_COLOR } from '../lib/theme'

const ITEMS = [
  { label: 'shared', color: LANE_COLOR.shared },
  { label: 'classical', color: LANE_COLOR.classical },
  { label: 'quantum', color: LANE_COLOR.quantum },
]

/** Reads the lane colours back to the user. Sits over the canvas, bottom left. */
export function LaneLegend() {
  return (
    <div
      className="absolute bottom-4 left-4 z-20 flex items-center gap-3.5 rounded-[9px] px-3 py-2"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06), 0 1px 2px rgba(0,0,0,0.8)',
      }}
    >
      {ITEMS.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span className="h-[5px] w-[5px] rounded-full" style={{ background: i.color }} />
          <span className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
            {i.label}
          </span>
        </span>
      ))}
    </div>
  )
}
