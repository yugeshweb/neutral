import { useEffect, useState } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'
import type { NodeStatus } from '../lib/pipeline/types'

type Props = {
  /** status of the first pipeline node, so the feed lights with it */
  status: NodeStatus
  reducedMotion: boolean
}

/**
 * Draws the feed from the input rail into the first graph node. It lives in an
 * overlay rather than the React Flow graph because its source is a DOM panel
 * outside the canvas; it measures both ends and re-measures on resize.
 */
export function SourceConnector({ status, reducedMotion }: Props) {
  const [geom, setGeom] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)

  useEffect(() => {
    const measure = () => {
      // Anchor on the input panel itself, not the scrolling rail, so the feed
      // leaves from the card edge rather than the container border.
      const rail = document.querySelector('[aria-label="Input data"]')
      const first = document.querySelector('.react-flow__node[data-id="ingest"]')
      const host = document.querySelector('main')
      if (!rail || !first || !host) return setGeom(null)

      const r = rail.getBoundingClientRect()
      const f = first.getBoundingClientRect()
      const h = host.getBoundingClientRect()

      const y1 = r.top + r.height / 2 - h.top
      const x2 = f.left - h.left
      // Below a usable horizontal run the feed would read as a stray mark.
      if (x2 < 40) return setGeom(null)

      setGeom({
        x1: 0,
        y1: Math.min(Math.max(y1, 10), h.height - 10),
        x2,
        y2: f.top + f.height / 2 - h.top,
      })
    }

    measure()
    // The node position shifts with pan/zoom, so poll on a frame loop while
    // mounted rather than guessing at React Flow's internal events.
    const id = setInterval(measure, 250)
    window.addEventListener('resize', measure)
    return () => {
      clearInterval(id)
      window.removeEventListener('resize', measure)
    }
  }, [])

  if (!geom || geom.x2 <= geom.x1) return null

  const accent = LANE_COLOR.shared
  const active = status === 'running'
  const done = status === 'done'
  // Keep the control points near the endpoints so the curve stays a gentle S
  // instead of bowing across the empty canvas.
  const dx = Math.min(64, (geom.x2 - geom.x1) * 0.45)
  const path = `M ${geom.x1} ${geom.y1} C ${geom.x1 + dx} ${geom.y1}, ${geom.x2 - dx} ${geom.y2}, ${geom.x2} ${geom.y2}`

  return (
    <svg
      className="pointer-events-none absolute inset-0 z-[5] h-full w-full"
      aria-hidden
      style={{ overflow: 'visible' }}
    >
      <path
        d={path}
        fill="none"
        stroke={active || done ? alpha(accent, done ? 0.32 : 0.9) : 'rgba(255,255,255,0.08)'}
        strokeWidth={1.5}
        strokeLinecap="round"
        style={{ transition: 'stroke 240ms ease-out' }}
      />

      {active && !reducedMotion && (
        <>
          <path
            d={path}
            fill="none"
            stroke={alpha(accent, 0.85)}
            strokeWidth={1.5}
            strokeDasharray="5 11"
            className="edge-flow"
          />
          <circle r={2.4} fill={accent}>
            <animateMotion dur="1.5s" repeatCount="indefinite" path={path} />
          </circle>
        </>
      )}

      {/* terminal pin at the panel edge */}
      <circle
        cx={geom.x1}
        cy={geom.y1}
        r={2.5}
        fill={active || done ? accent : 'rgba(255,255,255,0.18)'}
        style={{ transition: 'fill 240ms ease-out' }}
      />
    </svg>
  )
}
