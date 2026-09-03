import { useState } from 'react'
import type { Finding } from '../lib/findings'
import { SEVERITY_COLOR } from '../lib/findings'
import { getFindingMorphology } from '../lib/morphology'

interface Props {
  findings: Finding[]
  activeId: string | null
  onSelect?: (f: Finding | null) => void
  mode?: 'hybrid' | 'contours' | 'heatmap'
  compact?: boolean
  className?: string
}

/**
 * Clinical radiological contours overlay.
 *
 * Renders crisp anatomical lesion boundary paths (spiculated/lobulated vector contours),
 * multi-level Grad-CAM iso-activation curves, and clean center reticles.
 * Completely free of cluttering callout boxes, circular rings, or browser focus outlines.
 */
export function RadiologicalContours({
  findings,
  activeId,
  onSelect,
  mode = 'hybrid',
  compact = false,
  className = '',
}: Props) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  if (mode === 'heatmap') {
    return null
  }

  return (
    <div className={`absolute inset-0 select-none pointer-events-none ${className}`}>
      {/* SVG Layer for Organic Vector Contours & Reticles */}
      <svg
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full outline-none"
        style={{ overflow: 'visible', outline: 'none' }}
      >
        <defs>
          <filter id="contour-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {findings.map((f) => {
          const m = getFindingMorphology(f)
          const isActive = activeId === f.id
          const isHovered = hoveredId === f.id
          const isFocus = isActive || isHovered
          const color = SEVERITY_COLOR[f.severity]

          const cx = f.x * 1000
          const cy = f.y * 1000

          return (
            <g key={f.id} className="pointer-events-none">
              {/* Topographic Iso-Activation Contours (Iso-lines) */}
              {m.isoCurves.map((iso) => {
                const showIso = mode === 'contours' || isFocus
                const isoOpacity = showIso ? (isFocus ? iso.opacity : iso.opacity * 0.4) : 0
                return (
                  <path
                    key={iso.level}
                    d={iso.pathD}
                    fill="none"
                    stroke={color}
                    strokeWidth={isFocus ? (iso.level > 0.8 ? 1.6 : 1.0) : 0.8}
                    strokeDasharray={iso.dash}
                    style={{
                      opacity: isoOpacity,
                      transition: 'opacity 160ms ease-out, stroke-width 160ms ease-out',
                    }}
                  />
                )
              })}

              {/* Primary Biological Lesion Boundary Path */}
              <path
                d={m.lesionPath}
                fill={isFocus ? `${color}28` : `${color}0f`}
                stroke={color}
                strokeWidth={isFocus ? 2.2 : 1.5}
                filter={isFocus ? 'url(#contour-glow)' : undefined}
                style={{
                  transition: 'fill 160ms ease-out, stroke-width 160ms ease-out',
                }}
              />

              {/* Central Precision Reticle Crosshair */}
              <g
                style={{
                  opacity: isFocus ? 1 : 0.75,
                  transition: 'opacity 160ms ease-out',
                }}
              >
                <line
                  x1={cx - (compact ? 5 : 8)}
                  y1={cy}
                  x2={cx + (compact ? 5 : 8)}
                  y2={cy}
                  stroke="#FFFFFF"
                  strokeWidth="1.2"
                  opacity="0.8"
                />
                <line
                  x1={cx}
                  y1={cy - (compact ? 5 : 8)}
                  x2={cx}
                  y2={cy + (compact ? 5 : 8)}
                  stroke="#FFFFFF"
                  strokeWidth="1.2"
                  opacity="0.8"
                />
                <circle
                  cx={cx}
                  cy={cy}
                  r={compact ? 2 : 3}
                  fill={color}
                  stroke="#0A0B0E"
                  strokeWidth="1.2"
                />
              </g>

              {/* Clean Orthogonal Dimension Lines in Inspector (No Boxes) */}
              {!compact && isFocus && (
                <g opacity="0.75" className="transition-opacity duration-200 pointer-events-none">
                  <line
                    x1={m.calipers.major.x1}
                    y1={m.calipers.major.y1}
                    x2={m.calipers.major.x2}
                    y2={m.calipers.major.y2}
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="4 3"
                  />
                  <line
                    x1={m.calipers.minor.x1}
                    y1={m.calipers.minor.y1}
                    x2={m.calipers.minor.x2}
                    y2={m.calipers.minor.y2}
                    stroke={color}
                    strokeWidth="0.8"
                    strokeDasharray="3 3"
                  />
                </g>
              )}
            </g>
          )
        })}
      </svg>

      {/* Transparent Target Buttons (Generous hit target, zero hover box, zero circles, zero focus outline) */}
      {findings.map((f) => {
        const m = getFindingMorphology(f)
        const isActive = activeId === f.id

        // Generous size for target button so hovering/clicking is easy
        const hitSize = Math.max(54, Math.round(m.sigmaU * 1000 * 0.3))

        return (
          <button
            key={f.id}
            type="button"
            onClick={() => onSelect?.(isActive ? null : f)}
            onMouseEnter={() => setHoveredId(f.id)}
            onMouseLeave={() => setHoveredId((prev) => (prev === f.id ? null : prev))}
            aria-label={`${f.label} (${f.severity} severity)`}
            aria-pressed={isActive}
            className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer outline-none focus:outline-none focus:ring-0 border-none bg-transparent transition-transform duration-150 active:scale-95"
            style={{
              left: `${f.x * 100}%`,
              top: `${f.y * 100}%`,
              width: `${hitSize}px`,
              height: `${hitSize}px`,
              outline: 'none',
              WebkitTapHighlightColor: 'transparent',
            }}
          />
        )
      })}
    </div>
  )
}
