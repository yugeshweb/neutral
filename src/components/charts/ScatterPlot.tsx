import { useState } from 'react'

type Point = {
  x: number
  y: number
  z?: number
  label: number
  id?: string | number
}

type Props = {
  points: Point[]
  positiveLabel: string
  negativeLabel: string
  xLabel?: string
  yLabel?: string
  zLabel?: string
  height?: number
  is3d?: boolean
}

export function ScatterPlot({
  points,
  positiveLabel,
  negativeLabel,
  xLabel = 'Principal Component 1 (PC1)',
  yLabel = 'Principal Component 2 (PC2)',
  zLabel = 'Principal Component 3 (PC3)',
  height = 320,
  is3d = false,
}: Props) {
  const [rotX, setRotX] = useState(25)
  const [rotY, setRotY] = useState(35)
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null)

  if (!points || points.length === 0) {
    return (
      <div
        className="grid place-items-center rounded-[8px] panel-well"
        style={{ height }}
      >
        <span className="font-mono text-[13px] text-ink-faint">
          No feature projection data available
        </span>
      </div>
    )
  }

  const POS_COLOR = '#A3543D' // Coral / Positive / Risk
  const NEG_COLOR = '#5FA88C' // Sage / Negative / Healthy

  // 2D Projection limits
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)

  const spanX = Math.max(maxX - minX, 1e-4)
  const spanY = Math.max(maxY - minY, 1e-4)

  const w = 560
  const h = height
  const padL = 44
  const padR = 24
  const padT = 20
  const padB = 34

  // 3D projection transformation helper
  const project3D = (p: Point) => {
    const radX = (rotX * Math.PI) / 180
    const radY = (rotY * Math.PI) / 180

    // Normalize coordinates to -1 .. 1
    const nx = ((p.x - minX) / spanX) * 2 - 1
    const ny = ((p.y - minY) / spanY) * 2 - 1
    const nz = p.z !== undefined ? p.z * 2 - 1 : 0

    // Rotation around Y
    const x1 = nx * Math.cos(radY) + nz * Math.sin(radY)
    const z1 = -nx * Math.sin(radY) + nz * Math.cos(radY)

    // Rotation around X
    const y2 = ny * Math.cos(radX) - z1 * Math.sin(radX)
    const z2 = ny * Math.sin(radX) + z1 * Math.cos(radX)

    // Perspective projection
    const scale = 180 / (2.5 + z2 * 0.4)
    const screenX = w / 2 + x1 * scale
    const screenY = h / 2 - y2 * scale

    return { screenX, screenY, depth: z2 }
  }

  const projectedPoints = is3d
    ? points
        .map((p, i) => ({ ...p, id: i, ...project3D(p) }))
        .sort((a, b) => a.depth - b.depth)
    : points.map((p, i) => ({
        ...p,
        id: i,
        screenX: padL + ((p.x - minX) / spanX) * (w - padL - padR),
        screenY: padT + (1 - (p.y - minY) / spanY) * (h - padT - padB),
        depth: 0,
      }))

  return (
    <div className="relative select-none">
      {/* Interactive 3D Rotation Controls if in 3D Mode */}
      {is3d && (
        <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-[6px] bg-black/60 px-2 py-1 backdrop-blur-sm">
          <span className="font-mono text-[11px] text-ink-faint">Drag / Rotate:</span>
          <button
            type="button"
            onClick={() => {
              setRotY((r) => r + 15)
            }}
            className="cursor-pointer rounded bg-white/10 px-1.5 py-0.5 font-mono text-[11px] text-ink hover:bg-white/20"
          >
            Rot Y
          </button>
          <button
            type="button"
            onClick={() => {
              setRotX((r) => (r + 10) % 360)
            }}
            className="cursor-pointer rounded bg-white/10 px-1.5 py-0.5 font-mono text-[11px] text-ink hover:bg-white/20"
          >
            Rot X
          </button>
        </div>
      )}

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        style={{ cursor: is3d ? 'grab' : 'crosshair' }}
        onMouseMove={(e) => {
          if (is3d && e.buttons === 1) {
            setRotY((y) => y + e.movementX * 0.5)
            setRotX((x) => Math.max(-60, Math.min(60, x - e.movementY * 0.5)))
          }
        }}
        role="img"
        aria-label="Dimensionality Reduction Feature Space Projection"
      >
        {/* Background Grid for 2D Mode */}
        {!is3d && (
          <>
            {[0, 0.25, 0.5, 0.75, 1].map((f) => {
              const yVal = padT + f * (h - padT - padB)
              const xVal = padL + f * (w - padL - padR)
              return (
                <g key={f}>
                  <line
                    x1={padL}
                    y1={yVal}
                    x2={w - padR}
                    y2={yVal}
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth="1"
                  />
                  <line
                    x1={xVal}
                    y1={padT}
                    x2={xVal}
                    y2={h - padB}
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth="1"
                  />
                </g>
              )
            })}

            {/* Axes */}
            <line
              x1={padL}
              y1={padT}
              x2={padL}
              y2={h - padB}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth="1"
            />
            <line
              x1={padL}
              y1={h - padB}
              x2={w - padR}
              y2={h - padB}
              stroke="rgba(255,255,255,0.15)"
            />

            {/* Axis Titles */}
            <text
              x={w / 2}
              y={h - 8}
              textAnchor="middle"
              className="font-mono text-[11px] fill-ink-faint"
            >
              {xLabel} →
            </text>
            <text
              x={14}
              y={h / 2}
              textAnchor="middle"
              transform={`rotate(-90 14 ${h / 2})`}
              className="font-mono text-[11px] fill-ink-faint"
            >
              {yLabel} →
            </text>
          </>
        )}

        {/* 3D Wireframe Box Indicator */}
        {is3d && (
          <g opacity={0.25} stroke="#6A6C72" strokeWidth="0.8" fill="none">
            <ellipse
              cx={w / 2}
              cy={h / 2 + 50}
              rx={120}
              ry={35}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="3 3"
            />
            <text
              x={w / 2}
              y={h - 10}
              textAnchor="middle"
              className="font-mono text-[11px] fill-ink-faint"
            >
              Interactive 3D Projections ({xLabel}, {yLabel}, {zLabel})
            </text>
          </g>
        )}

        {/* Scatter Points */}
        {projectedPoints.map((p) => {
          const isPos = p.label === 1
          const color = isPos ? POS_COLOR : NEG_COLOR
          const radius = is3d ? Math.max(2.2, 4 + p.depth * 1.5) : 3.8

          return (
            <circle
              key={p.id}
              cx={p.screenX}
              cy={p.screenY}
              r={radius}
              fill={color}
              fillOpacity={0.82}
              stroke="#0D0E10"
              strokeWidth={0.7}
              className="transition-transform duration-75 hover:scale-150"
              onMouseEnter={() => setHoveredPoint(p)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
          )
        })}
      </svg>

      {/* Tooltip Overlay */}
      {hoveredPoint && (
        <div
          className="pointer-events-none absolute z-20 rounded-[6px] border border-white/10 bg-[#17181B] px-2.5 py-1.5 shadow-xl"
          style={{
            left: 20,
            top: 20,
          }}
        >
          <div className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{
                background: hoveredPoint.label === 1 ? POS_COLOR : NEG_COLOR,
              }}
            />
            <span className="font-mono text-[12px] font-medium text-ink">
              {hoveredPoint.label === 1 ? positiveLabel : negativeLabel}
            </span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-ink-faint">
            {xLabel.slice(0, 8)}: {hoveredPoint.x.toFixed(3)} | {yLabel.slice(0, 8)}:{' '}
            {hoveredPoint.y.toFixed(3)}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-2 flex items-center justify-between px-3">
        <div className="flex items-center gap-4 font-mono text-[12px]">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: NEG_COLOR }} />
            <span className="text-ink-dim">{negativeLabel}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: POS_COLOR }} />
            <span className="text-ink-dim">{positiveLabel}</span>
          </span>
        </div>
        <span className="font-mono text-[11px] text-ink-faint">
          {points.length} samples projected into reduced feature space
        </span>
      </div>
    </div>
  )
}
