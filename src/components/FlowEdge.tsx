import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react'
import { memo } from 'react'
import { alpha } from '../lib/theme'

export type FlowEdgeData = {
  accent: string
  /** 'idle' until the source node reports done, then 'active', then 'complete' */
  phase: 'idle' | 'active' | 'complete'
  reducedMotion: boolean
}

function FlowEdgeInner({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const d = (data ?? {}) as FlowEdgeData
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    curvature: 0.35,
  })

  const phase = d.phase ?? 'idle'
  const accent = d.accent ?? '#8A8F98'

  const stroke =
    phase === 'idle'
      ? 'rgba(255,255,255,0.08)'
      : phase === 'active'
        ? accent
        : alpha(accent, 0.32)

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke,
          strokeWidth: 1.5,
          transition: 'stroke 240ms ease-out',
        }}
      />

      {/* flowing dash overlay, active only */}
      {phase === 'active' && !d.reducedMotion && (
        <path
          d={path}
          fill="none"
          stroke={alpha(accent, 0.85)}
          strokeWidth={1.5}
          strokeDasharray="5 11"
          className="edge-flow"
        />
      )}

      {/* travelling dot */}
      {phase === 'active' && !d.reducedMotion && (
        <circle r={2.4} fill={accent}>
          <animateMotion dur="1.5s" repeatCount="indefinite" path={path} />
        </circle>
      )}
    </>
  )
}

export const FlowEdge = memo(FlowEdgeInner)
