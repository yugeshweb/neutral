import { Handle, Position, type NodeProps } from '@xyflow/react'
import { memo } from 'react'
import type { Lane, NodeStatus } from '../lib/pipeline/types'
import { LAMP_COLOR, STATUS_LABEL, alpha, laneColor } from '../lib/theme'
import { STAGE_ICON } from './icons'

export type StageNodeData = {
  stageId: string
  label: string
  subtitle: string
  lane: Lane
  status: NodeStatus
  progress: number
  metrics?: Record<string, number | string>
  selected?: boolean
}

/** Small recessed LED. The only element in the UI permitted to glow. */
function Lamp({ status }: { status: NodeStatus }) {
  const color = LAMP_COLOR[status]
  const lit = status === 'running' || status === 'done' || status === 'error'

  return (
    <span
      className="relative grid h-[14px] w-[14px] place-items-center rounded-full panel-well"
      title={STATUS_LABEL[status]}
    >
      <span
        className="h-[6px] w-[6px] rounded-full transition-colors duration-200"
        style={{
          background: color,
          boxShadow: lit ? `0 0 4px ${alpha(color, 0.9)}, 0 0 8px ${alpha(color, 0.45)}` : 'none',
        }}
      />
    </span>
  )
}

/** Recessed channel with a filled bar. Shimmers while indeterminate. */
function ProgressChannel({
  progress,
  accent,
  status,
}: {
  progress: number
  accent: string
  status: NodeStatus
}) {
  const indeterminate = status === 'running' && progress <= 0.001
  const pct = status === 'done' ? 100 : Math.round(progress * 100)

  return (
    <div className="h-[5px] w-full overflow-hidden rounded-full panel-well">
      {indeterminate ? (
        <div
          className="progress-shimmer h-full w-1/3 rounded-full"
          style={{ background: alpha(accent, 0.55) }}
        />
      ) : (
        <div
          className="h-full rounded-full transition-[width] duration-150 ease-out"
          style={{
            width: `${pct}%`,
            background: status === 'idle' ? 'transparent' : accent,
            opacity: status === 'done' ? 0.55 : 1,
          }}
        />
      )}
    </div>
  )
}

function StageNodeInner({ data, selected }: NodeProps) {
  const d = data as unknown as StageNodeData
  const accent = laneColor(d.lane, d.status)
  const Icon = STAGE_ICON[d.stageId] ?? null
  const idle = d.status === 'idle'
  const blocked = d.status === 'queued'

  const footer = d.metrics
    ? Object.entries(d.metrics)
        .slice(0, 3)
        .map(([k, v]) => `${k} ${v}`)
        .join('   ')
    : d.subtitle

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`${d.label}, ${d.lane} lane, ${STATUS_LABEL[d.status]}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          ;(e.currentTarget as HTMLElement).click()
        }
      }}
      className="w-[228px] cursor-pointer rounded-panel px-3 pb-2.5 pt-3 transition-[border-color] duration-200"
      style={{
        background: '#17181B',
        border: `1px solid ${
          selected
            ? 'rgba(255,255,255,0.16)'
            : d.status === 'error'
              ? alpha(accent, 0.4)
              : 'rgba(255,255,255,0.06)'
        }`,
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.6)',
        opacity: blocked ? 0.42 : 1,
      }}
    >
      <Handle type="target" position={Position.Left} />

      <div className="flex items-start gap-2.5">
        {/* icon in a recessed square well */}
        <span
          className="grid h-8 w-8 shrink-0 place-items-center rounded-[7px] panel-well"
          style={{ color: idle || blocked ? '#5A5D64' : accent }}
        >
          {Icon ? <Icon className="h-[17px] w-[17px]" /> : null}
        </span>

        <div className="min-w-0 flex-1 pt-px">
          <div className="truncate text-[12.5px] font-medium leading-tight text-ink">{d.label}</div>
          <div className="mt-1 truncate font-mono text-[10px] leading-tight text-ink-faint">
            {d.subtitle}
          </div>
        </div>

        <Lamp status={d.status} />
      </div>

      <div className="mt-3">
        <ProgressChannel progress={d.progress} accent={accent} status={d.status} />
      </div>

      <div
        className="mt-2 truncate font-mono text-[9.5px] leading-tight transition-colors duration-200"
        style={{ color: idle || blocked ? '#55575D' : '#8B8E95' }}
      >
        {blocked ? 'blocked / upstream failed' : footer}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
}

export const StageNode = memo(StageNodeInner)
