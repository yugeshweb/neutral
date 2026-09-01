import { useEffect, useRef } from 'react'
import type { LogLine, NodeState } from '../hooks/usePipeline'
import { NODE_BY_ID } from '../lib/pipeline/graph'
import type { LogLevel } from '../lib/pipeline/types'
import { LAMP_COLOR, STATUS_LABEL, alpha, formatClock, laneColor } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconClose } from './icons'

const LEVEL_COLOR: Record<LogLevel, string> = {
  info: '#7C7F86',
  warn: '#C08A3E',
  error: '#A3543D',
  success: '#5FA88C',
}

type Props = {
  nodeId: string | null
  state?: NodeState
  logs: LogLine[]
  onClose: () => void
}

function Row({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[5px]">
      <span className="font-mono text-[12.5px] text-ink-faint">{k}</span>
      <span className="truncate font-mono text-[12.5px] text-ink-dim">{v}</span>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-4 py-3.5" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="mb-1.5 font-mono text-[11.5px] font-medium tracking-[0.02em] text-ink-faint/70">
        {title}
      </div>
      {children}
    </div>
  )
}

export function NodeDrawer({ nodeId, state, logs, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const spec = nodeId ? NODE_BY_ID.get(nodeId) : null

  // Escape closes; focus moves into the drawer so Tab stays in context.
  useEffect(() => {
    if (!nodeId) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    panelRef.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [nodeId, onClose])

  if (!nodeId || !spec) return null

  const status = state?.status ?? 'idle'
  const accent = laneColor(spec.lane, status)

  return (
    <aside
      ref={panelRef}
      tabIndex={-1}
      role="dialog"
      aria-label={`${spec.label} details`}
      className="absolute right-0 top-0 z-30 flex h-full w-[320px] flex-col outline-none"
      style={{
        background: '#111214',
        borderLeft: '1px solid rgba(255,255,255,0.06)',
        boxShadow: '-14px 0 30px rgba(0,0,0,0.55)',
      }}
    >
      <div className="flex items-start gap-3 px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              className="h-[6px] w-[6px] shrink-0 rounded-full"
              style={{
                background: LAMP_COLOR[status],
                boxShadow: status === 'idle' ? 'none' : `0 0 4px ${alpha(LAMP_COLOR[status], 0.9)}`,
              }}
            />
            <h2 className="truncate text-[14.5px] font-medium text-ink">{spec.label}</h2>
          </div>
          <div className="mt-1 font-mono text-[12px] text-ink-faint">
            {spec.id} / {STATUS_LABEL[status]}
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Close details"
          className="grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-[7px] text-ink-faint transition-colors duration-150 hover:text-ink"
          style={{ background: '#17181B', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <IconClose className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* lane tag */}
      <div className="px-4 pb-3.5">
        <span
          className="inline-block rounded-[5px] px-2 py-[3px] font-mono text-[11.5px] tracking-[0.02em]"
          style={{
            color: accent,
            background: alpha(accent, 0.09),
            border: `1px solid ${alpha(accent, 0.22)}`,
          }}
        >
          {spec.lane} lane
        </span>
      </div>

      <div className="console-scroll flex-1 overflow-y-auto">
        <Section title="configuration">
          {Object.entries(spec.config).map(([k, v]) => (
            <Row key={k} k={k} v={v} />
          ))}
        </Section>

        <Section title="metrics">
          <div className="mb-2">
            <DemoChip />
          </div>
          {state?.metrics ? (
            Object.entries(state.metrics).map(([k, v]) => <Row key={k} k={k} v={v} />)
          ) : (
            <div className="py-1 font-mono text-[12.5px] text-ink-faint/60">
              no metrics until stage completes
            </div>
          )}
        </Section>

        <Section title={`log / ${logs.length}`}>
          {logs.length === 0 ? (
            <div className="py-1 font-mono text-[12.5px] text-ink-faint/60">no entries</div>
          ) : (
            logs.map((l) => (
              <div key={l.id} className="flex gap-2.5 py-[2px] font-mono text-[12px] leading-[1.55]">
                <span className="shrink-0 tabular-nums text-ink-faint/55">
                  {formatClock(l.timestamp)}
                </span>
                <span className="min-w-0 flex-1" style={{ color: LEVEL_COLOR[l.level] }}>
                  {l.message}
                </span>
              </div>
            ))
          )}
        </Section>
      </div>
    </aside>
  )
}
