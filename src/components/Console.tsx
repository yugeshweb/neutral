import { useEffect, useRef } from 'react'
import type { LogLine } from '../hooks/usePipeline'
import type { LogLevel } from '../lib/pipeline/types'
import { formatClock } from '../lib/theme'
import { IconChevron } from './icons'

const LEVEL_COLOR: Record<LogLevel, string> = {
  info: '#7C7F86',
  warn: '#C08A3E',
  error: '#A3543D',
  success: '#5FA88C',
}

type Props = {
  logs: LogLine[]
  open: boolean
  onToggle: () => void
}

export function Console({ logs, open, onToggle }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  // Follow the run, but yield to the user if they scroll up to read history.
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !open || !pinned.current) return
    el.scrollTop = el.scrollHeight
  }, [logs, open])

  return (
    <section
      className="shrink-0"
      style={{
        background: '#111214',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
      }}
      aria-label="Run console"
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center gap-2.5 px-4 py-2 text-left transition-colors duration-150 hover:bg-white/[0.02]"
      >
        <IconChevron
          className={`h-3.5 w-3.5 text-ink-faint transition-transform duration-200 ${
            open ? '' : '-rotate-90'
          }`}
        />
        <span className="font-mono text-[12px] font-medium tracking-[0.02em] text-ink-faint">
          console
        </span>
        <span className="font-mono text-[12px] text-ink-faint/70">{logs.length} lines</span>
      </button>

      {open && (
        <div
          ref={scrollRef}
          className="console-scroll h-40 overflow-y-auto px-4 pb-3 pt-1"
          style={{
            maskImage: 'linear-gradient(to bottom, transparent 0, #000 14px)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent 0, #000 14px)',
          }}
          onScroll={(e) => {
            const el = e.currentTarget
            pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
          }}
        >
          {logs.length === 0 ? (
            <div className="pt-1 font-mono text-[13px] text-ink-faint/60">
              waiting for run to start
            </div>
          ) : (
            logs.map((l) => (
              <div key={l.id} className="flex gap-2.5 py-[1.5px] font-mono text-[13px] leading-[1.5]">
                <span className="shrink-0 tabular-nums text-ink-faint/55">
                  {formatClock(l.timestamp)}
                </span>
                <span className="w-[88px] shrink-0 truncate text-ink-faint/75">{l.nodeId}</span>
                <span className="min-w-0 flex-1" style={{ color: LEVEL_COLOR[l.level] }}>
                  {l.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  )
}
