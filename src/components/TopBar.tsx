import { LAMP_COLOR, alpha, formatElapsed } from '../lib/theme'
import type { RunPhase } from '../hooks/usePipeline'
import { IconArrowLeft, IconChevron, IconPlay, IconReset, IconStop } from './icons'
import { PushButton } from './PushButton'
import { Wordmark } from './Wordmark'

const DATASETS = ['Wisconsin Breast Cancer', 'UCI Heart Disease']

const PHASE_LAMP: Record<RunPhase, string> = {
  idle: LAMP_COLOR.idle,
  running: LAMP_COLOR.running,
  stopped: LAMP_COLOR.queued,
  complete: LAMP_COLOR.done,
  failed: LAMP_COLOR.error,
}

const PHASE_TEXT: Record<RunPhase, string> = {
  idle: 'idle',
  running: 'running',
  stopped: 'stopped',
  complete: 'complete',
  failed: 'failed',
}

type Props = {
  phase: RunPhase
  elapsed: number
  dataset: string
  /** set when the user has uploaded a file; overrides the sample selector */
  uploadName: string | null
  onDataset: (d: string) => void
  onStart: () => void
  onStop: () => void
  onReset: () => void
  onHome: () => void
}

export function TopBar({
  phase,
  elapsed,
  dataset,
  uploadName,
  onDataset,
  onStart,
  onStop,
  onReset,
  onHome,
}: Props) {
  const lamp = PHASE_LAMP[phase]
  const running = phase === 'running'

  return (
    <header
      className="flex h-14 shrink-0 items-center gap-4 px-4"
      style={{
        background: '#111214',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
      }}
    >
      {/* wordmark doubles as the way back to the menu */}
      <button
        type="button"
        onClick={onHome}
        className="flex cursor-pointer items-center gap-2 rounded-[8px] px-2 py-1.5 text-ink-faint transition-colors duration-150 hover:text-ink"
        aria-label="Back to menu"
      >
        <IconArrowLeft className="h-3.5 w-3.5" />
        <Wordmark size={14} />
      </button>

      <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />

      <span className="text-[14.5px] text-ink-dim">Train</span>

      {/* dataset: the sample selector, or the uploaded file name */}
      {uploadName ? (
        <div
          className="flex items-center gap-2 rounded-[8px] py-1.5 pl-3 pr-3"
          style={{
            background: '#0D0E10',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
          }}
          title={uploadName}
        >
          <span className="font-mono text-[11px] tracking-[0.02em] text-ink-faint">
            upload
          </span>
          <span className="max-w-[190px] truncate font-mono text-[13px] text-ink-dim">
            {uploadName}
          </span>
        </div>
      ) : (
        <div className="relative">
          <label htmlFor="dataset" className="sr-only">
            Dataset
          </label>
          <select
            id="dataset"
            value={dataset}
            onChange={(e) => onDataset(e.target.value)}
            disabled={running}
            className="select bg-[right_10px_center] py-1.5 pl-3 pr-8 text-[14px] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {DATASETS.map((d) => (
              <option key={d} value={d} style={{ background: '#17181B' }}>
                {d}
              </option>
            ))}
          </select>
          <IconChevron className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
        </div>
      )}

      {/* status pill */}
      <div
        className="flex items-center gap-2 rounded-full py-1.5 pl-2.5 pr-3"
        style={{
          background: '#0D0E10',
          border: '1px solid rgba(255,255,255,0.05)',
          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
        }}
        role="status"
        aria-live="polite"
      >
        <span
          className="h-[6px] w-[6px] rounded-full transition-colors duration-200"
          style={{
            background: lamp,
            boxShadow: phase === 'idle' ? 'none' : `0 0 4px ${alpha(lamp, 0.9)}`,
          }}
        />
        <span className="font-mono text-[12px] tracking-tight text-ink-dim">
          {PHASE_TEXT[phase]}
        </span>
      </div>

      {/* elapsed */}
      <span className="font-mono text-[14px] tabular-nums text-ink-dim">
        {formatElapsed(elapsed)}
      </span>

      <div className="flex-1" />

      <div className="flex items-center gap-2">
        {running ? (
          <PushButton
            label="Stop"
            icon={<IconStop className="h-3.5 w-3.5" />}
            onClick={onStop}
            tone="primary"
            accent="#C08A3E"
          />
        ) : (
          <PushButton
            label="Run"
            icon={<IconPlay className="h-3.5 w-3.5" />}
            onClick={onStart}
            tone="primary"
            accent="#5FA88C"
            disabled={phase === 'complete' || phase === 'failed'}
          />
        )}
        <PushButton
          label="Reset"
          icon={<IconReset className="h-3.5 w-3.5" />}
          onClick={onReset}
          disabled={phase === 'idle'}
        />
      </div>
    </header>
  )
}
