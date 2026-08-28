import type { AppMode } from './LaunchScreen'
import { IconArrowLeft } from './icons'

const LABEL: Record<AppMode, string> = {
  train: 'Train',
  predict: 'Predict',
  compare: 'Compare',
}

type Props = {
  mode: AppMode
  onHome: () => void
  onMode: (m: AppMode) => void
}

/**
 * Slim header for the predict and compare views. The train view keeps its own
 * TopBar - it carries run controls this bar has no business duplicating.
 */
export function ModeBar({ mode, onHome, onMode }: Props) {
  return (
    <header
      className="flex h-14 shrink-0 items-center gap-4 px-4"
      style={{
        background: '#111214',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
      }}
    >
      <button
        type="button"
        onClick={onHome}
        className="flex cursor-pointer items-center gap-2 rounded-[8px] px-2 py-1.5 text-ink-faint transition-colors duration-150 hover:text-ink"
        aria-label="Back to menu"
      >
        <IconArrowLeft className="h-3.5 w-3.5" />
        <span className="text-[15px] font-semibold tracking-[-0.02em] text-ink">
          Netural
        </span>
      </button>

      <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />

      {/* mode switch, so the three views are reachable without going home */}
      <nav
        className="flex overflow-hidden rounded-[8px]"
        style={{
          background: '#0D0E10',
          border: '1px solid rgba(255,255,255,0.05)',
          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
        }}
        aria-label="Mode"
      >
        {(['train', 'predict', 'compare'] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => onMode(m)}
            aria-current={mode === m ? 'page' : undefined}
            className="cursor-pointer px-3 py-1.5 text-[11.5px] transition-colors duration-150"
            style={{
              background: mode === m ? 'rgba(255,255,255,0.06)' : 'transparent',
              color: mode === m ? '#E8E9EB' : '#6A6C72',
            }}
          >
            {LABEL[m]}
          </button>
        ))}
      </nav>
    </header>
  )
}
