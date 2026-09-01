import type { AppMode } from './LaunchScreen'
import { IconArrowLeft } from './icons'
import { Wordmark } from './Wordmark'

const LABEL: Record<AppMode, string> = {
  train: 'Train',
  predict: 'Predict',
  compare: 'Compare',
  conditions: 'Conditions',
}

type Props = {
  mode: AppMode
  onHome: () => void
}

/**
 * Slim header for the predict and compare views. The train view keeps its own
 * TopBar - it carries run controls this bar has no business duplicating.
 */
export function ModeBar({ mode, onHome }: Props) {
  return (
    <header
      className="flex h-14 shrink-0 items-center gap-3 px-4"
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
        <Wordmark size={14} />
      </button>

      <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />

      <span className="text-[14.5px] text-ink-dim">{LABEL[mode]}</span>
    </header>
  )
}
