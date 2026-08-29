import { LANE_COLOR, alpha } from '../lib/theme'
import { IconCheck } from './icons'

export type StepId =
  | 'data'
  | 'preprocess'
  | 'features'
  | 'model'
  | 'train'
  | 'results'
  | 'explain'

export const STEPS: { id: StepId; label: string }[] = [
  { id: 'data', label: 'Data' },
  { id: 'preprocess', label: 'Preprocess' },
  { id: 'features', label: 'Features' },
  { id: 'model', label: 'Model' },
  { id: 'train', label: 'Train' },
  { id: 'results', label: 'Results' },
  { id: 'explain', label: 'Explain' },
]

type Props = {
  current: StepId
  /** steps that have been configured or completed */
  done: Set<StepId>
  /** steps that cannot be opened yet, with the reason shown on hover */
  blocked?: Partial<Record<StepId, string>>
  onGo: (id: StepId) => void
}

/**
 * The pipeline stepper. Always visible, so the user knows where they are and
 * what is left - and so the flow reads as one pipeline rather than a menu of
 * unrelated screens.
 */
export function Stepper({ current, done, blocked = {}, onGo }: Props) {
  const currentIdx = STEPS.findIndex((s) => s.id === current)

  return (
    <nav
      className="flex shrink-0 items-center gap-0 overflow-x-auto px-4 py-2"
      style={{
        background: '#0E0F11',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
      aria-label="Pipeline progress"
    >
      {STEPS.map((step, i) => {
        const isCurrent = step.id === current
        const isDone = done.has(step.id)
        const reason = blocked[step.id]
        const isBlocked = Boolean(reason)
        const passed = i < currentIdx

        const color = isCurrent
          ? LANE_COLOR.quantum
          : isDone
            ? '#5FA88C'
            : '#6A6C72'

        return (
          <div key={step.id} className="flex shrink-0 items-center">
            {i > 0 && (
              <span
                className="mx-1 h-px w-5"
                style={{
                  background: passed || isDone ? alpha('#5FA88C', 0.4) : 'rgba(255,255,255,0.08)',
                }}
              />
            )}

            <button
              type="button"
              onClick={() => !isBlocked && onGo(step.id)}
              disabled={isBlocked}
              title={reason ?? step.label}
              aria-current={isCurrent ? 'step' : undefined}
              className="flex cursor-pointer items-center gap-1.5 rounded-[7px] px-2 py-1 transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
              style={{
                background: isCurrent ? alpha(LANE_COLOR.quantum, 0.1) : 'transparent',
              }}
            >
              <span
                className="grid h-[15px] w-[15px] shrink-0 place-items-center rounded-full font-mono text-[8.5px]"
                style={{
                  background: isDone ? alpha('#5FA88C', 0.16) : 'rgba(255,255,255,0.05)',
                  color,
                  boxShadow: isCurrent ? `0 0 0 1px ${alpha(LANE_COLOR.quantum, 0.5)}` : 'none',
                }}
              >
                {isDone ? <IconCheck className="h-[8px] w-[8px]" /> : i + 1}
              </span>
              <span
                className="whitespace-nowrap text-[11px]"
                style={{ color: isCurrent ? '#E8E9EB' : isDone ? '#9A9CA1' : '#6A6C72' }}
              >
                {step.label}
              </span>
            </button>
          </div>
        )
      })}
    </nav>
  )
}
