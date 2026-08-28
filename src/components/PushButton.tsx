import type { ReactNode } from 'react'

const RAISED =
  'inset 0 1px 0 rgba(255,255,255,0.08), 0 1px 2px rgba(0,0,0,0.8), 0 6px 14px rgba(0,0,0,0.5)'
const PRESSED = 'inset 0 1px 2px rgba(0,0,0,0.9), 0 1px 1px rgba(0,0,0,0.6)'

type Props = {
  label: string
  icon?: ReactNode
  onClick: () => void
  disabled?: boolean
  /** primary carries a faint accent tint on the icon; secondary stays neutral */
  tone?: 'primary' | 'secondary'
  accent?: string
}

/**
 * Physical push button. The press is real travel: the cap moves down 1px while
 * its drop shadow collapses, so the depth reads as motion, not a colour change.
 */
export function PushButton({
  label,
  icon,
  onClick,
  disabled,
  tone = 'secondary',
  accent = '#8A8F98',
}: Props) {
  const restore = (el: HTMLButtonElement) => {
    el.style.boxShadow = RAISED
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={[
        'inline-flex select-none items-center gap-2 rounded-[9px] px-3.5 py-2',
        'text-[12px] font-medium tracking-tight',
        'transition-[transform,box-shadow,background-color,color] duration-100 ease-out',
        'active:translate-y-[1px]',
        disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer hover:text-ink',
      ].join(' ')}
      style={{
        background: tone === 'primary' ? '#1D1E22' : '#17181B',
        color: tone === 'primary' ? '#E8E9EB' : '#9A9CA1',
        border: '1px solid rgba(255,255,255,0.07)',
        boxShadow: RAISED,
      }}
      onPointerDown={(e) => {
        if (!disabled) e.currentTarget.style.boxShadow = PRESSED
      }}
      onPointerUp={(e) => restore(e.currentTarget)}
      onPointerLeave={(e) => restore(e.currentTarget)}
      onBlur={(e) => restore(e.currentTarget)}
    >
      {icon ? (
        <span
          className="grid h-4 w-4 shrink-0 place-items-center"
          style={{ color: tone === 'primary' ? accent : '#7C7F86' }}
        >
          {icon}
        </span>
      ) : null}
      {label}
    </button>
  )
}
