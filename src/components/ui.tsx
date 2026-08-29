import type { ReactNode } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'

/** Shared surfaces, so every step screen shares one recipe. */

export function Panel({
  children,
  className = '',
  label,
}: {
  children: ReactNode
  className?: string
  label?: string
}) {
  return (
    <section
      className={`rounded-panel p-3.5 ${className}`}
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
      aria-label={label}
    >
      {children}
    </section>
  )
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h3 className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
      {children}
    </h3>
  )
}

export function Well({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-[8px] ${className}`}
      style={{
        background: '#0D0E10',
        border: '1px solid rgba(255,255,255,0.05)',
        boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
      }}
    >
      {children}
    </div>
  )
}

/** Segmented control. Options are compact enough to sit in a rail. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled,
  ariaLabel,
}: {
  options: { value: T; label: string; title?: string }[]
  value: T
  onChange: (v: T) => void
  disabled?: boolean
  ariaLabel?: string
}) {
  return (
    <div
      className="flex overflow-hidden rounded-[7px]"
      style={{
        background: '#0D0E10',
        border: '1px solid rgba(255,255,255,0.05)',
        boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
      }}
      role="group"
      aria-label={ariaLabel}
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className="flex-1 cursor-pointer whitespace-nowrap px-2 py-1.5 font-mono text-[9.5px] transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
          style={{
            background: value === o.value ? 'rgba(255,255,255,0.06)' : 'transparent',
            color: value === o.value ? '#E8E9EB' : '#6A6C72',
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div>
      <div className="mb-1.5 font-mono text-[10px] text-ink-dim">{label}</div>
      {children}
      {hint && (
        <p className="mt-1 font-mono text-[8.5px] leading-relaxed text-ink-faint/80">{hint}</p>
      )}
    </div>
  )
}

export function Slider({
  id,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
  format,
}: {
  id: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
  disabled?: boolean
  format?: (v: number) => string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="relative">
        <div className="pointer-events-none absolute inset-x-0 top-1/2 h-[4px] -translate-y-1/2 overflow-hidden rounded-full panel-well">
          <div
            className="h-full rounded-full"
            style={{ width: `${pct}%`, background: alpha(LANE_COLOR.quantum, 0.6) }}
          />
        </div>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className="feature-slider relative w-full cursor-pointer bg-transparent disabled:cursor-not-allowed disabled:opacity-50"
        />
      </div>
      {format && (
        <div className="mt-1 text-right font-mono text-[9.5px] tabular-nums text-ink">
          {format(value)}
        </div>
      )}
    </div>
  )
}

export function Checkbox({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex w-full cursor-pointer items-center gap-2 rounded-[6px] px-2 py-1.5 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
      style={{ background: checked ? alpha(LANE_COLOR.quantum, 0.07) : 'transparent' }}
    >
      <span
        className="grid h-[13px] w-[13px] shrink-0 place-items-center rounded-[3px]"
        style={{
          background: checked ? LANE_COLOR.quantum : '#0D0E10',
          border: `1px solid ${checked ? LANE_COLOR.quantum : 'rgba(255,255,255,0.14)'}`,
        }}
      >
        {checked && (
          <svg viewBox="0 0 24 24" className="h-[8px] w-[8px]" fill="none" stroke="#0B0B0D" strokeWidth="4">
            <path d="M5 12.5l4.5 4.5L19 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </span>
      <span className="font-mono text-[10px]" style={{ color: checked ? '#E8E9EB' : '#9A9CA1' }}>
        {label}
      </span>
    </button>
  )
}

/** Marks a value as computed by this platform rather than scripted. */
export function LiveChip({ label = 'computed' }: { label?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[9px] tracking-[0.02em]"
      style={{
        color: '#5FA88C',
        background: alpha('#5FA88C', 0.08),
        border: `1px solid ${alpha('#5FA88C', 0.22)}`,
      }}
    >
      <span className="h-[4px] w-[4px] rounded-full" style={{ background: '#5FA88C' }} />
      {label}
    </span>
  )
}
