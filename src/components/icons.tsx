import type { ReactElement } from 'react'

/**
 * Stage icons. Single 24x24 viewBox, 1.5 stroke, currentColor throughout so
 * each icon inherits its lane accent from the node header.
 */
type IconProps = { className?: string }

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

export function IconDatabase({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <ellipse cx="12" cy="6" rx="7" ry="3" />
      <path d="M5 6v6c0 1.66 3.13 3 7 3s7-1.34 7-3V6" />
      <path d="M5 12v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6" />
    </svg>
  )
}

export function IconFilter({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 5h16l-6 7v6l-4 2v-8L4 5z" />
    </svg>
  )
}

export function IconColumns({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <rect x="4" y="4" width="5" height="16" rx="1" />
      <rect x="11" y="4" width="5" height="10" rx="1" />
      <rect x="18" y="4" width="2" height="6" rx="1" />
    </svg>
  )
}

export function IconTree({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <rect x="3" y="4" width="6" height="4" rx="1" />
      <rect x="15" y="10" width="6" height="4" rx="1" />
      <rect x="15" y="16" width="6" height="4" rx="1" />
      <path d="M9 6h3v12h3M12 12h3" />
    </svg>
  )
}

export function IconTarget({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3.5" />
      <path d="M12 4v2M12 18v2M4 12h2M18 12h2" />
    </svg>
  )
}

export function IconWave({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M3 12c1.5-5 3-5 4.5 0S10.5 17 12 12s3-5 4.5 0 3 5 4.5 0" />
    </svg>
  )
}

export function IconCircuit({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M3 8h4M11 8h10M3 16h10M17 16h4" />
      <rect x="7" y="5.5" width="4" height="5" rx="1" />
      <rect x="13" y="13.5" width="4" height="5" rx="1" />
    </svg>
  )
}

export function IconGauge({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 17a8 8 0 1 1 16 0" />
      <path d="M12 17l4.5-5" />
      <circle cx="12" cy="17" r="1.2" />
    </svg>
  )
}

export function IconScale({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 4v16M6 8h12" />
      <path d="M3 15l3-7 3 7a3 3 0 0 1-6 0zM15 15l3-7 3 7a3 3 0 0 1-6 0z" />
    </svg>
  )
}

export function IconLayers({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 3l8 4.5-8 4.5-8-4.5L12 3z" />
      <path d="M4 12l8 4.5 8-4.5M4 16.5L12 21l8-4.5" />
    </svg>
  )
}

export function IconReport({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M6 3h8l4 4v14H6z" />
      <path d="M14 3v4h4M9 12h6M9 16h6" />
    </svg>
  )
}

export function IconChevron({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}

export function IconClose({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

export function IconPlay({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <path d="M8 5.5v13l11-6.5z" />
    </svg>
  )
}

export function IconStop({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
    </svg>
  )
}

export function IconReset({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 12a8 8 0 1 0 2.5-5.8" />
      <path d="M4 4v4h4" />
    </svg>
  )
}

export function IconUpload({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 16V4M8 8l4-4 4 4" />
      <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </svg>
  )
}

export function IconDownload({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M12 4v12M8 12l4 4 4-4" />
      <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />
    </svg>
  )
}

export function IconFlask({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M10 3v6.5L4.8 18a1.5 1.5 0 0 0 1.3 2.2h11.8a1.5 1.5 0 0 0 1.3-2.2L14 9.5V3" />
      <path d="M8.5 3h7M7.2 14h9.6" />
    </svg>
  )
}

export function IconPulse({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M3 12h3.5l2-5.5 3.5 11 2.5-7 1.5 1.5H21" />
    </svg>
  )
}

export function IconBars({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 20h16" />
      <rect x="6" y="11" width="3.5" height="6" rx="1" />
      <rect x="14.5" y="6" width="3.5" height="11" rx="1" />
    </svg>
  )
}

export function IconArrowRight({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M4 12h15M13 6l6 6-6 6" />
    </svg>
  )
}

export function IconArrowLeft({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M20 12H5M11 6l-6 6 6 6" />
    </svg>
  )
}

export function IconLock({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </svg>
  )
}

export function IconCheck({ className }: IconProps) {
  return (
    <svg {...base} className={className}>
      <path d="M5 12.5l4.5 4.5L19 7" />
    </svg>
  )
}

export const STAGE_ICON: Record<string, (p: IconProps) => ReactElement> = {
  ingest: IconDatabase,
  clean: IconFilter,
  features: IconColumns,
  baseline: IconTree,
  'classical-infer': IconTarget,
  encode: IconWave,
  vqc: IconCircuit,
  measure: IconGauge,
  benchmark: IconScale,
  explain: IconLayers,
  results: IconReport,
}
