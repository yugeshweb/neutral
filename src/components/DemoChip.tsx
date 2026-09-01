/**
 * Marks every number in this UI as fabricated. Required wherever a metric is
 * displayed so nothing can be mistaken for a measured result.
 */
export function DemoChip({ label = 'demo data' }: { label?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[11px] tracking-[0.02em]"
      style={{
        color: '#8A8F98',
        background: 'rgba(138,143,152,0.07)',
        border: '1px solid rgba(138,143,152,0.2)',
      }}
    >
      <span className="h-[4px] w-[4px] rounded-full" style={{ background: '#8A8F98' }} />
      {label}
    </span>
  )
}
