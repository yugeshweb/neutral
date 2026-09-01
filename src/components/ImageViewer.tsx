import { useEffect, useRef, useState } from 'react'
import type { DatasetSummary } from '../lib/dataset'
import { SEVERITY_COLOR, deriveFindings, type Finding } from '../lib/findings'
import { DemoChip } from './DemoChip'
import { IconClose } from './icons'

type Props = {
  upload: DatasetSummary
  onClose: () => void
}

/**
 * Full-bleed preview of an uploaded image with region-of-interest markers.
 * Markers are positioned in fractional coordinates so they track the rendered
 * image at any size. All findings are mocked - see lib/findings.ts.
 */
export function ImageViewer({ upload, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [findings] = useState<Finding[]>(() => deriveFindings(upload.name))
  const [active, setActive] = useState<Finding | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Escape steps back one level: detail first, then the viewer.
      if (active) setActive(null)
      else onClose()
    }
    window.addEventListener('keydown', onKey)
    ref.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [active, onClose])

  return (
    <div className="absolute inset-0 z-50 grid place-items-center p-6">
      <button
        type="button"
        aria-label="Dismiss preview"
        onClick={onClose}
        className="absolute inset-0 cursor-default"
        style={{ background: 'rgba(6,6,8,0.8)' }}
      />

      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-label={`Preview of ${upload.name}`}
        className="relative flex max-h-full w-full max-w-[980px] flex-col rounded-panel outline-none"
        style={{
          background: '#17181B',
          border: '1px solid rgba(255,255,255,0.07)',
          boxShadow:
            'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 24px 50px rgba(0,0,0,0.7)',
        }}
      >
        {/* header */}
        <div
          className="flex shrink-0 items-center gap-3 px-4 py-3"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
        >
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[14.5px] font-medium text-ink">{upload.name}</h2>
            <p className="mt-0.5 font-mono text-[12px] text-ink-faint">
              {upload.imageSize
                ? `${upload.imageSize.w} x ${upload.imageSize.h} px`
                : 'image'}{' '}
              / {findings.length} regions flagged
            </p>
          </div>
          <DemoChip label="simulated regions" />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close preview"
            className="grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-[7px] text-ink-faint transition-colors duration-150 hover:text-ink"
            style={{ background: '#111214', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <IconClose className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1">
          {/* image + overlay */}
          <div className="relative min-h-0 flex-1 p-4">
            <div className="relative mx-auto inline-block max-h-full">
              <img
                src={upload.objectUrl ?? ''}
                alt={`Uploaded scan ${upload.name}`}
                className="block max-h-[58vh] w-auto rounded-[8px]"
                style={{ border: '1px solid rgba(255,255,255,0.08)' }}
              />

              {/* markers, positioned as a fraction of the rendered image box */}
              {findings.map((f) => {
                const color = SEVERITY_COLOR[f.severity]
                const isActive = active?.id === f.id
                return (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setActive(isActive ? null : f)}
                    aria-label={`${f.label}, ${f.severity} severity. Show details`}
                    aria-pressed={isActive}
                    className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                    style={{ left: `${f.x * 100}%`, top: `${f.y * 100}%` }}
                  >
                    {/* ring showing the flagged area */}
                    <span
                      className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                      style={{
                        width: `${f.r * 460}px`,
                        height: `${f.r * 460}px`,
                        border: `2px solid ${color}`,
                        outline: '1px solid rgba(0,0,0,0.55)',
                        outlineOffset: '-3px',
                        boxShadow: `0 0 0 1px rgba(0,0,0,0.5), 0 0 12px ${color}55`,
                        opacity: isActive ? 1 : 0.8,
                        background: isActive ? `${color}1F` : 'transparent',
                        transition: 'opacity 160ms ease-out, background 160ms ease-out',
                      }}
                    />
                    {/* the blinking dot itself */}
                    <span
                      className="marker-pulse relative block h-[11px] w-[11px] rounded-full"
                      style={{
                        background: color,
                        border: '1.5px solid rgba(10,10,12,0.85)',
                        boxShadow: `0 0 6px ${color}, 0 0 14px ${color}90`,
                      }}
                    />
                  </button>
                )
              })}
            </div>
          </div>

          {/* detail rail */}
          <aside
            className="console-scroll w-[280px] shrink-0 overflow-y-auto p-4"
            style={{ borderLeft: '1px solid rgba(255,255,255,0.05)' }}
            aria-label="Region details"
          >
            {!active ? (
              <>
                <p className="font-mono text-[12px] leading-relaxed text-ink-faint">
                  Select a marker on the image to inspect a flagged region.
                </p>
                <div className="mt-4 space-y-2">
                  {findings.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setActive(f)}
                      className="flex w-full cursor-pointer items-center gap-2 rounded-[7px] px-2.5 py-2 text-left transition-colors duration-150 hover:bg-white/[0.03]"
                      style={{ background: 'rgba(255,255,255,0.02)' }}
                    >
                      <span
                        className="h-[6px] w-[6px] shrink-0 rounded-full"
                        style={{ background: SEVERITY_COLOR[f.severity] }}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[13px] text-ink-dim">{f.label}</span>
                        <span className="font-mono text-[11px] text-ink-faint">{f.id}</span>
                      </span>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="flex items-start gap-2">
                  <span
                    className="mt-[5px] h-[7px] w-[7px] shrink-0 rounded-full"
                    style={{ background: SEVERITY_COLOR[active.severity] }}
                  />
                  <div className="min-w-0 flex-1">
                    <h3 className="text-[14.5px] font-medium leading-tight text-ink">
                      {active.label}
                    </h3>
                    <p className="mt-1 font-mono text-[11.5px] text-ink-faint">
                      {active.id} / {active.severity} severity
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActive(null)}
                    aria-label="Back to region list"
                    className="cursor-pointer font-mono text-[11.5px] text-ink-faint transition-colors duration-150 hover:text-ink"
                  >
                    back
                  </button>
                </div>

                <div className="mt-3">
                  <DemoChip />
                </div>

                <div
                  className="mt-3 rounded-[8px] p-2.5"
                  style={{
                    background: '#0D0E10',
                    border: '1px solid rgba(255,255,255,0.05)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                  }}
                >
                  <div className="flex items-baseline justify-between py-[3px]">
                    <span className="font-mono text-[12px] text-ink-faint">confidence</span>
                    <span className="font-mono text-[12px] tabular-nums text-ink-dim">
                      {active.confidence.toFixed(2)}
                    </span>
                  </div>
                  {Object.entries(active.metrics).map(([k, v]) => (
                    <div key={k} className="flex items-baseline justify-between py-[3px]">
                      <span className="font-mono text-[12px] text-ink-faint">{k}</span>
                      <span className="font-mono text-[12px] tabular-nums text-ink-dim">{v}</span>
                    </div>
                  ))}
                </div>

                {active.notes.length > 0 && (
                  <ul className="mt-3 space-y-1.5">
                    {active.notes.map((n) => (
                      <li
                        key={n}
                        className="flex gap-2 font-mono text-[11.5px] leading-relaxed text-ink-faint"
                      >
                        <span className="text-ink-faint/50">-</span>
                        <span className="flex-1">{n}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </aside>
        </div>

        <div className="shrink-0 px-4 py-2.5" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          <p className="font-mono text-[11.5px] leading-relaxed text-ink-faint/70">
            Regions are generated for demonstration only. No detector was run on this image
            and nothing here is a diagnostic finding.
          </p>
        </div>
      </div>
    </div>
  )
}
