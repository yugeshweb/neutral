import { useEffect, useState } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'
import type { TourStop } from '../hooks/useTour'
import { IconArrowLeft, IconArrowRight, IconClose } from './icons'

type Rect = { top: number; left: number; width: number; height: number }

const PAD = 8

/**
 * A dark scrim with a spotlight cut around whatever element carries
 * data-tour="{stop.target}", plus a callout box explaining it.
 *
 * Re-measures on scroll/resize so the hole tracks the real element rather
 * than a stale position - the workspace and launch screen both scroll their
 * own content areas independently of the window.
 */
export function TourOverlay({
  stop,
  index,
  total,
  isFirst,
  isLast,
  onNext,
  onPrev,
  onClose,
}: {
  stop: TourStop
  index: number
  total: number
  isFirst: boolean
  isLast: boolean
  onNext: () => void
  onPrev: () => void
  onClose: () => void
}) {
  const [rect, setRect] = useState<Rect | null>(null)

  useEffect(() => {
    const measure = () => {
      const el = document.querySelector(`[data-tour="${stop.target}"]`)
      if (!el) {
        setRect(null)
        return
      }
      const r = el.getBoundingClientRect()
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height })
      // A step change can land on an element outside the current scroll
      // position (workspace steps scroll their own panel) - bring it into view.
      el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }

    measure()
    // scrollIntoView is async/animated, so re-measure a beat later too.
    const settle = window.setTimeout(measure, 260)
    window.addEventListener('resize', measure)
    window.addEventListener('scroll', measure, true)
    return () => {
      window.clearTimeout(settle)
      window.removeEventListener('resize', measure)
      window.removeEventListener('scroll', measure, true)
    }
  }, [stop.target])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight') onNext()
      if (e.key === 'ArrowLeft' && !isFirst) onPrev()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onNext, onPrev, isFirst])

  const hole = rect
    ? {
        top: rect.top - PAD,
        left: rect.left - PAD,
        width: rect.width + PAD * 2,
        height: rect.height + PAD * 2,
      }
    : null

  // Place the callout below the hole, flipping above when there is not
  // enough room beneath it.
  const placeAbove = hole ? hole.top + hole.height + 190 > window.innerHeight : false
  const calloutTop = hole ? (placeAbove ? hole.top - 12 : hole.top + hole.height + 12) : window.innerHeight / 2
  const calloutLeft = hole ? Math.min(Math.max(hole.left, 16), window.innerWidth - 336) : window.innerWidth / 2 - 160

  return (
    <div className="fixed inset-0 z-[200]" role="dialog" aria-label={`Tour: ${stop.title}`}>
      {/* scrim, punched with a spotlight rect via box-shadow rather than a
          clip-path - keeps the hole's edges soft and themeable */}
      <div
        className="absolute inset-0 transition-[top,left,width,height] duration-300 ease-out"
        style={
          hole
            ? {
                top: hole.top,
                left: hole.left,
                width: hole.width,
                height: hole.height,
                borderRadius: 12,
                boxShadow: `0 0 0 4000px rgba(6,6,8,0.78)`,
                border: `1px solid ${alpha(LANE_COLOR.quantum, 0.6)}`,
              }
            : { top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(6,6,8,0.78)' }
        }
      />
      {/* dismiss by clicking the scrim outside the hole */}
      <button
        type="button"
        aria-label="Skip tour"
        onClick={onClose}
        className="absolute inset-0 cursor-default"
        style={{ background: 'transparent' }}
      />

      <div
        className="absolute w-[320px] rounded-panel p-4 transition-[top,left] duration-300 ease-out"
        style={{
          top: calloutTop,
          left: calloutLeft,
          transform: placeAbove ? 'translateY(-100%)' : 'none',
          background: '#17181B',
          border: '1px solid rgba(255,255,255,0.1)',
          boxShadow:
            'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 24px 50px rgba(0,0,0,0.7)',
        }}
      >
        <div className="mb-2 flex items-start justify-between gap-3">
          <span className="font-mono text-[9px] tracking-[0.04em] text-ink-faint">
            {index + 1} / {total}
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Skip tour"
            className="grid h-5 w-5 shrink-0 cursor-pointer place-items-center rounded-[5px] text-ink-faint transition-colors duration-150 hover:text-ink"
          >
            <IconClose className="h-3 w-3" />
          </button>
        </div>

        <h3 className="text-[13px] font-medium text-ink">{stop.title}</h3>
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-dim">{stop.body}</p>

        <div className="mt-3.5 flex items-center gap-2">
          {!isFirst && (
            <button
              type="button"
              onClick={onPrev}
              className="flex cursor-pointer items-center gap-1 rounded-[7px] px-2 py-1.5 font-mono text-[10px] text-ink-faint transition-colors duration-150 hover:text-ink"
            >
              <IconArrowLeft className="h-3 w-3" />
              Back
            </button>
          )}
          <div className="flex-1" />
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-[7px] px-2 py-1.5 font-mono text-[10px] text-ink-faint transition-colors duration-150 hover:text-ink"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={onNext}
            className="flex cursor-pointer items-center gap-1.5 rounded-[7px] px-3 py-1.5 font-mono text-[10px] transition-colors duration-150"
            style={{ background: alpha(LANE_COLOR.quantum, 0.14), color: LANE_COLOR.quantum }}
          >
            {isLast ? 'Done' : 'Next'}
            {!isLast && <IconArrowRight className="h-3 w-3" />}
          </button>
        </div>
      </div>
    </div>
  )
}
