import { useEffect, useRef, useState } from 'react'

/**
 * The "i" affordance.
 *
 * Explanatory copy is the main thing that makes a dense instrument screen feel
 * cluttered, but the explanation still has to exist - a clinician meeting
 * "occlusion vs training mean" for the first time needs it. So it lives here:
 * out of the layout until asked for, one line of it in the corner of whatever
 * it explains.
 */
export function InfoDot({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Dismiss on outside click and on Escape: a popover that traps the pointer
  // is worse than no popover.
  useEffect(() => {
    if (!open) return

    const onPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }

    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={ref} className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        data-pressed={open}
        className="key relative grid h-[20px] w-[20px] cursor-pointer place-items-center rounded-full font-mono text-[12px] leading-none text-ink-faint after:absolute after:-inset-[12px] after:content-[''] hover:text-ink"
      >
        i
      </button>

      {open && (
        <div
          role="tooltip"
          className="panel-raised absolute right-0 top-[24px] z-40 w-[260px] rounded-[8px] p-3 text-[13px] leading-relaxed text-ink-dim"
        >
          {children}
        </div>
      )}
    </div>
  )
}
