/**
 * Two stacked layers behind the graph: a barely-there dot grid, and a fine
 * grain overlay from feTurbulence. Both are non-interactive and unanimated.
 */
export function CanvasBackdrop() {
  return (
    <div className="pointer-events-none absolute inset-0 z-0" aria-hidden>
      {/* dot grid */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />

      {/* grain */}
      <svg className="absolute inset-0 h-full w-full opacity-[0.035]">
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#grain)" />
      </svg>
    </div>
  )
}
