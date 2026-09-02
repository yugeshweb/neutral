/**
 * Tile vectors.
 *
 * The drawn artwork used on tiles across the app: the opening screen, the
 * condition pickers, and the prediction intake cards.
 *
 * All of them share one 120x120 grid, so a screen full of them reads as a
 * single set rather than as borrowed clip art. Every shape paints with a
 * `url(#flow-<id>)` gradient rather than a flat colour: three stops that
 * sweep across the shape over a few seconds via SMIL, so the artwork reads as
 * something live rather than a static icon. Each `Frame` gets its own
 * gradient (via `useId`) rather than one shared instance, so cards on screen
 * drift slightly out of phase with each other instead of ticking in lockstep
 * - closer to how light actually moves than a synchronised pulse would be.
 *
 * `useReducedMotion` gates the animation itself: `prefers-reduced-motion`
 * drops the `<animateTransform>` entirely and the gradient sits at its
 * resting paint, because SMIL time containers are not something the global
 * CSS reduced-motion rule can reach.
 *
 * These are decorative: every tile that uses one also carries a text label, so
 * each svg is `aria-hidden` and contributes nothing to the accessible name.
 */

import type { ReactElement, ReactNode } from 'react'
import { useId } from 'react'
import { useReducedMotion } from '../hooks/useReducedMotion'

const VB = '0 0 120 120'

/**
 * The animated paint source. One `<linearGradient>` per accent colour,
 * travelling along its own axis via SMIL so no per-frame JS or CSS animation
 * loop is needed. Declared with `gradientUnits="userSpaceOnUse"` and a wide
 * axis so the sweep reads as a pass of light through the shape rather than a
 * hard-edged wipe.
 */
function FlowGradient({ id, accent, still }: { id: string; accent: string; still: boolean }) {
  const stops = (
    <>
      <stop offset="0%" stopColor={accent} stopOpacity="0.55" />
      <stop offset="45%" stopColor={accent} stopOpacity="1" />
      <stop offset="100%" stopColor={accent} stopOpacity="0.55" />
    </>
  )
  return (
    <linearGradient
      id={id}
      gradientUnits="userSpaceOnUse"
      x1="-40"
      y1="0"
      x2="40"
      y2="120"
    >
      {stops}
      {!still && (
        <animateTransform
          attributeName="gradientTransform"
          type="translate"
          from="0 0"
          to="120 0"
          dur="3.4s"
          repeatCount="indefinite"
        />
      )}
    </linearGradient>
  )
}

function Frame({
  children,
  size,
  accent,
}: {
  children: (fill: string) => ReactNode
  size: number
  accent: string
}) {
  const reduced = useReducedMotion()
  const uid = useId().replace(/[:]/g, '')
  const gradId = `flow-${uid}`

  return (
    <svg
      viewBox={VB}
      className="tile-vector"
      style={{ height: size, width: size }}
      aria-hidden="true"
    >
      <defs>
        <FlowGradient id={gradId} accent={accent} still={reduced} />
      </defs>
      {children(`url(#${gradId})`)}
    </svg>
  )
}

/* --- stage vectors, used on the opening screen ---------------------------- */
/* Unchanged by request: the home screen keeps its current tiles as-is. These
   three stay available for it to import. */

const SW = 1.6

function StrokeFrame({ children, size }: { children: ReactNode; size: number }) {
  return (
    <svg
      viewBox={VB}
      className="tile-vector"
      style={{ height: size, width: size }}
      aria-hidden="true"
    >
      <g stroke="currentColor" strokeWidth={SW} fill="none" strokeLinecap="round" strokeLinejoin="round">
        {children}
      </g>
    </svg>
  )
}

export function VecCircuit({ size = 104 }: { size?: number }) {
  return (
    <StrokeFrame size={size}>
      {[38, 60, 82].map((y) => (
        <line key={y} x1={18} y1={y} x2={102} y2={y} opacity={0.42} />
      ))}
      <line x1={44} y1={38} x2={44} y2={60} opacity={0.75} />
      <line x1={76} y1={60} x2={76} y2={82} opacity={0.75} />
      <circle cx={44} cy={38} r={2.6} fill="currentColor" stroke="none" />
      <circle cx={76} cy={60} r={2.6} fill="currentColor" stroke="none" />
      {[
        [30, 38],
        [62, 60],
        [94, 38],
        [30, 82],
        [62, 82],
      ].map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x - 7} y={y - 7} width={14} height={14} rx={3} />
      ))}
      <circle cx={44} cy={60} r={5} />
      <circle cx={76} cy={82} r={5} />
    </StrokeFrame>
  )
}

export function VecTrace({ size = 104 }: { size?: number }) {
  return (
    <StrokeFrame size={size}>
      <line x1={16} y1={60} x2={104} y2={60} opacity={0.28} strokeDasharray="3 5" />
      <path d="M16 60 L30 60 L36 54 L42 66 L48 60 L60 60 L66 26 L72 92 L78 60 L90 60 L96 56 L104 60" />
      <circle cx={66} cy={26} r={9} opacity={0.5} />
      <circle cx={66} cy={26} r={3.2} fill="currentColor" stroke="none" />
      <line x1={66} y1={8} x2={66} y2={14} opacity={0.55} />
      <line x1={48} y1={26} x2={54} y2={26} opacity={0.55} />
      <line x1={78} y1={26} x2={84} y2={26} opacity={0.55} />
    </StrokeFrame>
  )
}

export function VecBars({ size = 104 }: { size?: number }) {
  const pairs: [number, number][] = [
    [34, 52],
    [50, 68],
    [42, 46],
    [58, 78],
  ]
  return (
    <StrokeFrame size={size}>
      {[36, 60, 84].map((y) => (
        <line key={y} x1={16} y1={y} x2={104} y2={y} opacity={0.16} strokeDasharray="2 6" />
      ))}
      <line x1={16} y1={100} x2={104} y2={100} opacity={0.45} />
      {pairs.map(([back, front], i) => {
        const x = 26 + i * 21
        return (
          <g key={x}>
            <rect x={x - 8} y={100 - back} width={9} height={back} rx={1.5} opacity={0.4} />
            <rect
              x={x}
              y={100 - front}
              width={9}
              height={front}
              rx={1.5}
              fill="currentColor"
              stroke="none"
              opacity={0.85}
            />
          </g>
        )
      })}
    </StrokeFrame>
  )
}

/* --- condition vectors -----------------------------------------------------
 * Solid, layered artwork rather than thin outlines: at the 38-56px these
 * render at, a 1.6px stroke loses its detail to antialiasing. Filled shapes
 * hold their read at any size. Depth comes from opacity tiers on whole
 * shapes; every filled region paints with the flowing gradient passed in by
 * `Frame`, so the colour genuinely moves through the drawing rather than
 * sitting flat.
 * ------------------------------------------------------------------------- */

/** Breast cancer: a lobed, spiculated mass under a scan reticle. */
export function VecLesion({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <circle cx="60" cy="60" r="46" fill="none" stroke={fill} strokeWidth="1" strokeDasharray="1 7" opacity="0.5" />
          {/* Spicules, radiating from behind the mass. */}
          {[0, 40, 80, 120, 160, 200, 240, 280, 320].map((deg) => (
            <rect
              key={deg}
              x="58.6"
              y="14"
              width="2.8"
              height="16"
              rx="1.4"
              fill={fill}
              opacity="0.42"
              transform={`rotate(${deg} 60 60)`}
            />
          ))}
          {/* The mass: an irregular, lobed outline rather than a clean circle. */}
          <path
            d="M60 32c9 0 15.5 3.6 20 9.6 4.3 5.8 5.6 13 3.6 19.8-2 6.9-7 12.7-13.4 16.4-6.8 3.9-15.4 4.7-22.6 1.6-7-3-12.7-9-14.8-16.6-2-7.3-.5-15.6 4.4-21.8C42 34.8 50.6 32 60 32Z"
            fill={fill}
          />
          <circle cx="60" cy="60" r="7.5" fill="#0b0c0e" opacity="0.55" />
        </>
      )}
    </Frame>
  )
}

/** Brain: a cranial mass with sulci and a focal lesion cut into it. */
export function VecBrain({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <path
            d="M60 18c14.9 0 27 11.4 27 27.4v.4c5.4 2 9 7.2 9 13.2 0 6.7-4.6 12.3-10.8 13.9-2.7 8.8-11.4 15.1-21.7 15.1-4 0-7.8-1-11-2.7-2 1.2-4.4 1.9-6.9 1.9-6.7 0-12.3-4.9-13.4-11.3C24.4 74.6 20 68 20 60.4c0-5.7 2.5-10.8 6.4-14.3C27.7 30.6 42.1 18 60 18Z"
            fill={fill}
            opacity="0.9"
          />
          {/* Sulci: the folded surface, so the mass reads as tissue. */}
          <g fill="none" stroke="#0b0c0e" strokeWidth="2.4" strokeLinecap="round" opacity="0.4">
            <path d="M42 36c6 5 6 13 0 18" />
            <path d="M56 30c6 6 6 15 0 21" />
            <path d="M74 34c6 6 6 15-1 22" />
            <path d="M40 62c6 4 12 4 18 0" />
          </g>
          {/* The focal lesion, cut in with a dark ring so it reads as a site
              within the tissue rather than a dot on top of it. */}
          <circle cx="78" cy="66" r="12" fill="#0b0c0e" opacity="0.55" />
          <circle cx="78" cy="66" r="7.5" fill={fill} />
        </>
      )}
    </Frame>
  )
}

/** Heart: a solid chamber with the rhythm cut through it in negative space. */
export function VecHeart({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <path
            d="M60 96C36.6 79.7 22 65.5 22 47.6 22 34.3 32 24 45 24c6.5 0 12.5 2.9 15 8 2.5-5.1 8.5-8 15-8 13 0 23 10.3 23 23.6 0 17.9-14.6 32.1-38 48.4Z"
            fill={fill}
          />
          {/* The rhythm, cut through as negative space so it reads at full
              contrast regardless of what the gradient is doing underneath. */}
          <path
            d="M22 51.5h14l4.4-8.6 5.6 20 6-24.4 6.4 30.6 4.6-17.6h27"
            fill="none"
            stroke="#0b0c0e"
            strokeWidth="3.4"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity="0.68"
          />
        </>
      )}
    </Frame>
  )
}

/** Alzheimer's: a cranial ring with the atrophied core and enlarged ventricles. */
export function VecNeuro({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          {/* Outer tissue: a ring, not a disc, so the receded boundary inside
              it is legible as volume loss rather than a random hole. */}
          <path
            d="M60 16c16.6 0 30 13.3 30 29.8 0 .5 0 1-.1 1.5 3.7 2.6 6.1 6.9 6.1 11.7 0 6.5-4.3 12-10.3 13.8C82.4 85.3 72.2 94 60 94S37.6 85.3 34.3 72.8C28.3 71 24 65.5 24 59c0-4.8 2.4-9.1 6.1-11.7-.1-.5-.1-1-.1-1.5C30 29.3 43.4 16 60 16Zm0 12.4c-9.7 0-17.6 7.7-17.6 17.2 0 10.3 7.3 20 17.6 20s17.6-9.7 17.6-20c0-9.5-7.9-17.2-17.6-17.2Z"
            fill={fill}
            fillRule="evenodd"
          />
          {/* Enlarged ventricles. */}
          <path d="M53.4 40.2c-4.4 4.6-4.4 15 0 19.6-7-2.9-9.8-16.8 0-19.6Z" fill="#0b0c0e" opacity="0.6" />
          <path d="M66.6 40.2c4.4 4.6 4.4 15 0 19.6 7-2.9 9.8-16.8 0-19.6Z" fill="#0b0c0e" opacity="0.6" />
        </>
      )}
    </Frame>
  )
}

/** Stroke: a branching vessel with a solid occlusion at the junction. */
export function VecVessel({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <path
            d="M18 98a3 3 0 0 1-2.2-5c6.4-7.1 12-12 17.6-15.8l6.2-4.1 3-2.6 7.7-16.4a3 3 0 0 1 5.5 2.4l-5.6 11.9 21.5-3.9a3 3 0 0 1 1.1 5.9l-24.6 4.5-3.2 2.7c-5.3 4.4-11 9.9-17.4 15.9l-6.9 5.4a3 3 0 0 1-2.7.1Z"
            fill={fill}
            opacity="0.85"
          />
          <path
            d="M52 33l6-4.6a3 3 0 1 1 3.7 4.7l-6 4.7L52 33Z"
            fill={fill}
            opacity="0.55"
          />
          {/* The occlusion, marked at full strength with a break in the vessel. */}
          <circle cx="52" cy="58" r="12" fill="#0b0c0e" opacity="0.55" />
          <circle cx="52" cy="58" r="7" fill={fill} />
        </>
      )}
    </Frame>
  )
}

/** Parkinson's: a steady signal breaking into a regular tremor. */
export function VecTremor({ size = 56, accent = 'currentColor' }: { size?: number; accent?: string }) {
  const pts: string[] = []
  for (let i = 0; i <= 60; i++) {
    const x = 20 + i * 1.35
    const ramp = Math.max(0, (i - 16) / 44)
    const y = 60 - Math.sin(i * 0.62) * 27 * ramp
    pts.push(`${x.toFixed(1)} ${y.toFixed(1)}`)
  }
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <line x1="16" y1="60" x2="104" y2="60" stroke={fill} strokeWidth="1" strokeDasharray="1 6" opacity="0.35" />
          <path
            d={`M${pts.join(' L')}`}
            fill="none"
            stroke={fill}
            strokeWidth="4.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="20" cy="60" r="3.4" fill={fill} opacity="0.7" />
        </>
      )}
    </Frame>
  )
}

/* --- intake vectors --------------------------------------------------------
 * Same treatment: solid geometry, painted with the flowing gradient.
 * ------------------------------------------------------------------------- */

/** A table: a solid sheet with the row/column grid cut into it. */
export function VecTable({ size = 44, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <rect x="20" y="26" width="80" height="68" rx="6" fill={fill} />
          <g stroke="#0b0c0e" strokeWidth="2.6" opacity="0.55">
            <line x1="20" y1="44" x2="100" y2="44" />
            <line x1="47" y1="26" x2="47" y2="94" />
            <line x1="73" y1="26" x2="73" y2="94" />
            <line x1="20" y1="62" x2="100" y2="62" />
            <line x1="20" y1="78" x2="100" y2="78" />
          </g>
        </>
      )}
    </Frame>
  )
}

/** An exchange record: a folded document with a transfer mark. */
export function VecExchange({ size = 44, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          {/* Two overlapping records, so the drawing reads as an exchange
              between systems rather than a single static file - and fills
              the frame the way the other intake vectors do, instead of one
              narrow portrait shape sitting in the middle of it. */}
          <path
            d="M20 24h20.6l13.4 13.4V90c0 2.8-2.2 5-5 5H20c-2.8 0-5-2.2-5-5V29c0-2.8 2.2-5 5-5Z"
            fill={fill}
            opacity="0.55"
          />
          <path
            d="M60 15h20.6L94 28.4V81c0 2.8-2.2 5-5 5H60c-2.8 0-5-2.2-5-5V20c0-2.8 2.2-5 5-5Z"
            fill={fill}
          />
          <path d="M80.6 15L94 28.4H85.6c-2.8 0-5-2.2-5-5V15Z" fill="#0b0c0e" opacity="0.4" />
          {/* Transfer arrow, cut in dark for contrast against the gradient. */}
          <path
            d="M30 62h34.6l-6.7-6.7a3.2 3.2 0 0 1 4.6-4.6l12.6 12.6a3.2 3.2 0 0 1 0 4.6L62.5 80.5a3.2 3.2 0 0 1-4.6-4.6l6.7-6.7H30a3.2 3.2 0 0 1 0-6.4Z"
            fill="#0b0c0e"
            opacity="0.6"
          />
        </>
      )}
    </Frame>
  )
}

/** A message feed: solid segments streaming in at increasing strength. */
export function VecFeed({ size = 44, accent = 'currentColor' }: { size?: number; accent?: string }) {
  const rows: [number, number, number][] = [
    [30, 34, 0.42],
    [50, 46, 0.62],
    [70, 28, 0.8],
    [90, 40, 1],
  ]
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          {rows.map(([y, w, o]) => (
            <g key={y} opacity={o}>
              <circle cx="18" cy={y} r="5" fill={fill} />
              <rect x="30" y={y - 4.4} width={w} height="8.8" rx="4.4" fill={fill} />
            </g>
          ))}
        </>
      )}
    </Frame>
  )
}

/** A scan: a framed viewport with corner brackets and a marked region. */
export function VecScan({ size = 44, accent = 'currentColor' }: { size?: number; accent?: string }) {
  return (
    <Frame size={size} accent={accent}>
      {(fill) => (
        <>
          <rect x="16" y="22" width="88" height="76" rx="6" fill={fill} opacity="0.28" />
          {/* Corner brackets, solid and full strength. */}
          {[
            'M16 40V28a6 6 0 0 1 6-6h12v8H24v10Z',
            'M104 40V28a6 6 0 0 0-6-6H86v8h10v10Z',
            'M16 80v12a6 6 0 0 0 6 6h12v-8H24V80Z',
            'M104 80v12a6 6 0 0 1-6 6H86v-8h10V80Z',
          ].map((d) => (
            <path key={d} d={d} fill={fill} />
          ))}
          <circle cx="60" cy="60" r="17" fill="none" stroke={fill} strokeWidth="2.6" opacity="0.7" />
          <circle cx="60" cy="60" r="7.5" fill={fill} />
        </>
      )}
    </Frame>
  )
}

/* --- lookup --------------------------------------------------------------- */

/**
 * Vector per condition id. Keyed by the registry ids rather than by name, so a
 * relabelled condition keeps its artwork.
 */
export const CONDITION_VECTOR: Record<
  string,
  (p: { size?: number; accent?: string }) => ReactElement
> = {
  'breast-cancer': VecLesion,
  'brain-seizure': VecBrain,
  'heart-disease': VecHeart,
  alzheimers: VecNeuro,
  // Only the four above are selectable in Train and Predict, but the Benchmark
  // tab lists the whole registry, so these two need drawings of their own
  // rather than falling back to an unrelated one.
  'stroke-risk': VecVessel,
  parkinsons: VecTremor,
}

/**
 * Vector per intake field id. `imaging` fields all resolve to the scan drawing
 * via the fallback, so a new imaging input does not need an entry here.
 */
export const INTAKE_VECTOR: Record<
  string,
  (p: { size?: number; accent?: string }) => ReactElement
> = {
  table: VecTable,
  fhir: VecExchange,
  hl7: VecFeed,
}

export function intakeVector(fieldId: string, imaging?: boolean) {
  return INTAKE_VECTOR[fieldId] ?? (imaging ? VecScan : VecTable)
}
