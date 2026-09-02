import type { ReactElement } from 'react'
import { LANE_COLOR } from '../lib/theme'

/**
 * The opening screen.
 *
 * Three squares, one per stage, in the order the platform is meant to be
 * worked through: train a model, score data with it, then compare it against
 * the classical baseline.
 *
 * Each tile carries a drawn vector rather than a UI icon. An icon at 16px
 * labels a control; these are the only thing on the screen, so they are given
 * room and drawn as small diagrams of what the stage actually does - a circuit
 * being parameterised, a trace being read, two runs being measured against
 * each other. The glow is a static filter, not an animation: infinite motion
 * on a landing screen is a distraction and a motion-sensitivity problem.
 *
 * Each tile is a link to its route rather than a button that only swaps state,
 * so it can be middle-clicked or copied like any other link. The click is
 * intercepted for the in-app transition; modifier-key and non-primary clicks
 * fall through to the browser so "open in new tab" still works.
 */

const QUANTUM = LANE_COLOR.quantum
const CLASSICAL = LANE_COLOR.classical

/* ---------------------------------------------------------------------------
 * Artwork.
 *
 * All three are drawn on the same 120x120 grid with the same 1.6 stroke, so
 * they read as one set rather than three borrowed graphics. `currentColor`
 * throughout, so the tile's accent drives them and the glow filter applies to
 * the whole shape.
 * ------------------------------------------------------------------------- */

const VIEWBOX = '0 0 120 120'
const STROKE = 1.6

/** Train: a parameterised circuit. Three wires, entangling links, rotation gates. */
function ArtTrain() {
  const wires = [38, 60, 82]
  return (
    <svg viewBox={VIEWBOX} className="tile-vector h-[104px] w-[104px]" aria-hidden="true">
      <g stroke="currentColor" strokeWidth={STROKE} fill="none" strokeLinecap="round">
        {/* The wires. */}
        {wires.map((y) => (
          <line key={y} x1={18} y1={y} x2={102} y2={y} opacity={0.42} />
        ))}

        {/* Entangling links between adjacent wires. */}
        <line x1={44} y1={38} x2={44} y2={60} opacity={0.75} />
        <line x1={76} y1={60} x2={76} y2={82} opacity={0.75} />

        {/* Control dots on the links. */}
        <circle cx={44} cy={38} r={2.6} fill="currentColor" stroke="none" />
        <circle cx={76} cy={60} r={2.6} fill="currentColor" stroke="none" />

        {/* Rotation gates: the parameters that training moves. */}
        {[
          [30, 38],
          [62, 60],
          [94, 38],
          [30, 82],
          [62, 82],
        ].map(([x, y]) => (
          <rect key={`${x}-${y}`} x={x - 7} y={y - 7} width={14} height={14} rx={3} />
        ))}

        {/* Targets on the entangling links. */}
        <circle cx={44} cy={60} r={5} />
        <circle cx={76} cy={82} r={5} />
      </g>
    </svg>
  )
}

/** Predict: a trace being read, with the decision point marked. */
function ArtPredict() {
  return (
    <svg viewBox={VIEWBOX} className="tile-vector h-[104px] w-[104px]" aria-hidden="true">
      <g stroke="currentColor" strokeWidth={STROKE} fill="none" strokeLinecap="round" strokeLinejoin="round">
        {/* Baseline. */}
        <line x1={16} y1={60} x2={104} y2={60} opacity={0.28} strokeDasharray="3 5" />

        {/* The trace: quiet, then one clear event. */}
        <path d="M16 60 L30 60 L36 54 L42 66 L48 60 L60 60 L66 26 L72 92 L78 60 L90 60 L96 56 L104 60" />

        {/* The read-out ring on the peak. */}
        <circle cx={66} cy={26} r={9} opacity={0.5} />
        <circle cx={66} cy={26} r={3.2} fill="currentColor" stroke="none" />

        {/* Crosshair ticks framing the event. */}
        <line x1={66} y1={8} x2={66} y2={14} opacity={0.55} />
        <line x1={48} y1={26} x2={54} y2={26} opacity={0.55} />
        <line x1={78} y1={26} x2={84} y2={26} opacity={0.55} />
      </g>
    </svg>
  )
}

/** Benchmark: two runs measured against a shared scale. */
function ArtBenchmark() {
  // Paired bars: the back one is the classical baseline, the front the quantum
  // run. Heights differ per pair so the graphic reads as a real comparison.
  const pairs: [number, number][] = [
    [34, 52],
    [50, 68],
    [42, 46],
    [58, 78],
  ]
  return (
    <svg viewBox={VIEWBOX} className="tile-vector h-[104px] w-[104px]" aria-hidden="true">
      <g stroke="currentColor" strokeWidth={STROKE} fill="none" strokeLinecap="round">
        {/* Shared scale. */}
        {[36, 60, 84].map((y) => (
          <line key={y} x1={16} y1={y} x2={104} y2={y} opacity={0.16} strokeDasharray="2 6" />
        ))}
        <line x1={16} y1={100} x2={104} y2={100} opacity={0.45} />

        {pairs.map(([back, front], i) => {
          const x = 26 + i * 21
          return (
            <g key={x}>
              {/* Baseline bar: outlined, sitting behind. */}
              <rect x={x - 8} y={100 - back} width={9} height={back} rx={1.5} opacity={0.4} />
              {/* Comparison bar: filled, in front. */}
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
      </g>
    </svg>
  )
}

type Tile = {
  id: 'train' | 'predict' | 'benchmark'
  step: string
  title: string
  blurb: string
  accent: string
  Art: () => ReactElement
}

const TILES: Tile[] = [
  {
    id: 'train',
    step: '01',
    title: 'Train',
    blurb: 'Fit a quantum and a classical model on the same split.',
    accent: QUANTUM,
    Art: ArtTrain,
  },
  {
    id: 'predict',
    step: '02',
    title: 'Predict',
    blurb: 'Score new records with the model training saved.',
    accent: CLASSICAL,
    Art: ArtPredict,
  },
  {
    id: 'benchmark',
    step: '03',
    title: 'Benchmark',
    blurb: 'Read the gap between the two on the same metrics.',
    accent: QUANTUM,
    Art: ArtBenchmark,
  },
]

export function HomeScreen({
  onOpen,
}: {
  onOpen: (id: 'train' | 'predict' | 'benchmark') => void
}) {
  return (
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen">
        <div className="border-b border-white/5 pb-4 text-center">
          <h1 className="text-[19px] font-medium text-ink">
            Hybrid quantum-classical disease detection
          </h1>
          {/* `mx-auto` alongside the measure cap: without it the paragraph keeps
              its 62ch width but sits flush left under a centred heading. */}
          <p className="mx-auto mt-1 max-w-[62ch] text-[13px] leading-relaxed text-ink-dim">
            Three stages, worked through in order. Start anywhere, but Predict
            needs a model from Train, and Benchmark reads what both produced.
          </p>
        </div>

        {/* Capped and centred: three squares stretched across a 1240px screen
            would be 390px wide each and stop reading as squares. */}
        <div className="mx-auto grid w-full max-w-[840px] grid-cols-1 gap-5 sm:grid-cols-3">
          {TILES.map(({ id, step, title, blurb, accent, Art }) => (
            <a
              key={id}
              href={`/${id}`}
              onClick={(e) => {
                // Let the browser handle new-tab and non-primary clicks.
                if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) {
                  return
                }
                e.preventDefault()
                onOpen(id)
              }}
              className="tile group flex cursor-pointer flex-col p-3 no-underline"
              style={{ ['--tile-accent' as string]: accent }}
            >
              {/* The lit plate. `aspect-square` is what makes the tile read as
                  a square regardless of how the text below wraps. */}
              <div className="tile-art aspect-square w-full">
                <Art />

                {/* Step number, set into the corner of the plate. */}
                <span className="engraved absolute left-2.5 top-2 font-mono text-[11px]">
                  {step}
                </span>
              </div>

              <div className="px-1 pb-1 pt-3">
                <h2 className="text-[15.5px] font-medium text-ink">{title}</h2>
                <p className="mt-1 text-[12.5px] leading-relaxed text-ink-dim">
                  {blurb}
                </p>
              </div>
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}
