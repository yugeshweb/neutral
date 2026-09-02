import type { ReactElement } from 'react'
import { LANE_COLOR } from '../lib/theme'
import { VecBars, VecCircuit, VecTrace } from './vectors'

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

type Tile = {
  id: 'train' | 'predict' | 'benchmark'
  step: string
  title: string
  blurb: string
  accent: string
  Art: (p: { size?: number }) => ReactElement
}

const TILES: Tile[] = [
  {
    id: 'train',
    step: '01',
    title: 'Train',
    blurb: 'Fit a quantum and a classical model on the same split.',
    accent: QUANTUM,
    Art: VecCircuit,
  },
  {
    id: 'predict',
    step: '02',
    title: 'Predict',
    blurb: 'Score new records with the model training saved.',
    accent: CLASSICAL,
    Art: VecTrace,
  },
  {
    id: 'benchmark',
    step: '03',
    title: 'Benchmark',
    blurb: 'Read the gap between the two on the same metrics.',
    accent: QUANTUM,
    Art: VecBars,
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
