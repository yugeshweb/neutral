import type { ReactNode } from 'react'
import { useTour, type TourStop } from '../hooks/useTour'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconBars, IconDatabase, IconFlask, IconPulse } from './icons'
import { TourOverlay } from './TourOverlay'
import { Wordmark } from './Wordmark'

export type AppMode = 'train' | 'predict' | 'compare' | 'conditions'

/**
 * Hover is a border glow rather than a lift: the accent bleeds into the edge
 * and a soft halo sits outside it, so the card lights up in place instead of
 * moving. Both are driven from the resting shadow so the bevel is preserved.
 */
const BASE_BORDER = 'rgba(255,255,255,0.06)'
const REST_SHADOW =
  'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)'

type Card = {
  mode: AppMode
  index: string
  title: string
  body: string
  accent: string
  icon: ReactNode
}

const CARDS: Card[] = [
  {
    mode: 'train',
    index: '01',
    title: 'Train',
    body: 'Ingest a dataset and run the hybrid pipeline.',
    accent: LANE_COLOR.classical,
    icon: <IconFlask className="h-[22px] w-[22px]" />,
  },
  {
    mode: 'predict',
    index: '02',
    title: 'Predict',
    body: 'Score a single case and see what drove it.',
    accent: LANE_COLOR.quantum,
    icon: <IconPulse className="h-[22px] w-[22px]" />,
  },
  {
    mode: 'compare',
    index: '03',
    title: 'Compare',
    body: 'Benchmark the quantum model against the classical baseline.',
    accent: LANE_COLOR.shared,
    icon: <IconBars className="h-[22px] w-[22px]" />,
  },
  {
    mode: 'conditions',
    index: '04',
    title: 'Neurological conditions',
    body: 'Browse the clinical condition registry and what each one needs.',
    accent: LANE_COLOR.shared,
    icon: <IconDatabase className="h-[22px] w-[22px]" />,
  },
]

type Props = {
  onSelect: (mode: AppMode) => void
}

const TOUR_STOPS: TourStop[] = [
  {
    target: 'card-train',
    title: '01 · Train',
    body: 'Start here. Pick a dataset, walk through preprocessing and feature selection, then train the quantum model against classical baselines - a real run, computed in your browser.',
  },
  {
    target: 'card-predict',
    title: '02 · Predict',
    body: 'Once a model is trained, score a single patient and see exactly which features drove the prediction, mapped back to clinical terms.',
  },
  {
    target: 'card-compare',
    title: '03 · Compare',
    body: 'The benchmark: accuracy, ROC curves and cross-validation spread for the quantum model against every classical baseline, on the same split.',
  },
  {
    target: 'card-conditions',
    title: '04 · Neurological conditions',
    body: 'A registry of six real neurological conditions this platform is built to generalise to, served by the Python backend.',
  },
]

/** Entry screen. The detail lives inside each mode. */
export function LaunchScreen({ onSelect }: Props) {
  const tour = useTour('launch', TOUR_STOPS)

  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto flex min-h-full w-full max-w-[1120px] flex-col justify-center px-8 py-14">
        <div className="mb-9 text-center">
          <h1 aria-label="Netural">
            <Wordmark size={38} />
          </h1>
          <p className="mt-4 text-[13px] text-ink-dim">
            Hybrid quantum-classical machine learning platform for early disease detection.
          </p>
          <button
            type="button"
            onClick={tour.start}
            className="mt-3 cursor-pointer font-mono text-[9.5px] text-ink-faint underline decoration-dotted underline-offset-2 transition-colors duration-150 hover:text-ink"
          >
            take the tour
          </button>
        </div>

        <div className="grid grid-cols-4 gap-4">
          {CARDS.map((card) => (
            <button
              key={card.mode}
              type="button"
              data-tour={`card-${card.mode}`}
              onClick={() => onSelect(card.mode)}
              className="group relative flex cursor-pointer flex-col rounded-panel p-5 text-left transition-[border-color,box-shadow] duration-200 ease-out"
              style={{
                background: '#17181B',
                border: `1px solid ${BASE_BORDER}`,
                boxShadow: REST_SHADOW,
              }}
              onPointerEnter={(e) => {
                e.currentTarget.style.borderColor = alpha(card.accent, 0.55)
                e.currentTarget.style.boxShadow = `${REST_SHADOW}, 0 0 0 1px ${alpha(card.accent, 0.18)}, 0 0 18px ${alpha(card.accent, 0.22)}`
              }}
              onPointerLeave={(e) => {
                e.currentTarget.style.borderColor = BASE_BORDER
                e.currentTarget.style.boxShadow = REST_SHADOW
              }}
            >
              {/* accent hairline along the top edge */}
              <span
                className="absolute inset-x-5 top-0 h-px opacity-60 transition-opacity duration-150 group-hover:opacity-100"
                style={{
                  background: `linear-gradient(90deg, transparent, ${card.accent}, transparent)`,
                }}
              />

              <div className="mb-4 flex items-start justify-between">
                <span
                  className="grid h-11 w-11 place-items-center rounded-[10px] panel-well"
                  style={{ color: card.accent }}
                >
                  {card.icon}
                </span>
                <span className="font-mono text-[10px] tracking-[0.06em] text-ink-faint">
                  {card.index}
                </span>
              </div>

              <h2 className="text-[16px] font-medium tracking-[-0.01em] text-ink">
                {card.title}
              </h2>
              <p className="mt-2 text-[11.5px] leading-relaxed text-ink-dim">{card.body}</p>
            </button>
          ))}
        </div>
      </div>

      {tour.active && tour.stop && (
        <TourOverlay
          stop={tour.stop}
          index={tour.index}
          total={tour.total}
          isFirst={tour.isFirst}
          isLast={tour.isLast}
          onNext={tour.next}
          onPrev={tour.prev}
          onClose={tour.close}
        />
      )}
    </div>
  )
}
