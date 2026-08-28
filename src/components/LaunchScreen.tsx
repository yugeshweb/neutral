import type { ReactNode } from 'react'
import { LANE_COLOR } from '../lib/theme'
import { IconBars, IconFlask, IconPulse } from './icons'

export type AppMode = 'train' | 'predict' | 'compare'

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
]

type Props = {
  onSelect: (mode: AppMode) => void
}

/** Entry screen. Three doors and nothing else - the detail lives inside each. */
export function LaunchScreen({ onSelect }: Props) {
  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto flex min-h-full w-full max-w-[900px] flex-col justify-center px-8 py-14">
        <div className="mb-9 text-center">
          <h1 className="text-[26px] font-semibold tracking-[-0.03em] text-ink">Netural</h1>
          <p className="mt-2 text-[13px] text-ink-dim">
            Hybrid quantum-classical machine learning for early disease detection.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {CARDS.map((card) => (
            <button
              key={card.mode}
              type="button"
              onClick={() => onSelect(card.mode)}
              className="group relative flex cursor-pointer flex-col rounded-panel p-5 text-left transition-[transform,box-shadow] duration-150 ease-out hover:-translate-y-[2px] active:translate-y-0"
              style={{
                background: '#17181B',
                border: '1px solid rgba(255,255,255,0.06)',
                boxShadow:
                  'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
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
    </div>
  )
}
