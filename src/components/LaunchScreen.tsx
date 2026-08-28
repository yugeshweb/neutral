import type { ReactNode } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconArrowRight, IconBars, IconCheck, IconFlask, IconLock, IconPulse } from './icons'

export type AppMode = 'train' | 'predict' | 'compare'

type Card = {
  mode: AppMode
  index: string
  title: string
  subtitle: string
  body: string
  bullets: string[]
  accent: string
  icon: ReactNode
}

const CARDS: Card[] = [
  {
    mode: 'train',
    index: '01',
    title: 'Train',
    subtitle: 'hybrid pipeline',
    body: 'Ingest a biomedical dataset and run it through the full hybrid graph - cleaning, feature selection, a classical baseline lane and a variational quantum lane in parallel.',
    bullets: ['11 stages', 'live stage log', 'classical + quantum lanes'],
    accent: LANE_COLOR.classical,
    icon: <IconFlask className="h-[22px] w-[22px]" />,
  },
  {
    mode: 'predict',
    index: '02',
    title: 'Predict',
    subtitle: 'single-case inference',
    body: 'Score one patient against the trained heads. Adjust the eight retained features and watch the malignant probability and its per-feature attribution update.',
    bullets: ['8 selected features', 'both heads side by side', 'SHAP-style attribution'],
    accent: LANE_COLOR.quantum,
    icon: <IconPulse className="h-[22px] w-[22px]" />,
  },
  {
    mode: 'compare',
    index: '03',
    title: 'Compare',
    subtitle: 'benchmark study',
    body: 'Put the quantum-enhanced classifier against the classical baseline on the held-out split - accuracy, ROC-AUC, sensitivity, specificity, and cost.',
    bullets: ['4 headline metrics', 'McNemar significance', 'cost & efficiency'],
    accent: LANE_COLOR.shared,
    icon: <IconBars className="h-[22px] w-[22px]" />,
  },
]

type Props = {
  onSelect: (mode: AppMode) => void
  /** a training run has completed at least once this session */
  trained: boolean
}

/**
 * Entry screen. Predict and Compare stay reachable before training, but are
 * marked as running against the bundled reference weights rather than a model
 * the user produced - so the distinction stays honest either way.
 */
export function LaunchScreen({ onSelect, trained }: Props) {
  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto flex min-h-full w-full max-w-[1080px] flex-col justify-center px-8 py-14">
        {/* masthead */}
        <div className="mb-10">
          <div className="flex items-center gap-3">
            <span className="text-[26px] font-semibold tracking-[-0.03em] text-ink">
              Netural
            </span>
            <DemoChip />
          </div>
          <p className="mt-3 max-w-[560px] text-[13px] leading-relaxed text-ink-dim">
            A hybrid quantum-classical machine learning platform for early disease
            detection. Choose where to begin.
          </p>

          <div className="mt-4 flex items-center gap-2">
            <span
              className="grid h-[15px] w-[15px] place-items-center rounded-full"
              style={{
                background: trained ? alpha('#5FA88C', 0.16) : 'rgba(255,255,255,0.04)',
                color: trained ? '#5FA88C' : '#6A6C72',
              }}
            >
              {trained ? <IconCheck className="h-[9px] w-[9px]" /> : null}
            </span>
            <span className="font-mono text-[10px] text-ink-faint">
              {trained
                ? 'model trained this session - predictions use the run above'
                : 'no model trained yet - predict and compare use bundled reference weights'}
            </span>
          </div>
        </div>

        {/* the three modes */}
        <div className="grid grid-cols-3 gap-4">
          {CARDS.map((card) => {
            const reference = card.mode !== 'train' && !trained
            return (
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
                <p
                  className="mt-0.5 font-mono text-[9.5px] tracking-[0.02em]"
                  style={{ color: card.accent }}
                >
                  {card.subtitle}
                </p>

                <p className="mt-3 flex-1 text-[11.5px] leading-relaxed text-ink-dim">
                  {card.body}
                </p>

                <ul className="mt-4 space-y-1.5">
                  {card.bullets.map((b) => (
                    <li
                      key={b}
                      className="flex items-center gap-2 font-mono text-[9.5px] text-ink-faint"
                    >
                      <span
                        className="h-[3px] w-[3px] shrink-0 rounded-full"
                        style={{ background: card.accent }}
                      />
                      {b}
                    </li>
                  ))}
                </ul>

                <div
                  className="mt-4 flex items-center justify-between pt-3"
                  style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
                >
                  <span className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-faint">
                    {reference && <IconLock className="h-3 w-3" />}
                    {reference ? 'reference weights' : 'ready'}
                  </span>
                  <span
                    className="grid h-6 w-6 place-items-center rounded-[6px] text-ink-faint transition-[transform,color] duration-150 group-hover:translate-x-[2px] group-hover:text-ink"
                    style={{ background: 'rgba(255,255,255,0.04)' }}
                  >
                    <IconArrowRight className="h-3.5 w-3.5" />
                  </span>
                </div>
              </button>
            )
          })}
        </div>

        <p className="mt-8 max-w-[620px] font-mono text-[9.5px] leading-relaxed text-ink-faint/70">
          Every metric in this platform is a placeholder produced by a mock runner. No
          model is trained and no dataset is evaluated. Do not cite these figures as
          results or use them for clinical decisions.
        </p>
      </div>
    </div>
  )
}
