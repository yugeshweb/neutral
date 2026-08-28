import { useMemo, useState } from 'react'
import {
  DEFAULT_VALUES,
  FEATURES,
  PRESETS,
  predict,
  type FeatureValues,
  type Prediction,
} from '../lib/predict'
import { LANE_COLOR, alpha } from '../lib/theme'
import { DemoChip } from './DemoChip'
import { IconReset } from './icons'

const MALIGNANT = '#A3543D'
const BENIGN = '#5FA88C'

/** Dial-style readout for one model head. */
function Readout({
  title,
  subtitle,
  accent,
  result,
}: {
  title: string
  subtitle: string
  accent: string
  result: Prediction
}) {
  const tone = result.label === 'malignant' ? MALIGNANT : BENIGN
  const pct = result.probability * 100

  return (
    <div
      className="rounded-panel p-4"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="h-[7px] w-[7px] rounded-full" style={{ background: accent }} />
        <div className="flex-1">
          <div className="text-[12.5px] font-medium leading-tight text-ink">{title}</div>
          <div className="mt-0.5 font-mono text-[9.5px] text-ink-faint">{subtitle}</div>
        </div>
      </div>

      {/* verdict */}
      <div className="flex items-baseline gap-2">
        <span
          className="font-mono text-[30px] font-medium leading-none tabular-nums"
          style={{ color: tone }}
        >
          {pct.toFixed(1)}
        </span>
        <span className="font-mono text-[12px] text-ink-faint">% malignant</span>
      </div>

      <div
        className="mt-2 inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[9.5px] tracking-[0.02em]"
        style={{
          color: tone,
          background: alpha(tone, 0.1),
          border: `1px solid ${alpha(tone, 0.28)}`,
        }}
      >
        <span className="h-[4px] w-[4px] rounded-full" style={{ background: tone }} />
        predicted {result.label}
      </div>

      {/* probability channel, with the 0.50 threshold marked */}
      <div className="relative mt-3 h-[6px] w-full overflow-hidden rounded-full panel-well">
        <div
          className="h-full rounded-full transition-[width] duration-300 ease-out"
          style={{ width: `${pct}%`, background: alpha(tone, 0.8) }}
        />
        <span
          className="absolute inset-y-0 left-1/2 w-px"
          style={{ background: 'rgba(255,255,255,0.28)' }}
        />
      </div>

      <div className="mt-1.5 flex justify-between font-mono text-[9px] text-ink-faint">
        <span>benign</span>
        <span>threshold 0.50</span>
        <span>malignant</span>
      </div>

      <div
        className="mt-3 flex justify-between pt-2.5 font-mono text-[10px]"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span className="text-ink-faint">decision margin</span>
        <span className="tabular-nums text-ink-dim">{result.confidence.toFixed(3)}</span>
      </div>
    </div>
  )
}

/** Signed contribution bars, diverging from a centre axis. */
function Attributions({ result }: { result: Prediction }) {
  const max = Math.max(...result.attributions.map((a) => Math.abs(a.contribution)), 0.001)

  return (
    <div
      className="rounded-panel p-4"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
    >
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="text-[12.5px] font-medium text-ink">Feature attribution</h3>
        <span className="font-mono text-[9.5px] text-ink-faint">quantum head</span>
      </div>
      <p className="mb-3.5 font-mono text-[9.5px] text-ink-faint">
        signed contribution to the malignant logit
      </p>

      <div className="space-y-2.5">
        {result.attributions.map((a) => {
          const pushesMalignant = a.contribution > 0
          const tone = pushesMalignant ? MALIGNANT : BENIGN
          const width = (Math.abs(a.contribution) / max) * 50

          return (
            <div key={a.id}>
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="truncate font-mono text-[10px] text-ink-dim">
                  {a.label}
                </span>
                <span
                  className="shrink-0 font-mono text-[10px] tabular-nums"
                  style={{ color: tone }}
                >
                  {pushesMalignant ? '+' : ''}
                  {a.contribution.toFixed(3)}
                </span>
              </div>

              {/* diverging bar: centre axis is zero contribution */}
              <div className="relative h-[5px] w-full rounded-full panel-well">
                <span
                  className="absolute inset-y-0 left-1/2 w-px"
                  style={{ background: 'rgba(255,255,255,0.18)' }}
                />
                <div
                  className="absolute top-0 h-full rounded-full transition-all duration-300 ease-out"
                  style={{
                    background: alpha(tone, 0.75),
                    width: `${width}%`,
                    left: pushesMalignant ? '50%' : `${50 - width}%`,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div
        className="mt-3.5 flex gap-4 pt-2.5 font-mono text-[9px] text-ink-faint"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: BENIGN }} />
          toward benign
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: MALIGNANT }} />
          toward malignant
        </span>
      </div>
    </div>
  )
}

/** One feature row: slider in a recessed channel, with a numeric entry. */
function FeatureRow({
  id,
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
}: {
  id: string
  label: string
  unit: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="font-mono text-[10px] text-ink-dim">
          {label}
        </label>
        <span className="flex items-baseline gap-1">
          <input
            type="number"
            aria-label={`${label} value`}
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => {
              const n = Number(e.target.value)
              if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)))
            }}
            className="w-[62px] rounded-[5px] px-1.5 py-[3px] text-right font-mono text-[10.5px] tabular-nums text-ink outline-none"
            style={{
              background: '#0D0E10',
              border: '1px solid rgba(255,255,255,0.05)',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          />
          {unit && (
            <span className="w-[22px] font-mono text-[9px] text-ink-faint">{unit}</span>
          )}
        </span>
      </div>

      <div className="relative">
        {/* the channel the thumb rides in */}
        <div className="pointer-events-none absolute inset-x-0 top-1/2 h-[4px] -translate-y-1/2 overflow-hidden rounded-full panel-well">
          <div
            className="h-full rounded-full"
            style={{ width: `${pct}%`, background: alpha(LANE_COLOR.quantum, 0.6) }}
          />
        </div>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="feature-slider relative w-full cursor-pointer bg-transparent"
        />
      </div>
    </div>
  )
}

type Props = {
  /** true once a training run has completed this session */
  trained: boolean
}

export function PredictView({ trained }: Props) {
  const [values, setValues] = useState<FeatureValues>(DEFAULT_VALUES)

  const quantum = useMemo(() => predict(values, 'quantum'), [values])
  const classical = useMemo(() => predict(values, 'classical'), [values])

  const set = (id: string, v: number) => setValues((prev) => ({ ...prev, [id]: v }))
  const agree = quantum.label === classical.label

  // Read the top attributions back as a sentence, so the explanation does not
  // depend on the reader interpreting the bar chart correctly.
  const rationale = useMemo(() => {
    const [first, second] = quantum.attributions
    const driversToward = quantum.attributions
      .filter((a) => (quantum.label === 'malignant' ? a.contribution > 0 : a.contribution < 0))
      .slice(0, 2)
      .map((a) => a.label.toLowerCase())

    const strength =
      quantum.confidence > 0.8 ? 'strongly' : quantum.confidence > 0.4 ? 'moderately' : 'weakly'

    const lead =
      driversToward.length > 0
        ? `driven mainly by ${driversToward.join(' and ')}`
        : `with no single feature dominating`

    const counter =
      second && Math.sign(second.contribution) !== Math.sign(first.contribution)
        ? ` ${second.label} pulls in the opposite direction, which is why the margin is not wider.`
        : ''

    return `The model ${strength} favours ${quantum.label}, ${lead}. ${first.label} carries the largest single contribution at ${first.contribution >= 0 ? '+' : ''}${first.contribution.toFixed(3)}.${counter}`
  }, [quantum])

  return (
    <div className="console-scroll h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[1120px] px-6 py-6">
        <div className="mb-5 flex items-start gap-3">
          <div className="flex-1">
            <h1 className="text-[16px] font-medium tracking-[-0.01em] text-ink">
              Prediction
            </h1>
            <p className="mt-1 font-mono text-[10px] text-ink-faint">
              single-case inference over the 8 retained features /{' '}
              {trained ? 'weights from this session' : 'bundled reference weights'}
            </p>
          </div>
          <DemoChip />
        </div>

        <div className="flex gap-4">
          {/* inputs */}
          <section
            className="w-[380px] shrink-0 rounded-panel p-4"
            style={{
              background: '#17181B',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow:
                'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
            }}
            aria-label="Case features"
          >
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-[12.5px] font-medium text-ink">Case features</h2>
              <button
                type="button"
                onClick={() => setValues(DEFAULT_VALUES)}
                className="flex cursor-pointer items-center gap-1 font-mono text-[9.5px] text-ink-faint transition-colors duration-150 hover:text-ink"
              >
                <IconReset className="h-3 w-3" />
                population mean
              </button>
            </div>

            {/* stored cases */}
            <div className="mb-4 flex gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setValues(p.values)}
                  className="flex-1 cursor-pointer rounded-[7px] px-2 py-2 text-left transition-colors duration-150"
                  style={{
                    background: '#0D0E10',
                    border: '1px solid rgba(255,255,255,0.05)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                  }}
                >
                  <div className="font-mono text-[10px] text-ink-dim">{p.label}</div>
                  <div className="mt-0.5 font-mono text-[9px] text-ink-faint">{p.note}</div>
                </button>
              ))}
            </div>

            <div className="space-y-3.5">
              {FEATURES.map((f) => (
                <FeatureRow
                  key={f.id}
                  id={f.id}
                  label={f.label}
                  unit={f.unit}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={values[f.id]}
                  onChange={(v) => set(f.id, v)}
                />
              ))}
            </div>
          </section>

          {/* outputs */}
          <div className="flex min-w-0 flex-1 flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <Readout
                title="Hybrid quantum"
                subtitle="8-qubit VQC / 4 layers"
                accent={LANE_COLOR.quantum}
                result={quantum}
              />
              <Readout
                title="Classical baseline"
                subtitle="XGBoost + RandomForest"
                accent={LANE_COLOR.classical}
                result={classical}
              />
            </div>

            {/* agreement between the two heads */}
            <div
              className="flex items-center gap-2.5 rounded-[9px] px-3 py-2.5"
              style={{
                background: '#0D0E10',
                border: '1px solid rgba(255,255,255,0.05)',
                boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
              }}
              role="status"
              aria-live="polite"
            >
              <span
                className="h-[6px] w-[6px] shrink-0 rounded-full"
                style={{ background: agree ? BENIGN : '#C08A3E' }}
              />
              <span className="font-mono text-[10px] text-ink-dim">
                {agree
                  ? `both heads agree on ${quantum.label}`
                  : `heads disagree - quantum says ${quantum.label}, classical says ${classical.label}`}
              </span>
              <span className="flex-1" />
              <span className="font-mono text-[10px] tabular-nums text-ink-faint">
                delta {(quantum.probability - classical.probability >= 0 ? '+' : '') +
                  (quantum.probability - classical.probability).toFixed(3)}
              </span>
            </div>

            <Attributions result={quantum} />

            {/* explainability, read back in words rather than bars */}
            <div
              className="rounded-panel p-4"
              style={{
                background: '#17181B',
                border: '1px solid rgba(255,255,255,0.06)',
                boxShadow:
                  'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
              }}
            >
              <h3 className="mb-2.5 text-[12.5px] font-medium text-ink">
                Why this prediction
              </h3>
              <p className="text-[11.5px] leading-relaxed text-ink-dim">{rationale}</p>

              <p
                className="mt-3 pt-2.5 font-mono text-[9px] leading-relaxed text-ink-faint/80"
                style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
              >
                Attribution is computed against the population mean, so a feature reads as
                neutral when it sits at its typical value and contributes only as it moves
                away. This is a decision-support view, not a diagnosis.
              </p>
            </div>

            <p className="font-mono text-[9px] leading-relaxed text-ink-faint/70">
              This readout is produced by a fixed logistic function over the eight
              features, not a trained model. It is deterministic and illustrative only -
              it must not inform any clinical decision.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
