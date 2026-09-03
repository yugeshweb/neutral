import { useMemo, useState } from 'react'
import {
  DEFAULT_VALUES,
  FEATURES,
  PRESETS,
  predict,
  type Attribution,
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
          <div className="text-[14.5px] font-medium leading-tight text-ink">{title}</div>
          <div className="mt-0.5 font-mono text-[11.5px] text-ink-faint">{subtitle}</div>
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
        <span className="font-mono text-[14px] text-ink-faint">% malignant</span>
      </div>

      <div
        className="mt-2 inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[11.5px] tracking-[0.02em]"
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

      <div className="mt-1.5 flex justify-between font-mono text-[11px] text-ink-faint">
        <span>benign</span>
        <span>threshold 0.50</span>
        <span>malignant</span>
      </div>

      <div
        className="mt-3 flex justify-between pt-2.5 font-mono text-[12px]"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span className="text-ink-faint" title="Decision margin from threshold — not a probability">decision margin</span>
        <span className="tabular-nums text-ink-dim" title="Decision margin from threshold — not a probability">{result.confidence.toFixed(3)}</span>
      </div>
    </div>
  )
}

/** Signed contribution bars, diverging from a centre axis with interactive inspection. */
function Attributions({
  result,
  selectedFeature,
  onSelectFeature,
}: {
  result: Prediction
  selectedFeature: string | null
  onSelectFeature: (id: string | null) => void
}) {
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
        <h3 className="text-[14.5px] font-medium text-ink">Feature Attribution & Directionality</h3>
        <span className="font-mono text-[11.5px] text-ink-faint">local occlusion</span>
      </div>
      <p className="mb-3.5 font-mono text-[11.5px] text-ink-faint">
        signed logit shift relative to population baseline (click a feature to inspect)
      </p>

      <div className="space-y-2.5">
        {result.attributions.map((a) => {
          const pushesMalignant = a.contribution > 0
          const tone = pushesMalignant ? MALIGNANT : BENIGN
          const width = (Math.abs(a.contribution) / max) * 50
          const isSelected = selectedFeature === a.id

          return (
            <div
              key={a.id}
              onClick={() => onSelectFeature(isSelected ? null : a.id)}
              className="group cursor-pointer rounded-[6px] p-1 transition-colors hover:bg-white/[0.03]"
              style={{
                background: isSelected ? 'rgba(255,255,255,0.04)' : 'transparent',
              }}
            >
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="truncate font-mono text-[12px] text-ink-dim group-hover:text-ink">
                  {a.label}
                </span>
                <span
                  className="shrink-0 font-mono text-[12px] tabular-nums"
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

              {isSelected && a.description && (
                <div
                  className="mt-2 rounded-[6px] p-2.5 text-[11.5px] leading-relaxed animate-fadeIn"
                  style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.05)' }}
                >
                  <div className="font-mono text-[10.5px] text-ink-faint mb-1">
                    Value: <span className="text-ink">{a.value} {a.unit}</span> (Baseline mean: {a.typical} {a.unit})
                  </div>
                  <p className="text-ink-dim">{a.description}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div
        className="mt-3.5 flex gap-4 pt-2.5 font-mono text-[11px] text-ink-faint"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: BENIGN }} />
          pulls toward benign (protective)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: MALIGNANT }} />
          pulls toward malignant (atypical)
        </span>
      </div>
    </div>
  )
}

/** One feature row: slider in a recessed channel, with numeric entry and clinical tooltip. */
function FeatureRow({
  id,
  label,
  unit,
  min,
  max,
  step,
  value,
  description,
  onChange,
}: {
  id: string
  label: string
  unit: string
  min: number
  max: number
  step: number
  value: number
  description?: string
  onChange: (v: number) => void
}) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label
          htmlFor={id}
          title={description}
          className="font-mono text-[12px] text-ink-dim cursor-help transition-colors hover:text-ink"
        >
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
            className="w-[62px] rounded-[5px] px-1.5 py-[3px] text-right font-mono text-[12.5px] tabular-nums text-ink outline-none"
            style={{
              background: '#0D0E10',
              border: '1px solid rgba(255,255,255,0.05)',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          />
          {unit && (
            <span className="w-[22px] font-mono text-[11px] text-ink-faint">{unit}</span>
          )}
        </span>
      </div>

      <div className="relative">
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
  trained: boolean
}

export function PredictView({ trained }: Props) {
  const [values, setValues] = useState<FeatureValues>(DEFAULT_VALUES)
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null)

  const quantum = useMemo(() => predict(values, 'quantum'), [values])
  const classical = useMemo(() => predict(values, 'classical'), [values])

  const set = (id: string, v: number) => setValues((prev) => ({ ...prev, [id]: v }))
  const agree = quantum.label === classical.label

  // Structured clinical rationale synthesis
  const clinicalReport = useMemo(() => {
    const isMalignant = quantum.label === 'malignant'
    const drivers = quantum.attributions.filter((a) =>
      isMalignant ? a.contribution > 0.1 : a.contribution < -0.1
    )
    const counters = quantum.attributions.filter((a) =>
      isMalignant ? a.contribution < -0.1 : a.contribution > 0.1
    )

    const topDriver = drivers[0] as Attribution | undefined
    const secondDriver = drivers[1] as Attribution | undefined
    const topCounter = counters[0] as Attribution | undefined

    const strength =
      quantum.confidence > 0.75
        ? 'high certainty'
        : quantum.confidence > 0.35
        ? 'moderate certainty'
        : 'borderline / equivocal margin'

    return {
      strength,
      topDriver,
      secondDriver,
      topCounter,
      driverCount: drivers.length,
      counterCount: counters.length,
    }
  }, [quantum])

  return (
    <div className="console-scroll h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[1120px] px-6 py-6">
        <div className="mb-5 flex items-start gap-3">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h1 className="text-[16px] font-medium tracking-[-0.01em] text-ink">
                Inference & Model Explainability
              </h1>
              <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[11px] text-ink-faint">
                WDBC Cohort
              </span>
            </div>
            <p className="mt-1 font-mono text-[12px] text-ink-faint">
              single-case cytological inference over 8 retained morphometric features /{' '}
              {trained ? 'weights from this session' : 'bundled reference weights'}
            </p>
          </div>
          <DemoChip label="investigational inference" />
        </div>

        <div className="flex flex-col lg:flex-row gap-4">
          {/* inputs column */}
          <section
            className="w-full lg:w-[380px] shrink-0 rounded-panel p-4"
            style={{
              background: '#17181B',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow:
                'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
            }}
            aria-label="Case features"
          >
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-[14.5px] font-medium text-ink">Cytological Morphometrics</h2>
              <button
                type="button"
                onClick={() => setValues(DEFAULT_VALUES)}
                className="flex cursor-pointer items-center gap-1 font-mono text-[11.5px] text-ink-faint transition-colors duration-150 hover:text-ink"
              >
                <IconReset className="h-3 w-3" />
                reset to mean
              </button>
            </div>

            {/* stored presets */}
            <div className="mb-4 grid grid-cols-3 gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setValues(p.values)}
                  className="cursor-pointer rounded-[7px] px-2 py-2 text-left transition-colors duration-150"
                  style={{
                    background: '#0D0E10',
                    border: '1px solid rgba(255,255,255,0.05)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                  }}
                >
                  <div className="font-mono text-[11.5px] text-ink-dim truncate">{p.label}</div>
                  <div className="mt-0.5 font-mono text-[10px] text-ink-faint truncate">{p.note}</div>
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
                  description={f.description}
                  onChange={(v) => set(f.id, v)}
                />
              ))}
            </div>

            <div className="mt-4 pt-3 border-t border-white/5 font-mono text-[11px] text-ink-faint leading-relaxed">
              Hover over feature labels to view clinical morphological definitions.
            </div>
          </section>

          {/* outputs and explainability column */}
          <div className="flex min-w-0 flex-1 flex-col gap-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Readout
                title="Hybrid Quantum VQC"
                subtitle="8-qubit / parameterized rotation ansatz"
                accent={LANE_COLOR.quantum}
                result={quantum}
              />
              <Readout
                title="Classical Baseline"
                subtitle="Gradient Boosted Decision Forest"
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
              <span className="font-mono text-[12px] text-ink-dim">
                {agree
                  ? `Both model architectures reach concordance on ${quantum.label.toUpperCase()}`
                  : `Architectural discordance: Quantum says ${quantum.label}, Classical baseline says ${classical.label}`}
              </span>
              <span className="flex-1" />
              <span className="font-mono text-[12px] tabular-nums text-ink-faint">
                delta {(quantum.probability - classical.probability >= 0 ? '+' : '') +
                  (quantum.probability - classical.probability).toFixed(3)}
              </span>
            </div>

            {/* Feature attribution chart */}
            <Attributions
              result={quantum}
              selectedFeature={selectedFeature}
              onSelectFeature={setSelectedFeature}
            />

            {/* Comprehensive clinical explainability narrative */}
            <div
              className="rounded-panel p-4"
              style={{
                background: '#17181B',
                border: '1px solid rgba(255,255,255,0.06)',
                boxShadow:
                  'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
              }}
            >
              <div className="flex items-center justify-between gap-2 mb-3">
                <h3 className="text-[14.5px] font-medium text-ink">
                  Pathological Decision Support & Insight Handling
                </h3>
                <span className="font-mono text-[11px] text-ink-faint">
                  Evidence Breakdown
                </span>
              </div>

              <div className="space-y-3 text-[12px] leading-relaxed text-ink-dim">
                {/* Executive summary */}
                <div className="p-3 rounded-[7px] bg-[#0D0E10] border border-white/5">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mb-1">
                    Executive Classification Verdict ({clinicalReport.strength})
                  </div>
                  <p className="text-ink">
                    The hybrid quantum-classical engine categorizes this cytological profile as{' '}
                    <strong style={{ color: quantum.label === 'malignant' ? MALIGNANT : BENIGN }}>
                      {quantum.label.toUpperCase()}
                    </strong>{' '}
                    with a {(quantum.probability * 100).toFixed(1)}% predicted probability and a decision margin of{' '}
                    {quantum.confidence.toFixed(3)}.
                  </p>
                </div>

                {/* Primary drivers */}
                {clinicalReport.topDriver && (
                  <div>
                    <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mb-1">
                      Dominant Pathological Drivers
                    </div>
                    <p className="text-ink-dim">
                      The primary feature pulling the prediction is{' '}
                      <span className="font-medium text-ink">{clinicalReport.topDriver.label}</span>{' '}
                      (logit contribution{' '}
                      <span
                        className="font-mono"
                        style={{
                          color: clinicalReport.topDriver.contribution > 0 ? MALIGNANT : BENIGN,
                        }}
                      >
                        {clinicalReport.topDriver.contribution >= 0 ? '+' : ''}
                        {clinicalReport.topDriver.contribution.toFixed(3)}
                      </span>
                      ). {clinicalReport.topDriver.description}
                      {clinicalReport.secondDriver && (
                        <>
                          {' '}Secondarily influenced by{' '}
                          <span className="font-medium text-ink">{clinicalReport.secondDriver.label}</span>{' '}
                          (+{clinicalReport.secondDriver.contribution.toFixed(3)}).
                        </>
                      )}
                    </p>
                  </div>
                )}

                {/* Counter evidence */}
                {clinicalReport.topCounter ? (
                  <div>
                    <div className="text-[11px] font-mono uppercase tracking-wider text-ink-faint mb-1">
                      Moderating Evidence / Counter-Factors
                    </div>
                    <p className="text-ink-dim">
                      <span className="font-medium text-ink">{clinicalReport.topCounter.label}</span>{' '}
                      opposes the verdict with a signed contribution of{' '}
                      <span
                        className="font-mono"
                        style={{
                          color: clinicalReport.topCounter.contribution > 0 ? MALIGNANT : BENIGN,
                        }}
                      >
                        {clinicalReport.topCounter.contribution.toFixed(3)}
                      </span>
                      , explaining why the decision boundary maintains an uncertainty envelope rather than 100% saturation.
                    </p>
                  </div>
                ) : (
                  <p className="text-ink-faint text-[11px]">
                    Uniform directional consensus: All evaluated morphological markers align toward the current class.
                  </p>
                )}

                {/* Safety & Protocol */}
                <div
                  className="pt-2.5 border-t border-white/5 font-mono text-[11px] leading-relaxed text-ink-faint/80"
                >
                  <strong>Clinical Caveat:</strong> Attributions represent local Taylor/occlusion perturbations around the population baseline. Nuclear morphological assessment via FNA does not assess architectural tissue invasion; tissue biopsy and histological staging are mandatory for clinical diagnosis.
                </div>
              </div>
            </div>

            <p className="font-mono text-[11px] leading-relaxed text-ink-faint/70">
              API-compatible contract: Endpoint /api/predict/explain. Replaceable via backend integration in lib/explainability.ts.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
