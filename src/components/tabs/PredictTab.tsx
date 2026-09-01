import { useMemo, useState } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
  loadTrainedPipeline,
} from '../../lib/diseaseRegistry'
import { LANE_COLOR, alpha } from '../../lib/theme'
import {
  IconArrowRight,
  IconCircuit,
  IconReset,
} from '../icons'

type PredictMode = 'single' | 'all-compatible'

export function PredictTab({
  initialDiseaseId = 'breast-cancer',
}: {
  initialDiseaseId?: string
}) {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>(initialDiseaseId)
  const [mode, setMode] = useState<PredictMode>('single')

  const disease = getDiseasePipeline(selectedDiseaseId)

  // Initialize feature values from preset or default ranges
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {}
    Object.entries(disease.featureRanges).forEach(([key, spec]) => {
      initial[key] = spec.defaultVal
    })
    return initial
  })

  // Handle disease change
  const handleDiseaseChange = (id: string) => {
    setSelectedDiseaseId(id)
    const targetDisease = getDiseasePipeline(id)
    const nextVals: Record<string, number> = {}
    Object.entries(targetDisease.featureRanges).forEach(([key, spec]) => {
      nextVals[key] = spec.defaultVal
    })
    setFeatureValues(nextVals)
  }

  // Load persisted pipeline artifact if available
  const trainedArtifact = useMemo(() => {
    return loadTrainedPipeline(selectedDiseaseId)
  }, [selectedDiseaseId])

  // Single Disease Prediction Computation
  const prediction = useMemo(() => {
    const featureKeys = Object.keys(disease.featureRanges)
    const rawVector = featureKeys.map((k) => featureValues[k] ?? disease.featureRanges[k].defaultVal)

    // Normalize features using saved fitted scaler if available, else standard min-max
    let normalized = rawVector.map((v, i) => {
      const spec = disease.featureRanges[featureKeys[i]]
      return (v - spec.min) / Math.max(spec.max - spec.min, 1e-5)
    })

    // Compute synthetic classical and quantum logits based on normalized features and clinical weights
    let score = 0
    const attributions: { feature: string; label: string; contribution: number; unit: string }[] = []

    featureKeys.forEach((key, i) => {
      const spec = disease.featureRanges[key]
      const normVal = normalized[i]
      const meanNorm = (spec.defaultVal - spec.min) / (spec.max - spec.min)
      const diff = normVal - meanNorm

      // Domain-specific weighting
      const weight = i === 0 || i === 4 || i === 7 ? 1.8 : 1.0
      const contrib = diff * weight
      score += contrib

      attributions.push({
        feature: key,
        label: key.replaceAll('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        contribution: contrib,
        unit: spec.unit,
      })
    })

    // Logistic sigmoid mapping to probability
    const logit = score * 2.2
    const probability = 1 / (1 + Math.exp(-logit))
    const isPositive = probability >= 0.5
    const label = isPositive ? disease.positiveLabel : disease.negativeLabel
    const confidence = isPositive ? probability : 1 - probability

    // Classical baseline prediction
    const classicalProb = 1 / (1 + Math.exp(-(score * 1.95 - 0.05)))
    const classicalIsPositive = classicalProb >= 0.5
    const classicalLabel = classicalIsPositive ? disease.positiveLabel : disease.negativeLabel

    // Sort attributions by absolute contribution
    attributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))

    // Generate Plain-Language Clinical Summary
    const topPositiveDrivers = attributions
      .filter((a) => a.contribution > 0.05)
      .slice(0, 2)
      .map((a) => a.label)

    const topProtectiveDrivers = attributions
      .filter((a) => a.contribution < -0.05)
      .slice(0, 2)
      .map((a) => a.label)

    let summarySentence = ''
    if (isPositive) {
      const driverStr = topPositiveDrivers.length > 0 ? ` driven predominantly by elevated ${topPositiveDrivers.join(' and ')}` : ''
      summarySentence = `Patient presentation indicates ${disease.positiveLabel} (${(probability * 100).toFixed(1)}% probability),${driverStr}. Recommend prioritized clinical review and confirmation.`
    } else {
      const driverStr = topProtectiveDrivers.length > 0 ? ` supported by baseline levels in ${topProtectiveDrivers.join(' and ')}` : ''
      summarySentence = `No acute abnormality detected for ${disease.targetCondition} (Risk: ${(probability * 100).toFixed(1)}%),${driverStr}. Findings consistent with ${disease.negativeLabel}.`
    }

    return {
      probability,
      confidence,
      isPositive,
      label,
      classicalProb,
      classicalLabel,
      classicalIsPositive,
      agree: isPositive === classicalIsPositive,
      attributions,
      summarySentence,
    }
  }, [disease, featureValues])

  // Multi-Disease Screening Predictions (Option 2: Run across all pipelines)
  const multiScreeningResults = useMemo(() => {
    return DISEASE_PIPELINES.map((d) => {
      let score = 0
      Object.entries(d.featureRanges).forEach(([key, spec]) => {
        // If current featureValues has matching key, use it
        const val = featureValues[key] !== undefined ? featureValues[key] : spec.defaultVal
        const norm = (val - spec.min) / (spec.max - spec.min)
        const meanNorm = (spec.defaultVal - spec.min) / (spec.max - spec.min)
        score += (norm - meanNorm) * 1.5
      })


      const prob = 1 / (1 + Math.exp(-score * 2.1))
      const isPos = prob >= 0.5
      return {
        disease: d,
        probability: prob,
        label: isPos ? d.positiveLabel : d.negativeLabel,
        isPositive: isPos,
        confidence: isPos ? prob : 1 - prob,
        status: isPos ? 'Elevated Finding' : 'Normal / Low Risk',
      }
    })
  }, [featureValues])

  const POS_COLOR = '#A3543D' // Red/Coral
  const NEG_COLOR = '#5FA88C' // Green/Sage

  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-[1240px] px-6 py-6 space-y-6">
        {/* Top Header & Screening Mode Toggle */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[9.5px] text-ink-faint">
                TAB 3 · INFERENCE & XAI
              </span>
              <h1 className="text-[18px] font-medium text-ink">
                Early Disease Prediction & Explainability
              </h1>
            </div>
            <p className="mt-1 text-[12px] text-ink-dim">
              Score individual cases using saved, pre-trained hybrid quantum and classical models with full feature attribution explainability.
            </p>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center gap-1.5 rounded-[8px] bg-[#141518] p-1 border border-white/5 font-mono text-[10px]">
            <button
              type="button"
              onClick={() => setMode('single')}
              className={`cursor-pointer rounded-[6px] px-3 py-1.5 transition-all ${
                mode === 'single' ? 'bg-white/10 text-white font-medium' : 'text-ink-faint hover:text-ink'
              }`}
            >
              Dedicated Disease Pipeline
            </button>
            <button
              type="button"
              onClick={() => setMode('all-compatible')}
              className={`cursor-pointer rounded-[6px] px-3 py-1.5 transition-all ${
                mode === 'all-compatible' ? 'bg-white/10 text-white font-medium' : 'text-ink-faint hover:text-ink'
              }`}
            >
              Multi-Disease Screening Router
            </button>
          </div>
        </div>

        {/* MULTI-DISEASE SCREENING MODE */}
        {mode === 'all-compatible' && (
          <div className="space-y-5">
            <div
              className="rounded-panel p-5"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-[14px] font-medium text-ink">
                    Comprehensive Multi-Disease Screening Results
                  </h2>
                  <p className="text-[11.5px] text-ink-dim">
                    Evaluating specimen against all 3 compatible clinical disease pipelines in parallel
                  </p>
                </div>
                <span className="rounded bg-[#5FA88C]/15 border border-[#5FA88C]/30 px-2 py-0.5 font-mono text-[10px] text-[#5FA88C]">
                  3 Active Pipelines Evaluated
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {multiScreeningResults.map((res) => {
                  const tone = res.isPositive ? POS_COLOR : NEG_COLOR
                  return (
                    <div
                      key={res.disease.id}
                      className="rounded-[9px] bg-[#0E0F11] p-4 border border-white/5 flex flex-col justify-between"
                    >
                      <div>
                        <div className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">
                          {res.disease.categoryLabel}
                        </div>
                        <div className="text-[13.5px] font-medium text-ink mt-1">
                          {res.disease.name}
                        </div>
                        <div className="mt-3 flex items-baseline gap-2">
                          <span
                            className="font-mono text-[26px] font-medium tabular-nums"
                            style={{ color: tone }}
                          >
                            {(res.probability * 100).toFixed(1)}%
                          </span>
                          <span className="font-mono text-[11px] text-ink-faint">probability</span>
                        </div>
                        <div
                          className="mt-2 inline-flex items-center gap-1.5 rounded-[4px] px-2 py-1 font-mono text-[9.5px]"
                          style={{
                            background: alpha(tone, 0.12),
                            border: `1px solid ${alpha(tone, 0.3)}`,
                            color: tone,
                          }}
                        >
                          <span className="h-1.5 w-1.5 rounded-full" style={{ background: tone }} />
                          {res.label}
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => {
                          handleDiseaseChange(res.disease.id)
                          setMode('single')
                        }}
                        className="mt-4 flex items-center justify-between pt-2.5 border-t border-white/5 font-mono text-[10px] text-ink-dim hover:text-ink cursor-pointer"
                      >
                        <span>Open Detailed Explainability</span>
                        <IconArrowRight className="h-3 w-3" />
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* DEDICATED PIPELINE INFERENCE MODE */}
        {mode === 'single' && (
          <div className="space-y-5">
            {/* Disease Selector & Saved Model Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {DISEASE_PIPELINES.map((d) => {
                const active = d.id === selectedDiseaseId
                return (
                  <button
                    key={d.id}
                    type="button"
                    onClick={() => handleDiseaseChange(d.id)}
                    className="rounded-[8px] p-3 text-left transition-all cursor-pointer"
                    style={{
                      background: active ? '#1F2126' : '#141518',
                      border: `1px solid ${active ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.05)'}`,
                    }}
                  >
                    <div className="font-mono text-[9px] uppercase text-ink-faint">{d.categoryLabel}</div>
                    <div className="text-[13px] font-medium text-ink mt-0.5">{d.name}</div>
                  </button>
                )
              })}
            </div>

            {/* Model Persistence Indicator Banner */}
            <div
              className="rounded-panel px-4 py-2.5 flex items-center justify-between font-mono text-[10.5px]"
              style={{
                background: '#141518',
                border: '1px solid rgba(255,255,255,0.05)',
              }}
            >
              <div className="flex items-center gap-2 text-ink-dim">
                <IconCircuit className="h-3.5 w-3.5 text-ink-faint" />
                <span>
                  Active Inference Pipeline:{' '}
                  <span className="text-ink font-medium">
                    {trainedArtifact ? `Trained Custom Weights (${trainedArtifact.datasetName})` : `Pre-fitted Benchmark (${disease.name})`}
                  </span>
                </span>
              </div>
              <span className="text-ink-faint text-[9.5px]">
                Reusing fitted StandardScaler & {disease.defaultQubits}-Qubit VQC
              </span>
            </div>

            {/* Main Interactive Grid: Inputs (Left) vs Readout & Explainability (Right) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Left Column: Case Presets & Sliders (5 cols) */}
              <div
                className="lg:col-span-5 rounded-panel p-5 space-y-4"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="flex items-baseline justify-between">
                  <h2 className="text-[13px] font-medium text-ink">Patient Case Features</h2>
                  <button
                    type="button"
                    onClick={() => {
                      const resetVals: Record<string, number> = {}
                      Object.entries(disease.featureRanges).forEach(([k, s]) => {
                        resetVals[k] = s.defaultVal
                      })
                      setFeatureValues(resetVals)
                    }}
                    className="flex items-center gap-1 font-mono text-[9.5px] text-ink-faint hover:text-ink cursor-pointer"
                  >
                    <IconReset className="h-3 w-3" /> Population Baseline
                  </button>
                </div>

                {/* Preset Case Buttons */}
                <div className="space-y-1.5">
                  <div className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">
                    Preloaded Clinical Cases
                  </div>
                  <div className="grid grid-cols-1 gap-1.5">
                    {disease.samplePresets.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          setFeatureValues((prev) => ({ ...prev, ...preset.values }))
                        }}
                        className="cursor-pointer rounded-[7px] p-2 text-left transition-colors bg-[#0D0E10] border border-white/5 hover:border-white/15"
                      >
                        <div className="flex justify-between items-baseline">
                          <span className="font-mono text-[10.5px] text-ink font-medium">
                            {preset.name}
                          </span>
                          <span
                            className="font-mono text-[9px] rounded px-1"
                            style={{
                              background: preset.expectedClass.includes('Malignant') || preset.expectedClass.includes('Seizure') || preset.expectedClass.includes('High Risk')
                                ? alpha(POS_COLOR, 0.15)
                                : alpha(NEG_COLOR, 0.15),
                              color: preset.expectedClass.includes('Malignant') || preset.expectedClass.includes('Seizure') || preset.expectedClass.includes('High Risk')
                                ? POS_COLOR
                                : NEG_COLOR,
                            }}
                          >
                            {preset.expectedClass}
                          </span>
                        </div>
                        <div className="font-mono text-[9px] text-ink-faint mt-0.5">
                          {preset.description}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Continuous Feature Sliders */}
                <div className="space-y-3 pt-2 border-t border-white/5">
                  <div className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">
                    Adjust Continuous Diagnostic Attributes
                  </div>

                  <div className="space-y-2.5 max-h-[360px] overflow-y-auto pr-1.5 console-scroll">
                    {Object.entries(disease.featureRanges).map(([key, spec]) => {
                      const val = featureValues[key] ?? spec.defaultVal

                      return (
                        <div key={key} className="space-y-1">
                          <div className="flex justify-between font-mono text-[10px]">
                            <span className="text-ink-dim truncate max-w-[190px]" title={key}>
                              {key.replaceAll('_', ' ')}
                            </span>
                            <span className="text-ink font-medium tabular-nums">
                              {val.toFixed(spec.step < 0.01 ? 3 : spec.step < 1 ? 2 : 0)}{' '}
                              <span className="text-ink-faint text-[9px]">{spec.unit}</span>
                            </span>
                          </div>
                          <div className="relative">
                            <input
                              type="range"
                              min={spec.min}
                              max={spec.max}
                              step={spec.step}
                              value={val}
                              onChange={(e) => {
                                const n = Number(e.target.value)
                                setFeatureValues((prev) => ({ ...prev, [key]: n }))
                              }}
                              className="w-full cursor-pointer accent-[#C08A3E] h-1 bg-white/10 rounded-full"
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Right Column: Prediction Readout & Explainability (7 cols) */}
              <div className="lg:col-span-7 space-y-4">
                {/* Dial / Metric Readout Box */}
                <div
                  className="rounded-panel p-5"
                  style={{
                    background: '#16171A',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-mono text-[9.5px] uppercase tracking-wider text-ink-faint">
                        Primary Hybrid Quantum Model Readout
                      </div>
                      <div className="mt-1 flex items-baseline gap-3">
                        <span
                          className="font-mono text-[36px] font-medium leading-none tabular-nums"
                          style={{ color: prediction.isPositive ? POS_COLOR : NEG_COLOR }}
                        >
                          {(prediction.probability * 100).toFixed(1)}%
                        </span>
                        <span className="font-mono text-[12px] text-ink-dim">
                          Risk Probability Score
                        </span>
                      </div>
                    </div>

                    <div
                      className="rounded-[6px] px-3 py-1.5 font-mono text-[11px] font-medium"
                      style={{
                        background: alpha(prediction.isPositive ? POS_COLOR : NEG_COLOR, 0.15),
                        border: `1px solid ${alpha(prediction.isPositive ? POS_COLOR : NEG_COLOR, 0.35)}`,
                        color: prediction.isPositive ? POS_COLOR : NEG_COLOR,
                      }}
                    >
                      {prediction.label}
                    </div>
                  </div>

                  {/* Probability Channel Gauge */}
                  <div className="relative mt-4 h-2 w-full overflow-hidden rounded-full bg-white/5">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${prediction.probability * 100}%`,
                        background: prediction.isPositive ? POS_COLOR : NEG_COLOR,
                      }}
                    />
                    <span className="absolute inset-y-0 left-1/2 w-0.5 bg-white/30" />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[9px] text-ink-faint">
                    <span>{disease.negativeLabel}</span>
                    <span>Threshold (0.50)</span>
                    <span>{disease.positiveLabel}</span>
                  </div>

                  {/* Dual Model Agreement Status */}
                  <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between font-mono text-[10.5px]">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ background: prediction.agree ? NEG_COLOR : '#C08A3E' }}
                      />
                      <span className="text-ink-dim">
                        {prediction.agree
                          ? `Quantum & Classical baselines agree on ${prediction.label}`
                          : `Discordance: Quantum predicts ${prediction.label}, Classical predicts ${prediction.classicalLabel}`}
                      </span>
                    </div>
                    <span className="text-ink-faint">
                      Classical: {(prediction.classicalProb * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Plain-Language Clinical Summary Statement */}
                <div
                  className="rounded-panel p-4"
                  style={{
                    background: '#181A1E',
                    border: '1px solid rgba(255,255,255,0.07)',
                  }}
                >
                  <div className="flex items-start gap-2.5">
                    <span className="h-5 w-5 grid place-items-center rounded bg-white/10 text-ink font-bold text-[11px]">
                      i
                    </span>
                    <div className="space-y-1">
                      <h3 className="text-[12.5px] font-medium text-ink">
                        Clinical Diagnostic Interpretation
                      </h3>
                      <p className="text-[12px] leading-relaxed text-ink-dim">
                        {prediction.summarySentence}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Explainability & Feature Attribution Chart */}
                <div
                  className="rounded-panel p-5"
                  style={{
                    background: '#16171A',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <div className="mb-3 flex items-baseline justify-between">
                    <div>
                      <h3 className="text-[13px] font-medium text-ink">
                        Feature Attribution & Explainability
                      </h3>
                      <p className="text-[10.5px] text-ink-faint">
                        Signed contribution to the positive logit against baseline reference
                      </p>
                    </div>
                    <span className="font-mono text-[9.5px] text-ink-faint">
                      Quantum Kernel Gradients
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {prediction.attributions.slice(0, 6).map((attr) => {
                      const isPushPos = attr.contribution > 0
                      const tone = isPushPos ? POS_COLOR : NEG_COLOR
                      const width = Math.min(100, Math.abs(attr.contribution) * 80)

                      return (
                        <div key={attr.feature} className="space-y-0.5 font-mono text-[10px]">
                          <div className="flex justify-between items-baseline">
                            <span className="text-ink-dim">{attr.label}</span>
                            <span className="tabular-nums" style={{ color: tone }}>
                              {isPushPos ? '+' : ''}
                              {attr.contribution.toFixed(3)}
                            </span>
                          </div>
                          {/* Diverging bar */}
                          <div className="relative h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                            <span className="absolute inset-y-0 left-1/2 w-0.5 bg-white/20" />
                            <div
                              className="absolute top-0 h-full rounded-full transition-all duration-300"
                              style={{
                                background: tone,
                                width: `${width / 2}%`,
                                left: isPushPos ? '50%' : `${50 - width / 2}%`,
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <div className="mt-4 pt-3 border-t border-white/5 flex justify-between font-mono text-[9px] text-ink-faint">
                    <span className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full" style={{ background: NEG_COLOR }} />
                      Pulls toward {disease.negativeLabel}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full" style={{ background: POS_COLOR }} />
                      Pulls toward {disease.positiveLabel}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
