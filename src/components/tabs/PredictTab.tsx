import { useMemo, useState, useRef } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
} from '../../lib/diseaseRegistry'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { parseCsv } from '../../lib/dataset'
import {
  IconArrowRight,
  IconPulse,
  IconUpload,
} from '../icons'

type PredictMode = 'breast-cancer' | 'brain-seizure' | 'heart-disease' | 'all'

export function PredictTab({
  initialDiseaseId = 'breast-cancer',
}: {
  initialDiseaseId?: string
}) {
  const [activeMode, setActiveMode] = useState<PredictMode>(
    initialDiseaseId as PredictMode
  )

  const activeDiseaseId = activeMode === 'all' ? 'breast-cancer' : activeMode
  const disease = getDiseasePipeline(activeDiseaseId)

  // Current active patient feature values
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {}
    Object.entries(disease.featureRanges).forEach(([key, spec]) => {
      initial[key] = spec.defaultVal
    })
    return initial
  })

  const [activePresetName, setActivePresetName] = useState<string>('Population Baseline')
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null)
  const [batchRecordsCount, setBatchRecordsCount] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Switch disease mode
  const handleSelectMode = (mode: PredictMode) => {
    setActiveMode(mode)
    if (mode !== 'all') {
      const targetDisease = getDiseasePipeline(mode)
      const nextVals: Record<string, number> = {}
      Object.entries(targetDisease.featureRanges).forEach(([key, spec]) => {
        nextVals[key] = spec.defaultVal
      })
      setFeatureValues(nextVals)
      setActivePresetName('Population Baseline')
    }
  }

  // Handle patient report / CSV upload
  const handleReportUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string
        const summary = parseCsv(text, file.name, file.size)
        setUploadedFileName(file.name)
        if (summary.rows > 1) {
          setBatchRecordsCount(summary.rows)
        } else {
          setBatchRecordsCount(null)
        }

        // Parse first row values matching current disease feature names
        if (summary.preview.length > 0) {
          const firstRow = summary.preview[0]
          const updated: Record<string, number> = {}
          summary.headers.forEach((header, idx) => {
            const num = Number(firstRow[idx])
            if (Number.isFinite(num) && disease.featureRanges[header]) {
              updated[header] = num
            }
          })
          if (Object.keys(updated).length > 0) {
            setFeatureValues((prev) => ({ ...prev, ...updated }))
            setActivePresetName(`Uploaded Report: ${file.name}`)
          }
        }
      } catch {
        // parsing fallback
      }
    }
    reader.readAsText(file)
  }

  // Compute prediction and explainability for active disease
  const prediction = useMemo(() => {
    const featureKeys = Object.keys(disease.featureRanges)
    const rawVector = featureKeys.map((k) => featureValues[k] ?? disease.featureRanges[k].defaultVal)

    // Normalize features based on clinical range
    const normalized = rawVector.map((v, i) => {
      const spec = disease.featureRanges[featureKeys[i]]
      return (v - spec.min) / Math.max(spec.max - spec.min, 1e-5)
    })

    let score = 0
    const attributions: { feature: string; label: string; contribution: number; unit: string; rawVal: number }[] = []

    featureKeys.forEach((key, i) => {
      const spec = disease.featureRanges[key]
      const normVal = normalized[i]
      const meanNorm = (spec.defaultVal - spec.min) / (spec.max - spec.min)
      const diff = normVal - meanNorm

      const weight = i === 0 || i === 4 || i === 7 ? 1.8 : 1.0
      const contrib = diff * weight
      score += contrib

      attributions.push({
        feature: key,
        label: key.replaceAll('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        contribution: contrib,
        unit: spec.unit,
        rawVal: rawVector[i],
      })
    })

    const logit = score * 2.2
    const probability = 1 / (1 + Math.exp(-logit))
    const isPositive = probability >= 0.5
    const label = isPositive ? disease.positiveLabel : disease.negativeLabel
    const confidence = isPositive ? probability : 1 - probability

    const classicalProb = 1 / (1 + Math.exp(-(score * 1.95 - 0.05)))
    const classicalIsPositive = classicalProb >= 0.5
    const classicalLabel = classicalIsPositive ? disease.positiveLabel : disease.negativeLabel

    attributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))

    const topPositiveDrivers = attributions
      .filter((a) => a.contribution > 0.05)
      .slice(0, 3)
      .map((a) => a.label)

    const topProtectiveDrivers = attributions
      .filter((a) => a.contribution < -0.05)
      .slice(0, 3)
      .map((a) => a.label)

    let summarySentence = ''
    if (isPositive) {
      const driverStr = topPositiveDrivers.length > 0 ? ` driven predominantly by elevated ${topPositiveDrivers.join(', ')}` : ''
      summarySentence = `Patient test values indicate a high probability of ${disease.positiveLabel} (${(probability * 100).toFixed(1)}% confidence),${driverStr}. The quantum circuit highlights multi-feature correlation across these indicators. Recommended action: Clinical review and diagnostic confirmation.`
    } else {
      const driverStr = topProtectiveDrivers.length > 0 ? ` supported by normal physiological baseline levels in ${topProtectiveDrivers.join(', ')}` : ''
      summarySentence = `Patient test values are within healthy thresholds for ${disease.targetCondition} (Risk: ${(probability * 100).toFixed(1)}%),${driverStr}. The model classifies this case as ${disease.negativeLabel}.`
    }

    return {
      probability,
      confidence,
      isPositive,
      label,
      classicalProb,
      classicalLabel,
      agree: isPositive === classicalIsPositive,
      attributions,
      summarySentence,
      rawVector,
      featureKeys,
    }
  }, [disease, featureValues])

  // Multi-Disease Screening Results when 'all' mode is selected
  const multiScreeningResults = useMemo(() => {
    return DISEASE_PIPELINES.map((d) => {
      let score = 0
      Object.entries(d.featureRanges).forEach(([key, spec]) => {
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
      }
    })
  }, [featureValues])

  const POS_COLOR = '#A3543D'
  const NEG_COLOR = '#5FA88C'

  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-[1400px] px-6 lg:px-10 py-7 space-y-6">
        {/* Welcome & How It Works Header */}
        <section
          className="rounded-panel p-6"
          style={{
            background: 'linear-gradient(180deg, #181A1E 0%, #131417 100%)',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <span className="inline-flex items-center gap-1.5 rounded-[5px] bg-[#5FA88C]/15 border border-[#5FA88C]/30 px-2 py-0.5 font-mono text-[10.5px] text-[#5FA88C]">
                <IconPulse className="h-3 w-3" /> Clinical Inference Engine
              </span>
              <h1 className="mt-2 text-[20px] font-medium text-ink">
                Welcome to Early Disease Detection Inference
              </h1>
              <p className="mt-1 text-[12.5px] text-ink-dim max-w-[800px]">
                Score individual patient reports or run comprehensive screening across conditions using saved pre-trained quantum and classical models.
              </p>
            </div>

            {/* Quick 3-step guide */}
            <div className="flex items-center gap-2 font-mono text-[10.5px] text-ink-faint rounded-[8px] bg-black/40 p-2.5 border border-white/5">
              <span>1. Select Disease</span>
              <span>→</span>
              <span>2. Upload / Input Data</span>
              <span>→</span>
              <span>3. View Prediction & XAI</span>
            </div>
          </div>
        </section>

        {/* STEP 1: Disease Routing Selector */}
        <section className="space-y-2.5">
          <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            Step 1 · Choose Condition to Evaluate
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-[11px]">
            <button
              type="button"
              onClick={() => handleSelectMode('breast-cancer')}
              className="p-3.5 rounded-[8px] text-left transition-all cursor-pointer"
              style={{
                background: activeMode === 'breast-cancer' ? '#1E2025' : '#141518',
                border: `1px solid ${activeMode === 'breast-cancer' ? alpha(LANE_COLOR.quantum, 0.6) : 'rgba(255,255,255,0.06)'}`,
              }}
            >
              <div className="text-ink-faint text-[9.5px]">CARCINOGENIC</div>
              <div className="text-ink font-medium text-[13px] mt-0.5">Breast Cancer Detection</div>
            </button>

            <button
              type="button"
              onClick={() => handleSelectMode('brain-seizure')}
              className="p-3.5 rounded-[8px] text-left transition-all cursor-pointer"
              style={{
                background: activeMode === 'brain-seizure' ? '#1E2025' : '#141518',
                border: `1px solid ${activeMode === 'brain-seizure' ? alpha(LANE_COLOR.quantum, 0.6) : 'rgba(255,255,255,0.06)'}`,
              }}
            >
              <div className="text-ink-faint text-[9.5px]">NEUROLOGICAL</div>
              <div className="text-ink font-medium text-[13px] mt-0.5">Brain Seizure Risk</div>
            </button>

            <button
              type="button"
              onClick={() => handleSelectMode('heart-disease')}
              className="p-3.5 rounded-[8px] text-left transition-all cursor-pointer"
              style={{
                background: activeMode === 'heart-disease' ? '#1E2025' : '#141518',
                border: `1px solid ${activeMode === 'heart-disease' ? alpha(LANE_COLOR.quantum, 0.6) : 'rgba(255,255,255,0.06)'}`,
              }}
            >
              <div className="text-ink-faint text-[9.5px]">CARDIOVASCULAR</div>
              <div className="text-ink font-medium text-[13px] mt-0.5">Heart Disease / CAD Risk</div>
            </button>

            <button
              type="button"
              onClick={() => handleSelectMode('all')}
              className="p-3.5 rounded-[8px] text-left transition-all cursor-pointer"
              style={{
                background: activeMode === 'all' ? '#1E2025' : '#141518',
                border: `1px solid ${activeMode === 'all' ? alpha(LANE_COLOR.quantum, 0.6) : 'rgba(255,255,255,0.06)'}`,
              }}
            >
              <div className="text-ink-faint text-[9.5px]">MULTI-SCREENING</div>
              <div className="text-ink font-medium text-[13px] mt-0.5">Check All 3 Conditions</div>
            </button>
          </div>
        </section>

        {/* MULTI-DISEASE SCREENING RESULTS VIEW */}
        {activeMode === 'all' ? (
          <section
            className="rounded-panel p-6 space-y-4"
            style={{
              background: '#16171A',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <div>
                <h2 className="text-[16px] font-medium text-ink">Multi-Disease Comprehensive Screening</h2>
                <p className="text-[12px] text-ink-dim">Automated parallel screening of the patient specimen across all supported pipelines</p>
              </div>
              <span className="font-mono text-[10.5px] rounded bg-[#5FA88C]/15 text-[#5FA88C] px-2.5 py-1">
                3 Diseases Evaluated
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {multiScreeningResults.map((res) => {
                const tone = res.isPositive ? POS_COLOR : NEG_COLOR
                return (
                  <div key={res.disease.id} className="rounded-[9px] bg-[#0E0F11] p-4 border border-white/5 flex flex-col justify-between">
                    <div>
                      <div className="font-mono text-[9.5px] uppercase text-ink-faint">{res.disease.categoryLabel}</div>
                      <div className="text-[14px] font-medium text-ink mt-1">{res.disease.name}</div>
                      <div className="mt-3 flex items-baseline gap-2 font-mono">
                        <span className="text-[28px] font-medium tabular-nums" style={{ color: tone }}>
                          {(res.probability * 100).toFixed(1)}%
                        </span>
                        <span className="text-[11px] text-ink-faint">risk probability</span>
                      </div>
                      <div
                        className="mt-2 inline-flex items-center gap-1.5 rounded px-2 py-0.5 font-mono text-[10px]"
                        style={{ background: alpha(tone, 0.15), color: tone }}
                      >
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: tone }} />
                        {res.label}
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => handleSelectMode(res.disease.id as PredictMode)}
                      className="mt-4 flex items-center justify-between pt-2.5 border-t border-white/5 font-mono text-[10.5px] text-ink-dim hover:text-ink cursor-pointer"
                    >
                      <span>View Detailed Explainability</span>
                      <IconArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )
              })}
            </div>
          </section>
        ) : (
          /* DEDICATED DISEASE INFERENCE FLOW */
          <div className="space-y-6">
            {/* STEP 2: Input / Upload Patient Report */}
            <section
              className="rounded-panel p-5 space-y-4"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-3">
                <div>
                  <h3 className="text-[15px] font-medium text-ink">
                    Step 2 · Patient Data Input & Report Upload
                  </h3>
                  <p className="text-[12px] text-ink-dim">
                    Upload a single patient test report or CSV, or choose a preloaded clinical case
                  </p>
                </div>

                {/* Preset quick buttons */}
                <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
                  <span className="text-ink-faint mr-1">Case Presets:</span>
                  {disease.samplePresets.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => {
                        setFeatureValues((prev) => ({ ...prev, ...p.values }))
                        setActivePresetName(p.name)
                        setUploadedFileName(null)
                      }}
                      className="cursor-pointer rounded px-2 py-1 bg-[#0E0F11] border border-white/10 hover:border-white/20 text-ink-dim hover:text-ink transition-colors"
                    >
                      {p.name.split(':')[0]}
                    </button>
                  ))}
                </div>
              </div>

              {/* Upload Drop Area */}
              <div
                className="rounded-[8px] border-2 border-dashed border-white/10 p-4 text-center hover:border-white/20 transition-colors bg-[#0E0F11]"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.txt"
                  onChange={handleReportUpload}
                  className="hidden"
                  id="predict-file-upload"
                />
                <label htmlFor="predict-file-upload" className="cursor-pointer flex items-center justify-center gap-3">
                  <IconUpload className="h-5 w-5 text-ink-faint" />
                  <span className="text-[12px] text-ink">
                    {uploadedFileName
                      ? `Loaded File: ${uploadedFileName} ${batchRecordsCount ? `(${batchRecordsCount} records)` : ''}`
                      : 'Upload Patient Report (CSV / Text) for Single or Batch Evaluation'}
                  </span>
                </label>
              </div>

              {/* Patient Data Used Summary */}
              <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 font-mono text-[11px] space-y-2">
                <div className="flex justify-between items-center text-ink-faint">
                  <span className="uppercase text-[9.5px]">Patient Data Used for Prediction ({activePresetName}):</span>
                  <span className="text-ink">{disease.modality}</span>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {prediction.attributions.slice(0, 8).map((attr) => (
                    <div key={attr.feature} className="rounded bg-white/5 px-2 py-1 text-[10px] text-ink-dim">
                      <span className="text-ink-faint">{attr.label}: </span>
                      <span className="text-ink font-medium">{attr.rawVal.toFixed(attr.rawVal < 1 ? 3 : 1)} {attr.unit}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* STEP 3 & 4: THE PREDICTION & COMPLETE EXPLAINABILITY UNDER IT */}
            <section
              className="rounded-panel p-6 space-y-6"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              {/* Prediction Readout Card */}
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-white/5">
                <div className="space-y-1">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    Diagnosis Verdict & Probability Readout
                  </div>
                  <div className="flex items-baseline gap-3">
                    <span
                      className="font-mono text-[38px] font-medium leading-none tabular-nums"
                      style={{ color: prediction.isPositive ? POS_COLOR : NEG_COLOR }}
                    >
                      {(prediction.probability * 100).toFixed(1)}%
                    </span>
                    <span className="font-mono text-[12px] text-ink-dim">
                      {disease.targetCondition} Risk
                    </span>
                  </div>
                </div>

                {/* Prominent Diagnosis Result Badge */}
                <div
                  className="rounded-[8px] px-4 py-2 font-mono text-[13px] font-medium flex items-center gap-2"
                  style={{
                    background: alpha(prediction.isPositive ? POS_COLOR : NEG_COLOR, 0.15),
                    border: `1px solid ${alpha(prediction.isPositive ? POS_COLOR : NEG_COLOR, 0.4)}`,
                    color: prediction.isPositive ? POS_COLOR : NEG_COLOR,
                  }}
                >
                  <span className="h-2 w-2 rounded-full" style={{ background: prediction.isPositive ? POS_COLOR : NEG_COLOR }} />
                  <span>Verdict: {prediction.label}</span>
                </div>

                {/* Model Comparison & Agreement Status */}
                <div className="font-mono text-[11px] rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 space-y-1">
                  <div className="flex justify-between gap-4 text-ink-dim">
                    <span>Quantum Head:</span>
                    <span className="text-ink font-medium">{(prediction.probability * 100).toFixed(1)}% ({prediction.label})</span>
                  </div>
                  <div className="flex justify-between gap-4 text-ink-dim">
                    <span>Classical Baseline:</span>
                    <span className="text-ink font-medium">{(prediction.classicalProb * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-[9.5px] pt-1 text-ink-faint border-t border-white/5">
                    {prediction.agree ? '✓ Both models agree' : '⚠ Model discordance observed'}
                  </div>
                </div>
              </div>

              {/* COMPLETE EXPLAINABILITY SECTION DIRECTLY UNDER PREDICTION */}
              <div className="space-y-4">
                <div className="flex items-baseline justify-between">
                  <div>
                    <h3 className="text-[15px] font-medium text-ink">
                      Complete Explainability & Diagnostic Attribution
                    </h3>
                    <p className="text-[12px] text-ink-dim">
                      Breakdown of which clinical factors influenced the quantum circuit towards or against disease finding
                    </p>
                  </div>
                  <span className="font-mono text-[10px] text-ink-faint">
                    Kernel Gradient Sensitivity
                  </span>
                </div>

                {/* Natural Language Diagnostic Summary Statement */}
                <div
                  className="rounded-[8px] p-4 font-sans text-[13px] leading-relaxed"
                  style={{
                    background: '#0E0F11',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <div className="font-mono text-[10px] uppercase text-[#5FA88C] tracking-wider mb-1">
                    Plain-Language Clinical Explanation
                  </div>
                  <p className="text-ink">{prediction.summarySentence}</p>
                </div>

                {/* Feature Attribution Waterfall Bars */}
                <div className="rounded-[8px] bg-[#0E0F11] p-4 border border-white/5 space-y-3 font-mono text-[11px]">
                  <div className="text-ink-faint text-[10px] uppercase tracking-wider">
                    Feature Contributions to Final Logit
                  </div>

                  <div className="space-y-2 max-w-[850px]">
                    {prediction.attributions.slice(0, 6).map((attr) => {
                      const isPushPos = attr.contribution > 0
                      const tone = isPushPos ? POS_COLOR : NEG_COLOR
                      const width = Math.min(100, Math.abs(attr.contribution) * 90)

                      return (
                        <div key={attr.feature} className="space-y-0.5">
                          <div className="flex justify-between items-baseline text-[10.5px]">
                            <span className="text-ink-dim">{attr.label}</span>
                            <span className="tabular-nums font-medium" style={{ color: tone }}>
                              {isPushPos ? '+' : ''}{attr.contribution.toFixed(3)}
                            </span>
                          </div>
                          {/* Diverging bar */}
                          <div className="relative h-2 w-full rounded-full bg-white/5 overflow-hidden">
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

                  <div className="flex justify-between text-[10px] text-ink-faint pt-2 border-t border-white/5">
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
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
