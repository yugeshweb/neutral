import { useState } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
} from '../../lib/diseaseRegistry'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { RocChart } from '../charts'
import { IconCheck } from '../icons'

export function BenchmarkTab() {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('breast-cancer')
  const disease = getDiseasePipeline(selectedDiseaseId)

  const classical = disease.classicalModel
  const quantum = disease.quantumModel

  const metricsList = [
    { key: 'accuracy', label: 'Accuracy', desc: 'Overall correct prediction rate' },
    { key: 'precision', label: 'Precision', desc: 'Positive predictive value' },
    { key: 'sensitivity', label: 'Sensitivity / Recall', desc: 'True positive rate (catching positive cases)' },
    { key: 'specificity', label: 'Specificity', desc: 'True negative rate (clearing healthy cases)' },
    { key: 'f1', label: 'F1 Score', desc: 'Harmonic balance of precision & sensitivity' },
    { key: 'rocAuc', label: 'AUC-ROC', desc: 'Area under ROC curve' },
    { key: 'trainingTime', label: 'Training Time', desc: 'Wall-clock convergence duration' },
    { key: 'inferenceTime', label: 'Inference Latency', desc: 'Single-case prediction time' },
  ] as const

  const rocCurves = [
    {
      label: `Classical (${classical.name.split('(')[0].trim()})`,
      color: LANE_COLOR.classical,
      points: classical.rocPoints,
      auc: classical.metrics.rocAuc,
    },
    {
      label: `Quantum (${quantum.name.split('(')[0].trim()})`,
      color: LANE_COLOR.quantum,
      points: quantum.rocPoints,
      auc: quantum.metrics.rocAuc,
    },
  ]

  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-[1400px] px-6 lg:px-10 py-7 space-y-6">
        {/* Disease Pipeline Selector Cards */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
          {DISEASE_PIPELINES.map((d) => {
            const active = d.id === selectedDiseaseId
            return (
              <button
                key={d.id}
                type="button"
                onClick={() => setSelectedDiseaseId(d.id)}
                className="group relative flex flex-col rounded-[10px] p-4 text-left transition-all duration-150 cursor-pointer"
                style={{
                  background: active ? '#1E2025' : '#141518',
                  border: `1px solid ${active ? alpha(LANE_COLOR.quantum, 0.55) : 'rgba(255,255,255,0.06)'}`,
                  boxShadow: active
                    ? `0 0 0 1px ${alpha(LANE_COLOR.quantum, 0.25)}, 0 6px 20px rgba(0,0,0,0.6)`
                    : 'none',
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    {d.categoryLabel}
                  </span>
                  {active && (
                    <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />
                  )}
                </div>
                <div className="mt-1.5 text-[15px] font-medium text-ink group-hover:text-white">
                  {d.name}
                </div>
                <div className="mt-1 text-[11.5px] text-ink-dim line-clamp-1">{d.tagline}</div>
                <div className="mt-3 flex items-center justify-between font-mono text-[10px] text-ink-faint border-t border-white/5 pt-2">
                  <span>{d.modality}</span>
                  <span>{d.totalSamples} records</span>
                </div>
              </button>
            )
          })}
        </section>

        {/* Dataset Used & Specifications in Points */}
        <section
          className="rounded-panel p-5 space-y-4"
          style={{
            background: '#16171A',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 border-b border-white/5 pb-3">
            <div>
              <h2 className="text-[17px] font-medium text-ink flex items-center gap-2.5">
                <span>{disease.name}</span>
                <span className="rounded bg-white/6 px-2 py-0.5 font-mono text-[10px] text-ink-dim">
                  {disease.categoryLabel}
                </span>
              </h2>
              <p className="mt-1 text-[12px] text-ink-dim">
                Target Condition: <span className="text-ink font-medium">{disease.targetCondition}</span> ({disease.positiveLabel} vs {disease.negativeLabel})
              </p>
            </div>
            <div className="font-mono text-[11px] text-ink-dim">
              Dataset: <span className="text-ink font-medium">{disease.datasetName}</span> ({disease.datasetSource})
            </div>
          </div>

          {/* Specifications in Clean Points */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-[11px]">
            <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-2">
              <div className="text-ink-faint uppercase text-[9.5px] tracking-wider">Dataset Specifications</div>
              <ul className="space-y-1.5 text-ink-dim text-[11px] list-disc pl-4">
                <li><span className="text-ink">Total Cohort:</span> {disease.totalSamples} verified patient records</li>
                <li><span className="text-ink">Input Dimensionality:</span> {disease.inputDimensionality}</li>
                <li><span className="text-ink">Modality:</span> {disease.modality}</li>
                <li><span className="text-ink">Validation Split:</span> 70% Train / 15% Validation / 15% Holdout</li>
              </ul>
            </div>

            <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-2">
              <div className="text-ink-faint uppercase text-[9.5px] tracking-wider" style={{ color: LANE_COLOR.classical }}>
                Classical Baseline Specs
              </div>
              <ul className="space-y-1.5 text-ink-dim text-[11px] list-disc pl-4">
                <li><span className="text-ink">Architecture:</span> {classical.name}</li>
                <li><span className="text-ink">Configuration:</span> {classical.parameters}</li>
                <li><span className="text-ink">Hardware:</span> {classical.hardware}</li>
                <li><span className="text-ink">Rationale:</span> {classical.rationale}</li>
              </ul>
            </div>

            <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-2">
              <div className="text-ink-faint uppercase text-[9.5px] tracking-wider" style={{ color: LANE_COLOR.quantum }}>
                Hybrid Quantum Model Specs
              </div>
              <ul className="space-y-1.5 text-ink-dim text-[11px] list-disc pl-4">
                <li><span className="text-ink">Circuit:</span> {quantum.name}</li>
                <li><span className="text-ink">Quantum Specs:</span> {quantum.parameters}</li>
                <li><span className="text-ink">Reduced Features:</span> {disease.reducedDimensionality}</li>
                <li><span className="text-ink">Rationale:</span> {quantum.rationale}</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Side-by-Side Vertical Metrics Table & ROC Overlay */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Vertical Metrics Comparison Table (7 cols) */}
          <div
            className="lg:col-span-7 rounded-panel p-5"
            style={{
              background: '#16171A',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div className="mb-3.5 flex items-baseline justify-between">
              <h3 className="font-mono text-[12px] font-medium uppercase tracking-wider text-ink-faint">
                Vertical Evaluation Metrics Comparison
              </h3>
              <div className="flex items-center gap-4 font-mono text-[11px]">
                <span className="flex items-center gap-1.5" style={{ color: LANE_COLOR.classical }}>
                  <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.classical }} />
                  Classical Baseline
                </span>
                <span className="flex items-center gap-1.5" style={{ color: LANE_COLOR.quantum }}>
                  <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />
                  Hybrid Quantum
                </span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-[12px]">
                <thead>
                  <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-ink-faint">
                    <th className="py-2.5 font-medium">Evaluation Metric</th>
                    <th className="py-2.5 text-right font-medium">Classical</th>
                    <th className="py-2.5 text-right font-medium">Quantum</th>
                    <th className="py-2.5 text-right font-medium">Difference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {metricsList.map((m) => {
                    const cVal = classical.metrics[m.key]
                    const qVal = quantum.metrics[m.key]
                    const isNumeric = typeof cVal === 'number' && typeof qVal === 'number'
                    const delta = isNumeric ? (qVal as number) - (cVal as number) : null
                    const quantumWins = delta !== null && delta > 0.001
                    const classicalWins = delta !== null && delta < -0.001

                    return (
                      <tr key={m.key} className="hover:bg-white/[0.02]">
                        <td className="py-3">
                          <div className="font-medium text-ink">{m.label}</div>
                          <div className="text-[10px] text-ink-faint font-sans">{m.desc}</div>
                        </td>
                        <td className="py-3 text-right text-ink-dim tabular-nums">
                          {isNumeric ? (cVal as number).toFixed(3) : cVal}
                        </td>
                        <td
                          className="py-3 text-right font-medium tabular-nums"
                          style={{
                            color: quantumWins
                              ? LANE_COLOR.quantum
                              : classicalWins
                              ? '#9A9CA1'
                              : '#E8E9EB',
                          }}
                        >
                          {isNumeric ? (qVal as number).toFixed(3) : qVal}
                        </td>
                        <td className="py-3 text-right tabular-nums">
                          {delta !== null ? (
                            <span
                              className="rounded px-2 py-0.5 text-[10.5px] font-medium"
                              style={{
                                background:
                                  delta > 0
                                    ? alpha('#5FA88C', 0.15)
                                    : delta < 0
                                    ? alpha('#A3543D', 0.15)
                                    : 'transparent',
                                color: delta > 0 ? '#5FA88C' : delta < 0 ? '#A3543D' : '#6A6C72',
                              }}
                            >
                              {delta >= 0 ? '+' : ''}
                              {delta.toFixed(3)}
                            </span>
                          ) : (
                            <span className="text-ink-faint">—</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* ROC Curves Overlay (5 cols) */}
          <div
            className="lg:col-span-5 rounded-panel p-5 flex flex-col justify-between"
            style={{
              background: '#16171A',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <h3 className="font-mono text-[12px] font-medium uppercase tracking-wider text-ink-faint">
                  ROC Curve Discrimination
                </h3>
                <span className="font-mono text-[10px] text-ink-faint">Holdout Validation</span>
              </div>
              <div className="py-2">
                <RocChart curves={rocCurves} size={260} />
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-white/5 font-mono text-[10.5px] text-ink-dim flex justify-between">
              <span>Classical AUC: <span className="text-ink font-medium">{classical.metrics.rocAuc.toFixed(3)}</span></span>
              <span>Quantum AUC: <span className="text-ink font-medium" style={{ color: LANE_COLOR.quantum }}>{quantum.metrics.rocAuc.toFixed(3)}</span></span>
            </div>
          </div>
        </div>

        {/* Clean Conclusion Box at End */}
        <section
          className="rounded-panel p-5"
          style={{
            background: '#181A1E',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          <div className="flex items-start gap-3.5">
            <span
              className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-[6px] text-black font-bold text-[12px]"
              style={{ background: '#5FA88C' }}
            >
              <IconCheck className="h-3.5 w-3.5" />
            </span>
            <div className="space-y-1.5 flex-1">
              <h3 className="text-[14px] font-medium text-ink">
                Conclusion: {disease.honestCallout.title}
              </h3>
              <p className="text-[12.5px] leading-relaxed text-ink-dim">
                {disease.honestCallout.summary} {disease.honestCallout.nuance}
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
