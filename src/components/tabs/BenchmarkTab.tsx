import { useState } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
} from '../../lib/diseaseRegistry'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { RocChart } from '../charts'
import { DemoChip } from '../DemoChip'
import { IconCheck, IconPulse, IconTree } from '../icons'

export function BenchmarkTab() {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('breast-cancer')
  const disease = getDiseasePipeline(selectedDiseaseId)

  const classical = disease.classicalModel
  const quantum = disease.quantumModel

  const metricsList = [
    { key: 'accuracy', label: 'Accuracy', desc: 'Overall correct classifications' },
    { key: 'precision', label: 'Precision', desc: 'Positive predictive value' },
    { key: 'sensitivity', label: 'Sensitivity / Recall', desc: 'True positive rate (critical for misses)' },
    { key: 'specificity', label: 'Specificity', desc: 'True negative rate (false alarm avoidance)' },
    { key: 'f1', label: 'F1 Score', desc: 'Harmonic mean of precision and sensitivity' },
    { key: 'rocAuc', label: 'AUC-ROC', desc: 'Threshold-independent discrimination quality' },
    { key: 'trainingTime', label: 'Training Time', desc: 'Wall-clock time to converge' },
    { key: 'inferenceTime', label: 'Inference Latency', desc: 'Single-case prediction latency' },
  ] as const

  const keyComparisonMetrics = [
    { label: 'Accuracy', classical: classical.metrics.accuracy, quantum: quantum.metrics.accuracy },
    { label: 'Sensitivity', classical: classical.metrics.sensitivity, quantum: quantum.metrics.sensitivity },
    { label: 'Specificity', classical: classical.metrics.specificity, quantum: quantum.metrics.specificity },
    { label: 'Precision', classical: classical.metrics.precision, quantum: quantum.metrics.precision },
    { label: 'F1 Score', classical: classical.metrics.f1, quantum: quantum.metrics.f1 },
  ]

  const rocCurves = [
    {
      label: classical.name,
      color: LANE_COLOR.classical,
      points: classical.rocPoints,
      auc: classical.metrics.rocAuc,
    },
    {
      label: quantum.name,
      color: LANE_COLOR.quantum,
      points: quantum.rocPoints,
      auc: quantum.metrics.rocAuc,
    },
  ]

  return (
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen">
        {/* Top Platform Overview Header */}
        <section className="panel-raised rounded-panel panel-pad">
          <div className="flex items-start justify-between gap-6">
            <div className="max-w-[780px]">
              <div className="flex items-center gap-2.5">
                <span className="inline-flex items-center gap-1.5 rounded-[4px] bg-[#3E8C9E]/15 border border-[#3E8C9E]/30 px-2 py-0.5 font-mono text-[12px] text-[#3E8C9E]">
                  <IconPulse className="h-3 w-3" /> Pre-computed Benchmarks
                </span>
                <span className="font-mono text-[12px] text-ink-faint">
                  Reference Evaluation Registry · Dual-Lane Quantum/Classical
                </span>
              </div>
              <h1 className="mt-2.5 flex items-center gap-2 text-[26px] font-medium tracking-[-0.02em] text-ink">
                Benchmarks
                <InfoDot label="About these benchmarks">
                  Pre-computed evaluations across the platform's disease pipelines.
                  Each runs a classical baseline and a hybrid variational quantum
                  model on the same split, so the comparison is like for like.
                </InfoDot>
              </h1>
            </div>
            <div className="hidden sm:flex flex-col items-end gap-2 shrink-0">
              <DemoChip />
              <div className="readout px-3 py-2 text-right">
                <div className="font-mono text-[13px] text-ink">3 Core Disease Pipelines</div>
                <div className="font-mono text-[11.5px] text-ink-faint">Tabular · EEG Biosignals · Hemodynamic</div>
              </div>
            </div>
          </div>

          {/* Disease Coverage Badges */}
          <div className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-3 pt-4 border-t border-white/5">
            {DISEASE_PIPELINES.map((d) => {
              const active = d.id === selectedDiseaseId
              return (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setSelectedDiseaseId(d.id)}
                  data-pressed={active}
                  title={d.tagline}
                  className="key group relative flex cursor-pointer flex-col rounded-[8px] p-3.5 text-left"
                  style={
                    active
                      ? { borderColor: alpha(LANE_COLOR.quantum, 0.5) }
                      : undefined
                  }
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px] text-ink-faint">
                      {d.categoryLabel}
                    </span>
                    {active && (
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: LANE_COLOR.quantum }} />
                    )}
                  </div>
                  <div className="mt-1 text-[15px] font-medium text-ink group-hover:text-white">
                    {d.name}
                  </div>
                  {/*
                    * The tagline used to sit here under `line-clamp-1`, which
                    * truncated all six mid-word with no way to read the rest.
                    * Since none of them said much a title and a metric do not,
                    * the space now carries the accuracy of each lane, which is
                    * the number you actually pick a benchmark on. The full
                    * tagline is still available as the button's tooltip.
                    */}
                  <div className="mt-2 flex items-baseline gap-3 font-mono text-[11px]">
                    <span className="text-ink-faint">
                      classical{' '}
                      <span className="tabular-nums" style={{ color: LANE_COLOR.classical }}>
                        {d.classicalModel.metrics.accuracy.toFixed(3)}
                      </span>
                    </span>
                    <span className="text-ink-faint">
                      quantum{' '}
                      <span className="tabular-nums" style={{ color: LANE_COLOR.quantum }}>
                        {d.quantumModel.metrics.accuracy.toFixed(3)}
                      </span>
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between font-mono text-[11px] text-ink-faint">
                    <span className="truncate">{d.modality}</span>
                    <span className="shrink-0">{d.totalSamples} samples</span>
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {/*
         * Laid out on the same two-column grid as the model cards below, so
         * the description wraps at the left card's edge instead of running the
         * full width of the screen and leaving a long unbroken line.
         */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 border-b border-white/5 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-[19px] font-medium text-ink">{disease.name}</h2>
              <span className="rounded-[4px] bg-white/5 px-2 py-0.5 font-mono text-[11.5px] text-ink-dim">
                {disease.categoryLabel}
              </span>
            </div>
            <p className="mt-1 font-mono text-[12.5px] leading-relaxed text-ink-faint">
              Target: <span className="text-ink-dim">{disease.targetCondition}</span> · Input:{' '}
              <span className="text-ink-dim">{disease.inputDimensionality}</span> → Reduced:{' '}
              <span className="text-ink-dim">{disease.reducedDimensionality}</span>
            </p>
          </div>
          <div className="font-mono text-[12px] leading-relaxed text-ink-faint lg:text-right">
            Dataset: <span className="text-ink">{disease.datasetName}</span> ({disease.datasetSource})
          </div>
        </div>

        {/* Architectures and Rationale Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="panel-raised rounded-panel panel-pad">
            <div className="mb-3 flex items-baseline justify-between gap-2">
              <div>
                <div className="engraved font-mono text-[11.5px]">
                  Classical baseline
                </div>
                <h3 className="mt-0.5 text-[14.5px] font-medium text-ink">{classical.name}</h3>
              </div>
              <InfoDot label="About the classical baseline">
                <div className="space-y-2">
                  <p>{classical.description}</p>
                  <p>{classical.rationale}</p>
                </div>
              </InfoDot>
            </div>
            <div className="panel-well well-pad flex justify-between rounded-[6px] font-mono text-[12px]">
              <span className="text-ink-faint">Parameters</span>
              <span className="text-ink">{classical.parameters}</span>
            </div>
          </div>

          <div className="panel-raised rounded-panel panel-pad">
            <div className="mb-3 flex items-baseline justify-between gap-2">
              <div>
                <div className="engraved font-mono text-[11.5px]">
                  Hybrid quantum
                </div>
                <h3 className="mt-0.5 text-[14.5px] font-medium text-ink">{quantum.name}</h3>
              </div>
              <InfoDot label="About the quantum model">
                <div className="space-y-2">
                  <p>{quantum.description}</p>
                  <p>{quantum.rationale}</p>
                </div>
              </InfoDot>
            </div>
            <div className="panel-well well-pad flex justify-between rounded-[6px] font-mono text-[12px]">
              <span className="text-ink-faint">Circuit</span>
                <span style={{ color: LANE_COLOR.quantum }}>{quantum.parameters}</span>
            </div>
          </div>
        </div>

        {/* Side-by-Side Unified Metrics Table & Visual Comparisons */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Metrics Table (7 cols) */}
          <div className="lg:col-span-7 panel-raised rounded-panel panel-pad">
            <div className="mb-3 flex items-baseline justify-between">
              <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                Unified Evaluation Metrics Comparison
              </h3>
              <div className="flex items-center gap-4 font-mono text-[12px]">
                <span className="flex items-center gap-1.5" style={{ color: LANE_COLOR.classical }}>
                  <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.classical }} />
                  Classical
                </span>
                <span className="flex items-center gap-1.5" style={{ color: LANE_COLOR.quantum }}>
                  <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />
                  Quantum
                </span>
              </div>
            </div>

            <div className="console-scroll overflow-x-auto">
              <table className="w-full text-left font-mono text-[13px]">
                <thead>
                  <tr className="border-b border-white/10 text-[11.5px] text-ink-faint">
                    <th className="py-2 font-normal">Metric</th>
                    <th className="py-2 text-right font-normal">Classical</th>
                    <th className="py-2 text-right font-normal">Quantum</th>
                    <th className="py-2 text-right font-normal">Delta (Q - C)</th>
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
                        <td className="py-2.5">
                          <div className="font-medium text-ink">{m.label}</div>
                          <div className="text-[11px] text-ink-faint font-sans">{m.desc}</div>
                        </td>
                        <td className="py-2.5 text-right text-ink-dim tabular-nums">
                          {isNumeric ? (cVal as number).toFixed(3) : cVal}
                        </td>
                        <td
                          className="py-2.5 text-right font-medium tabular-nums"
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
                        <td className="py-2.5 text-right tabular-nums">
                          {delta !== null ? (
                            <span
                              className="rounded px-1.5 py-0.5 text-[12px]"
                              style={{
                                background:
                                  delta > 0
                                    ? alpha('#3E8C9E', 0.15)
                                    : delta < 0
                                    ? alpha('#A3543D', 0.15)
                                    : 'transparent',
                                color: delta > 0 ? '#3E8C9E' : delta < 0 ? '#A3543D' : '#6A6C72',
                              }}
                            >
                              {delta >= 0 ? '+' : ''}
                              {delta.toFixed(3)}
                            </span>
                          ) : (
                            <span className="text-ink-faint">n/a</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts (5 cols): ROC Overlay + Paired Bars */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            {/* ROC Curve Overlay */}
            <div className="panel-raised rounded-panel panel-pad">
              <div className="mb-2 flex items-baseline justify-between">
                <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                  ROC Curve Overlay
                </h3>
                <span className="font-mono text-[11px] text-ink-faint">AUC Comparison</span>
              </div>
              <RocChart curves={rocCurves} size={220} />
            </div>

            {/* Paired Bar Comparisons */}
            <div className="panel-raised rounded-panel panel-pad">
              <div className="mb-3 flex items-baseline justify-between">
                <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                  Key Metrics Comparison
                </h3>
                <span className="font-mono text-[11px] text-ink-faint">0.0 → 1.0</span>
              </div>
              <div className="space-y-3">
                {keyComparisonMetrics.map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between font-mono text-[12px] text-ink-dim mb-1">
                      <span>{item.label}</span>
                      <span>
                        <span style={{ color: LANE_COLOR.classical }}>
                          {item.classical.toFixed(3)}
                        </span>{' '}
                        vs{' '}
                        <span style={{ color: LANE_COLOR.quantum }}>
                          {item.quantum.toFixed(3)}
                        </span>
                      </span>
                    </div>
                    <div className="space-y-1">
                      {/* Classical bar */}
                      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${item.classical * 100}%`,
                            background: alpha(LANE_COLOR.classical, 0.8),
                          }}
                        />
                      </div>
                      {/* Quantum bar */}
                      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${item.quantum * 100}%`,
                            background: alpha(LANE_COLOR.quantum, 0.85),
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Honest Callout & Assessment Box */}
        <section
          className="rounded-panel panel-pad"
          style={{
            background: '#181A1E',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          <div className="flex items-start gap-3.5">
            <span
              className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-[6px] text-black font-bold text-[14.5px]"
              style={{ background: '#C08A3E' }}
            >
              !
            </span>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h3 className="text-[15.5px] font-medium text-ink">
                  Rigorous & Honest Quantum Advantage Analysis
                </h3>
                <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[11px] text-ink-faint">
                  Empirical Framing
                </span>
              </div>
              <p className="mt-1 text-[14px] leading-relaxed text-ink">
                {disease.honestCallout.title}: {disease.honestCallout.summary}
              </p>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="panel-well well-pad rounded-[8px]">
                  <div className="font-mono text-[12px] text-[#3E8C9E] mb-2 flex items-center gap-1.5">
                    <IconCheck className="h-3.5 w-3.5" /> Quantum Model Strengths
                  </div>
                  <ul className="space-y-1.5 text-[13px] text-ink-dim list-disc pl-4">
                    {disease.honestCallout.quantumPros.map((pro, i) => (
                      <li key={i}>{pro}</li>
                    ))}
                  </ul>
                </div>

                <div className="panel-well well-pad rounded-[8px]">
                  <div className="font-mono text-[12px] text-[#C08A3E] mb-2 flex items-center gap-1.5">
                    <IconTree className="h-3.5 w-3.5" /> Classical Model Strengths & Latency
                  </div>
                  <ul className="space-y-1.5 text-[13px] text-ink-dim list-disc pl-4">
                    {disease.honestCallout.classicalPros.map((pro, i) => (
                      <li key={i}>{pro}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <p className="mt-3.5 font-mono text-[11.5px] leading-relaxed text-ink-faint pt-2.5 border-t border-white/5">
                {disease.honestCallout.nuance}
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
