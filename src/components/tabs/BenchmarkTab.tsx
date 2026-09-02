import { useState } from 'react'
import { CONDITION_VECTOR, VecLesion } from '../vectors'
import { getDiseasePipeline } from '../../lib/diseaseRegistry'
import { INTAKE_DISEASE_IDS } from '../../lib/intakeSpec'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { RocChart } from '../charts'
import { IconCheck, IconTree } from '../icons'

/*
 * The conditions this tab benchmarks: the same four Train and Predict offer.
 * Driven off the shared list rather than the whole registry, so adding a
 * condition in one place cannot leave the tabs disagreeing about coverage.
 */
const BENCHMARK_PIPELINES = INTAKE_DISEASE_IDS.map(getDiseasePipeline)

export function BenchmarkTab() {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('breast-cancer')
  const disease = getDiseasePipeline(selectedDiseaseId)

  const classical = disease.classicalModel
  const quantum = disease.quantumModel

  const metricsList = [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'precision', label: 'Precision' },
    { key: 'sensitivity', label: 'Sensitivity / Recall' },
    { key: 'specificity', label: 'Specificity' },
    { key: 'f1', label: 'F1 Score' },
    { key: 'rocAuc', label: 'AUC-ROC' },
    { key: 'trainingTime', label: 'Training Time' },
    { key: 'inferenceTime', label: 'Inference Latency' },
  ] as const

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
          <div className="mx-auto max-w-[780px] text-center">
            <h1 className="flex items-center justify-center gap-2 text-[26px] font-medium tracking-[-0.02em] text-ink">
              Pre-computed Benchmarks
              <InfoDot label="About these benchmarks">
                Pre-computed evaluations across the platform's disease pipelines.
                Each runs a classical baseline and a hybrid variational quantum
                model on the same split, so the comparison is like for like.
              </InfoDot>
            </h1>
            <p className="mt-1.5 font-mono text-[12.5px] text-ink-faint">
              Reference Evaluation Registry · Dual-Lane Quantum/Classical
            </p>
          </div>

          {/* Disease Coverage Badges.

              The same four conditions Train and Predict offer, so the tabs do
              not disagree about which pipelines the platform covers. */}
          <div className="mt-5 grid grid-cols-1 gap-3 border-t border-white/5 pt-4 sm:grid-cols-2 lg:grid-cols-4">
            {BENCHMARK_PIPELINES.map((d) => {
              const active = d.id === selectedDiseaseId
              const Vec = CONDITION_VECTOR[d.id] ?? VecLesion
              return (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setSelectedDiseaseId(d.id)}
                  data-pressed={active}
                  title={d.tagline}
                  className="tile group relative flex cursor-pointer flex-col p-3.5 text-left"
                  style={{ ['--tile-accent' as string]: LANE_COLOR.quantum }}
                >
                  {/* A shallow plate rather than the full square used in Train
                      and Predict: this card carries metrics below it, so the
                      artwork gets a strip across the top instead of the whole
                      tile. Still the same drawing, same flowing paint. */}
                  <div className="tile-art h-14 w-full">
                    <Vec size={40} accent={LANE_COLOR.quantum} />
                  </div>
                  {/* `shortName`, matching the pickers on Train and Predict.
                      The full name is still available as the button's tooltip. */}
                  <div className="mt-2.5 truncate text-[15px] font-medium text-ink group-hover:text-white">
                    {d.shortName}
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
                  <div className="tile-muted mt-2 flex items-center justify-between font-mono text-[11px]">
                    <span className="truncate">{d.modality}</span>
                    <span className="shrink-0">{d.totalSamples} samples</span>
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {/* The pipeline detail that used to run out as a full sentence
            (target, input dimensionality, reduction, dataset and source) now
            sits behind the info dot: the two numbers worth seeing at a glance
            are the qubit count and the sample size, both already on the
            condition card above. */}
        <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-3">
          <div className="flex items-center gap-2">
            <h2 className="text-[19px] font-medium text-ink">{disease.name}</h2>
            <span className="rounded-[4px] bg-white/5 px-2 py-0.5 font-mono text-[11.5px] text-ink-dim">
              {disease.categoryLabel}
            </span>
          </div>
          <InfoDot label="Pipeline detail">
            <div className="space-y-1">
              <p>Target: {disease.targetCondition}</p>
              <p>
                Input: {disease.inputDimensionality} → Reduced: {disease.reducedDimensionality}
              </p>
              <p>
                Dataset: {disease.datasetName} ({disease.datasetSource})
              </p>
            </div>
          </InfoDot>
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

        {/* Side-by-Side Unified Metrics Table & ROC Overlay.

            The table used to carry a description under every metric name
            ("Overall correct classifications" under "Accuracy" and so on) and
            sat beside a bar chart repeating five of its eight rows in a second
            form. Both were the same information said twice. The descriptions
            moved to one info dot on the header; the bar chart is gone, since
            the table already gives classical, quantum and the delta at a
            glance, and the ROC curve is the one comparison the table cannot
            show. */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Metrics Table (8 cols) */}
          <div className="lg:col-span-8 panel-raised rounded-panel panel-pad">
            <div className="mb-3 flex items-baseline justify-between">
              <div className="flex items-center gap-2">
                <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                  Evaluation Metrics
                </h3>
                <InfoDot label="What each metric means">
                  <div className="space-y-1.5">
                    <p>Accuracy: overall correct classifications.</p>
                    <p>Precision: positive predictive value.</p>
                    <p>Sensitivity / Recall: true positive rate, critical for misses.</p>
                    <p>Specificity: true negative rate, false alarm avoidance.</p>
                    <p>F1: harmonic mean of precision and sensitivity.</p>
                    <p>AUC-ROC: threshold-independent discrimination quality.</p>
                  </div>
                </InfoDot>
              </div>
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
                        <td className="py-2.5 font-medium text-ink">{m.label}</td>
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

          {/* ROC Curve Overlay (4 cols) */}
          <div className="lg:col-span-4 panel-raised rounded-panel panel-pad">
            <div className="mb-2 flex items-baseline justify-between">
              <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                ROC Curve Overlay
              </h3>
              <span className="font-mono text-[11px] text-ink-faint">AUC</span>
            </div>
            <RocChart curves={rocCurves} size={220} />
          </div>
        </div>

        {/* Honest Callout.

            Was a full-sentence summary, two bulleted lists, and a footnote
            paragraph under a redundant "Empirical Framing" chip that just
            restated the section's own point. The chip is gone, the summary is
            cut to its claim, and the nuance paragraph - caveats on reading the
            comparison, not a strength of either lane - moved into the info
            dot rather than sitting on the page as a third paragraph. */}
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
                <h3 className="text-[15.5px] font-medium text-ink">{disease.honestCallout.title}</h3>
                <InfoDot label="Reading this comparison">{disease.honestCallout.nuance}</InfoDot>
              </div>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-dim">
                {disease.honestCallout.summary}
              </p>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="panel-well well-pad rounded-[8px]">
                  <div className="font-mono text-[12px] text-[#3E8C9E] mb-2 flex items-center gap-1.5">
                    <IconCheck className="h-3.5 w-3.5" /> Quantum strengths
                  </div>
                  <ul className="space-y-1.5 text-[13px] text-ink-dim list-disc pl-4">
                    {disease.honestCallout.quantumPros.map((pro, i) => (
                      <li key={i}>{pro}</li>
                    ))}
                  </ul>
                </div>

                <div className="panel-well well-pad rounded-[8px]">
                  <div className="font-mono text-[12px] text-[#C08A3E] mb-2 flex items-center gap-1.5">
                    <IconTree className="h-3.5 w-3.5" /> Classical strengths
                  </div>
                  <ul className="space-y-1.5 text-[13px] text-ink-dim list-disc pl-4">
                    {disease.honestCallout.classicalPros.map((pro, i) => (
                      <li key={i}>{pro}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
