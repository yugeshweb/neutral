import { useMemo, useRef, useState } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
  loadTrainedPipeline,
} from '../../lib/diseaseRegistry'
import { parseCsv, splitRow } from '../../lib/dataset'
import {
  isReplayable,
  scoreBatch,
  scoreCase,
  type BatchResult,
} from '../../lib/ml/inference'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { IconCircuit, IconFlask, IconReset, IconUpload } from '../icons'

/**
 * Inference and explainability.
 *
 * Every number on this screen comes from the circuit trained in the Train tab:
 * the saved parameters are reloaded, the saved preprocessing is replayed, and
 * the same model scores the case. When no model has been trained for the
 * selected condition the screen says so and offers no prediction, rather than
 * substituting an approximation that would look identical to a real one.
 */

// The palette is the four lane hues and nothing else.
const QUANTUM = LANE_COLOR.quantum // teal - the model and its output
const CLASSICAL = LANE_COLOR.classical // amber - warnings, elevated risk
const NEUTRAL = LANE_COLOR.shared // grey - structure and labels

export function PredictTab({
  initialDiseaseId = 'breast-cancer',
}: {
  initialDiseaseId?: string
}) {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>(initialDiseaseId)
  const disease = getDiseasePipeline(selectedDiseaseId)

  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {}
    Object.entries(disease.featureRanges).forEach(([key, spec]) => {
      initial[key] = spec.defaultVal
    })
    return initial
  })

  const handleDiseaseChange = (id: string) => {
    setSelectedDiseaseId(id)
    const target = getDiseasePipeline(id)
    const next: Record<string, number> = {}
    Object.entries(target.featureRanges).forEach(([key, spec]) => {
      next[key] = spec.defaultVal
    })
    setFeatureValues(next)
  }

  const artifact = useMemo(() => loadTrainedPipeline(selectedDiseaseId), [selectedDiseaseId])
  const ready = isReplayable(artifact)

  // Batch scoring from an uploaded table.
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [batch, setBatch] = useState<{ file: string; result: BatchResult } | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !ready) return
    setBatchError(null)

    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string
        const summary = parseCsv(text, file.name, file.size)

        // `summary.preview` is only the first few rows; score the whole file.
        const lines = (summary.content ?? text)
          .split(/\r\n|\n|\r/)
          .filter((l) => l.trim().length > 0)
        const rows = lines.slice(1).map(splitRow)

        const result = scoreBatch(artifact, summary.headers, rows)
        if (result.rows.length === 0) {
          setBatchError(
            `No row could be scored. The model needs columns named: ${artifact.featureNames
              .slice(0, 4)
              .join(', ')}${artifact.featureNames.length > 4 ? ', ...' : ''}`,
          )
          setBatch(null)
          return
        }
        setBatch({ file: file.name, result })
      } catch (err) {
        setBatchError(err instanceof Error ? err.message : 'could not read that file')
        setBatch(null)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  /**
   * Real inference. The slider panel is keyed by the disease's own feature
   * ranges, so values are mapped onto the artifact's feature order by name;
   * anything the trained model expects but this form does not expose is left
   * as NaN and filled by the artifact's stored imputation value.
   */
  const result = useMemo(() => {
    if (!ready) return null

    const raw = artifact.featureNames.map((name) => {
      const v = featureValues[name]
      return typeof v === 'number' ? v : Number.NaN
    })

    try {
      return scoreCase(artifact, raw)
    } catch {
      return null
    }
  }, [ready, artifact, featureValues])

  const covered = useMemo(() => {
    if (!ready) return { known: 0, total: 0 }
    const known = artifact.featureNames.filter(
      (n) => typeof featureValues[n] === 'number',
    ).length
    return { known, total: artifact.featureNames.length }
  }, [ready, artifact, featureValues])

  const positive = result ? result.probability >= 0.5 : false
  const tone = positive ? CLASSICAL : QUANTUM

  return (
    /*
     * The page itself does not scroll: the slider list is capped to the
     * viewport and scrolls internally, so only one scrollbar ever appears.
     * `overflow-y-auto` rather than `hidden` is a deliberate safety valve -
     * below roughly 620px of height the fixed chrome alone exceeds the window
     * and clipping would hide the score entirely, so in that case the page is
     * allowed to scroll rather than swallow content.
     */
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <h1 className="text-[19px] font-medium text-ink">Predict</h1>
          <InfoDot label="About this screen">
            Every number here comes from the circuit trained in the Train tab. Its
            saved parameters are reloaded and the same preprocessing is replayed. No
            prediction is shown for a condition that has not been trained.
          </InfoDot>
        </div>

        {/* Condition selector: physical keys */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {DISEASE_PIPELINES.map((d) => {
            const active = d.id === selectedDiseaseId
            return (
              <button
                key={d.id}
                type="button"
                data-pressed={active}
                onClick={() => handleDiseaseChange(d.id)}
                className="key cursor-pointer rounded-[8px] px-3.5 py-2.5 text-left"
              >
                <div
                  className="text-[14.5px] font-medium"
                  style={{ color: active ? '#E8E9EB' : '#9A9CA1' }}
                >
                  {d.name}
                </div>
              </button>
            )
          })}
        </div>

        {/* No trained model: state it plainly, predict nothing. */}
        {!ready && (
          <div className="panel-raised rounded-panel panel-pad text-center">
            <div
              className="mx-auto grid h-10 w-10 place-items-center rounded-full"
              style={{ background: alpha(NEUTRAL, 0.1) }}
            >
              <IconFlask className="h-5 w-5 text-ink-faint" />
            </div>
            <h2 className="mt-3 text-[15.5px] font-medium text-ink">No trained model yet</h2>
            <p className="mx-auto mt-1.5 max-w-[380px] text-[14px] text-ink-dim">
              Train {disease.name.toLowerCase()} first. Its model loads here
              automatically.
            </p>
          </div>
        )}

        {ready && result && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Case inputs */}
            <div className="lg:col-span-5 panel-raised rounded-panel panel-pad">
              <div className="flex items-baseline justify-between">
                <h2 className="text-[14.5px] font-medium text-ink">Case</h2>
                <button
                  type="button"
                  onClick={() => {
                    const reset: Record<string, number> = {}
                    Object.entries(disease.featureRanges).forEach(([k, s]) => {
                      reset[k] = s.defaultVal
                    })
                    setFeatureValues(reset)
                  }}
                  className="flex cursor-pointer items-center gap-1 font-mono text-[11.5px] text-ink-faint hover:text-ink"
                >
                  <IconReset className="h-3 w-3" /> Baseline
                </button>
              </div>

              {disease.samplePresets.length > 0 && (
                <div className="mt-3 grid grid-cols-2 gap-1.5">
                  {disease.samplePresets.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() =>
                        setFeatureValues((prev) => ({ ...prev, ...preset.values }))
                      }
                      className="key cursor-pointer rounded-[6px] px-2.5 py-1.5 text-left font-mono text-[12.5px] text-ink-dim hover:text-ink"
                    >
                      {preset.name}
                    </button>
                  ))}
                </div>
              )}

              {/*
                * The slider list is the only thing on this screen that scrolls.
                * It is capped relative to the viewport rather than a fixed
                * pixel height, so the whole page fits at any window size and
                * the browser never adds a second, outer scrollbar next to this
                * one. `min-h-0` is what lets a flex child actually shrink.
                */}
              <div className="console-scroll mt-3 max-h-[calc(100vh-30rem)] min-h-0 space-y-2 overflow-y-auto border-t border-white/5 pr-1.5 pt-3">
                {Object.entries(disease.featureRanges).map(([key, spec]) => {
                  const val = featureValues[key] ?? spec.defaultVal
                  return (
                    <div key={key} className="space-y-0.5">
                      <div className="flex justify-between font-mono text-[12px]">
                        <span className="truncate text-ink-dim" title={key}>
                          {key.replaceAll('_', ' ')}
                        </span>
                        <span className="tabular-nums text-ink">
                          {val.toFixed(spec.step < 0.01 ? 3 : spec.step < 1 ? 2 : 0)}
                          <span className="ml-1 text-[11px] text-ink-faint">{spec.unit}</span>
                        </span>
                      </div>
                      {/* Recessed channel with the shared skeuomorphic thumb
                          riding in it, rather than a bare native track. */}
                      <div className="relative flex h-[14px] items-center">
                        {/* At 3px the channel is too shallow for a shadow to
                            resolve, so it is simply a darker track. */}
                        <span className="absolute inset-x-0 h-[3px] rounded-full bg-black/45" />
                        <span
                          className="absolute h-[3px] rounded-full"
                          style={{
                            width: `${((val - spec.min) / Math.max(spec.max - spec.min, 1e-5)) * 100}%`,
                            background: alpha(QUANTUM, 0.55),
                          }}
                        />
                        <input
                          type="range"
                          aria-label={key.replaceAll('_', ' ')}
                          min={spec.min}
                          max={spec.max}
                          step={spec.step}
                          value={val}
                          onChange={(e) =>
                            setFeatureValues((prev) => ({
                              ...prev,
                              [key]: Number(e.target.value),
                            }))
                          }
                          className="feature-slider relative w-full cursor-pointer bg-transparent"
                        />
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Batch scoring: every row in the file goes through the same
                  trained model as the sliders above. */}
              {/* Upload control and its label share one row: the heading was
                  costing a line for two words. */}
              <div className="mt-3 flex items-center gap-2 border-t border-white/5 pt-3">
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv,.txt"
                  className="sr-only"
                  onChange={handleUpload}
                  id="predict-batch-file"
                />
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  className="key flex min-w-0 flex-1 cursor-pointer items-center justify-center gap-2 rounded-[6px] px-3 py-1.5 text-[12.5px] text-ink-dim hover:text-ink"
                >
                  <IconUpload className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">
                    {batch ? batch.file : 'Score a CSV file'}
                  </span>
                </button>
                <InfoDot label="About batch scoring">
                  Every row is scored with this same trained model. Columns are
                  matched to the model's features by name, so column order does not
                  matter and extra columns are ignored. A feature the file omits
                  falls back to its training-set average.
                </InfoDot>
              </div>
              <div>
                {batchError && (
                  <p className="mt-2 text-[12px] leading-relaxed" style={{ color: CLASSICAL }}>
                    {batchError}
                  </p>
                )}

                {batch && (
                  <div className="readout mt-2 px-3 py-2.5">
                    <div className="flex items-baseline justify-between font-mono text-[13px]">
                      <span className="text-ink-faint">scored</span>
                      <span className="tabular-nums text-ink">
                        {batch.result.rows.length} rows
                      </span>
                    </div>
                    <div className="mt-1 flex items-baseline justify-between font-mono text-[13px]">
                      <span className="text-ink-faint">
                        {disease.positiveLabel.toLowerCase()}
                      </span>
                      <span className="tabular-nums" style={{ color: CLASSICAL }}>
                        {batch.result.positiveCount}
                        <span className="ml-1 text-ink-faint">
                          (
                          {(
                            (batch.result.positiveCount / batch.result.rows.length) *
                            100
                          ).toFixed(1)}
                          %)
                        </span>
                      </span>
                    </div>

                    {/* State plainly when the file did not supply everything. */}
                    {batch.result.missing.length > 0 && (
                      <p className="mt-2 border-t border-white/5 pt-2 font-mono text-[11.5px] leading-relaxed text-ink-faint">
                        {batch.result.matched.length} of{' '}
                        {batch.result.matched.length + batch.result.missing.length}{' '}
                        features matched by name. Missing ones use their training
                        average, so these scores are weaker than a complete row.
                      </p>
                    )}
                    {batch.result.skipped > 0 && (
                      <p className="mt-1 font-mono text-[11.5px] text-ink-faint">
                        {batch.result.skipped} row(s) skipped, nothing numeric to read.
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Readout and explanation */}
            <div className="space-y-4 lg:col-span-7">
              <div className="panel-raised rounded-panel panel-pad">
                <div className="readout rounded-[8px] px-5 py-4">
                  <div className="engraved font-mono text-[11.5px]">
                    {disease.targetCondition}
                  </div>
                  <div className="mt-1.5 flex items-baseline gap-3">
                    <span
                      className="font-mono text-[46px] font-medium leading-none tabular-nums"
                      style={{ color: tone }}
                    >
                      {(result.probability * 100).toFixed(1)}%
                    </span>
                    <span
                      className="rounded-[4px] px-2 py-1 font-mono text-[13px]"
                      style={{
                        background: alpha(tone, 0.14),
                        border: `1px solid ${alpha(tone, 0.32)}`,
                        color: tone,
                      }}
                    >
                      {positive ? disease.positiveLabel : disease.negativeLabel}
                    </span>
                  </div>

                  <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-black/45">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${result.probability * 100}%`,
                        background: tone,
                      }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[11px] text-ink-faint">
                    <span>{disease.negativeLabel}</span>
                    <span>0.50</span>
                    <span>{disease.positiveLabel}</span>
                  </div>
                </div>

                {/* Provenance: exactly which model produced the number. */}
                <div className="mt-3 flex items-center justify-between font-mono text-[11.5px] text-ink-faint">
                  <span className="flex items-center gap-1.5">
                    <IconCircuit className="h-3 w-3" />
                    {artifact.quantumModelName}
                  </span>
                  <InfoDot label="Model provenance">
                    <div className="space-y-1">
                      <div>Qubits: {artifact.quantumConfig?.qubits}</div>
                      <div>Trained on: {artifact.rows} rows</div>
                      <div>Dataset: {artifact.datasetName}</div>
                      <div>
                        Inputs supplied: {covered.known} of {covered.total}; the rest use
                        their training-set fill value.
                      </div>
                    </div>
                  </InfoDot>
                </div>
              </div>

              {/* Attribution: occlusion against the training mean. */}
              <div className="panel-raised rounded-panel panel-pad">
                <div className="flex items-center justify-between">
                  <h3 className="text-[14.5px] font-medium text-ink">Why</h3>
                  <InfoDot label="How attribution is computed">
                    Each bar is the change in predicted probability when that input is
                    replaced by its training-set average, holding the rest fixed. Bars
                    to the right push toward {disease.positiveLabel.toLowerCase()}.
                  </InfoDot>
                </div>

                <div className="mt-4 space-y-2.5">
                  {result.attributions.slice(0, 6).map((attr) => {
                    const pushesPositive = attr.value > 0
                    const barTone = pushesPositive ? CLASSICAL : QUANTUM
                    const width = Math.min(100, Math.abs(attr.value) * 180)
                    return (
                      <div key={attr.name} className="space-y-1 font-mono text-[12px]">
                        <div className="flex items-baseline justify-between">
                          <span className="truncate text-ink-dim">
                            {attr.name.replaceAll('_', ' ')}
                          </span>
                          <span className="tabular-nums" style={{ color: barTone }}>
                            {pushesPositive ? '+' : ''}
                            {attr.value.toFixed(3)}
                          </span>
                        </div>
                        <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-black/45">
                          <span className="absolute inset-y-0 left-1/2 w-px bg-white/15" />
                          <div
                            className="absolute top-0 h-full rounded-full transition-all duration-300"
                            style={{
                              background: barTone,
                              width: `${width / 2}%`,
                              left: pushesPositive ? '50%' : `${50 - width / 2}%`,
                            }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>

              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
