import { useMemo, useRef, useState } from 'react'
import { getDiseasePipeline, loadTrainedPipeline } from '../../lib/diseaseRegistry'
import { splitRow } from '../../lib/dataset'
import { ingest } from '../../lib/ingest'
import { NotImplementedError } from '../../lib/ingest/types'
import { intakeFor, INTAKE_DISEASE_IDS, type IntakeField } from '../../lib/intakeSpec'
import { isReplayable, scoreBatch, type BatchResult } from '../../lib/ml/inference'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { ProcessingBay } from '../ProcessingBay'
import { PredictionResult } from '../PredictionResult'
import { IconArrowLeft, IconArrowRight, IconCheck, IconUpload } from '../icons'

/**
 * Inference intake.
 *
 * Two steps, in order: choose a condition, then supply data for it. The intake
 * fields are declared per condition in `lib/intakeSpec`, because what a seizure
 * model can read is not what a breast-cancer model can read, and showing the
 * same four boxes everywhere would imply otherwise.
 *
 * Scoring only ever happens through the model trained in the Train tab. A
 * format with no parser behind it says so and scores nothing, rather than
 * accepting the file and producing a number that looks identical to a real one.
 */

const QUANTUM = LANE_COLOR.quantum
const CLASSICAL = LANE_COLOR.classical

type FieldState =
  | { kind: 'idle' }
  | { kind: 'image'; file: string }
  | { kind: 'scored'; file: string; result: BatchResult }
  | { kind: 'parsed'; file: string; rows: number; columns: number }
  | { kind: 'unavailable'; file: string; requires: string }
  | { kind: 'error'; message: string }

export function PredictTab({
  initialDiseaseId = 'breast-cancer',
}: {
  initialDiseaseId?: string
}) {
  const [selectedId, setSelectedId] = useState<string>(
    INTAKE_DISEASE_IDS.includes(initialDiseaseId) ? initialDiseaseId : INTAKE_DISEASE_IDS[0],
  )
  const [states, setStates] = useState<Record<string, FieldState>>({})
  /*
   * One card at a time: choose the condition, then supply data for it.
   *
   * Always starts on the condition step. `initialDiseaseId` cannot be used to
   * infer "already chosen" because App defaults it to a real id, so testing it
   * would skip step one on every visit.
   */
  const [step, setStep] = useState<'condition' | 'intake' | 'processing' | 'result'>(
    'condition',
  )

  /*
   * What gets carried into the result screen. The image is kept separately
   * from the scoring result because they arrive from different fields: a scan
   * supplies pixels and no rows, a table supplies rows and no pixels, and the
   * result screen has to cope with either.
   */
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [lastFile, setLastFile] = useState<string>('')

  const disease = getDiseasePipeline(selectedId)
  const fields = intakeFor(selectedId)
  const artifact = useMemo(() => loadTrainedPipeline(selectedId), [selectedId])
  const trained = isReplayable(artifact)

  // Which conditions already have a model, for the badge on each card. Read
  // once per render rather than inside the map, which would hit localStorage
  // four times on every keystroke elsewhere in the tree.
  const trainedIds = useMemo(
    () =>
      new Set(
        INTAKE_DISEASE_IDS.filter((id) => isReplayable(loadTrainedPipeline(id))),
      ),
    [],
  )

  const setField = (id: string, s: FieldState) =>
    setStates((prev) => ({ ...prev, [id]: s }))

  // Something has to have been supplied before analysis means anything. An
  // unreadable or unsupported file still counts as supplied: the result screen
  // reports honestly on those rather than pretending nothing happened.
  const anyUpload = Object.values(states).some((s) => s.kind !== 'idle')

  // The first field that actually scored. Only one result is shown, because
  // scoring the same patient from two sources would need a merge policy that
  // does not exist here.
  const scoredResult =
    Object.values(states).find((s): s is Extract<FieldState, { kind: 'scored' }> =>
      s.kind === 'scored',
    )?.result ?? null

  const handleFile = async (field: IntakeField, file: File) => {
    // A format with no parser is reported as such, before the file is touched.
    if (!field.wired) {
      setLastFile(file.name)
      setField(field.id, {
        kind: 'unavailable',
        file: file.name,
        requires: field.requires ?? 'This input is not yet wired to a parser.',
      })
      return
    }

    try {
      const parsed = await ingest(file)
      const summary = parsed.dataset
      setLastFile(file.name)

      // A picture is displayed with the region overlay. It carries no rows, so
      // there is nothing to score - the result screen shows the image and says
      // no detector ran rather than pairing it with invented numbers.
      if (summary.kind === 'image' && summary.objectUrl) {
        setImageUrl(summary.objectUrl)
        setField(field.id, { kind: 'image', file: file.name })
        return
      }

      // Without a trained model there is nothing to score against, so the file
      // is reported as read and nothing more is claimed.
      if (!trained) {
        setField(field.id, {
          kind: 'parsed',
          file: file.name,
          rows: summary.rows,
          columns: summary.columns,
        })
        return
      }

      const lines = (summary.content ?? '')
        .split(/\r\n|\n|\r/)
        .filter((l) => l.trim().length > 0)
      const rows = lines.slice(1).map(splitRow)
      const result = scoreBatch(artifact, summary.headers, rows)

      if (result.rows.length === 0) {
        setField(field.id, {
          kind: 'parsed',
          file: file.name,
          rows: summary.rows,
          columns: summary.columns,
        })
        return
      }
      setField(field.id, { kind: 'scored', file: file.name, result })
    } catch (err) {
      if (err instanceof NotImplementedError) {
        setField(field.id, { kind: 'unavailable', file: file.name, requires: err.requires })
        return
      }
      setField(field.id, {
        kind: 'error',
        message: err instanceof Error ? err.message : 'could not read that file',
      })
    }
  }

  return (
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen">
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <h1 className="text-[19px] font-medium text-ink">Predict</h1>
          <InfoDot label="About this screen">
            Pick a condition, then supply data for it. Files are scored with the
            model trained in the Train tab. Inputs with no parser behind them say
            so and score nothing.
          </InfoDot>
        </div>

        {/* Step one: the condition. */}
        {step === 'condition' && (
          <div className="panel-raised rounded-panel panel-pad flow-step">
            <div className="flex items-baseline justify-between">
              <h2 className="text-[14.5px] font-medium text-ink">
                <span className="engraved mr-2 font-mono text-[12px]">1</span>
                Choose a condition
              </h2>
            </div>

            {/* The cards fill the card's height evenly, so the grid reads as a
                deliberate 2x2 rather than four controls floating at the top. */}
            <div className="flow-body mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {INTAKE_DISEASE_IDS.map((id) => {
                const d = getDiseasePipeline(id)
                const active = id === selectedId
                const trainedHere = trainedIds.has(id)
                const accent = active ? QUANTUM : 'rgba(255,255,255,0.14)'
                return (
                  <button
                    key={id}
                    type="button"
                    data-pressed={active}
                    onClick={() => {
                      setSelectedId(id)
                      setStates({})
                    }}
                    className="key flex cursor-pointer flex-col justify-between rounded-[8px] p-4 text-left"
                    style={active ? { borderColor: alpha(QUANTUM, 0.45) } : undefined}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <span
                          className="text-[15.5px] font-medium leading-snug"
                          style={{ color: active ? '#E8E9EB' : '#9A9CA1' }}
                        >
                          {d.name}
                        </span>
                        {/* Selected marker: a filled ring, so the choice is
                            legible without relying on the border alone. */}
                        <span
                          className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full"
                          style={{ border: `1.5px solid ${accent}` }}
                        >
                          {active && (
                            <span
                              className="h-1.5 w-1.5 rounded-full"
                              style={{ background: QUANTUM }}
                            />
                          )}
                        </span>
                      </div>
                      <div className="engraved mt-1.5 font-mono text-[11px]">
                        {d.categoryLabel}
                      </div>
                    </div>

                    <div className="mt-3 flex items-end justify-between gap-2 border-t border-white/5 pt-2.5 font-mono text-[11px]">
                      <span className="text-ink-faint">
                        {d.totalSamples.toLocaleString()} samples
                      </span>
                      <span
                        style={{ color: trainedHere ? QUANTUM : '#6A6C72' }}
                        title={
                          trainedHere
                            ? 'A model has been trained for this condition'
                            : 'Not trained yet: files will be read but not scored'
                        }
                      >
                        {trainedHere ? 'model ready' : 'not trained'}
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>

            <div className="mt-4 flex shrink-0 justify-end border-t border-white/5 pt-4">
              <button
                type="button"
                onClick={() => setStep('intake')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink hover:text-white"
              >
                Next
                <IconArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Step two: the inputs that condition accepts. */}
        {step === 'intake' && (
          <div className="panel-raised rounded-panel panel-pad flow-step">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-[14.5px] font-medium text-ink">
                <span className="engraved mr-2 font-mono text-[12px]">2</span>
                Data for {disease.name.toLowerCase()}
              </h2>
              <InfoDot label="About these inputs">
                These are the sources this condition's model can use. CSV, FHIR R4
                and HL7 v2 have working parsers. The others are part of the intended
                pipeline but are not built yet, and will tell you what they would
                take rather than accepting the file quietly.
              </InfoDot>
            </div>

            {!trained && (
              <p className="mt-2 text-[13px] text-ink-dim">
                No model has been trained for this condition yet, so files will be read
                and described but not scored. Train it first for predictions.
              </p>
            )}

            <div className="flow-body mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {fields.map((field) => (
                <IntakeCard
                  key={field.id}
                  field={field}
                  state={states[field.id] ?? { kind: 'idle' }}
                  positiveLabel={disease.positiveLabel}
                  onFile={(f) => void handleFile(field, f)}
                />
              ))}
            </div>

            <div className="mt-4 flex shrink-0 items-center justify-between border-t border-white/5 pt-4">
              <button
                type="button"
                onClick={() => setStep('condition')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink-dim hover:text-ink"
              >
                <IconArrowLeft className="h-3.5 w-3.5" />
                Back
              </button>

              <div className="flex items-center gap-3">
                {!anyUpload && (
                  <span className="font-mono text-[11px] text-ink-faint">
                    supply at least one input
                  </span>
                )}
                <button
                  type="button"
                  disabled={!anyUpload}
                  onClick={() => setStep('processing')}
                  className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Analyse
                  <IconArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step three: the machine, on a fixed five-second sequence. */}
        {step === 'processing' && (
          <ProcessingBay
            fileName={lastFile || 'input'}
            onDone={() => setStep('result')}
          />
        )}

        {/* Step four: metrics and the region view. */}
        {step === 'result' && (
          <>
            <PredictionResult
              fileName={lastFile || 'input'}
              result={scoredResult}
              imageUrl={imageUrl}
              conditionId={selectedId}
              positiveLabel={disease.positiveLabel}
              negativeLabel={disease.negativeLabel}
            />
            <div className="flex justify-start">
              <button
                type="button"
                onClick={() => setStep('intake')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink-dim hover:text-ink"
              >
                <IconArrowLeft className="h-3.5 w-3.5" />
                Back to inputs
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function IntakeCard({
  field,
  state,
  positiveLabel,
  onFile,
}: {
  field: IntakeField
  state: FieldState
  positiveLabel: string
  onFile: (file: File) => void
}) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [dragging, setDragging] = useState(false)

  return (
    <div className="panel-well well-pad flex h-full flex-col rounded-[8px]">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium text-ink">{field.label}</span>
        <span
          className="shrink-0 rounded-[4px] px-1.5 py-0.5 font-mono text-[11px]"
          style={
            field.wired
              ? { color: QUANTUM, background: alpha(QUANTUM, 0.12) }
              : { color: '#6A6C72', background: 'rgba(255,255,255,0.05)' }
          }
        >
          {field.wired ? 'supported' : 'not built'}
        </span>
      </div>
      <p className="engraved mt-1 font-mono text-[11px]">
        {field.system} · {field.accept}
      </p>
      <p className="mt-2 text-[12.5px] leading-relaxed text-ink-dim">{field.hint}</p>

      <input
        ref={inputRef}
        type="file"
        accept={field.accept}
        className="sr-only"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const f = e.dataTransfer.files?.[0]
          if (f) onFile(f)
        }}
        className="mt-auto flex w-full cursor-pointer flex-col items-center justify-center gap-1.5 rounded-[6px] px-3 py-4 text-[12.5px] text-ink-dim transition-colors hover:text-ink"
        style={{
          background: '#131417',
          border: `1px dashed ${dragging ? alpha(QUANTUM, 0.6) : 'rgba(255,255,255,0.12)'}`,
        }}
      >
        <IconUpload className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">
          {state.kind === 'idle' || state.kind === 'error'
            ? 'Choose a file, or drop one here'
            : state.file}
        </span>
      </button>

      <Outcome state={state} positiveLabel={positiveLabel} />
    </div>
  )
}

function Outcome({ state, positiveLabel }: { state: FieldState; positiveLabel: string }) {
  if (state.kind === 'idle') return null

  if (state.kind === 'error') {
    return (
      <p className="mt-2 text-[12px] leading-relaxed" style={{ color: CLASSICAL }}>
        {state.message}
      </p>
    )
  }

  // The honest path: a declared input with nothing behind it yet.
  if (state.kind === 'unavailable') {
    return (
      <div className="readout mt-2 flex items-center justify-between gap-2 px-3 py-2">
        <span className="font-mono text-[11px]" style={{ color: CLASSICAL }}>
          not scored, parser not built
        </span>
        <InfoDot label="What this input would need">{state.requires}</InfoDot>
      </div>
    )
  }

  if (state.kind === 'parsed') {
    return (
      <div className="readout mt-2 px-3 py-2 font-mono text-[12px]">
        <div className="flex justify-between">
          <span className="text-ink-faint">read</span>
          <span className="tabular-nums text-ink">
            {state.rows} rows x {state.columns} cols
          </span>
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-ink-faint">
          Not scored: no trained model, or no column matched one the model uses.
        </p>
      </div>
    )
  }

  // An image is loaded for display only: it carries no rows to score, and no
  // detector runs on the pixels.
  if (state.kind === 'image') {
    return (
      <div className="readout mt-2 flex items-center justify-between gap-2 px-3 py-2">
        <span className="font-mono text-[11px]" style={{ color: QUANTUM }}>
          image loaded
        </span>
        <InfoDot label="What happens to this image">
          The image is shown on the result screen with region markers over it.
          Those markers come from a hash of the file name, not from a detector,
          and nothing is scored from the pixels.
        </InfoDot>
      </div>
    )
  }

  const { result } = state
  const share = (result.positiveCount / result.rows.length) * 100

  return (
    <div className="readout mt-2 px-3 py-2 font-mono text-[12px]">
      <div className="flex items-baseline justify-between">
        <span className="flex items-center gap-1.5 text-ink-faint">
          <span style={{ color: QUANTUM }} className="flex">
            <IconCheck className="h-3 w-3" />
          </span>
          scored
        </span>
        <span className="tabular-nums text-ink">{result.rows.length} rows</span>
      </div>
      <div className="mt-1 flex items-baseline justify-between">
        <span className="text-ink-faint">{positiveLabel.toLowerCase()}</span>
        <span className="tabular-nums" style={{ color: CLASSICAL }}>
          {result.positiveCount}
          <span className="ml-1 text-ink-faint">({share.toFixed(1)}%)</span>
        </span>
      </div>
      {result.missing.length > 0 && (
        <div className="mt-1.5 flex items-center justify-between gap-2 border-t border-white/5 pt-1.5">
          <span className="font-mono text-[11px] text-ink-faint">
            {result.matched.length}/{result.matched.length + result.missing.length} features
          </span>
          <InfoDot label="About the missing features">
            {result.missing.length} feature(s) the model uses were not present in this
            file, so they fell back to their training average. These scores are
            weaker than a complete row would give.
          </InfoDot>
        </div>
      )}
      {result.skipped > 0 && (
        <p className="mt-1 text-[11px] text-ink-faint">
          {result.skipped} row(s) skipped, nothing numeric to read.
        </p>
      )}
    </div>
  )
}
