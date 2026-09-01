import { useMemo, useRef, useState } from 'react'
import { getDiseasePipeline, loadTrainedPipeline } from '../../lib/diseaseRegistry'
import { splitRow } from '../../lib/dataset'
import { ingest } from '../../lib/ingest'
import { NotImplementedError } from '../../lib/ingest/types'
import { intakeFor, INTAKE_DISEASE_IDS, type IntakeField } from '../../lib/intakeSpec'
import { isReplayable, scoreBatch, type BatchResult } from '../../lib/ml/inference'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { IconCheck, IconUpload } from '../icons'

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

  const disease = getDiseasePipeline(selectedId)
  const fields = intakeFor(selectedId)
  const artifact = useMemo(() => loadTrainedPipeline(selectedId), [selectedId])
  const trained = isReplayable(artifact)

  const setField = (id: string, s: FieldState) =>
    setStates((prev) => ({ ...prev, [id]: s }))

  const handleFile = async (field: IntakeField, file: File) => {
    // A format with no parser is reported as such, before the file is touched.
    if (!field.wired) {
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

        {/* Card one: the condition. */}
        <div className="panel-raised rounded-panel panel-pad">
          <h2 className="text-[14.5px] font-medium text-ink">Condition</h2>
          <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {INTAKE_DISEASE_IDS.map((id) => {
              const d = getDiseasePipeline(id)
              const active = id === selectedId
              return (
                <button
                  key={id}
                  type="button"
                  data-pressed={active}
                  onClick={() => {
                    setSelectedId(id)
                    setStates({})
                  }}
                  className="key cursor-pointer rounded-[8px] px-3.5 py-3 text-left"
                >
                  <div
                    className="text-[14.5px] font-medium"
                    style={{ color: active ? '#E8E9EB' : '#9A9CA1' }}
                  >
                    {d.name}
                  </div>
                  <div className="engraved mt-1 font-mono text-[11px]">{d.modality}</div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Card two: the inputs that condition accepts. */}
        <div className="panel-raised rounded-panel panel-pad">
          <div className="flex items-center justify-between">
            <h2 className="text-[14.5px] font-medium text-ink">
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

          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
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
        </div>
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
    <div className="panel-well well-pad flex flex-col rounded-[8px]">
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
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">{field.hint}</p>

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
        className="mt-3 flex w-full cursor-pointer items-center justify-center gap-2 rounded-[6px] px-3 py-2 text-[12.5px] text-ink-dim transition-colors hover:text-ink"
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
      <div className="readout mt-2 px-3 py-2">
        <div className="font-mono text-[11px]" style={{ color: CLASSICAL }}>
          not scored
        </div>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">{state.requires}</p>
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
        <p className="mt-1.5 border-t border-white/5 pt-1.5 text-[11px] leading-relaxed text-ink-faint">
          {result.matched.length} of {result.matched.length + result.missing.length} features
          matched by name. The rest use their training average, so these scores are
          weaker than a complete row.
        </p>
      )}
      {result.skipped > 0 && (
        <p className="mt-1 text-[11px] text-ink-faint">
          {result.skipped} row(s) skipped, nothing numeric to read.
        </p>
      )}
    </div>
  )
}
