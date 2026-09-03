import { useRef } from 'react'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconCheck, IconClose, IconUpload } from './icons'
import { INPUT_KINDS, kindOf, type InputKind, type InputRow } from '../lib/inputKinds'

/**
 * The training intake builder.
 *
 * A cohort is rarely one file. Clinical/EHR data arrives as a table, a FHIR
 * bundle, an HL7 feed or a PDF report, imaging as a separate export - so
 * rather than fixing the number of inputs, the user adds a row per source
 * and says what type each one is.
 *
 * The `ehr` kind is parsed into a trainable matrix, through the same
 * `ingest()` adapter registry the Predict tab uses - it auto-detects which
 * of CSV/FHIR/HL7/PDF a row's file actually is, so the user names the kind
 * of data, not the file format. Every imaging kind has no feature extraction
 * behind it yet, and the row says plainly that it will not contribute to
 * training, rather than being silently ignored once uploaded.
 */

const QUANTUM = LANE_COLOR.quantum
const CLASSICAL = LANE_COLOR.classical


export function InputBuilder({
  rows,
  onAdd,
  onRemove,
  onKind,
  onFile,
}: {
  rows: InputRow[]
  onAdd: () => void
  onRemove: (id: string) => void
  onKind: (id: string, kind: InputKind) => void
  onFile: (id: string, file: File) => void
}) {
  return (
    <div className="space-y-2">
      {rows.map((row, i) => (
        <InputRowView
          key={row.id}
          row={row}
          index={i}
          canRemove={rows.length > 1}
          onRemove={() => onRemove(row.id)}
          onKind={(k) => onKind(row.id, k)}
          onFile={(f) => onFile(row.id, f)}
        />
      ))}

      <button
        type="button"
        onClick={onAdd}
        className="key flex w-full cursor-pointer items-center justify-center gap-2 rounded-[6px] px-3 py-2 text-[13px] text-ink-dim hover:text-ink"
      >
        <span className="text-[15px] leading-none">+</span>
        Add another input
      </button>
    </div>
  )
}

function InputRowView({
  row,
  index,
  canRemove,
  onRemove,
  onKind,
  onFile,
}: {
  row: InputRow
  index: number
  canRemove: boolean
  onRemove: () => void
  onKind: (kind: InputKind) => void
  onFile: (file: File) => void
}) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const spec = kindOf(row.kind)

  return (
    <div className="panel-well flex items-center gap-2 rounded-[6px] px-2.5 py-2">
      <span className="engraved w-4 shrink-0 font-mono text-[11px]">{index + 1}</span>

      {/* Type picker for this row. */}
      <select
        aria-label={`Input ${index + 1} type`}
        value={row.kind}
        onChange={(e) => onKind(e.target.value as InputKind)}
        className="select w-[130px] shrink-0 py-1 text-[12px]"
      >
        {INPUT_KINDS.map((k) => (
          <option key={k.value} value={k.value}>
            {k.label}
          </option>
        ))}
      </select>

      <input
        ref={inputRef}
        type="file"
        accept={spec.accept}
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
        className="key flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-[6px] px-2.5 py-1 text-[12px] text-ink-dim hover:text-ink"
      >
        <IconUpload className="h-3 w-3 shrink-0" />
        <span className="truncate">{row.fileName ?? `Choose a file (${spec.accept})`}</span>
      </button>

      {/* Outcome for this row, kept to one line so the list stays scannable. */}
      <span className="w-[112px] shrink-0 text-right font-mono text-[11px]">
        {row.note ? (
          <span style={{ color: CLASSICAL }}>{row.note}</span>
        ) : row.rows !== null ? (
          <span className="inline-flex items-center gap-1" style={{ color: QUANTUM }}>
            <IconCheck className="h-3 w-3" />
            {row.rows} rows
          </span>
        ) : (
          !spec.trains && <span className="text-ink-faint">reference</span>
        )}
      </span>

      <button
        type="button"
        onClick={onRemove}
        disabled={!canRemove}
        aria-label={`Remove input ${index + 1}`}
        className="grid h-5 w-5 shrink-0 cursor-pointer place-items-center rounded-[4px] text-ink-faint hover:text-ink disabled:cursor-not-allowed disabled:opacity-30"
        style={{ background: alpha('#FFFFFF', 0.04) }}
      >
        <IconClose className="h-3 w-3" />
      </button>
    </div>
  )
}
