import { useRef, useState } from 'react'
import { MAX_UPLOAD_BYTES, formatBytes, type DatasetSummary } from '../lib/dataset'
import {
  ACCEPT_ATTR,
  NotImplementedError,
  adapterFor,
  ingest,
  type IngestAdapter,
  type SchemaField,
} from '../lib/ingest'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconClose, IconUpload } from './icons'
import { SourcePicker } from './SourcePicker'

type Props = {
  upload: DatasetSummary | null
  onUpload: (d: DatasetSummary | null) => void
  /** opens the image preview; only offered for image uploads */
  onView: () => void
  /** uploads are locked while a run is in flight */
  locked: boolean
}

/** A format the user selected that has no parser yet. */
type Blocked = { format: string; label: string; requires: string }

const TYPE_COLOR: Record<SchemaField['type'], string> = {
  numeric: '#9A9CA1',
  categorical: '#C08A3E',
  identifier: '#6A6C72',
  label: '#5FA88C',
}

export function InputPanel({ upload, onUpload, onView, locked }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [blocked, setBlocked] = useState<Blocked | null>(null)
  const [dragging, setDragging] = useState(false)
  const [schema, setSchema] = useState<SchemaField[]>([])
  const [notes, setNotes] = useState<string[]>([])
  const [format, setFormat] = useState<string | null>(null)
  const [showSources, setShowSources] = useState(false)

  const reset = () => {
    setError(null)
    setBlocked(null)
  }

  async function accept(file: File) {
    reset()

    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`file is ${formatBytes(file.size)}, limit is ${formatBytes(MAX_UPLOAD_BYTES)}`)
      return
    }

    try {
      const result = await ingest(file)
      onUpload(result.dataset)
      setSchema(result.schema)
      setNotes(result.notes)
      setFormat(adapterFor(file)?.format ?? null)
      setShowSources(false)
    } catch (e) {
      // An unbuilt format is not a user error - say what it would take instead.
      if (e instanceof NotImplementedError) {
        setBlocked({
          format: e.format,
          label: e.message,
          requires: e.requires,
        })
        return
      }
      setError(e instanceof Error ? e.message : 'could not read file')
    }
  }

  /** Picking a source opens the file dialog, or explains why it cannot. */
  const pickSource = (a: IngestAdapter) => {
    reset()
    if (a.status === 'not-implemented') {
      void a.parse(new File([], 'x')).catch((e: unknown) => {
        if (e instanceof NotImplementedError) {
          setBlocked({ format: e.format, label: e.message, requires: e.requires })
        }
      })
      return
    }
    inputRef.current?.click()
  }

  const clear = () => {
    onUpload(null)
    setSchema([])
    setNotes([])
    setFormat(null)
    reset()
  }

  return (
    <section
      className="rounded-panel p-3"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
      aria-label="Input data"
    >
      <div className="mb-2.5 flex items-baseline justify-between">
        <h2 className="font-mono text-[11.5px] font-medium tracking-[0.02em] text-ink-faint">
          input data
        </h2>
        {upload ? (
          <button
            type="button"
            onClick={clear}
            disabled={locked}
            className="flex cursor-pointer items-center gap-1 font-mono text-[11.5px] text-ink-faint transition-colors duration-150 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            <IconClose className="h-3 w-3" />
            clear
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setShowSources((s) => !s)}
            className="cursor-pointer font-mono text-[11.5px] text-ink-faint transition-colors duration-150 hover:text-ink"
            aria-expanded={showSources}
          >
            {showSources ? 'hide sources' : 'sources'}
          </button>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="sr-only"
        disabled={locked}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void accept(f)
          e.target.value = ''
        }}
      />

      {!upload ? (
        <>
          <button
            type="button"
            disabled={locked}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              if (!locked) setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              if (locked) return
              const f = e.dataTransfer.files?.[0]
              if (f) void accept(f)
            }}
            className="flex w-full cursor-pointer flex-col items-center gap-1.5 rounded-[9px] px-3 py-5 transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
            style={{
              background: '#0D0E10',
              border: `1px dashed ${dragging ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.09)'}`,
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            <IconUpload className="h-4 w-4 text-ink-faint" />
            <span className="text-[11.5px] text-ink-dim">Drop a file, or browse</span>
            <span className="font-mono text-[11.5px] text-ink-faint">
              csv, fhir json or image / max {formatBytes(MAX_UPLOAD_BYTES)}
            </span>
          </button>

          {showSources && (
            <div className="mt-2.5">
              <p className="mb-2 font-mono text-[11px] leading-relaxed text-ink-faint/80">
                Hospital data rarely arrives as a CSV. Each source below is a separate
                adapter converging on one numeric matrix.
              </p>
              <SourcePicker activeFormat={format} onPick={pickSource} disabled={locked} />
            </div>
          )}
        </>
      ) : (
        <div
          className="rounded-[9px] p-2.5"
          style={{
            background: '#0D0E10',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
          }}
        >
          <div className="flex items-baseline gap-1.5">
            <div className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-ink" title={upload.name}>
              {upload.name}
            </div>
            {format && (
              <span
                className="shrink-0 rounded-[4px] px-1.5 py-[1px] font-mono text-[11px]"
                style={{
                  color: LANE_COLOR.quantum,
                  background: alpha(LANE_COLOR.quantum, 0.1),
                }}
              >
                {format}
              </span>
            )}
          </div>

          {upload.kind === 'image' ? (
            <>
              <div className="mt-1.5 flex gap-3 font-mono text-[11.5px] text-ink-faint">
                <span>
                  {upload.imageSize?.w} x {upload.imageSize?.h}
                </span>
                <span>{formatBytes(upload.sizeBytes)}</span>
              </div>

              <button
                type="button"
                onClick={onView}
                className="mt-2 block w-full cursor-pointer overflow-hidden rounded-[7px]"
                style={{ border: '1px solid rgba(255,255,255,0.08)' }}
                aria-label={`View ${upload.name} with flagged regions`}
              >
                <img
                  src={upload.objectUrl ?? ''}
                  alt=""
                  className="h-[92px] w-full object-cover"
                />
              </button>

              <button
                type="button"
                onClick={onView}
                className="mt-2 w-full cursor-pointer rounded-[7px] py-1.5 font-mono text-[11.5px] text-ink-dim transition-colors duration-150 hover:text-ink"
                style={{ background: 'rgba(255,255,255,0.04)' }}
              >
                view flagged regions
              </button>
            </>
          ) : (
            <>
              <div className="mt-1.5 flex gap-3 font-mono text-[11.5px] text-ink-faint">
                <span>{upload.rows} rows</span>
                <span>{upload.columns} cols</span>
                <span>{formatBytes(upload.sizeBytes)}</span>
              </div>

              {/* what the adapter did, so the transform is not a black box */}
              {notes.length > 0 && (
                <ul className="mt-2 space-y-[3px]">
                  {notes.map((n) => (
                    <li
                      key={n}
                      className="flex gap-1.5 font-mono text-[11px] leading-relaxed text-ink-faint/85"
                    >
                      <span className="text-ink-faint/50">-</span>
                      {n}
                    </li>
                  ))}
                </ul>
              )}

              {/* resolved schema, with the coding system each column came from */}
              {schema.length > 0 && (
                <div className="mt-2.5">
                  <div className="mb-1 font-mono text-[11px] tracking-[0.02em] text-ink-faint/70">
                    schema / {schema.length} columns
                  </div>
                  <div className="console-scroll max-h-[168px] space-y-[2px] overflow-y-auto pr-1">
                    {schema.map((f) => (
                      <div key={f.name} className="flex items-baseline gap-1.5">
                        <span
                          className="h-[3px] w-[3px] shrink-0 rounded-full"
                          style={{ background: TYPE_COLOR[f.type] }}
                        />
                        <span
                          className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-dim"
                          title={f.name}
                        >
                          {f.name}
                        </span>
                        {f.coding && (
                          <span
                            className="shrink-0 font-mono text-[10.5px] text-ink-faint/70"
                            title={`${f.coding.system} ${f.coding.code}`}
                          >
                            {f.coding.system}
                          </span>
                        )}
                        <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-faint/70">
                          {f.present}/{upload.rows}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* first columns, so the user can confirm the right file landed */}
              <div className="mt-2 flex flex-wrap gap-1">
                {upload.headers.slice(0, 5).map((h) => (
                  <span
                    key={h}
                    className="truncate rounded-[4px] px-1.5 py-[2px] font-mono text-[11px] text-ink-faint"
                    style={{ background: 'rgba(255,255,255,0.04)', maxWidth: '96px' }}
                    title={h}
                  >
                    {h || '(blank)'}
                  </span>
                ))}
                {upload.headers.length > 5 && (
                  <span className="px-1 py-[2px] font-mono text-[11px] text-ink-faint">
                    +{upload.headers.length - 5}
                  </span>
                )}
              </div>
            </>
          )}

          {upload.warnings.map((w) => (
            <div key={w} className="mt-2 font-mono text-[11.5px]" style={{ color: '#C08A3E' }}>
              {w}
            </div>
          ))}
        </div>
      )}

      {/* a format with no parser yet - states the gap rather than failing vaguely */}
      {blocked && (
        <div
          className="mt-2 rounded-[8px] p-2.5"
          style={{
            background: alpha('#C08A3E', 0.06),
            border: `1px solid ${alpha('#C08A3E', 0.24)}`,
          }}
          role="status"
        >
          <div className="flex items-baseline justify-between gap-2">
            <span className="font-mono text-[11.5px]" style={{ color: '#C08A3E' }}>
              {blocked.label}
            </span>
            <button
              type="button"
              onClick={() => setBlocked(null)}
              aria-label="Dismiss"
              className="shrink-0 cursor-pointer text-ink-faint transition-colors duration-150 hover:text-ink"
            >
              <IconClose className="h-3 w-3" />
            </button>
          </div>
          <p className="mt-1.5 font-mono text-[11px] leading-relaxed text-ink-faint">
            <span className="text-ink-dim">Would require:</span> {blocked.requires}
          </p>
        </div>
      )}

      {error && (
        <div className="mt-2 font-mono text-[11.5px]" style={{ color: '#A3543D' }} role="alert">
          {error}
        </div>
      )}

      <p className="mt-2.5 font-mono text-[11px] leading-relaxed text-ink-faint/70">
        Files are parsed in-browser for display only. The demo pipeline does not train on
        uploaded data, and nothing is sent to a server.
      </p>
    </section>
  )
}
