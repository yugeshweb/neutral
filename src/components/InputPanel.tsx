import { useRef, useState } from 'react'
import {
  IMAGE_TYPES,
  MAX_UPLOAD_BYTES,
  formatBytes,
  loadImage,
  parseCsv,
  type DatasetSummary,
} from '../lib/dataset'
import { IconClose, IconUpload } from './icons'

type Props = {
  upload: DatasetSummary | null
  onUpload: (d: DatasetSummary | null) => void
  /** opens the image preview; only offered for image uploads */
  onView: () => void
  /** uploads are locked while a run is in flight */
  locked: boolean
}

export function InputPanel({ upload, onUpload, onView, locked }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  async function accept(file: File) {
    setError(null)

    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`file is ${formatBytes(file.size)}, limit is ${formatBytes(MAX_UPLOAD_BYTES)}`)
      return
    }
    const isImage = IMAGE_TYPES.test(file.name)
    if (!isImage && !/\.csv$/i.test(file.name)) {
      setError('expected a .csv or image file')
      return
    }

    try {
      onUpload(isImage ? await loadImage(file) : parseCsv(await file.text(), file.name, file.size))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'could not read file')
    }
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
        <h2 className="font-mono text-[9.5px] font-medium tracking-[0.02em] text-ink-faint">
          input data
        </h2>
        {upload && (
          <button
            type="button"
            onClick={() => {
              onUpload(null)
              setError(null)
            }}
            disabled={locked}
            className="flex cursor-pointer items-center gap-1 font-mono text-[9.5px] text-ink-faint transition-colors duration-150 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            <IconClose className="h-3 w-3" />
            clear
          </button>
        )}
      </div>

      {!upload ? (
        <>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv,image/png,image/jpeg,image/webp,image/bmp"
            className="sr-only"
            disabled={locked}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void accept(f)
              e.target.value = ''
            }}
          />
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
            <span className="font-mono text-[9.5px] text-ink-faint">
              csv or image / max {formatBytes(MAX_UPLOAD_BYTES)}
            </span>
          </button>
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
          <div className="truncate font-mono text-[10.5px] text-ink" title={upload.name}>
            {upload.name}
          </div>

          {upload.kind === 'image' ? (
            <>
              <div className="mt-1.5 flex gap-3 font-mono text-[9.5px] text-ink-faint">
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
                className="mt-2 w-full cursor-pointer rounded-[7px] py-1.5 font-mono text-[9.5px] text-ink-dim transition-colors duration-150 hover:text-ink"
                style={{ background: 'rgba(255,255,255,0.04)' }}
              >
                view flagged regions
              </button>
            </>
          ) : (
            <>
              <div className="mt-1.5 flex gap-3 font-mono text-[9.5px] text-ink-faint">
                <span>{upload.rows} rows</span>
                <span>{upload.columns} cols</span>
                <span>{formatBytes(upload.sizeBytes)}</span>
              </div>

              {/* first columns, so the user can confirm the right file landed */}
              <div className="mt-2 flex flex-wrap gap-1">
                {upload.headers.slice(0, 5).map((h) => (
                  <span
                    key={h}
                    className="truncate rounded-[4px] px-1.5 py-[2px] font-mono text-[9px] text-ink-faint"
                    style={{ background: 'rgba(255,255,255,0.04)', maxWidth: '96px' }}
                    title={h}
                  >
                    {h || '(blank)'}
                  </span>
                ))}
                {upload.headers.length > 5 && (
                  <span className="px-1 py-[2px] font-mono text-[9px] text-ink-faint">
                    +{upload.headers.length - 5}
                  </span>
                )}
              </div>
            </>
          )}

          {upload.warnings.map((w) => (
            <div key={w} className="mt-2 font-mono text-[9.5px]" style={{ color: '#C08A3E' }}>
              {w}
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="mt-2 font-mono text-[9.5px]" style={{ color: '#A3543D' }} role="alert">
          {error}
        </div>
      )}

      <p className="mt-2.5 font-mono text-[9px] leading-relaxed text-ink-faint/70">
        Parsed in-browser for display only. The demo pipeline does not train on
        uploaded data.
      </p>
    </section>
  )
}
