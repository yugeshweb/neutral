import { useRef, useState } from 'react'
import { formatBytes, MAX_UPLOAD_BYTES, type DatasetSummary } from '../lib/dataset'
import { ACCEPT_ATTR, ingest, NotImplementedError } from '../lib/ingest'
import {
  convertUpload,
  CustomDatasetError,
  previewColumns,
  suggestLabelColumn,
  type ColumnPreview,
} from '../lib/ml/customDataset'
import { registerCustomDataset } from '../lib/ml/datasets'
import type { Dataset } from '../lib/ml/pipeline'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconCheck, IconUpload } from './icons'
import { PushButton } from './PushButton'

type Stage =
  | { step: 'pick' }
  | { step: 'error'; message: string }
  | { step: 'preview'; summary: DatasetSummary; columns: ColumnPreview[]; labelColumn: string }

type Props = {
  /** called once the user confirms a label column and the file converts cleanly */
  onReady: (dataset: Dataset) => void
  disabled?: boolean
}

/**
 * Upload → parse → pick a label column → convert → register, in one place.
 *
 * Deliberately separate from InputPanel: that component is a sidebar preview
 * with no path into the training pipeline. This one exists specifically to
 * close that gap, so it owns the one extra step InputPanel never needed -
 * telling the pipeline which column is the thing being predicted.
 */
export function UploadPanel({ onReady, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [stage, setStage] = useState<Stage>({ step: 'pick' })
  const [dragging, setDragging] = useState(false)

  async function accept(file: File) {
    if (file.size > MAX_UPLOAD_BYTES) {
      setStage({
        step: 'error',
        message: `file is ${formatBytes(file.size)}, limit is ${formatBytes(MAX_UPLOAD_BYTES)}`,
      })
      return
    }

    try {
      const result = await ingest(file)
      if (result.dataset.kind === 'image') {
        setStage({ step: 'error', message: 'images cannot be trained on directly here - upload a CSV, FHIR bundle or HL7 v2 feed' })
        return
      }
      const columns = previewColumns(result.dataset)
      const suggested = suggestLabelColumn(columns)
      if (!suggested) {
        setStage({
          step: 'error',
          message: 'no column with exactly two distinct values was found to use as the label - this pipeline only trains binary classifiers',
        })
        return
      }
      setStage({ step: 'preview', summary: result.dataset, columns, labelColumn: suggested })
    } catch (e) {
      if (e instanceof NotImplementedError) {
        setStage({ step: 'error', message: `${e.message} - would require: ${e.requires}` })
        return
      }
      setStage({ step: 'error', message: e instanceof Error ? e.message : 'could not read file' })
    }
  }

  function confirm() {
    if (stage.step !== 'preview') return
    try {
      const { dataset } = convertUpload(stage.summary, stage.labelColumn)
      const registered = registerCustomDataset(dataset, stage.summary.name)
      onReady(registered)
    } catch (e) {
      setStage({
        step: 'error',
        message: e instanceof CustomDatasetError ? e.message : e instanceof Error ? e.message : 'could not convert file',
      })
    }
  }

  if (stage.step === 'preview') {
    const labelCol = stage.columns.find((c) => c.name === stage.labelColumn)
    const binaryColumns = stage.columns.filter((c) => c.distinctValues === 2)

    return (
      <div
        className="rounded-panel p-3.5"
        style={{
          background: '#17181B',
          border: `1px solid ${alpha(LANE_COLOR.quantum, 0.3)}`,
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
        }}
      >
        <div className="mb-2.5 flex items-baseline justify-between">
          <span className="font-mono text-[11.5px] font-medium tracking-[0.02em] text-ink-faint">
            {stage.summary.name}
          </span>
          <span className="font-mono text-[11px] text-ink-faint">
            {stage.summary.rows} rows x {stage.summary.columns} cols
          </span>
        </div>

        <label htmlFor="label-col" className="mb-1.5 block font-mono text-[12px] text-ink-dim">
          label column - what the model predicts
        </label>
        <select
          id="label-col"
          value={stage.labelColumn}
          onChange={(e) => setStage({ ...stage, labelColumn: e.target.value })}
          className="select w-full px-2 py-1.5 font-mono text-[13px]"
        >
          {binaryColumns.length === 0 && <option value={stage.labelColumn}>{stage.labelColumn}</option>}
          {binaryColumns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>

        {labelCol && (
          <p className="mt-1.5 font-mono text-[11px] leading-relaxed text-ink-faint/80">
            values seen: {labelCol.sample.join(', ')}
            {binaryColumns.length === 0 && ' - only binary columns can be a label; none were found'}
          </p>
        )}

        <div className="mt-3 flex gap-2">
          <PushButton
            label="Use this dataset"
            icon={<IconCheck className="h-3.5 w-3.5" />}
            onClick={confirm}
            tone="primary"
            accent="#5FA88C"
            disabled={binaryColumns.length === 0}
          />
          <PushButton label="Cancel" onClick={() => setStage({ step: 'pick' })} />
        </div>
      </div>
    )
  }

  return (
    <div
      className="rounded-panel p-3.5"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="sr-only"
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void accept(f)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (disabled) return
          const f = e.dataTransfer.files?.[0]
          if (f) void accept(f)
        }}
        className="flex w-full cursor-pointer flex-col items-center gap-1.5 rounded-[9px] px-3 py-5 transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
        style={{
          background: '#0D0E10',
          border: `1px dashed ${dragging ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.09)'}`,
          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
        }}
      >
        <IconUpload className="h-4 w-4 text-ink-faint" />
        <span className="text-[11.5px] text-ink-dim">Drop a file, or browse</span>
        <span className="font-mono text-[11.5px] text-ink-faint">
          CSV, FHIR JSON or HL7 v2 / max {formatBytes(MAX_UPLOAD_BYTES)}
        </span>
      </button>

      {stage.step === 'error' && (
        <div className="mt-2 font-mono text-[11.5px]" style={{ color: '#A3543D' }} role="alert">
          {stage.message}
        </div>
      )}

      <p className="mt-2.5 font-mono text-[11px] leading-relaxed text-ink-faint/70">
        Parsed in your browser. The label column must have exactly two values - this
        pipeline trains binary classifiers only.
      </p>
    </div>
  )
}
