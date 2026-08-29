import { useState } from 'react'
import { MAX_UPLOAD_BYTES, formatBytes } from '../lib/dataset'
import { adapterFor, ingest, type IngestResult, type SchemaField } from '../lib/ingest'
import {
  postEhrValidation,
  type EhrValidationResult,
  type RoutingDecision,
} from '../lib/platform'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconArrowLeft, IconCheck, IconUpload } from './icons'
import { Wordmark } from './Wordmark'

const PANEL = {
  background: '#17181B',
  border: '1px solid rgba(255,255,255,0.06)',
  boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
}

type LoadedCohort = {
  name: string
  sourceFormat: string
  result: IngestResult
}

function statusTone(status: string) {
  if (status === 'passed' || status === 'compatible') {
    return { color: '#5FA88C', background: alpha('#5FA88C', 0.1) }
  }
  if (status === 'warning' || status === 'insufficient data') {
    return { color: '#C08A3E', background: alpha('#C08A3E', 0.1) }
  }
  if (status === 'incompatible' || status === 'not available') {
    return { color: '#8A8F98', background: alpha('#8A8F98', 0.1) }
  }
  return { color: '#A3543D', background: alpha('#A3543D', 0.1) }
}

function SchemaTable({ fields, rows }: { fields: SchemaField[]; rows: string[][] }) {
  return (
    <div className="console-scroll overflow-x-auto">
      <table className="w-full min-w-[580px] border-collapse text-left">
        <thead>
          <tr>
            {['field', 'type', 'present', 'sample'].map((heading) => (
              <th
                key={heading}
                scope="col"
                className="border-b pb-2 font-mono text-[9px] font-medium uppercase tracking-[0.04em] text-ink-faint"
                style={{ borderColor: 'rgba(255,255,255,0.08)' }}
              >
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((field, index) => (
            <tr key={field.name}>
              <td className="max-w-[220px] truncate border-b py-2 pr-3 font-mono text-[9.5px] text-ink-dim" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                {field.name}
              </td>
              <td className="border-b py-2 pr-3 font-mono text-[9px] text-ink-faint" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                {field.type}
              </td>
              <td className="border-b py-2 pr-3 font-mono text-[9px] tabular-nums text-ink-faint" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                {field.present}
              </td>
              <td className="max-w-[220px] truncate border-b py-2 font-mono text-[9px] text-ink-faint" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                {rows[0]?.[index] || '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RoutingTable({ decisions }: { decisions: RoutingDecision[] }) {
  return (
    <div className="console-scroll overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse text-left">
        <thead>
          <tr>
            {['model', 'status', 'reason'].map((heading) => (
              <th
                key={heading}
                scope="col"
                className="border-b pb-2 font-mono text-[9px] font-medium uppercase tracking-[0.04em] text-ink-faint"
                style={{ borderColor: 'rgba(255,255,255,0.08)' }}
              >
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {decisions.map((decision) => {
            const tone = statusTone(decision.status)
            return (
              <tr key={`${decision.model_id}-${decision.model_version}`}>
                <td className="border-b py-2.5 pr-3 font-mono text-[9px] text-ink" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                  {decision.model_id}
                  <span className="ml-2 text-ink-faint">v{decision.model_version}</span>
                </td>
                <td className="border-b py-2.5 pr-3" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                  <span className="rounded px-1.5 py-1 font-mono text-[8.5px]" style={{ color: tone.color, background: tone.background }}>
                    {decision.status}
                  </span>
                </td>
                <td className="border-b py-2.5 text-[10px] leading-relaxed text-ink-faint" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                  {decision.reason}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ValidationResult({ result }: { result: EhrValidationResult }) {
  return (
    <section className="space-y-4" aria-label="Validation result" aria-live="polite">
      <div className="flex items-center gap-2 rounded-panel px-4 py-3" style={{ background: alpha('#5FA88C', 0.1), border: `1px solid ${alpha('#5FA88C', 0.25)}` }}>
        <IconCheck className="h-3.5 w-3.5 text-[#5FA88C]" />
        <span className="font-mono text-[10px] text-ink">cohort contract validated</span>
        <span className="ml-auto font-mono text-[9px] text-ink-faint">{result.endpoint}</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        {[
          ['rows', result.dataset.rows.toLocaleString()],
          ['features', result.dataset.features.toLocaleString()],
          ['missing cells', result.dataset.missing_cells.toLocaleString()],
          ['compatible models', String(result.checks.find((check) => check.name === 'model_routing')?.detail.match(/^\d+/)?.[0] ?? 0)],
        ].map(([label, value]) => (
          <div key={label} className="rounded-panel px-3 py-3" style={PANEL}>
            <div className="font-mono text-[9px] uppercase tracking-[0.04em] text-ink-faint">{label}</div>
            <div className="mt-1 font-mono text-[17px] tabular-nums text-ink">{value}</div>
          </div>
        ))}
      </div>

      <div className="rounded-panel p-4" style={PANEL}>
        <h2 className="text-[12px] font-medium text-ink">Checks</h2>
        <div className="mt-3 space-y-2">
          {result.checks.map((check) => {
            const tone = statusTone(check.status)
            return (
              <div key={check.name} className="flex items-start gap-2 text-[10px]">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: tone.color }} />
                <span className="w-32 shrink-0 font-mono text-ink-dim">{check.name}</span>
                <span className="leading-relaxed text-ink-faint">{check.detail}</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="rounded-panel p-4" style={PANEL}>
        <div className="mb-3 flex items-baseline justify-between gap-3">
          <h2 className="text-[12px] font-medium text-ink">Registered neurological models</h2>
          <span className="font-mono text-[9px] text-ink-faint">routing preview only</span>
        </div>
        <RoutingTable decisions={result.routing} />
      </div>

      <p className="font-mono text-[9px] leading-relaxed text-ink-faint/80">{result.disclaimer}</p>
    </section>
  )
}

export function EhrValidationView() {
  const [loaded, setLoaded] = useState<LoadedCohort | null>(null)
  const [validation, setValidation] = useState<EhrValidationResult | null>(null)
  const [target, setTarget] = useState('label')
  const [idColumn, setIdColumn] = useState('patient_id')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [sampleLoading, setSampleLoading] = useState(false)

  async function acceptFile(file: File) {
    setError(null)
    setValidation(null)
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`file is ${formatBytes(file.size)}, limit is ${formatBytes(MAX_UPLOAD_BYTES)}`)
      return
    }

    try {
      const result = await ingest(file)
      if (result.dataset.kind !== 'csv' || !result.dataset.content) {
        throw new Error('the validation surface accepts CSV or FHIR bundles that flatten to a cohort table')
      }
      setLoaded({
        name: file.name,
        sourceFormat: adapterFor(file)?.format ?? 'csv',
        result,
      })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not read the cohort')
    }
  }

  async function loadSample() {
    setSampleLoading(true)
    setError(null)
    try {
      const response = await fetch('/samples/sample-fhir-bundle.json')
      if (!response.ok) throw new Error(`could not load sample bundle (${response.status})`)
      const blob = await response.blob()
      await acceptFile(new File([blob], 'sample-fhir-bundle.json', { type: 'application/json' }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not load sample bundle')
    } finally {
      setSampleLoading(false)
    }
  }

  async function validate() {
    if (!loaded?.result.dataset.content) {
      setError('load a CSV or FHIR bundle first')
      return
    }
    if (!target.trim()) {
      setError('target column is required')
      return
    }

    setBusy(true)
    setError(null)
    try {
      const result = await postEhrValidation({
        csv_text: loaded.result.dataset.content,
        dataset_name: loaded.name,
        source_format: loaded.sourceFormat,
        target: target.trim(),
        id_column: idColumn.trim() || undefined,
      })
      setValidation(result)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'validation request failed')
    } finally {
      setBusy(false)
    }
  }

  const dataset = loaded?.result.dataset

  return (
    <div className="min-h-full overflow-y-auto bg-canvas">
      <header className="flex h-14 items-center gap-3 border-b px-4" style={{ background: '#111214', borderColor: 'rgba(255,255,255,0.06)' }}>
        <a href="/" className="flex items-center gap-2 rounded-[8px] px-2 py-1.5 text-ink-faint transition-colors hover:text-ink" aria-label="Back to main dashboard">
          <IconArrowLeft className="h-3.5 w-3.5" />
          <Wordmark size={14} />
        </a>
        <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />
        <span className="text-[13px] text-ink-dim">EHR validation</span>
        <span className="ml-auto font-mono text-[9px] text-ink-faint">/ehr-validation</span>
      </header>

      <main className="mx-auto w-full max-w-[1180px] px-6 py-10">
        <div className="mb-8 max-w-[720px]">
          <div className="mb-2 font-mono text-[9px] uppercase tracking-[0.08em]" style={{ color: LANE_COLOR.quantum }}>isolated test surface</div>
          <h1 className="text-[25px] font-semibold tracking-[-0.03em] text-ink">Validate an EHR cohort</h1>
          <p className="mt-3 text-[12px] leading-relaxed text-ink-dim">
            Upload a CSV export or FHIR R4 Bundle. The existing adapter normalizes it to a numeric cohort, then the separate validation endpoint checks labels, missingness, fingerprints, and every registered neurological model contract.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
          <section className="rounded-panel p-4" style={PANEL} aria-label="Cohort input">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="font-mono text-[9.5px] uppercase tracking-[0.04em] text-ink-faint">cohort input</h2>
              <span className="font-mono text-[9px] text-ink-faint">CSV / FHIR JSON</span>
            </div>

            <label className="flex cursor-pointer flex-col items-center gap-2 rounded-[9px] px-3 py-6 text-center" style={{ background: '#0D0E10', border: '1px dashed rgba(255,255,255,0.11)' }}>
              <IconUpload className="h-4 w-4 text-ink-faint" />
              <span className="text-[11px] text-ink-dim">Choose a cohort file</span>
              <span className="font-mono text-[9px] text-ink-faint">max {formatBytes(MAX_UPLOAD_BYTES)}</span>
              <input className="sr-only" type="file" accept=".csv,.json" onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void acceptFile(file)
                event.target.value = ''
              }} />
            </label>

            <button type="button" onClick={() => void loadSample()} disabled={sampleLoading || busy} className="mt-2 w-full cursor-pointer rounded-[7px] py-2 font-mono text-[9.5px] text-ink-dim transition-colors hover:text-ink disabled:cursor-not-allowed disabled:opacity-50" style={{ background: alpha(LANE_COLOR.quantum, 0.08), border: `1px solid ${alpha(LANE_COLOR.quantum, 0.2)}` }}>
              {sampleLoading ? 'loading sample…' : 'load sample FHIR bundle'}
            </button>

            {loaded && (
              <div className="mt-3 rounded-[8px] p-2.5" style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div className="truncate font-mono text-[10px] text-ink" title={loaded.name}>{loaded.name}</div>
                <div className="mt-1 flex gap-3 font-mono text-[9px] text-ink-faint">
                  <span>{loaded.sourceFormat}</span>
                  <span>{dataset?.rows} rows</span>
                  <span>{dataset?.columns} cols</span>
                </div>
              </div>
            )}

            <div className="mt-4 space-y-3">
              <div>
                <label htmlFor="ehr-target" className="mb-1 block font-mono text-[9px] text-ink-faint">target column</label>
                <input id="ehr-target" value={target} onChange={(event) => setTarget(event.target.value)} className="w-full rounded-[6px] px-2.5 py-2 font-mono text-[10px] text-ink outline-none" style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.07)' }} />
              </div>
              <div>
                <label htmlFor="ehr-id" className="mb-1 block font-mono text-[9px] text-ink-faint">patient ID column <span className="text-ink-faint/60">optional</span></label>
                <input id="ehr-id" value={idColumn} onChange={(event) => setIdColumn(event.target.value)} className="w-full rounded-[6px] px-2.5 py-2 font-mono text-[10px] text-ink outline-none" style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.07)' }} />
              </div>
            </div>

            <button type="button" onClick={() => void validate()} disabled={!loaded || busy || sampleLoading} className="mt-4 w-full cursor-pointer rounded-[7px] py-2.5 font-mono text-[10px] font-medium text-ink transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40" style={{ background: LANE_COLOR.classical }}>
              {busy ? 'validating cohort…' : 'validate cohort'}
            </button>

            {error && <div className="mt-3 font-mono text-[9.5px] leading-relaxed" style={{ color: '#A3543D' }} role="alert">{error}</div>}

            <p className="mt-4 font-mono text-[9px] leading-relaxed text-ink-faint/70">
              The endpoint receives only the adapter’s canonical table. It does not accept live EHR credentials, store PHI, train a model, or produce a patient diagnosis.
            </p>
          </section>

          <section className="rounded-panel p-4" style={PANEL} aria-label="Normalized cohort">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="font-mono text-[9.5px] uppercase tracking-[0.04em] text-ink-faint">normalized cohort</h2>
              {dataset && <span className="font-mono text-[9px] text-ink-faint">adapter output</span>}
            </div>
            {!loaded || !dataset ? (
              <div className="grid min-h-[290px] place-items-center rounded-[8px] text-center" style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div>
                  <div className="text-[11px] text-ink-dim">No cohort loaded</div>
                  <div className="mt-1 font-mono text-[9px] text-ink-faint">The normalized schema and preview will appear here.</div>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-4 flex flex-wrap gap-2">
                  {[
                    ['rows', dataset.rows],
                    ['columns', dataset.columns],
                    ['warnings', dataset.warnings.length],
                  ].map(([label, value]) => (
                    <span key={label} className="rounded-[5px] px-2 py-1 font-mono text-[9px] text-ink-faint" style={{ background: '#0D0E10' }}>{label} <span className="text-ink-dim">{value}</span></span>
                  ))}
                </div>
                <SchemaTable fields={loaded.result.schema} rows={dataset.preview} />
                {loaded.result.notes.length > 0 && (
                  <div className="mt-4 space-y-1 border-t pt-3" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
                    {loaded.result.notes.map((note) => <div key={note} className="font-mono text-[9px] leading-relaxed text-ink-faint">— {note}</div>)}
                  </div>
                )}
                {dataset.warnings.length > 0 && (
                  <div className="mt-3 space-y-1">
                    {dataset.warnings.map((warning) => <div key={warning} className="font-mono text-[9px] leading-relaxed" style={{ color: '#C08A3E' }}>! {warning}</div>)}
                  </div>
                )}
              </>
            )}
          </section>
        </div>

        {validation && <div className="mt-5"><ValidationResult result={validation} /></div>}
      </main>
    </div>
  )
}
