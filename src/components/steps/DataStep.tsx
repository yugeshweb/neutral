import { useMemo, useState } from 'react'
import {
  CUSTOM_DATASET_ID,
  DATASET_META,
  datasetMeta,
  hasCustomDataset,
  loadDataset,
} from '../../lib/ml/datasets'
import type { RunConfig } from '../../lib/ml/pipeline'
import { classCounts, mean, std } from '../../lib/ml/stats'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { IconUpload } from '../icons'
import { Panel, SectionLabel } from '../ui'
import { UploadPanel } from '../UploadPanel'

type Props = {
  config: RunConfig
  patch: (p: Partial<RunConfig>) => void
  locked: boolean
}

export function DataStep({ config, patch, locked }: Props) {
  const [uploading, setUploading] = useState(false)
  const data = useMemo(() => loadDataset(config.datasetId), [config.datasetId])
  // datasetMeta() (not the DATASET_META array) also resolves the uploaded
  // dataset's synthesised meta - the array deliberately excludes it, since it
  // also drives the preset picker below and a file that has not been
  // uploaded yet must not appear there as a fourth preset.
  const meta = datasetMeta(config.datasetId)!

  const health = useMemo(() => {
    const d = data.featureNames.length
    return data.featureNames.map((name, j) => {
      const col = data.X.map((r) => r[j])
      const missing = col.filter((v) => !Number.isFinite(v)).length
      const observed = col.filter(Number.isFinite)
      const m = mean(observed)
      return {
        name,
        missing,
        mean: m,
        std: std(observed, m),
        min: Math.min(...observed),
        max: Math.max(...observed),
        type: 'numeric' as const,
        index: j,
        total: d,
      }
    })
  }, [data])

  const counts = classCounts(data.y)
  const positive = counts.get(1) ?? 0
  const negative = counts.get(0) ?? 0
  const posPct = (positive / data.y.length) * 100

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        {DATASET_META.map((m) => {
          const active = m.id === config.datasetId
          return (
            <button
              key={m.id}
              type="button"
              disabled={locked}
              onClick={() => {
                setUploading(false)
                patch({ datasetId: m.id, targetColumn: 'diagnosis' })
              }}
              className="cursor-pointer rounded-panel p-3.5 text-left transition-[border-color,box-shadow] duration-150 disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                background: '#17181B',
                border: `1px solid ${active ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.06)'}`,
                boxShadow: active
                  ? `inset 0 1px 0 rgba(255,255,255,0.07), 0 0 14px ${alpha(LANE_COLOR.quantum, 0.18)}`
                  : 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
              }}
            >
              <div className="flex items-baseline justify-between">
                <span className="text-[14.5px] font-medium text-ink">{m.name}</span>
                {active && (
                  <span
                    className="h-[6px] w-[6px] rounded-full"
                    style={{ background: LANE_COLOR.quantum }}
                  />
                )}
              </div>
              <div className="mt-1 font-mono text-[11px] text-ink-faint">{m.source}</div>
              <div className="mt-2 flex gap-3 font-mono text-[11.5px] text-ink-dim">
                <span>{m.rows} rows</span>
                <span>{Object.keys(m.featureDescriptions).length} features</span>
              </div>
            </button>
          )
        })}

        {/* upload: the fourth option, closing the "your own data" gap */}
        {(() => {
          const active = !uploading && config.datasetId === CUSTOM_DATASET_ID
          return (
            <button
              type="button"
              disabled={locked}
              onClick={() => {
                if (hasCustomDataset() && !uploading) {
                  patch({ datasetId: CUSTOM_DATASET_ID, targetColumn: 'diagnosis' })
                } else {
                  setUploading(true)
                }
              }}
              className="cursor-pointer rounded-panel p-3.5 text-left transition-[border-color,box-shadow] duration-150 disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                background: '#17181B',
                border: `1px dashed ${active ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.14)'}`,
                boxShadow: active
                  ? `inset 0 1px 0 rgba(255,255,255,0.07), 0 0 14px ${alpha(LANE_COLOR.quantum, 0.18)}`
                  : 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
              }}
            >
              <div className="flex items-baseline justify-between">
                <span className="flex items-center gap-1.5 text-[14.5px] font-medium text-ink">
                  <IconUpload className="h-3.5 w-3.5 text-ink-faint" />
                  {hasCustomDataset() ? datasetMeta(CUSTOM_DATASET_ID)?.name : 'Upload'}
                </span>
                {active && (
                  <span
                    className="h-[6px] w-[6px] rounded-full"
                    style={{ background: LANE_COLOR.quantum }}
                  />
                )}
              </div>
              <div className="mt-1 font-mono text-[11px] text-ink-faint">
                CSV, FHIR, HL7 v2
              </div>
              <div className="mt-2 flex gap-3 font-mono text-[11.5px] text-ink-dim">
                {hasCustomDataset() ? (
                  <span>{datasetMeta(CUSTOM_DATASET_ID)?.rows} rows</span>
                ) : (
                  <span>your own data</span>
                )}
              </div>
            </button>
          )
        })()}
      </div>

      {uploading && (
        <UploadPanel
          disabled={locked}
          onReady={(dataset) => {
            setUploading(false)
            patch({ datasetId: dataset.id, targetColumn: 'diagnosis' })
          }}
        />
      )}

      <div className="grid grid-cols-[1fr_320px] gap-4">
        {/* preview table */}
        <Panel>
          <div className="mb-2.5 flex items-baseline justify-between">
            <SectionLabel>table preview</SectionLabel>
            <span className="font-mono text-[11.5px] text-ink-faint">
              {data.X.length} rows x {data.featureNames.length} columns
            </span>
          </div>

          <div className="console-scroll overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr>
                  <th
                    className="whitespace-nowrap pb-1.5 pr-3 font-mono text-[11px] text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                  >
                    #
                  </th>
                  {data.featureNames.slice(0, 7).map((h) => (
                    <th
                      key={h}
                      className="whitespace-nowrap pb-1.5 pr-3 font-mono text-[11px] text-ink-faint"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                      title={meta.featureDescriptions[h]}
                    >
                      {h}
                    </th>
                  ))}
                  <th
                    className="pb-1.5 font-mono text-[11px]"
                    style={{
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                      color: LANE_COLOR.classical,
                    }}
                  >
                    target
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.X.slice(0, 6).map((row, i) => (
                  <tr key={i}>
                    <td
                      className="py-1 pr-3 font-mono text-[11px] tabular-nums text-ink-faint"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                    >
                      {i}
                    </td>
                    {row.slice(0, 7).map((v, j) => (
                      <td
                        key={j}
                        className="whitespace-nowrap py-1 pr-3 font-mono text-[11.5px] tabular-nums text-ink-dim"
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                      >
                        {v < 0.01 ? v.toExponential(1) : v.toFixed(2)}
                      </td>
                    ))}
                    <td
                      className="py-1 font-mono text-[11.5px]"
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                        color: data.y[i] === 1 ? '#A3543D' : '#5FA88C',
                      }}
                    >
                      {data.y[i] === 1 ? meta.positiveLabel : meta.negativeLabel}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 font-mono text-[11px] text-ink-faint/70">
            showing 6 of {data.X.length} rows, 7 of {data.featureNames.length} columns
          </p>
        </Panel>

        {/* split + seed */}
        <Panel>
          <SectionLabel>split & seed</SectionLabel>

          <div className="mt-3">
            <div className="mb-1.5 flex items-baseline justify-between">
              <label htmlFor="split" className="font-mono text-[12px] text-ink-dim">
                train / test split
              </label>
              <span className="font-mono text-[12px] tabular-nums text-ink">
                {Math.round((1 - config.testFraction) * 100)} / {Math.round(config.testFraction * 100)}
              </span>
            </div>
            <input
              id="split"
              type="range"
              min={0.1}
              max={0.4}
              step={0.05}
              value={config.testFraction}
              disabled={locked}
              onChange={(e) => patch({ testFraction: Number(e.target.value) })}
              className="feature-slider w-full cursor-pointer"
            />
            <div className="mt-1 flex justify-between font-mono text-[11px] text-ink-faint">
              <span>{Math.round(data.X.length * (1 - config.testFraction))} train</span>
              <span>{Math.round(data.X.length * config.testFraction)} test</span>
            </div>
          </div>

          <div className="mt-4">
            <label htmlFor="seed" className="mb-1.5 block font-mono text-[12px] text-ink-dim">
              random seed
            </label>
            <input
              id="seed"
              type="number"
              value={config.seed}
              disabled={locked}
              onChange={(e) => patch({ seed: Number(e.target.value) || 0 })}
              className="w-full rounded-[6px] px-2 py-1.5 font-mono text-[13px] tabular-nums text-ink outline-none disabled:opacity-50"
              style={{
                background: '#0D0E10',
                border: '1px solid rgba(255,255,255,0.05)',
                boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
              }}
            />
            <p className="mt-1.5 font-mono text-[11px] leading-relaxed text-ink-faint/80">
              Both lanes use this seed, so the comparison is on an identical split.
            </p>
          </div>

          <div className="mt-4">
            <SectionLabel>class balance</SectionLabel>
            <div className="mt-2 flex h-[6px] overflow-hidden rounded-full panel-well">
              <div style={{ width: `${100 - posPct}%`, background: alpha('#5FA88C', 0.7) }} />
              <div style={{ width: `${posPct}%`, background: alpha('#A3543D', 0.7) }} />
            </div>
            <div className="mt-1.5 flex justify-between font-mono text-[11px]">
              <span style={{ color: '#5FA88C' }}>
                {meta.negativeLabel} {negative}
              </span>
              <span style={{ color: '#A3543D' }}>
                {positive} {meta.positiveLabel}
              </span>
            </div>
            {Math.abs(posPct - 50) > 20 && (
              <p className="mt-1.5 font-mono text-[11px]" style={{ color: LANE_COLOR.classical }}>
                imbalanced: consider handling this in preprocessing
              </p>
            )}
          </div>
        </Panel>
      </div>

      {/* data health */}
      <Panel>
        <div className="mb-2.5 flex items-baseline justify-between">
          <SectionLabel>data health</SectionLabel>
          <span className="font-mono text-[11.5px] text-ink-faint">
            {health.reduce((n, h) => n + h.missing, 0)} missing cells across{' '}
            {health.length} columns
          </span>
        </div>

        <div className="console-scroll max-h-[240px] overflow-y-auto">
          <table className="w-full border-collapse text-left">
            <thead className="sticky top-0" style={{ background: '#17181B' }}>
              <tr>
                {['column', 'type', 'missing', 'mean', 'std', 'range'].map((h) => (
                  <th
                    key={h}
                    className="pb-1.5 pr-3 font-mono text-[11px] text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {health.map((h) => (
                <tr key={h.name}>
                  <td
                    className="py-1 pr-3 font-mono text-[11.5px] text-ink-dim"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                    title={meta.featureDescriptions[h.name]}
                  >
                    {h.name}
                  </td>
                  <td
                    className="py-1 pr-3 font-mono text-[11px] text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    {h.type}
                  </td>
                  <td
                    className="py-1 pr-3 font-mono text-[11px] tabular-nums"
                    style={{
                      borderBottom: '1px solid rgba(255,255,255,0.04)',
                      color: h.missing > 0 ? LANE_COLOR.classical : '#6A6C72',
                    }}
                  >
                    {h.missing}
                  </td>
                  <td
                    className="py-1 pr-3 font-mono text-[11px] tabular-nums text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    {h.mean < 0.01 ? h.mean.toExponential(1) : h.mean.toFixed(2)}
                  </td>
                  <td
                    className="py-1 pr-3 font-mono text-[11px] tabular-nums text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    {h.std < 0.01 ? h.std.toExponential(1) : h.std.toFixed(2)}
                  </td>
                  <td
                    className="py-1 font-mono text-[11px] tabular-nums text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    {h.min.toFixed(1)} - {h.max.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}
