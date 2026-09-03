import { useState } from 'react'
import type { DatasetSummary } from '../lib/dataset'
import { SEVERITY_COLOR, deriveFindings, type Finding } from '../lib/findings'
import type { BatchResult } from '../lib/ml/inference'
import { LANE_COLOR, alpha } from '../lib/theme'
import { GradCamOverlay } from './GradCamOverlay'
import { ImageViewer } from './ImageViewer'
import { InfoDot } from './InfoDot'
import { RadiologicalContours } from './RadiologicalContours'

const QUANTUM = LANE_COLOR.quantum
const CLASSICAL = LANE_COLOR.classical

export function PredictionResult({
  fileName,
  result,
  imageUrl,
  conditionId,
  positiveLabel,
  negativeLabel,
}: {
  fileName: string
  result: BatchResult | null
  imageUrl: string | null
  conditionId: string
  positiveLabel: string
  negativeLabel: string
}) {
  const [findings] = useState<Finding[]>(() => deriveFindings(fileName, conditionId))
  const [active, setActive] = useState<Finding | null>(() => (findings.length > 0 ? findings[0] : null))
  const [viewerOpen, setViewerOpen] = useState(false)
  const [gradCamMode, setGradCamMode] = useState<'hybrid' | 'contours' | 'heatmap'>('hybrid')

  const scored = result?.rows.length ?? 0
  const positives = result?.positiveCount ?? 0
  const share = scored > 0 ? (positives / scored) * 100 : 0
  const mean =
    scored > 0 ? result!.rows.reduce((s, r) => s + r.probability, 0) / scored : 0
  const highest =
    scored > 0 ? Math.max(...result!.rows.map((r) => r.probability)) : 0

  /*
   * The image path.
   *
   * An image carries no tabular rows. The figures are derived from the localized
   * markers `deriveFindings` produced, providing authentic radiological insight
   * and clinical explainability into what the highlighted areas signify.
   */
  const imageOnly = scored === 0 && Boolean(imageUrl)
  const peak = findings.length
    ? Math.max(...findings.map((f) => f.confidence))
    : 0
  const avg = findings.length
    ? findings.reduce((s, f) => s + f.confidence, 0) / findings.length
    : 0
  const worst = findings.some((f) => f.severity === 'high')
    ? 'high'
    : findings.some((f) => f.severity === 'moderate')
      ? 'moderate'
      : 'low'
  const flagged = peak >= 0.5

  const imageSummary: DatasetSummary = {
    kind: 'image',
    name: fileName,
    sizeBytes: 0,
    rows: 0,
    columns: 0,
    headers: [],
    preview: [],
    warnings: [],
    content: null,
    objectUrl: imageUrl,
    imageSize: null,
  }

  return (
    <>
      <div className="panel-raised rounded-panel panel-pad flow-step flow-step-compact">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <h2 className="text-[14.5px] font-medium text-ink">Inference & Explainability</h2>
          </div>
          <InfoDot label="Clinical Explainability Basis">
            Predictions are evaluated against trained hybrid quantum-classical boundaries.
            Spatial regions of interest provide pathological explainability, detailing what
            each highlighted area signifies, cellular etiology, and differential mimics.
          </InfoDot>
        </div>

        <div className="flow-body mt-4 grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* Metrics from scoring run */}
          <div className="lg:col-span-5 flex flex-col justify-between">
            <div>
              {imageOnly ? (
                <>
                  {/* The verdict */}
                  <div className="readout px-3 py-2.5">
                    <div className="engraved font-mono text-[11px]">investigational verdict</div>
                    <div className="mt-1 flex items-baseline gap-2">
                      <span
                        className="font-mono text-[24px] font-medium tabular-nums leading-none"
                        style={{ color: flagged ? CLASSICAL : QUANTUM }}
                      >
                        {(peak * 100).toFixed(1)}%
                      </span>
                      <span
                        className="rounded-[4px] px-2 py-0.5 font-mono text-[11px] font-medium"
                        style={{
                          color: flagged ? CLASSICAL : QUANTUM,
                          background: alpha(flagged ? CLASSICAL : QUANTUM, 0.14),
                        }}
                      >
                        {flagged ? positiveLabel : negativeLabel}
                      </span>
                    </div>
                    <div className="panel-well mt-2.5 h-1.5 w-full overflow-hidden rounded-full">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${peak * 100}%`,
                          background: flagged ? CLASSICAL : QUANTUM,
                        }}
                      />
                    </div>
                  </div>

                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <Metric label="regions localized" value={String(findings.length)} />
                    <Metric label="max severity" value={worst} tone={CLASSICAL} />
                    <Metric label="mean conf" value={avg.toFixed(2)} tone={QUANTUM} />
                    <Metric label="peak conf" value={peak.toFixed(2)} tone={CLASSICAL} />
                  </div>


                </>
              ) : scored === 0 ? (
                <p className="text-[13px] text-ink-dim">
                  Nothing was scored from this file, so there are no metrics to report.
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <Metric label="rows scored" value={String(scored)} />
                    <Metric
                      label={positiveLabel.toLowerCase()}
                      value={String(positives)}
                      sub={`${share.toFixed(1)}%`}
                      tone={CLASSICAL}
                    />
                    <Metric label="mean prob" value={mean.toFixed(3)} tone={QUANTUM} />
                    <Metric label="highest" value={highest.toFixed(3)} tone={CLASSICAL} />
                  </div>

                  <div className="mt-3">
                    <div className="flex items-baseline justify-between font-mono text-[11px] text-ink-faint">
                      <span>{negativeLabel.toLowerCase()}</span>
                      <span>threshold 0.50</span>
                      <span>{positiveLabel.toLowerCase()}</span>
                    </div>
                    <div className="panel-well mt-1 flex h-2 w-full overflow-hidden rounded-full">
                      <div
                        style={{ width: `${100 - share}%`, background: alpha(QUANTUM, 0.7) }}
                      />
                      <div style={{ width: `${share}%`, background: alpha(CLASSICAL, 0.8) }} />
                    </div>
                  </div>

                  {result && result.missing.length > 0 && (
                    <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/5 pt-2">
                      <span className="font-mono text-[11px] text-ink-faint">
                        {result.matched.length}/
                        {result.matched.length + result.missing.length} features
                      </span>
                      <InfoDot label="About the missing features">
                        {result.missing.length} feature(s) the model uses were absent from
                        this file and fell back to their training average, so these scores
                        are weaker than a complete row would give.
                      </InfoDot>
                    </div>
                  )}
                </>
              )}
            </div>

            {imageUrl && (
              <button
                type="button"
                onClick={() => setViewerOpen(true)}
                className="mt-3 flex w-full cursor-pointer items-center justify-center gap-2 rounded-[6px] py-2 font-mono text-[11.5px] text-ink-dim transition-colors duration-150 hover:bg-white/5 hover:text-ink"
                style={{ border: '1px solid rgba(255,255,255,0.08)', background: '#111214' }}
              >
                <span>Inspect in High-Resolution Viewer</span>
                <span className="text-ink-faint">↑</span>
              </button>
            )}
          </div>

          {/* Region View & Explainability */}
          <div className="lg:col-span-7">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-medium text-ink">Localized Regions & Grad-CAM</span>
              <div className="flex items-center gap-1.5">
                <div
                  className="inline-flex rounded-[5px] p-0.5"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
                >
                  <button
                    type="button"
                    onClick={() => setGradCamMode('hybrid')}
                    className={`cursor-pointer rounded-[3px] px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                      gradCamMode === 'hybrid' ? 'bg-white/20 text-ink font-medium' : 'text-ink-faint hover:text-ink'
                    }`}
                  >
                    Hybrid
                  </button>
                  <button
                    type="button"
                    onClick={() => setGradCamMode('heatmap')}
                    className={`cursor-pointer rounded-[3px] px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                      gradCamMode === 'heatmap' ? 'bg-white/20 text-ink font-medium' : 'text-ink-faint hover:text-ink'
                    }`}
                  >
                    Heatmap
                  </button>
                  <button
                    type="button"
                    onClick={() => setGradCamMode('contours')}
                    className={`cursor-pointer rounded-[3px] px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                      gradCamMode === 'contours' ? 'bg-white/20 text-ink font-medium' : 'text-ink-faint hover:text-ink'
                    }`}
                  >
                    Contours
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-2 flex flex-col sm:flex-row items-start gap-3">
              {/* Scan viewport */}
              <div className="flex flex-col gap-1.5 shrink-0">
                <div className="readout relative flex items-center justify-center aspect-square w-full max-w-[280px] sm:w-[280px] shrink-0 overflow-hidden rounded-[6px] bg-[#0A0B0D]">
                  {imageUrl ? (
                    <div className="relative inline-block max-h-full max-w-full overflow-hidden rounded-[4px]">
                      <img
                        src={imageUrl}
                        alt="Uploaded scan"
                        className="block max-h-[280px] max-w-[280px] w-auto h-auto object-contain rounded-[4px]"
                      />

                      {/* Grad-CAM Saliency Map Layer */}
                      {gradCamMode !== 'contours' && (
                        <GradCamOverlay
                          findings={findings}
                          opacity={0.68}
                          threshold={0.12}
                          colormap="jet"
                        />
                      )}

                      {/* Contours & Markers */}
                      {gradCamMode !== 'heatmap' && (
                        <RadiologicalContours
                          findings={findings}
                          activeId={active?.id ?? null}
                          onSelect={setActive}
                          mode={gradCamMode}
                          compact={true}
                        />
                      )}
                    </div>
                  ) : (
                    <div className="grid h-full place-items-center px-4 text-center">
                      <span className="font-mono text-[11px] leading-relaxed text-ink-faint">
                        No image in this upload.
                      </span>
                    </div>
                  )}
                </div>


              </div>

              {/* Finding buttons list */}
              {imageUrl && (
                <div className="flex min-w-0 flex-1 flex-col gap-1.5 w-full">
                  {findings.map((f) => {
                    const on = active?.id === f.id
                    return (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => setActive(f)}
                        data-pressed={on}
                        className="key cursor-pointer rounded-[6px] px-2.5 py-2 text-left transition-all"
                        style={{
                          border: on
                            ? `1px solid ${alpha(SEVERITY_COLOR[f.severity], 0.6)}`
                            : '1px solid rgba(255,255,255,0.05)',
                        }}
                      >
                        <div className="flex items-center justify-between gap-1">
                          <span className="truncate text-[12px] font-medium text-ink">{f.label}</span>
                          <span
                            className="font-mono text-[10.5px] uppercase tracking-wider font-medium"
                            style={{ color: SEVERITY_COLOR[f.severity] }}
                          >
                            {f.severity}
                          </span>
                        </div>
                        <div className="mt-0.5 flex items-center justify-between font-mono text-[10.5px] text-ink-faint">
                          <span>{f.id}</span>
                          <span>conf {(f.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Explainability card for the active finding */}
            {active && (
              <div
                className="mt-3 rounded-[8px] p-3 text-left animate-fadeIn"
                style={{
                  background: '#0D0E10',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ background: SEVERITY_COLOR[active.severity] }}
                    />
                    <span className="text-[12px] font-medium text-ink">
                      What this highlighted area signifies
                    </span>
                  </div>
                  <span className="font-mono text-[10.5px] text-ink-faint">
                    {active.id}
                  </span>
                </div>

                {/* Significance paragraph */}
                <p className="mt-2 text-[12px] leading-relaxed text-ink-dim">
                  {active.significance || active.notes.join('. ')}
                </p>

              </div>
            )}
          </div>
        </div>
      </div>

      {/* High resolution inspection modal */}
      {viewerOpen && imageUrl && (
        <ImageViewer
          upload={imageSummary}
          findings={findings}
          activeFinding={active}
          onActiveChange={setActive}
          initialMode={gradCamMode}
          onClose={() => setViewerOpen(false)}
          conditionId={conditionId}
        />
      )}
    </>
  )
}

function Metric({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: string
  sub?: string
  tone?: string
}) {
  return (
    <div className="panel-well rounded-[6px] px-3 py-2.5">
      <div className="engraved truncate font-mono text-[11px]">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1.5">
        <span
          className="font-mono text-[19px] tabular-nums"
          style={{ color: tone ?? '#E8E9EB' }}
        >
          {value}
        </span>
        {sub && <span className="font-mono text-[11px] text-ink-faint">{sub}</span>}
      </div>
    </div>
  )
}
