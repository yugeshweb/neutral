import { useState } from 'react'
import { SEVERITY_COLOR, deriveFindings, type Finding } from '../lib/findings'
import type { BatchResult } from '../lib/ml/inference'
import { LANE_COLOR, alpha } from '../lib/theme'
import { InfoDot } from './InfoDot'

/**
 * The prediction readout: metrics beside the region view, in one card.
 *
 * Deliberately a single panel rather than two stacked ones. Stacked, this step
 * ran to roughly 760px and pushed the page into scrolling on anything shorter
 * than a 1080p screen; side by side it matches the other steps in the flow.
 *
 * The metrics are real - they come from scoring the uploaded rows with the
 * model trained in the Train tab.
 *
 * The region overlay is not. `deriveFindings` places markers from a hash of
 * the file name, so a given image always marks the same spots; no detector
 * runs. That is stated on the panel rather than left for the viewer to infer,
 * because a marked-up scan is exactly the kind of output someone would
 * otherwise take at face value.
 */

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
  const [active, setActive] = useState<Finding | null>(null)

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
   * An image carries no rows, so nothing can be scored from it. These figures
   * are derived from the markers `deriveFindings` produced, which are
   * themselves hashed from the file name - so they are stable per file and
   * consistent with what is drawn on the picture, but they are not a
   * measurement of anything. The regions panel carries the "illustrative only"
   * badge that governs this whole screen.
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

  return (
    <div className="panel-raised rounded-panel panel-pad flow-step flow-step-compact">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-[14.5px] font-medium text-ink">
          <span className="engraved mr-2 font-mono text-[12px]">4</span>
          Result
        </h2>
        <InfoDot label="Where these numbers come from">
          The metrics are produced by scoring the uploaded rows with the model
          trained in the Train tab. The region markers are not: they are placed
          from a hash of the file name, and no detector runs.
        </InfoDot>
      </div>

      <div className="flow-body mt-4 grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Metrics from the actual scoring run. */}
        <div className="lg:col-span-5">
          {imageOnly ? (
            <>
              {/* The verdict, stated up front. */}
              <div className="readout px-3 py-2.5">
                <div className="engraved font-mono text-[11px]">prediction</div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span
                    className="font-mono text-[24px] font-medium tabular-nums leading-none"
                    style={{ color: flagged ? CLASSICAL : QUANTUM }}
                  >
                    {(peak * 100).toFixed(1)}%
                  </span>
                  <span
                    className="rounded-[4px] px-2 py-0.5 font-mono text-[11px]"
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
                    className="h-full rounded-full"
                    style={{
                      width: `${peak * 100}%`,
                      background: flagged ? CLASSICAL : QUANTUM,
                    }}
                  />
                </div>
              </div>

              <div className="mt-2 grid grid-cols-2 gap-2">
                <Metric label="regions" value={String(findings.length)} />
                <Metric label="severity" value={worst} tone={CLASSICAL} />
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
                  <span>0.50</span>
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

        {/* Region view. */}
        <div className="lg:col-span-7">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[13px] font-medium text-ink">Suspected regions</span>
            <span
              className="shrink-0 rounded-[4px] px-2 py-0.5 font-mono text-[11px]"
              style={{ color: CLASSICAL, background: alpha(CLASSICAL, 0.12) }}
            >
              illustrative only
            </span>
          </div>

          <div className="mt-2 flex items-start gap-3">
            {/* A square frame, which is what a scan viewport normally is, and
                which keeps the region markers circular rather than stretched.
                `object-contain` letterboxes whatever aspect ratio arrives, so
                a wide or tall upload is shown whole rather than cropped. */}
            <div className="readout relative aspect-square w-[240px] shrink-0 overflow-hidden rounded-[6px]">
              {imageUrl ? (
                <img
                  src={imageUrl}
                  alt="Uploaded scan"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="grid h-full place-items-center px-4 text-center">
                  <span className="font-mono text-[11px] leading-relaxed text-ink-faint">
                    No image in this upload.
                  </span>
                </div>
              )}

              {imageUrl &&
                findings.map((f) => {
                  const on = active?.id === f.id
                  return (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setActive(on ? null : f)}
                      aria-label={f.label}
                      className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer rounded-full"
                      style={{
                        left: `${f.x * 100}%`,
                        top: `${f.y * 100}%`,
                        width: `${f.r * 100}%`,
                        aspectRatio: '1',
                        border: `2px solid ${SEVERITY_COLOR[f.severity]}`,
                        background: alpha(SEVERITY_COLOR[f.severity], on ? 0.22 : 0.1),
                      }}
                    />
                  )
                })}
            </div>

            {imageUrl && (
              <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                {findings.map((f) => {
                  const on = active?.id === f.id
                  return (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setActive(on ? null : f)}
                      data-pressed={on}
                      title={f.notes.join(' · ')}
                      className="key cursor-pointer rounded-[6px] px-2 py-1.5 text-left"
                    >
                      <div className="truncate text-[12px] text-ink">{f.label}</div>
                      <div
                        className="font-mono text-[11px]"
                        style={{ color: SEVERITY_COLOR[f.severity] }}
                      >
                        {f.severity}
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
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
