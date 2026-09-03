import { useEffect, useRef, useState } from 'react'
import type { DatasetSummary } from '../lib/dataset'
import { fetchExplainability } from '../lib/explainability'
import { SEVERITY_COLOR, deriveFindings, type Finding } from '../lib/findings'
import { DemoChip } from './DemoChip'
import { GradCamControls, type GradCamViewMode } from './GradCamControls'
import { GradCamOverlay } from './GradCamOverlay'
import { IconClose } from './icons'

type Props = {
  upload: DatasetSummary
  onClose: () => void
  conditionId?: string
}

/**
 * Full-bleed preview of an uploaded image with Grad-CAM Saliency heatmap
 * and comprehensive pathological explainability.
 *
 * Supports toggleable Grad-CAM activation overlays, continuous thermal color ramps,
 * and detailed clinical interpretations of what each highlighted region signifies.
 */
export function ImageViewer({ upload, onClose, conditionId = 'breast-cancer' }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [findings] = useState<Finding[]>(() => deriveFindings(upload.name, conditionId))
  const [active, setActive] = useState<Finding | null>(null)

  // Grad-CAM Controls state
  const [gradCamMode, setGradCamMode] = useState<GradCamViewMode>('hybrid')
  const [gradCamOpacity, setGradCamOpacity] = useState(0.68)
  const [gradCamThreshold, setGradCamThreshold] = useState(0.12)
  const [gradCamMatrix, setGradCamMatrix] = useState<number[][] | undefined>(undefined)
  const [targetLayer, setTargetLayer] = useState<string>('backbone.layer4 (ResNet-50)')

  // Fetch API / Mock Grad-CAM matrix if available
  useEffect(() => {
    let unmounted = false
    void fetchExplainability(conditionId).then((data) => {
      if (unmounted) return
      if (data.model.gradcam) {
        setTargetLayer(data.model.gradcam.target_layer)
        if (data.model.gradcam.matrix) {
          setGradCamMatrix(data.model.gradcam.matrix)
        }
      }
    })
    return () => {
      unmounted = true
    }
  }, [conditionId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (active) setActive(null)
      else onClose()
    }
    window.addEventListener('keydown', onKey)
    ref.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [active, onClose])

  return (
    <div className="absolute inset-0 z-50 grid place-items-center p-3 lg:p-5">
      <button
        type="button"
        aria-label="Dismiss preview"
        onClick={onClose}
        className="absolute inset-0 cursor-default"
        style={{ background: 'rgba(6,6,8,0.85)', backdropFilter: 'blur(3px)' }}
      />

      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-label={`Preview and ROI analysis of ${upload.name}`}
        className="relative flex h-full max-h-[94vh] w-full max-w-[1180px] flex-col rounded-panel outline-none"
        style={{
          background: '#17181B',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow:
            'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 24px 50px rgba(0,0,0,0.7)',
        }}
      >
        {/* Header */}
        <div
          className="flex shrink-0 items-center gap-3 px-5 py-3"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-[15px] font-medium text-ink">{upload.name}</h2>
            </div>
            <p className="mt-0.5 font-mono text-[12px] text-ink-faint">
              {findings.length} region
              {findings.length === 1 ? '' : 's'} flagged
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close preview"
            className="grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-[7px] text-ink-faint transition-colors duration-150 hover:text-ink"
            style={{ background: '#111214', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <IconClose className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Content body */}
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          {/* Scan viewport & Grad-CAM canvas */}
          <div className="relative flex min-h-0 flex-1 flex-col items-center justify-between p-4 bg-[#0A0B0D] overflow-hidden">
            <div className="relative flex-1 flex items-center justify-center w-full min-h-0">
              <div className="relative inline-block max-h-full max-w-full">
                <img
                  src={upload.objectUrl ?? ''}
                  alt={`Uploaded scan ${upload.name}`}
                  className="block max-h-[56vh] w-auto max-w-full rounded-[8px] object-contain"
                  style={{ border: '1px solid rgba(255,255,255,0.08)' }}
                />

                {/* Grad-CAM Heatmap Layer */}
                {gradCamMode !== 'contours' && (
                  <GradCamOverlay
                    findings={findings}
                    matrix={gradCamMatrix}
                    opacity={gradCamOpacity}
                    threshold={gradCamThreshold}
                    colormap="jet"
                  />
                )}

                {/* ROI Vector Bounding Rings & Markers */}
                {gradCamMode !== 'heatmap' &&
                  findings.map((f) => {
                    const color = SEVERITY_COLOR[f.severity]
                    const isActive = active?.id === f.id
                    return (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => setActive(isActive ? null : f)}
                        aria-label={`${f.label}, ${f.severity} severity. Inspect finding`}
                        aria-pressed={isActive}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-transform duration-150 hover:scale-105"
                        style={{ left: `${f.x * 100}%`, top: `${f.y * 100}%` }}
                      >
                        {/* Ring showing flagged boundary */}
                        <span
                          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none"
                          style={{
                            width: `${f.r * 480}px`,
                            height: `${f.r * 480}px`,
                            border: `2px solid ${color}`,
                            outline: '1px solid rgba(0,0,0,0.65)',
                            outlineOffset: '-2px',
                            boxShadow: `0 0 0 1px rgba(0,0,0,0.5), 0 0 16px ${color}66`,
                            opacity: isActive ? 1 : 0.75,
                            background: isActive ? `${color}28` : 'transparent',
                            transition: 'opacity 160ms ease-out, background 160ms ease-out',
                          }}
                        />
                        {/* Blinking central node */}
                        <span
                          className="marker-pulse relative block h-[12px] w-[12px] rounded-full"
                          style={{
                            background: color,
                            border: '2px solid rgba(10,10,12,0.9)',
                            boxShadow: `0 0 8px ${color}, 0 0 16px ${color}90`,
                          }}
                        />
                      </button>
                    )
                  })}
              </div>
            </div>

            {/* Grad-CAM Controls Toolbar */}
            <div className="w-full max-w-[560px] mt-3">
              <GradCamControls
                mode={gradCamMode}
                onModeChange={setGradCamMode}
                opacity={gradCamOpacity}
                onOpacityChange={setGradCamOpacity}
                threshold={gradCamThreshold}
                onThresholdChange={setGradCamThreshold}
                targetLayer={targetLayer}
              />
            </div>
          </div>

          {/* Explainability Detail Rail */}
          <aside
            className="console-scroll w-full md:w-[410px] shrink-0 overflow-y-auto p-4 md:p-5"
            style={{
              borderLeft: '1px solid rgba(255,255,255,0.06)',
              background: '#121316',
            }}
            aria-label="Region details and clinical explainability"
          >
            {!active ? (
              <div className="space-y-4">
                <div>
                  <h3 className="text-[13px] font-medium text-ink">Grad-CAM Attention & Localized ROI</h3>
                  <p className="mt-1 font-mono text-[11.5px] leading-relaxed text-ink-faint">
                    The thermal heatmap visualizes the final convolutional layer feature map activations.
                    Select any localized region or marker to inspect what that highlighted area signifies.
                  </p>
                </div>

                <div className="space-y-2">
                  {findings.map((f) => {
                    const color = SEVERITY_COLOR[f.severity]
                    return (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => setActive(f)}
                        className="group flex w-full cursor-pointer flex-col gap-1 rounded-[8px] p-3 text-left transition-all duration-150 hover:bg-white/[0.04]"
                        style={{
                          background: 'rgba(255,255,255,0.02)',
                          border: '1px solid rgba(255,255,255,0.04)',
                        }}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="flex items-center gap-2">
                            <span className="h-[7px] w-[7px] rounded-full" style={{ background: color }} />
                            <span className="font-mono text-[11px] uppercase tracking-wider" style={{ color }}>
                              {f.severity} severity
                            </span>
                          </span>
                          <span className="font-mono text-[11px] text-ink-faint">
                            saliency {(f.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <span className="text-[13px] font-medium text-ink group-hover:text-white">
                          {f.label}
                        </span>
                        {f.significance && (
                          <p className="line-clamp-2 text-[11px] leading-relaxed text-ink-faint">
                            {f.significance}
                          </p>
                        )}
                      </button>
                    )
                  })}
                </div>


              </div>
            ) : (
              <div className="space-y-4">
                {/* Active finding header */}
                <div className="flex items-start justify-between gap-2 pb-3 border-b border-white/5">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-[7px] w-[7px] shrink-0 rounded-full"
                        style={{ background: SEVERITY_COLOR[active.severity] }}
                      />
                      <span
                        className="font-mono text-[11px] uppercase tracking-wider font-medium"
                        style={{ color: SEVERITY_COLOR[active.severity] }}
                      >
                        {active.severity} severity
                      </span>
                      <span className="font-mono text-[11px] text-ink-faint">
                        ({active.id})
                      </span>
                    </div>
                    <h3 className="mt-1 text-[14.5px] font-medium leading-snug text-ink">
                      {active.label}
                    </h3>
                  </div>

                  <button
                    type="button"
                    onClick={() => setActive(null)}
                    aria-label="Back to region list"
                    className="cursor-pointer rounded px-2 py-1 font-mono text-[11px] text-ink-faint transition-colors duration-150 hover:bg-white/5 hover:text-ink"
                  >
                    view all
                  </button>
                </div>

                {/* What this highlighted area signifies */}
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-[11.5px] font-medium uppercase tracking-wider text-ink-dim">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500/80" />
                    What This Highlighted Area Signifies
                  </div>
                  <div
                    className="rounded-[8px] p-3 text-[12px] leading-relaxed text-ink-dim"
                    style={{
                      background: '#0D0E10',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  >
                    {active.significance || active.notes.join('. ')}
                  </div>
                </div>

                {/* Pathological mechanism */}
                {active.pathological_mechanism && (
                  <div className="space-y-1.5">
                    <div className="text-[11.5px] font-medium uppercase tracking-wider text-ink-dim">
                      Pathological Mechanism
                    </div>
                    <p className="text-[11.5px] leading-relaxed text-ink-faint">
                      {active.pathological_mechanism}
                    </p>
                  </div>
                )}

                {/* Differential diagnoses */}
                {active.differential_diagnoses && active.differential_diagnoses.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="text-[11.5px] font-medium uppercase tracking-wider text-ink-dim">
                      Differential Diagnoses (Mimics)
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {active.differential_diagnoses.map((diff) => (
                        <span
                          key={diff}
                          className="rounded-[5px] px-2 py-1 font-mono text-[11px] text-ink-faint"
                          style={{
                            background: 'rgba(255,255,255,0.03)',
                            border: '1px solid rgba(255,255,255,0.06)',
                          }}
                        >
                          {diff}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Measurable metrics */}
                <div className="space-y-1.5">
                  <div className="text-[11.5px] font-medium uppercase tracking-wider text-ink-dim">
                    Localized Morphometrics & Quantitative Values
                  </div>
                  <div
                    className="rounded-[8px] p-2.5 divide-y divide-white/5"
                    style={{
                      background: '#0D0E10',
                      border: '1px solid rgba(255,255,255,0.05)',
                    }}
                  >
                    <div className="flex items-baseline justify-between py-1.5">
                      <span className="font-mono text-[11.5px] text-ink-faint">Peak Saliency Attention</span>
                      <span className="font-mono text-[12px] tabular-nums text-ink">
                        {(active.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {Object.entries(active.metrics).map(([k, v]) => (
                      <div key={k} className="flex items-baseline justify-between py-1.5">
                        <span className="font-mono text-[11.5px] text-ink-faint">
                          {k.replace(/_/g, ' ')}
                        </span>
                        <span className="font-mono text-[12px] tabular-nums text-ink-dim">
                          {v}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>


              </div>
            )}
          </aside>
        </div>

      </div>
    </div>
  )
}
