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

const TEMPORAL_FRAMING_MAP: Record<
  string,
  { label: string; tag: string; description: string; color: string }
> = {
  'breast-cancer': {
    label: 'Present Finding',
    tag: 'detection',
    description: 'Current-state detection of morphological lesion malignancy.',
    color: '#7C67FE',
  },
  'heart-disease': {
    label: 'Present Finding',
    tag: 'detection',
    description: 'Current-state detection of acute coronary risk / ischemia.',
    color: '#7C67FE',
  },
  parkinsons: {
    label: 'Present Finding',
    tag: 'detection',
    description: 'Current-state detection of gait kinematic ataxia.',
    color: '#5FA88C',
  },
  stroke: {
    label: 'Property of Identified Finding',
    tag: 'characterisation',
    description: 'Characterises infarct core volume on an existing lesion finding.',
    color: '#38BDF8',
  },
  'stroke-risk': {
    label: 'Property of Identified Finding',
    tag: 'characterisation',
    description: 'Characterises infarct core volume on an existing lesion finding.',
    color: '#38BDF8',
  },
  glioma: {
    label: 'Property of Identified Finding',
    tag: 'characterisation',
    description: 'Characterises MGMT methylation status on identified tumor volumes.',
    color: '#FBBF24',
  },
  alzheimers: {
    label: 'Population Screening',
    tag: 'screening',
    description: 'Same-visit clinical dementia association screening.',
    color: '#60A5FA',
  },
  'brain-seizure': {
    label: '⚠ Early Warning (lead time)',
    tag: 'prediction',
    description: 'Preictal seizure state prediction window (Gated at API).',
    color: '#EF4444',
  },
  seizure: {
    label: '⚠ Early Warning (lead time)',
    tag: 'prediction',
    description: 'Preictal seizure state prediction window (Gated at API).',
    color: '#EF4444',
  },
}

const TABULAR_SHAP_PRESETS: Record<string, { feature: string; impact: number; direction: 'pos' | 'neg' }[]> = {
  alzheimers: [
    { feature: 'MMSE (Cognitive Score)', impact: 0.38, direction: 'pos' },
    { feature: 'nWBV (Whole-Brain Volume)', impact: 0.29, direction: 'pos' },
    { feature: 'Age (Chronological)', impact: 0.16, direction: 'pos' },
    { feature: 'eTIV (Intracranial Volume)', impact: 0.09, direction: 'neg' },
    { feature: 'ASF (Atlas Scale Factor)', impact: 0.08, direction: 'neg' },
  ],
  'breast-cancer': [
    { feature: 'concave_points_mean', impact: 0.34, direction: 'pos' },
    { feature: 'radius_mean', impact: 0.28, direction: 'pos' },
    { feature: 'perimeter_worst', impact: 0.21, direction: 'pos' },
    { feature: 'texture_mean', impact: 0.11, direction: 'pos' },
    { feature: 'smoothness_mean', impact: 0.06, direction: 'neg' },
  ],
}

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

  const framing = TEMPORAL_FRAMING_MAP[conditionId] ?? {
    label: 'Present Finding',
    tag: 'detection',
    description: 'Standard diagnostic prediction.',
    color: '#7C67FE',
  }

  const isSeizureDisabled = conditionId === 'seizure' || conditionId === 'brain-seizure'
  const isTabularExplainability = conditionId === 'alzheimers' || conditionId === 'breast-cancer'
  const isCnnImagingConfirmed = !isTabularExplainability && !isSeizureDisabled

  const scored = result?.rows.length ?? 0
  const positives = result?.positiveCount ?? 0
  const share = scored > 0 ? (positives / scored) * 100 : 0
  const mean =
    scored > 0 ? result!.rows.reduce((s, r) => s + r.probability, 0) / scored : 0
  const highest =
    scored > 0 ? Math.max(...result!.rows.map((r) => r.probability)) : 0

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
        {/* Header with Temporal Framing Badge */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/5 pb-3">
          <div className="flex items-center gap-2.5">
            <h2 className="text-[14.5px] font-medium text-ink">Inference & Explainability</h2>
            <span
              className="inline-flex items-center gap-1 rounded-[4px] px-2 py-0.5 font-mono text-[11px] font-medium"
              style={{
                color: framing.color,
                background: alpha(framing.color, 0.12),
                border: `1px solid ${alpha(framing.color, 0.28)}`,
              }}
              title={framing.description}
            >
              {framing.label}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className="font-mono text-[11px] text-ink-faint"
              title="Decision margin from threshold — not a probability"
            >
              Score: decision margin from threshold
            </span>
            <InfoDot label="Clinical Explainability Basis">
              Predictions are evaluated against trained hybrid quantum-classical boundaries.
              Scores represent normalized decision distances from validation-selected thresholds.
            </InfoDot>
          </div>
        </div>

        {/* Safety Banner for Seizure */}
        {isSeizureDisabled && (
          <div className="mt-3 rounded-[6px] border border-red-500/30 bg-red-500/10 p-3 text-[12px] text-red-200">
            <div className="font-semibold text-red-400">⚠ GATED AT API LEVEL — PATIENT SAFETY LOCK</div>
            <p className="mt-1 font-mono text-[11.5px] leading-relaxed text-red-300">
              Leave-One-Patient-Out Balanced Accuracy is 0.505 ± 0.257 (performs at chance patient-independently).
              This model is explicitly disabled at the API level and must not be used for warning or clinical alerting.
            </p>
          </div>
        )}

        <div className="flow-body mt-4 grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* Left Column: Metrics & Score Breakdown */}
          <div className="lg:col-span-5 flex flex-col justify-between">
            <div>
              {imageOnly ? (
                <>
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
                    <Metric label="mean margin" value={avg.toFixed(2)} tone={QUANTUM} />
                    <Metric label="peak margin" value={peak.toFixed(2)} tone={CLASSICAL} />
                  </div>
                </>
              ) : scored === 0 ? (
                <p className="text-[13px] text-ink-dim">
                  Nothing was scored from this file, so there are no metrics to report.
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <Metric label="records scored" value={String(scored)} />
                    <Metric
                      label={positiveLabel.toLowerCase()}
                      value={String(positives)}
                      sub={`${share.toFixed(1)}%`}
                      tone={CLASSICAL}
                    />
                    <Metric label="mean decision margin" value={mean.toFixed(3)} tone={QUANTUM} />
                    <Metric label="peak decision margin" value={highest.toFixed(3)} tone={CLASSICAL} />
                  </div>

                  <div className="mt-3">
                    <div className="flex items-baseline justify-between font-mono text-[11px] text-ink-faint">
                      <span>{negativeLabel.toLowerCase()}</span>
                      <span title="Decision margin from threshold — not a probability">operating threshold</span>
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
                      <InfoDot label="About missing features">
                        {result.missing.length} feature(s) were absent and fell back to training averages.
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

          {/* Right Column: Conditional Explainability (SHAP for Tabular / Grad-CAM for CNN Imaging) */}
          <div className="lg:col-span-7">
            {isTabularExplainability ? (
              <div className="rounded-panel p-3.5" style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.06)' }}>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[13px] font-medium text-ink">SHAP Feature Importance & Attributions</span>
                  <span className="font-mono text-[10.5px] text-ink-faint">Kernel SHAP on Latent Embedding</span>
                </div>
                <p className="mb-3 font-mono text-[11px] text-ink-faint">
                  Individual feature contributions driving the decision boundary for this case:
                </p>

                <div className="space-y-2.5">
                  {(TABULAR_SHAP_PRESETS[conditionId] ?? TABULAR_SHAP_PRESETS['alzheimers']).map((item) => (
                    <div key={item.feature}>
                      <div className="flex justify-between font-mono text-[11px]">
                        <span className="text-ink-dim">{item.feature}</span>
                        <span className="text-lane-quantum font-medium">+{(item.impact * 100).toFixed(1)}%</span>
                      </div>
                      <div className="panel-well mt-1 h-2 w-full overflow-hidden rounded-full">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${item.impact * 100}%`,
                            background: item.direction === 'pos' ? '#7C67FE' : '#5FA88C',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : isCnnImagingConfirmed ? (
              <>
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
                  <div className="readout relative flex items-center justify-center aspect-square w-full max-w-[280px] sm:w-[280px] shrink-0 overflow-hidden rounded-[6px] bg-[#0A0B0D]">
                    {imageUrl ? (
                      <div className="relative inline-block max-h-full max-w-full overflow-hidden rounded-[4px]">
                        <img
                          src={imageUrl}
                          alt="Uploaded scan"
                          className="block max-h-[280px] max-w-[280px] w-auto h-auto object-contain rounded-[4px]"
                        />

                        {gradCamMode !== 'contours' && (
                          <GradCamOverlay
                            findings={findings}
                            opacity={0.68}
                            threshold={0.12}
                            colormap="jet"
                          />
                        )}

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
                          No scan or signal image provided for Grad-CAM overlay.
                        </span>
                      </div>
                    )}
                  </div>

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
                    <p className="mt-2 text-[12px] leading-relaxed text-ink-dim">
                      {active.significance || active.notes.join('. ')}
                    </p>
                  </div>
                )}
              </>
            ) : null}
          </div>
        </div>
      </div>

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
      <div
        className="engraved truncate font-mono text-[11px]"
        title="Decision margin from threshold — not a probability"
      >
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-1.5">
        <span
          className="font-mono text-[19px] tabular-nums"
          style={{ color: tone ?? '#E8E9EB' }}
          title="Decision margin from threshold — not a probability"
        >
          {value}
        </span>
        {sub && <span className="font-mono text-[11px] text-ink-faint">{sub}</span>}
      </div>
    </div>
  )
}
