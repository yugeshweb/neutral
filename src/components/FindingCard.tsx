import type { Finding } from '../lib/platform'

type Props = {
  finding: Finding
}

function getStatusColor(status: Finding['status']): { bg: string; text: string; label: string } {
  switch (status) {
    case 'not evaluated':
      return { bg: 'rgba(106, 108, 114, 0.2)', text: '#9A9CA1', label: 'Not evaluated' }
    case 'not available':
      return { bg: 'rgba(138, 143, 152, 0.2)', text: '#8A8F98', label: 'Not available' }
    case 'insufficient data':
      return { bg: 'rgba(160, 120, 60, 0.2)', text: '#C08A3E', label: 'Insufficient data' }
    case 'negative':
      return { bg: 'rgba(163, 84, 61, 0.2)', text: '#A3543D', label: 'Negative' }
    case 'positive':
      return { bg: 'rgba(95, 168, 140, 0.2)', text: '#5FA88C', label: 'Positive' }
    case 'abstained':
      return { bg: 'rgba(62, 140, 158, 0.2)', text: '#3E8C9E', label: 'Abstained' }
  }
}

function getExplanationMessage(status: Finding['explanation_status']): string {
  switch (status) {
    case 'available':
      return ''
    case 'unavailable':
      return 'Explanation unavailable'
    case 'not_applicable':
      return 'Explanation not applicable'
    case 'surrogate':
      return 'Surrogate features used'
  }
}

export function FindingCard({ finding }: Props) {
  const statusColor = getStatusColor(finding.status)
  const explanationMsg = getExplanationMessage(finding.explanation_status)

  return (
    <div
      className="rounded-panel p-5 text-left"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8)',
      }}
    >
      {/* Synthetic banner if present */}
      {finding.synthetic && (
        <div
          className="mb-4 inline-block rounded px-2 py-1 font-mono text-[9px] font-medium tracking-[0.05em]"
          style={{ background: 'rgba(163, 84, 61, 0.3)', color: '#A3543D' }}
        >
          SYNTHETIC / DEMO
        </div>
      )}

      {/* Header: condition, status, model version */}
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="flex-1">
          <h3 className="text-[13px] font-medium text-ink">{finding.condition_name}</h3>
          <p className="mt-0.5 text-[10px] text-ink-faint">{finding.task_type}</p>
        </div>
        <span
          className="shrink-0 rounded px-2 py-1 font-mono text-[9px] font-medium"
          style={{ background: statusColor.bg, color: statusColor.text }}
        >
          {statusColor.label}
        </span>
      </div>

      {/* Model and score info */}
      <div className="mb-4 space-y-1 border-t border-solid border-line pt-3">
        <div className="flex justify-between text-[10px]">
          <span className="text-ink-faint">Model version:</span>
          <span className="font-mono text-ink">{finding.model_version}</span>
        </div>
        {finding.score !== null && (
          <div className="flex justify-between text-[10px]">
            <span className="text-ink-faint">Score:</span>
            <span className="font-mono text-ink">
              {finding.score.toFixed(3)} ({finding.score_type})
              {finding.threshold !== null && ` / threshold ${finding.threshold.toFixed(3)}`}
            </span>
          </div>
        )}
      </div>

      {/* Uncertainty */}
      {finding.uncertainty && (
        <div className="mb-4 space-y-1 border-t border-solid border-line pt-3">
          <div className="text-[10px] font-medium text-ink-dim">Uncertainty</div>
          <div className="ml-1 space-y-0.5 text-[9.5px]">
            <div className="flex justify-between">
              <span className="text-ink-faint">Kind:</span>
              <span className="text-ink">{finding.uncertainty.kind}</span>
            </div>
            {finding.uncertainty.value !== null && (
              <div className="flex justify-between">
                <span className="text-ink-faint">Value:</span>
                <span className="font-mono text-ink">{finding.uncertainty.value.toFixed(4)}</span>
              </div>
            )}
            {finding.uncertainty.lower !== null && finding.uncertainty.upper !== null && (
              <div className="flex justify-between">
                <span className="text-ink-faint">Range:</span>
                <span className="font-mono text-ink">
                  {finding.uncertainty.lower.toFixed(4)} – {finding.uncertainty.upper.toFixed(4)}
                </span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-ink-faint">Calibration:</span>
              <span className="text-ink">{finding.uncertainty.calibration_status}</span>
            </div>
          </div>
        </div>
      )}

      {/* Input coverage */}
      {finding.input_coverage && (
        <div className="mb-4 space-y-2 border-t border-solid border-line pt-3">
          <div className="text-[10px] font-medium text-ink-dim">Input coverage</div>
          <div className="ml-1 space-y-1.5 text-[9.5px]">
            <div className="flex items-center justify-between">
              <span className="text-ink-faint">
                {finding.input_coverage.required_present} / {finding.input_coverage.required_total}{' '}
                required
              </span>
              <span className="font-mono text-ink">
                {(finding.input_coverage.coverage_ratio * 100).toFixed(0)}%
              </span>
            </div>
            {/* Coverage bar */}
            <div
              className="h-1.5 rounded-full"
              style={{
                background: '#0D0E10',
                border: '1px solid rgba(255,255,255,0.08)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${finding.input_coverage.coverage_ratio * 100}%`,
                  height: '100%',
                  background: '#5FA88C',
                  transition: 'width 0.2s ease',
                }}
              />
            </div>
            {finding.input_coverage.missing.length > 0 && (
              <div className="text-[9px] text-ink-faint">
                Missing: {finding.input_coverage.missing.join(', ')}
              </div>
            )}
            {finding.input_coverage.quality_failed.length > 0 && (
              <div className="text-[9px] text-lane-error">
                Quality issues: {finding.input_coverage.quality_failed.join(', ')}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Evidence or explanation message */}
      {explanationMsg && (
        <div className="mb-4 border-t border-solid border-line pt-3">
          <div className="text-[10px] text-ink-dim">{explanationMsg}</div>
        </div>
      )}

      {finding.evidence.length > 0 && explanationMsg === '' && (
        <div className="mb-4 space-y-1.5 border-t border-solid border-line pt-3">
          <div className="text-[10px] font-medium text-ink-dim">Evidence</div>
          <div className="ml-1 space-y-1 text-[9.5px]">
            {finding.evidence.map((item) => (
              <div key={item.evidence_id} className="flex justify-between">
                <span className="text-ink-faint">{item.label}</span>
                <span className="font-mono text-ink">
                  {item.value !== null ? item.value.toFixed(4) : '—'}
                  {item.unit ? ` ${item.unit}` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Limitations */}
      {finding.limitations.length > 0 && (
        <div className="mb-4 space-y-1 border-t border-solid border-line pt-3">
          <div className="text-[10px] font-medium text-ink-dim">Limitations</div>
          <ul className="ml-3 space-y-0.5 text-[9.5px] text-ink-faint">
            {finding.limitations.map((lim) => (
              <li key={lim} className="list-disc">
                {lim}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimer — always shown */}
      <div
        className="border-t border-solid border-line pt-3"
        style={{ borderColor: 'rgba(255,255,255,0.08)' }}
      >
        <div className="text-[9px] leading-relaxed text-ink-faint">{finding.disclaimer}</div>
      </div>
    </div>
  )
}
