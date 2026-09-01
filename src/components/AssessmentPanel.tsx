import { useState } from 'react'
import type { AssessmentRun } from '../lib/platform'
import { FindingCard } from './FindingCard'
import { IconChevron } from './icons'

type Props = {
  run: AssessmentRun
}

function getModelStatusColor(
  status: AssessmentRun['routing'][number]['status'],
): { bg: string; text: string } {
  switch (status) {
    case 'compatible':
      return { bg: 'rgba(95, 168, 140, 0.2)', text: '#5FA88C' }
    case 'incompatible':
      return { bg: 'rgba(163, 84, 61, 0.2)', text: '#A3543D' }
    case 'not available':
      return { bg: 'rgba(138, 143, 152, 0.2)', text: '#8A8F98' }
    case 'insufficient data':
      return { bg: 'rgba(160, 120, 60, 0.2)', text: '#C08A3E' }
    case 'ready':
      return { bg: 'rgba(95, 168, 140, 0.2)', text: '#5FA88C' }
    case 'running':
      return { bg: 'rgba(192, 138, 62, 0.2)', text: '#C08A3E' }
    case 'completed':
      return { bg: 'rgba(95, 168, 140, 0.2)', text: '#5FA88C' }
    case 'abstained':
      return { bg: 'rgba(62, 140, 158, 0.2)', text: '#3E8C9E' }
    case 'failed':
      return { bg: 'rgba(163, 84, 61, 0.2)', text: '#A3543D' }
  }
}

export function AssessmentPanel({ run }: Props) {
  const [errorsOpen, setErrorsOpen] = useState(false)

  return (
    <div className="space-y-4">
      {/* Synthetic banner if present */}
      {run.synthetic && (
        <div
          className="rounded-panel px-4 py-3 font-mono text-[11px] font-medium tracking-[0.05em]"
          style={{ background: 'rgba(163, 84, 61, 0.3)', color: '#A3543D' }}
        >
          SYNTHETIC / DEMO RUN
        </div>
      )}

      {/* Run-level disclaimer */}
      <div
        className="rounded-panel border border-solid px-4 py-3"
        style={{ borderColor: 'rgba(255,255,255,0.08)', background: 'rgba(160,120,60,0.08)' }}
      >
        <div className="text-[11.5px] leading-relaxed text-ink-faint">{run.disclaimer}</div>
      </div>

      {/* Routing table */}
      <div
        className="rounded-panel overflow-hidden"
        style={{
          background: '#17181B',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        <div className="text-[12px]">
          {/* Header */}
          <div
            className="flex border-b border-solid px-4 py-2 font-medium text-ink-dim"
            style={{ borderColor: 'rgba(255,255,255,0.08)' }}
          >
            <div className="flex-1">Model</div>
            <div className="w-24">Status</div>
            <div className="flex-1">Reason</div>
          </div>

          {/* Routing decisions */}
          {run.routing.map((decision) => {
            const statusColor = getModelStatusColor(decision.status)
            return (
              <div
                key={`${decision.model_id}-${decision.model_version}`}
                className="flex border-b border-solid px-4 py-2.5 text-ink-faint last:border-b-0"
                style={{ borderColor: 'rgba(255,255,255,0.04)' }}
              >
                <div className="flex-1">
                  <div className="font-mono text-[11px] text-ink">{decision.model_id}</div>
                  <div className="text-[11px] text-ink-faint">{decision.model_version}</div>
                </div>
                <div
                  className="w-24 rounded px-2 py-1 text-center font-medium"
                  style={{ background: statusColor.bg, color: statusColor.text }}
                >
                  {decision.status}
                </div>
                <div className="flex-1 text-[11.5px] leading-snug text-ink-faint">
                  {decision.reason}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Findings list */}
      <div className="space-y-3">
        <div className="text-[13px] font-medium text-ink-dim">Findings</div>
        <div className="space-y-3">
          {run.findings.map((finding) => (
            <FindingCard key={finding.finding_id} finding={finding} />
          ))}
        </div>
      </div>

      {/* Errors section (if any) */}
      {run.errors.length > 0 && (
        <div
          className="rounded-panel border border-solid"
          style={{
            background: '#17181B',
            border: '1px solid rgba(255,255,255,0.06)',
          }}
        >
          <button
            type="button"
            onClick={() => setErrorsOpen(!errorsOpen)}
            className="flex w-full items-center justify-between px-4 py-3 text-left transition-opacity hover:opacity-80"
          >
            <span className="text-[12px] font-medium text-lane-error">Errors ({run.errors.length})</span>
            <span
              className="inline-flex transition-transform"
              style={{ transform: errorsOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              <IconChevron className="h-3 w-3 text-lane-error" />
            </span>
          </button>

          {errorsOpen && (
            <div
              className="border-t border-solid px-4 py-3"
              style={{ borderColor: 'rgba(255,255,255,0.08)' }}
            >
              <div className="mb-2 text-[11px] leading-relaxed text-ink-faint">
                One or more models encountered errors, but completed findings shown above remain valid and
                should not be discredited.
              </div>
              <div className="space-y-1.5">
                {run.errors.map((err, idx) => (
                  <div
                    key={idx}
                    className="rounded px-2 py-1 font-mono text-[11px] leading-relaxed text-lane-error"
                    style={{ background: 'rgba(163, 84, 61, 0.1)' }}
                  >
                    {err}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
