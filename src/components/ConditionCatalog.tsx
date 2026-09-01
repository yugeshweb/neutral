import { useEffect, useState, type ReactNode } from 'react'
import { fetchCatalog } from '../lib/platform'
import type { CatalogEntry } from '../lib/platform'
import { IconArrowLeft, IconChevron } from './icons'
import { Wordmark } from './Wordmark'

type Props = { onHome: () => void }

/** Shell so the loading, error and loaded states all keep a way back. */
function CatalogShell({ onHome, children }: Props & { children: ReactNode }) {
  return (
    <div className="flex h-full flex-col bg-canvas">
      <header
        className="flex h-14 shrink-0 items-center gap-3 px-4"
        style={{
          background: '#111214',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
        }}
      >
        <button
          type="button"
          onClick={onHome}
          className="flex cursor-pointer items-center gap-2 rounded-[8px] px-2 py-1.5 text-ink-faint transition-colors duration-150 hover:text-ink"
          aria-label="Back to menu"
        >
          <IconArrowLeft className="h-3.5 w-3.5" />
          <Wordmark size={14} />
        </button>

        <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />

        <div>
          <div className="text-[14.5px] text-ink">Neurological conditions</div>
          <div className="font-mono text-[11px] text-ink-faint">
            clinical registry served by the backend
          </div>
        </div>
      </header>

      <div className="min-h-0 flex-1">{children}</div>
    </div>
  )
}

export function ConditionCatalog({ onHome }: Props) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedConditions, setExpandedConditions] = useState<Set<string>>(new Set())

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        const data = await fetchCatalog()
        setCatalog(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load catalog')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const toggleCondition = (id: string) => {
    setExpandedConditions((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  if (loading) {
    return (
      <CatalogShell onHome={onHome}>
        <div className="console-scroll h-full overflow-y-auto bg-canvas">
          <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col justify-center px-8 py-14">
            <div className="text-center">
              <div className="text-[14.5px] text-ink-dim">Loading condition catalog…</div>
              <div className="mt-2 font-mono text-[11.5px] text-ink-faint">GET /api/catalog</div>
            </div>
          </div>
        </div>
      </CatalogShell>
    )
  }

  if (error) {
    return (
      <CatalogShell onHome={onHome}>
        <div className="console-scroll h-full overflow-y-auto bg-canvas">
          <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col justify-center px-8 py-14">
            <div className="rounded-panel border border-solid border-lane-error/30 bg-canvas p-4">
              <div className="text-[14.5px] text-lane-error">{error}</div>
              <p className="mt-2.5 font-mono text-[11.5px] leading-relaxed text-ink-faint">
                The condition registry is served by the Python backend. Start it with{' '}
                <span className="text-ink-dim">python start_api.py 8765</span> from the
                backend directory, then set{' '}
                <span className="text-ink-dim">VITE_QML_API_URL=http://127.0.0.1:8765</span>.
                Train, Predict and Compare run entirely in-browser and do not need it.
              </p>
            </div>
          </div>
        </div>
      </CatalogShell>
    )
  }

  // Group by readiness tier
  const highRef = catalog.filter((e) => e.condition.readiness_tier === 'high_reference')
  const researchOnly = catalog.filter((e) => e.condition.readiness_tier === 'research_only')

  return (
    <CatalogShell onHome={onHome}>
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-4xl px-8 py-10">
        {/* the header carries the title; this states the scope caveat */}
        <div className="mb-10">
          <p className="max-w-2xl text-[14.5px] leading-relaxed text-ink-dim">
            This catalog is exhaustive only for the conditions listed below. It does not cover
            every neurological disease. High-reference entries have a comparatively strong reference
            standard for research evaluation; research-only entries have a weaker or unapproved
            label policy and must be read as risk/progression signals, never a diagnosis.
          </p>
        </div>

        {/* High Reference section */}
        {highRef.length > 0 && (
          <div className="mb-10">
            <div className="mb-4 flex items-center gap-2">
              <h2 className="text-[16px] font-medium text-ink">High Reference</h2>
              <span className="font-mono text-[12px] text-ink-faint">
                {highRef.length} condition{highRef.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className="space-y-2">
              {highRef.map((entry) => (
                <ConditionItem
                  key={entry.condition.condition_id}
                  entry={entry}
                  expanded={expandedConditions.has(entry.condition.condition_id)}
                  onToggle={() => toggleCondition(entry.condition.condition_id)}
                  tier="high_reference"
                />
              ))}
            </div>
          </div>
        )}

        {/* Research Only section */}
        {researchOnly.length > 0 && (
          <div>
            <div className="mb-4 flex items-center gap-2">
              <h2 className="text-[16px] font-medium text-ink">Research Only</h2>
              <span className="font-mono text-[12px] text-ink-faint">
                {researchOnly.length} condition{researchOnly.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className="space-y-2">
              {researchOnly.map((entry) => (
                <ConditionItem
                  key={entry.condition.condition_id}
                  entry={entry}
                  expanded={expandedConditions.has(entry.condition.condition_id)}
                  onToggle={() => toggleCondition(entry.condition.condition_id)}
                  tier="research_only"
                />
              ))}
            </div>
          </div>
        )}

        {catalog.length === 0 && (
          <div className="rounded-panel border border-solid border-line p-6 text-center">
            <div className="text-[14.5px] text-ink-dim">No conditions available in the catalog.</div>
          </div>
        )}
      </div>
    </div>
    </CatalogShell>
  )
}

type ItemProps = {
  entry: CatalogEntry
  expanded: boolean
  onToggle: () => void
  tier: 'high_reference' | 'research_only'
}

function ConditionItem({ entry, expanded, onToggle, tier }: ItemProps) {
  const availabilityColor =
    entry.availability === 'available'
      ? { bg: 'rgba(95, 168, 140, 0.2)', text: '#5FA88C', label: 'Available' }
      : { bg: 'rgba(138, 143, 152, 0.2)', text: '#8A8F98', label: 'Not available' }

  return (
    <div
      className="rounded-panel overflow-hidden"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      {/* Collapsed header */}
      <button
        type="button"
        onClick={onToggle}
        className="w-full px-5 py-4 text-left transition-opacity hover:opacity-80"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h3 className="text-[14.5px] font-medium text-ink">{entry.condition.name}</h3>
            <div className="mt-1.5 flex items-center gap-2">
              <span className="font-mono text-[11px] text-ink-faint">
                {entry.condition.task_type}
              </span>
              {tier === 'research_only' && (
                <span
                  className="rounded px-1.5 py-0.5 font-mono text-[10.5px] font-medium"
                  style={{ background: 'rgba(160, 120, 60, 0.2)', color: '#C08A3E' }}
                >
                  RESEARCH ONLY
                </span>
              )}
              <span
                className="rounded px-1.5 py-0.5 font-mono text-[10.5px] font-medium"
                style={{ background: availabilityColor.bg, color: availabilityColor.text }}
              >
                {availabilityColor.label}
              </span>
            </div>
          </div>
          <span
            className="mt-0.5 inline-flex shrink-0 transition-transform"
            style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
          >
            <IconChevron className="h-4 w-4 text-ink-faint" />
          </span>
        </div>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div
          className="border-t border-solid px-5 py-4"
          style={{ borderColor: 'rgba(255,255,255,0.08)' }}
        >
          <div className="space-y-4 text-[12px]">
            {/* Modalities */}
            <div>
              <div className="font-medium text-ink-dim">Required modalities</div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {entry.condition.required_modalities.map((m) => (
                  <span
                    key={m}
                    className="rounded px-2 py-0.5 font-mono text-[11px]"
                    style={{ background: '#0D0E10', color: '#8A8F98' }}
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>

            {entry.condition.optional_modalities.length > 0 && (
              <div>
                <div className="font-medium text-ink-dim">Optional modalities</div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {entry.condition.optional_modalities.map((m) => (
                    <span
                      key={m}
                      className="rounded px-2 py-0.5 font-mono text-[11px]"
                      style={{ background: '#0D0E10', color: '#6A6C72' }}
                    >
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Expected output */}
            <div>
              <div className="font-medium text-ink-dim">Expected output</div>
              <p className="mt-1 leading-relaxed text-ink-faint">{entry.condition.expected_output}</p>
            </div>

            {/* Reference datasets */}
            {entry.condition.reference_datasets.length > 0 && (
              <div>
                <div className="font-medium text-ink-dim">Reference datasets</div>
                <ul className="mt-1 space-y-0.5">
                  {entry.condition.reference_datasets.map((ref, idx) => (
                    <li key={idx}>
                      <a
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-lane-quantum hover:opacity-80"
                      >
                        {ref.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Limitations */}
            {entry.condition.limitations.length > 0 && (
              <div>
                <div className="font-medium text-ink-dim">Limitations</div>
                <ul className="mt-1 space-y-0.5 pl-4">
                  {entry.condition.limitations.map((lim) => (
                    <li key={lim} className="list-disc text-ink-faint">
                      {lim}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Models */}
            {entry.models.length > 0 && (
              <div>
                <div className="mb-2 font-medium text-ink-dim">Models</div>
                <div className="space-y-2 border-t border-solid pt-3" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
                  {entry.models.map((model) => (
                    <div
                      key={model.model_id}
                      className="rounded px-3 py-2"
                      style={{ background: '#0D0E10', border: '1px solid rgba(255,255,255,0.04)' }}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="font-medium text-ink">{model.display_name}</div>
                          <div className="mt-0.5 text-[11px] text-ink-faint">v{model.version}</div>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          <span
                            className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                            style={{
                              background:
                                model.lifecycle === 'operational_reference'
                                  ? 'rgba(95, 168, 140, 0.2)'
                                  : 'rgba(160, 120, 60, 0.2)',
                              color:
                                model.lifecycle === 'operational_reference'
                                  ? '#5FA88C'
                                  : '#C08A3E',
                            }}
                          >
                            {model.lifecycle}
                          </span>
                          {model.quantum && (
                            <span
                              className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                              style={{ background: 'rgba(62, 140, 158, 0.2)', color: '#3E8C9E' }}
                            >
                              HYBRID
                            </span>
                          )}
                          {!model.quantum && (
                            <span
                              className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                              style={{ background: 'rgba(138, 143, 152, 0.2)', color: '#8A8F98' }}
                            >
                              CLASSICAL
                            </span>
                          )}
                        </div>
                      </div>
                      {model.safety.disclaimer && (
                        <div className="mt-2 border-t border-solid pt-2 text-[11px] leading-relaxed text-ink-faint" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
                          {model.safety.disclaimer}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
