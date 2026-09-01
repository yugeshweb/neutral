import { ADAPTERS } from '../lib/ingest'
import type { IngestAdapter } from '../lib/ingest'
import { LANE_COLOR, alpha } from '../lib/theme'
import { IconCheck, IconLock } from './icons'

const SYSTEM_NOTE: Record<string, string> = {
  'EHR / EMR': 'hospital record system',
  PACS: 'imaging archive',
  LIS: 'lab information system',
  Sequencing: 'genomics pipeline',
  'Manual export': 'clinician export',
}

type Props = {
  /** highlighted because it produced the loaded dataset */
  activeFormat: string | null
  onPick: (a: IngestAdapter) => void
  disabled: boolean
}

/**
 * Lists every source format the platform knows about, implemented or not.
 *
 * Showing the unimplemented ones is deliberate: it makes the adapter layer's
 * shape visible and states plainly what is and is not built, which is a more
 * defensible answer than quietly offering only CSV.
 */
export function SourcePicker({ activeFormat, onPick, disabled }: Props) {
  return (
    <div className="space-y-1.5">
      {ADAPTERS.map((a) => {
        const ready = a.status === 'implemented'
        const active = activeFormat === a.format
        const accent = ready ? LANE_COLOR.quantum : LANE_COLOR.shared

        return (
          <button
            key={a.format}
            type="button"
            onClick={() => onPick(a)}
            disabled={disabled}
            title={a.description}
            className="w-full cursor-pointer rounded-[8px] px-2.5 py-2 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
            style={{
              background: active ? alpha(accent, 0.08) : '#0D0E10',
              border: `1px solid ${active ? alpha(accent, 0.3) : 'rgba(255,255,255,0.05)'}`,
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
            }}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="grid h-[13px] w-[13px] shrink-0 place-items-center rounded-full"
                style={{
                  background: ready ? alpha(accent, 0.16) : 'rgba(255,255,255,0.04)',
                  color: ready ? accent : '#6A6C72',
                }}
              >
                {ready ? (
                  <IconCheck className="h-[8px] w-[8px]" />
                ) : (
                  <IconLock className="h-[8px] w-[8px]" />
                )}
              </span>
              <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-dim">
                {a.label}
              </span>
              <span className="shrink-0 font-mono text-[11px] text-ink-faint">
                {a.extensions[0]}
              </span>
            </div>

            <div className="mt-1 flex items-center gap-1.5 pl-[19px]">
              <span className="font-mono text-[11px] text-ink-faint/80">
                {SYSTEM_NOTE[a.system] ?? a.system}
              </span>
              {a.vocabularies.length > 0 && (
                <>
                  <span className="text-ink-faint/40">/</span>
                  <span className="font-mono text-[11px] text-ink-faint/80">
                    {a.vocabularies.join(' + ')}
                  </span>
                </>
              )}
            </div>
          </button>
        )
      })}
    </div>
  )
}
