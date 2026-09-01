import { LANE_COLOR, alpha } from '../lib/theme'
import { Tooltip } from './Tooltip'

type Props = {
  qubits: number
  /** shown alongside, when the circuit is configured */
  depth?: number
  gates?: number
  params?: number
  compact?: boolean
}

/**
 * The qubit count, shown on every screen after feature selection.
 *
 * It is the constraint everything else bends around: one retained feature is
 * one qubit, and circuit width drives simulation cost, depth and trainability.
 * Keeping it permanently visible is the clearest way to show the platform is
 * built around that limit rather than ignoring it.
 */
export function QubitBadge({ qubits, depth, gates, params, compact }: Props) {
  const accent = LANE_COLOR.quantum

  return (
    <Tooltip
      term="Qubits"
      body={`The quantum register is exactly as wide as the feature set: ${qubits} retained features means ${qubits} qubits. Simulating it costs 2^${qubits} = ${(2 ** qubits).toLocaleString()} amplitudes, which is why feature selection is the binding constraint on this platform rather than an optional tidy-up.`}
    >
      <span
        className="inline-flex items-center gap-2 rounded-[6px] px-2 py-[3px] font-mono text-[12px]"
        style={{
          color: accent,
          background: alpha(accent, 0.09),
          border: `1px solid ${alpha(accent, 0.24)}`,
        }}
      >
        <span className="flex items-center gap-1">
          {/* register glyph: one tick per wire */}
          <span className="flex items-center gap-[2px]" aria-hidden>
            {Array.from({ length: Math.min(qubits, 10) }).map((_, i) => (
              <span
                key={i}
                className="block h-[9px] w-[1.5px] rounded-full"
                style={{ background: accent, opacity: 0.45 + (i / 20) }}
              />
            ))}
          </span>
          <span className="tabular-nums">{qubits}</span>
          <span className="text-ink-faint">qubits</span>
        </span>

        {!compact && depth !== undefined && (
          <>
            <span className="text-ink-faint/40">/</span>
            <span className="tabular-nums text-ink-dim">
              depth {depth}
            </span>
          </>
        )}
        {!compact && gates !== undefined && (
          <>
            <span className="text-ink-faint/40">/</span>
            <span className="tabular-nums text-ink-dim">{gates} gates</span>
          </>
        )}
        {!compact && params !== undefined && (
          <>
            <span className="text-ink-faint/40">/</span>
            <span className="tabular-nums text-ink-dim">{params} params</span>
          </>
        )}
      </span>
    </Tooltip>
  )
}
