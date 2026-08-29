import { useMemo } from 'react'
import { circuitStats, rotationsPerQubit, type VqcConfig } from '../lib/quantum/vqc'
import { LANE_COLOR, alpha } from '../lib/theme'

/**
 * Renders the actual circuit the model runs.
 *
 * The gate list is derived from the same config the simulator uses, so what is
 * drawn is what executes - the diagram cannot drift out of sync with the model.
 */

type Gate =
  | { kind: 'rot'; label: string; wire: number; col: number; encoding?: boolean }
  | { kind: 'cnot'; control: number; target: number; col: number }
  | { kind: 'measure'; wire: number; col: number }

function buildGates(cfg: VqcConfig): { gates: Gate[]; columns: number } {
  const gates: Gate[] = []
  const n = cfg.qubits
  let col = 0

  // ---- encoding ----
  if (cfg.featureMap === 'angle') {
    for (let q = 0; q < n; q++) {
      gates.push({ kind: 'rot', label: 'RY', wire: q, col, encoding: true })
    }
    col++
  } else if (cfg.featureMap === 'amplitude') {
    for (let q = 0; q < n; q++) gates.push({ kind: 'rot', label: 'RY', wire: q, col, encoding: true })
    col++
    for (let q = 0; q < n; q++) gates.push({ kind: 'rot', label: 'RZ', wire: q, col, encoding: true })
    col++
  } else {
    for (let q = 0; q < n; q++) gates.push({ kind: 'rot', label: 'H', wire: q, col, encoding: true })
    col++
    for (let q = 0; q < n; q++) gates.push({ kind: 'rot', label: 'RZ', wire: q, col, encoding: true })
    col++
    for (let q = 0; q < n - 1; q++) {
      gates.push({ kind: 'cnot', control: q, target: q + 1, col })
      col++
    }
  }

  // ---- variational layers ----
  const rot = rotationsPerQubit(cfg.ansatz)
  const labels =
    cfg.ansatz === 'strongly-entangling'
      ? ['RZ', 'RY', 'RZ']
      : cfg.ansatz === 'hardware-efficient'
        ? ['RY', 'RZ']
        : ['RY']

  // Cap what is drawn: beyond a few layers the picture stops being readable
  // and the stats line carries the real numbers.
  const drawLayers = Math.min(cfg.layers, 3)

  for (let l = 0; l < drawLayers; l++) {
    for (let r = 0; r < rot; r++) {
      for (let q = 0; q < n; q++) {
        gates.push({ kind: 'rot', label: labels[r], wire: q, col })
      }
      col++
    }
    if (cfg.ansatz === 'hardware-efficient') {
      for (let q = 0; q < n - 1; q++) {
        gates.push({ kind: 'cnot', control: q, target: q + 1, col })
        col++
      }
    } else {
      const stride = cfg.ansatz === 'strongly-entangling' ? (l % Math.max(1, n - 1)) + 1 : 1
      for (let q = 0; q < n; q++) {
        gates.push({ kind: 'cnot', control: q, target: (q + stride) % n, col })
        col++
      }
    }
  }

  gates.push({ kind: 'measure', wire: 0, col })
  return { gates, columns: col + 1 }
}

type Props = {
  config: VqcConfig
  /** feature names, labelling each wire's input */
  featureNames?: string[]
}

export function CircuitDiagram({ config, featureNames }: Props) {
  const { gates, columns } = useMemo(() => buildGates(config), [config])
  const stats = useMemo(() => circuitStats(config), [config])

  const n = config.qubits
  const rowH = 30
  const colW = 34
  const padL = 84
  const padR = 44
  const padT = 26
  const width = padL + columns * colW + padR
  const height = padT + n * rowH + 14
  const truncated = config.layers > 3

  const wireY = (q: number) => padT + q * rowH + rowH / 2
  const colX = (c: number) => padL + c * colW + colW / 2

  return (
    <div className="w-full">
      <div className="console-scroll overflow-x-auto">
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Quantum circuit: ${n} qubits, ${config.layers} layers, ${stats.gates} gates, depth ${stats.depth}`}
          style={{ minWidth: width }}
        >
          {/* wires */}
          {Array.from({ length: n }).map((_, q) => (
            <g key={q}>
              <line
                x1={padL - 8}
                y1={wireY(q)}
                x2={width - padR + 8}
                y2={wireY(q)}
                stroke="rgba(255,255,255,0.14)"
                strokeWidth="1"
              />
              <text
                x={padL - 16}
                y={wireY(q) + 3.5}
                textAnchor="end"
                fontSize="9.5"
                fill="#6A6C72"
                fontFamily="Inter, system-ui, sans-serif"
              >
                {featureNames?.[q]
                  ? `q${q} ${featureNames[q].slice(0, 9)}`
                  : `q${q} |0⟩`}
              </text>
            </g>
          ))}

          {/* gates */}
          {gates.map((g, i) => {
            if (g.kind === 'cnot') {
              const cy = wireY(g.control)
              const ty = wireY(g.target)
              const x = colX(g.col)
              return (
                <g key={i}>
                  <line
                    x1={x}
                    y1={cy}
                    x2={x}
                    y2={ty}
                    stroke={alpha(LANE_COLOR.quantum, 0.65)}
                    strokeWidth="1.2"
                  />
                  <circle cx={x} cy={cy} r="3" fill={LANE_COLOR.quantum} />
                  <circle
                    cx={x}
                    cy={ty}
                    r="6"
                    fill="none"
                    stroke={LANE_COLOR.quantum}
                    strokeWidth="1.2"
                  />
                  <line
                    x1={x - 6}
                    y1={ty}
                    x2={x + 6}
                    y2={ty}
                    stroke={LANE_COLOR.quantum}
                    strokeWidth="1.2"
                  />
                </g>
              )
            }

            if (g.kind === 'measure') {
              const x = colX(g.col)
              const y = wireY(g.wire)
              return (
                <g key={i}>
                  <rect
                    x={x - 11}
                    y={y - 10}
                    width="22"
                    height="20"
                    rx="4"
                    fill="#1B1C20"
                    stroke="rgba(255,255,255,0.22)"
                    strokeWidth="1"
                  />
                  {/* meter dial */}
                  <path
                    d={`M ${x - 6} ${y + 4} A 6 6 0 0 1 ${x + 6} ${y + 4}`}
                    fill="none"
                    stroke="#E8E9EB"
                    strokeWidth="1.1"
                  />
                  <line
                    x1={x}
                    y1={y + 4}
                    x2={x + 4}
                    y2={y - 3}
                    stroke="#E8E9EB"
                    strokeWidth="1.1"
                  />
                </g>
              )
            }

            const x = colX(g.col)
            const y = wireY(g.wire)
            const tone = g.encoding ? LANE_COLOR.classical : LANE_COLOR.quantum
            return (
              <g key={i}>
                <rect
                  x={x - 12}
                  y={y - 9}
                  width="24"
                  height="18"
                  rx="4"
                  fill="#17181B"
                  stroke={alpha(tone, 0.5)}
                  strokeWidth="1"
                />
                <text
                  x={x}
                  y={y + 3.5}
                  textAnchor="middle"
                  fontSize="9"
                  fill={tone}
                  fontFamily="Inter, system-ui, sans-serif"
                >
                  {g.label}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {/* legend + live stats */}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
        <span className="flex items-center gap-1.5 font-mono text-[9px] text-ink-faint">
          <span
            className="h-[8px] w-[8px] rounded-[2px] border"
            style={{ borderColor: alpha(LANE_COLOR.classical, 0.6) }}
          />
          data encoding
        </span>
        <span className="flex items-center gap-1.5 font-mono text-[9px] text-ink-faint">
          <span
            className="h-[8px] w-[8px] rounded-[2px] border"
            style={{ borderColor: alpha(LANE_COLOR.quantum, 0.6) }}
          />
          trainable
        </span>
        <span className="flex-1" />
        {truncated && (
          <span className="font-mono text-[9px]" style={{ color: LANE_COLOR.classical }}>
            showing 3 of {config.layers} layers
          </span>
        )}
      </div>
    </div>
  )
}
