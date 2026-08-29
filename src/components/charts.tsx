import type { BoxStats } from '../lib/ml/stats'
import { LANE_COLOR, alpha } from '../lib/theme'

/**
 * Small chart primitives, drawn as inline SVG.
 *
 * No chart library: these are simple enough to draw directly, and doing so
 * keeps them on the same palette and hairline weights as the rest of the UI.
 */

const AXIS = 'rgba(255,255,255,0.14)'
const GRID = 'rgba(255,255,255,0.05)'

// ---- convergence -----------------------------------------------------------

export function ConvergenceChart({
  points,
  height = 150,
  showAccuracy = true,
}: {
  points: { epoch: number; loss: number; trainAccuracy: number }[]
  height?: number
  showAccuracy?: boolean
}) {
  const w = 520
  const h = height
  const padL = 34
  const padR = 34
  const padB = 22
  const padT = 10

  if (points.length === 0) {
    return (
      <div
        className="grid place-items-center rounded-[8px] panel-well"
        style={{ height }}
      >
        <span className="font-mono text-[10px] text-ink-faint">
          convergence appears once training starts
        </span>
      </div>
    )
  }

  const maxLoss = Math.max(...points.map((p) => p.loss), 0.1)
  const n = Math.max(points.length, 2)

  const x = (i: number) => padL + (i / (n - 1)) * (w - padL - padR)
  const yLoss = (v: number) => padT + (1 - v / maxLoss) * (h - padT - padB)
  const yAcc = (v: number) => padT + (1 - v) * (h - padT - padB)

  const lossPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${yLoss(p.loss)}`).join(' ')
  const accPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${yAcc(p.trainAccuracy)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Training convergence">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <line
          key={f}
          x1={padL}
          y1={padT + f * (h - padT - padB)}
          x2={w - padR}
          y2={padT + f * (h - padT - padB)}
          stroke={GRID}
          strokeWidth="1"
        />
      ))}

      <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke={AXIS} strokeWidth="1" />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke={AXIS} strokeWidth="1" />

      {/* loss */}
      <path d={lossPath} fill="none" stroke={LANE_COLOR.classical} strokeWidth="1.6" />
      {/* accuracy */}
      {showAccuracy && (
        <path
          d={accPath}
          fill="none"
          stroke={LANE_COLOR.quantum}
          strokeWidth="1.4"
          strokeDasharray="3 3"
        />
      )}

      {/* last-point markers */}
      <circle cx={x(n - 1)} cy={yLoss(points[points.length - 1].loss)} r="2.5" fill={LANE_COLOR.classical} />
      {showAccuracy && (
        <circle cx={x(n - 1)} cy={yAcc(points[points.length - 1].trainAccuracy)} r="2.5" fill={LANE_COLOR.quantum} />
      )}

      {/* axis labels */}
      <text x={padL - 6} y={padT + 4} textAnchor="end" fontSize="8" fill="#6A6C72">
        {maxLoss.toFixed(2)}
      </text>
      <text x={padL - 6} y={h - padB} textAnchor="end" fontSize="8" fill="#6A6C72">
        0
      </text>
      <text x={w - padR + 6} y={padT + 4} fontSize="8" fill="#6A6C72">
        100%
      </text>
      <text x={w - padR + 6} y={h - padB} fontSize="8" fill="#6A6C72">
        0%
      </text>
      <text x={padL} y={h - 6} fontSize="8" fill="#6A6C72">
        epoch 1
      </text>
      <text x={w - padR} y={h - 6} textAnchor="end" fontSize="8" fill="#6A6C72">
        {points[points.length - 1].epoch}
      </text>
    </svg>
  )
}

// ---- ROC -------------------------------------------------------------------

export function RocChart({
  curves,
  size = 210,
}: {
  curves: { label: string; color: string; points: { fpr: number; tpr: number }[]; auc: number }[]
  size?: number
}) {
  const pad = 26
  const inner = size - pad * 2
  const x = (v: number) => pad + v * inner
  const y = (v: number) => pad + (1 - v) * inner

  return (
    <div>
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full" role="img" aria-label="ROC curves">
        {[0.25, 0.5, 0.75].map((f) => (
          <g key={f}>
            <line x1={x(f)} y1={pad} x2={x(f)} y2={y(0)} stroke={GRID} strokeWidth="1" />
            <line x1={pad} y1={y(f)} x2={x(1)} y2={y(f)} stroke={GRID} strokeWidth="1" />
          </g>
        ))}

        {/* chance diagonal */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="rgba(255,255,255,0.18)"
          strokeWidth="1"
          strokeDasharray="3 3"
        />

        <line x1={pad} y1={pad} x2={pad} y2={y(0)} stroke={AXIS} strokeWidth="1" />
        <line x1={pad} y1={y(0)} x2={x(1)} y2={y(0)} stroke={AXIS} strokeWidth="1" />

        {curves.map((c) => (
          <path
            key={c.label}
            d={c.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.fpr)} ${y(p.tpr)}`).join(' ')}
            fill="none"
            stroke={c.color}
            strokeWidth="1.7"
            strokeLinejoin="round"
          />
        ))}

        <text x={pad - 4} y={pad + 4} textAnchor="end" fontSize="8" fill="#6A6C72">1</text>
        <text x={pad - 4} y={y(0)} textAnchor="end" fontSize="8" fill="#6A6C72">0</text>
        <text x={x(1)} y={size - 8} textAnchor="end" fontSize="8" fill="#6A6C72">FPR 1</text>
        <text x={pad} y={size - 8} fontSize="8" fill="#6A6C72">0</text>
      </svg>

      <div className="mt-1.5 space-y-1">
        {curves.map((c) => (
          <div key={c.label} className="flex items-center gap-2">
            <span className="h-[2px] w-[12px] shrink-0 rounded-full" style={{ background: c.color }} />
            <span className="min-w-0 flex-1 truncate font-mono text-[9.5px] text-ink-dim">
              {c.label}
            </span>
            <span className="font-mono text-[9.5px] tabular-nums text-ink">
              {c.auc.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---- box plot --------------------------------------------------------------

/**
 * Cross-validation spread. A single accuracy figure proves nothing; the range
 * across folds is what shows whether the model generalises.
 */
export function BoxPlot({
  series,
  height = 130,
}: {
  series: { label: string; color: string; stats: BoxStats; folds?: number[] }[]
  height?: number
}) {
  const w = 400
  const padL = 8
  const padR = 8
  const padT = 12
  const padB = 26
  const rowW = (w - padL - padR) / Math.max(1, series.length)

  const all = series.flatMap((s) => [s.stats.min, s.stats.max, ...(s.folds ?? [])])
  const lo = Math.min(...all, 1)
  const hi = Math.max(...all, 0)
  const span = Math.max(hi - lo, 0.02)
  const yLo = lo - span * 0.15
  const yHi = hi + span * 0.15

  const y = (v: number) => padT + (1 - (v - yLo) / (yHi - yLo)) * (height - padT - padB)

  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="w-full" role="img" aria-label="Cross-validation spread">
      {[0, 0.5, 1].map((f) => {
        const v = yLo + f * (yHi - yLo)
        return (
          <g key={f}>
            <line x1={padL} y1={y(v)} x2={w - padR} y2={y(v)} stroke={GRID} strokeWidth="1" />
            <text x={w - padR} y={y(v) - 3} textAnchor="end" fontSize="8" fill="#6A6C72">
              {v.toFixed(3)}
            </text>
          </g>
        )
      })}

      {series.map((s, i) => {
        const cx = padL + rowW * i + rowW / 2
        const bw = Math.min(rowW * 0.42, 46)
        const { stats } = s

        return (
          <g key={s.label}>
            {/* whiskers */}
            <line x1={cx} y1={y(stats.max)} x2={cx} y2={y(stats.q3)} stroke={s.color} strokeWidth="1" />
            <line x1={cx} y1={y(stats.q1)} x2={cx} y2={y(stats.min)} stroke={s.color} strokeWidth="1" />
            <line x1={cx - bw / 3} y1={y(stats.max)} x2={cx + bw / 3} y2={y(stats.max)} stroke={s.color} strokeWidth="1" />
            <line x1={cx - bw / 3} y1={y(stats.min)} x2={cx + bw / 3} y2={y(stats.min)} stroke={s.color} strokeWidth="1" />

            {/* box */}
            <rect
              x={cx - bw / 2}
              y={y(stats.q3)}
              width={bw}
              height={Math.max(2, y(stats.q1) - y(stats.q3))}
              rx="2"
              fill={alpha(s.color, 0.16)}
              stroke={s.color}
              strokeWidth="1.1"
            />
            {/* median */}
            <line
              x1={cx - bw / 2}
              y1={y(stats.median)}
              x2={cx + bw / 2}
              y2={y(stats.median)}
              stroke={s.color}
              strokeWidth="1.8"
            />

            {/* individual folds, so the reader sees the actual observations */}
            {s.folds?.map((f, k) => (
              <circle
                key={k}
                cx={cx + (k - (s.folds!.length - 1) / 2) * 4.5}
                cy={y(f)}
                r="1.6"
                fill="#E8E9EB"
                opacity="0.5"
              />
            ))}

            <text x={cx} y={height - 8} textAnchor="middle" fontSize="8.5" fill="#9A9CA1">
              {s.label.length > 14 ? s.label.slice(0, 13) + '…' : s.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ---- horizontal bars -------------------------------------------------------

/** Ranked feature importance, with a kept/dropped cut line. */
export function ImportanceBars({
  items,
  keepCount,
  height = 8,
}: {
  items: { name: string; score: number; kept?: boolean }[]
  keepCount?: number
  height?: number
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.score)), 1e-9)

  return (
    <div className="space-y-[3px]">
      {items.map((item, i) => {
        const kept = item.kept ?? (keepCount !== undefined && i < keepCount)
        const color = kept ? LANE_COLOR.quantum : '#4A4C52'
        return (
          <div key={item.name}>
            {keepCount !== undefined && i === keepCount && (
              <div className="my-1.5 flex items-center gap-2">
                <span className="h-px flex-1" style={{ background: alpha(LANE_COLOR.classical, 0.4) }} />
                <span className="font-mono text-[8.5px]" style={{ color: LANE_COLOR.classical }}>
                  cut — {keepCount} kept above
                </span>
                <span className="h-px flex-1" style={{ background: alpha(LANE_COLOR.classical, 0.4) }} />
              </div>
            )}
            <div className="flex items-center gap-2">
              <span
                className="w-[132px] shrink-0 truncate font-mono text-[9.5px]"
                style={{ color: kept ? '#9A9CA1' : '#6A6C72' }}
                title={item.name}
              >
                {item.name}
              </span>
              <div className="flex-1 overflow-hidden rounded-full panel-well" style={{ height }}>
                <div
                  className="h-full rounded-full transition-[width] duration-300"
                  style={{
                    width: `${(Math.abs(item.score) / max) * 100}%`,
                    background: alpha(color, 0.8),
                  }}
                />
              </div>
              <span className="w-[42px] shrink-0 text-right font-mono text-[9px] tabular-nums text-ink-faint">
                {item.score.toFixed(3)}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** Diverging bars for signed attributions. */
export function DivergingBars({
  items,
  positiveLabel = 'toward positive',
  negativeLabel = 'toward negative',
  positiveColor = '#A3543D',
  negativeColor = '#5FA88C',
}: {
  items: { name: string; value: number; detail?: string }[]
  positiveLabel?: string
  negativeLabel?: string
  positiveColor?: string
  negativeColor?: string
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9)

  return (
    <div>
      <div className="space-y-2">
        {items.map((item) => {
          const positive = item.value > 0
          const color = positive ? positiveColor : negativeColor
          const width = (Math.abs(item.value) / max) * 50
          return (
            <div key={item.name}>
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="min-w-0 truncate font-mono text-[10px] text-ink-dim" title={item.name}>
                  {item.name}
                </span>
                <span className="shrink-0 font-mono text-[10px] tabular-nums" style={{ color }}>
                  {positive ? '+' : ''}
                  {item.value.toFixed(4)}
                </span>
              </div>
              <div className="relative h-[5px] w-full rounded-full panel-well">
                <span className="absolute inset-y-0 left-1/2 w-px" style={{ background: 'rgba(255,255,255,0.18)' }} />
                <div
                  className="absolute top-0 h-full rounded-full transition-all duration-300"
                  style={{
                    background: alpha(color, 0.75),
                    width: `${width}%`,
                    left: positive ? '50%' : `${50 - width}%`,
                  }}
                />
              </div>
              {item.detail && (
                <div className="mt-0.5 font-mono text-[8.5px] leading-relaxed text-ink-faint/80">
                  {item.detail}
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-3 flex gap-4 font-mono text-[9px] text-ink-faint">
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: negativeColor }} />
          {negativeLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-[3px] w-[8px] rounded-full" style={{ background: positiveColor }} />
          {positiveLabel}
        </span>
      </div>
    </div>
  )
}
