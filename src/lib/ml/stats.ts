/**
 * Classical statistics and preprocessing, computed for real.
 *
 * Everything here runs on the actual uploaded matrix - no scripted values.
 * Deterministic given a seed, so a run is reproducible from its config.
 */

export type Matrix = number[][]

export function mean(xs: number[]): number {
  if (xs.length === 0) return 0
  let s = 0
  for (const x of xs) s += x
  return s / xs.length
}

export function std(xs: number[], m = mean(xs)): number {
  if (xs.length < 2) return 0
  let s = 0
  for (const x of xs) s += (x - m) * (x - m)
  return Math.sqrt(s / (xs.length - 1))
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  if (s.length === 0) return 0
  const mid = s.length >> 1
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

export function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return lo === hi ? sorted[lo] : sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
}

/** Five-number summary plus outliers, for the box plot. */
export type BoxStats = {
  min: number
  q1: number
  median: number
  q3: number
  max: number
  outliers: number[]
}

export function boxStats(xs: number[]): BoxStats {
  const s = [...xs].sort((a, b) => a - b)
  if (s.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, outliers: [] }
  }
  const q1 = quantile(s, 0.25)
  const q3 = quantile(s, 0.75)
  const iqr = q3 - q1
  const lo = q1 - 1.5 * iqr
  const hi = q3 + 1.5 * iqr
  const inliers = s.filter((v) => v >= lo && v <= hi)
  return {
    min: inliers.length ? inliers[0] : s[0],
    q1,
    median: quantile(s, 0.5),
    q3,
    max: inliers.length ? inliers[inliers.length - 1] : s[s.length - 1],
    outliers: s.filter((v) => v < lo || v > hi),
  }
}

export function column(X: Matrix, j: number): number[] {
  return X.map((r) => r[j])
}

/** Pearson correlation, used for the collinearity drop. */
export function correlation(a: number[], b: number[]): number {
  const ma = mean(a)
  const mb = mean(b)
  let num = 0
  let da = 0
  let db = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma
    const y = b[i] - mb
    num += x * y
    da += x * x
    db += y * y
  }
  const den = Math.sqrt(da * db)
  return den === 0 ? 0 : num / den
}

// ---- missing values -------------------------------------------------------

export type ImputeStrategy = 'drop' | 'mean' | 'median'

export type ImputeResult = {
  X: Matrix
  y: number[]
  /** per-column count of cells that were filled */
  filled: number[]
  /** rows removed, only non-zero for `drop` */
  rowsDropped: number
}

/**
 * Missing cells arrive as NaN from the parser. Column statistics are computed
 * over observed values only, so a filled cell never influences its own fill.
 */
export function imputeMissing(
  X: Matrix,
  y: number[],
  strategy: ImputeStrategy,
): ImputeResult {
  const cols = X[0]?.length ?? 0
  const filled = new Array(cols).fill(0)

  if (strategy === 'drop') {
    const keep: number[] = []
    for (let i = 0; i < X.length; i++) {
      if (X[i].every((v) => Number.isFinite(v))) keep.push(i)
    }
    return {
      X: keep.map((i) => X[i]),
      y: keep.map((i) => y[i]),
      filled,
      rowsDropped: X.length - keep.length,
    }
  }

  const out = X.map((r) => [...r])
  for (let j = 0; j < cols; j++) {
    const observed = out.map((r) => r[j]).filter((v) => Number.isFinite(v))
    const fill = strategy === 'mean' ? mean(observed) : median(observed)
    for (let i = 0; i < out.length; i++) {
      if (!Number.isFinite(out[i][j])) {
        out[i][j] = fill
        filled[j]++
      }
    }
  }
  return { X: out, y, filled, rowsDropped: 0 }
}

// ---- scaling --------------------------------------------------------------

export type Scaler = {
  kind: 'standard' | 'minmax'
  center: number[]
  scale: number[]
}

/**
 * Fitted on the training fold ONLY, then applied to test. Fitting on the full
 * matrix leaks test statistics into training and inflates every score.
 */
export function fitScaler(X: Matrix, kind: Scaler['kind'] = 'standard'): Scaler {
  const cols = X[0]?.length ?? 0
  const center: number[] = []
  const scale: number[] = []

  for (let j = 0; j < cols; j++) {
    const col = column(X, j)
    if (kind === 'standard') {
      const m = mean(col)
      const s = std(col, m)
      center.push(m)
      scale.push(s === 0 ? 1 : s)
    } else {
      const lo = Math.min(...col)
      const hi = Math.max(...col)
      center.push(lo)
      scale.push(hi - lo === 0 ? 1 : hi - lo)
    }
  }
  return { kind, center, scale }
}

export function applyScaler(X: Matrix, s: Scaler): Matrix {
  return X.map((r) => r.map((v, j) => (v - s.center[j]) / s.scale[j]))
}

// ---- splitting ------------------------------------------------------------

export type Split = {
  trainIdx: number[]
  testIdx: number[]
}

function seededShuffle<T>(arr: T[], rng: () => number): T[] {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

/**
 * Stratified train/test split: class proportions are preserved in both folds,
 * which matters on imbalanced medical data where a random split can leave a
 * fold with almost no positives.
 */
export function stratifiedSplit(
  y: number[],
  testFraction: number,
  rng: () => number,
): Split {
  const byClass = new Map<number, number[]>()
  y.forEach((label, i) => {
    const list = byClass.get(label)
    if (list) list.push(i)
    else byClass.set(label, [i])
  })

  const trainIdx: number[] = []
  const testIdx: number[] = []

  for (const idx of byClass.values()) {
    const shuffled = seededShuffle(idx, rng)
    const nTest = Math.max(1, Math.round(shuffled.length * testFraction))
    testIdx.push(...shuffled.slice(0, nTest))
    trainIdx.push(...shuffled.slice(nTest))
  }
  return { trainIdx: trainIdx.sort((a, b) => a - b), testIdx: testIdx.sort((a, b) => a - b) }
}

/** Stratified k-fold indices, for cross-validation spread. */
export function stratifiedFolds(y: number[], k: number, rng: () => number): number[][] {
  const byClass = new Map<number, number[]>()
  y.forEach((label, i) => {
    const list = byClass.get(label)
    if (list) list.push(i)
    else byClass.set(label, [i])
  })

  const folds: number[][] = Array.from({ length: k }, () => [])
  for (const idx of byClass.values()) {
    const shuffled = seededShuffle(idx, rng)
    // Deal round-robin so each fold gets a proportional share of every class.
    shuffled.forEach((i, pos) => folds[pos % k].push(i))
  }
  return folds
}

// ---- class imbalance ------------------------------------------------------

export type BalanceStrategy = 'none' | 'oversample' | 'class-weight'

export function classCounts(y: number[]): Map<number, number> {
  const c = new Map<number, number>()
  for (const v of y) c.set(v, (c.get(v) ?? 0) + 1)
  return c
}

/** Random oversampling of minority classes up to the majority count. */
export function oversample(
  X: Matrix,
  y: number[],
  rng: () => number,
): { X: Matrix; y: number[]; added: number } {
  const counts = classCounts(y)
  const target = Math.max(...counts.values())

  const outX = [...X]
  const outY = [...y]
  let added = 0

  for (const [label, n] of counts) {
    const idx = y.map((v, i) => (v === label ? i : -1)).filter((i) => i >= 0)
    for (let i = n; i < target; i++) {
      const pick = idx[Math.floor(rng() * idx.length)]
      outX.push([...X[pick]])
      outY.push(label)
      added++
    }
  }
  return { X: outX, y: outY, added }
}

/** Inverse-frequency weights, the alternative to resampling. */
export function classWeights(y: number[]): Map<number, number> {
  const counts = classCounts(y)
  const n = y.length
  const k = counts.size
  const w = new Map<number, number>()
  for (const [label, c] of counts) w.set(label, n / (k * c))
  return w
}
