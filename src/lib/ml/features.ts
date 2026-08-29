import { column, correlation, mean, std, type Matrix } from './stats'

/**
 * Feature selection, computed for real.
 *
 * Every method returns the same `Ranking` shape so the UI can switch between
 * them without special-casing, and PCA additionally returns its loadings -
 * which is what makes tracing a component back to clinical features possible.
 */

export type SelectionMethod = 'pca' | 'mutual-info' | 'anova' | 'rfe'

export type Ranking = {
  /** original column index */
  index: number
  name: string
  /** method-specific importance, higher is better */
  score: number
}

// ---- mutual information ---------------------------------------------------

/**
 * Mutual information between a continuous feature and a binary label,
 * estimated by histogram binning.
 *
 * I(X;Y) = sum p(x,y) log( p(x,y) / (p(x) p(y)) )
 *
 * Binning is crude next to a k-NN estimator, but it is transparent and stable
 * at the sample sizes this platform handles.
 */
export function mutualInformation(x: number[], y: number[], bins = 10): number {
  const lo = Math.min(...x)
  const hi = Math.max(...x)
  if (hi === lo) return 0

  const labels = [...new Set(y)]
  const n = x.length

  // joint[bin][labelIndex]
  const joint = Array.from({ length: bins }, () => new Array(labels.length).fill(0))
  const px = new Array(bins).fill(0)
  const py = new Array(labels.length).fill(0)

  for (let i = 0; i < n; i++) {
    let b = Math.floor(((x[i] - lo) / (hi - lo)) * bins)
    if (b >= bins) b = bins - 1
    if (b < 0) b = 0
    const li = labels.indexOf(y[i])
    joint[b][li]++
    px[b]++
    py[li]++
  }

  let mi = 0
  for (let b = 0; b < bins; b++) {
    for (let li = 0; li < labels.length; li++) {
      if (joint[b][li] === 0) continue
      const pxy = joint[b][li] / n
      mi += pxy * Math.log2(pxy / ((px[b] / n) * (py[li] / n)))
    }
  }
  return Math.max(0, mi)
}

// ---- ANOVA F ---------------------------------------------------------------

/**
 * One-way ANOVA F statistic: between-group variance over within-group
 * variance. Large F means the classes separate on this feature.
 */
export function anovaF(x: number[], y: number[]): number {
  const labels = [...new Set(y)]
  const grand = mean(x)
  const n = x.length

  let ssBetween = 0
  let ssWithin = 0

  for (const label of labels) {
    const group = x.filter((_, i) => y[i] === label)
    if (group.length === 0) continue
    const gm = mean(group)
    ssBetween += group.length * (gm - grand) ** 2
    for (const v of group) ssWithin += (v - gm) ** 2
  }

  const dfBetween = labels.length - 1
  const dfWithin = n - labels.length
  if (dfBetween <= 0 || dfWithin <= 0 || ssWithin === 0) return 0
  return ssBetween / dfBetween / (ssWithin / dfWithin)
}

// ---- PCA -------------------------------------------------------------------

export type PcaResult = {
  /** eigenvector per component: loadings[c][j] weights original feature j */
  loadings: number[][]
  /** variance explained by each component, as a fraction of the total */
  explained: number[]
  /** cumulative explained variance */
  cumulative: number[]
  /** column means used to centre, needed to project new samples */
  center: number[]
  scale: number[]
}

/**
 * PCA by power iteration with deflation.
 *
 * Avoids a full eigendecomposition: we only ever need the top k components,
 * and k is small (it is the qubit count). Each component is found by iterating
 * the covariance action, then subtracted so the next iteration finds the next.
 */
export function pca(X: Matrix, k: number, seed = 42): PcaResult {
  const n = X.length
  const d = X[0]?.length ?? 0
  const comps = Math.min(k, d)

  // Standardise first: PCA on raw units would let large-scale features
  // dominate purely because of their units.
  const center: number[] = []
  const scale: number[] = []
  for (let j = 0; j < d; j++) {
    const col = column(X, j)
    const m = mean(col)
    const s = std(col, m)
    center.push(m)
    scale.push(s === 0 ? 1 : s)
  }
  const Z = X.map((r) => r.map((v, j) => (v - center[j]) / scale[j]))

  // Covariance matrix, d x d.
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      let s = 0
      for (let i = 0; i < n; i++) s += Z[i][a] * Z[i][b]
      const v = s / Math.max(1, n - 1)
      cov[a][b] = v
      cov[b][a] = v
    }
  }

  const totalVar = cov.reduce((acc, row, i) => acc + row[i], 0)

  let rng = seed >>> 0
  const nextRand = () => {
    rng = (rng * 1664525 + 1013904223) >>> 0
    return rng / 4294967296 - 0.5
  }

  const loadings: number[][] = []
  const explained: number[] = []

  for (let c = 0; c < comps; c++) {
    // Power iteration for the dominant eigenvector of the deflated matrix.
    let v = Array.from({ length: d }, () => nextRand())
    let norm = Math.hypot(...v)
    v = v.map((x) => x / norm)

    let eigenvalue = 0
    for (let iter = 0; iter < 300; iter++) {
      const w = new Array(d).fill(0)
      for (let a = 0; a < d; a++) {
        let s = 0
        for (let b = 0; b < d; b++) s += cov[a][b] * v[b]
        w[a] = s
      }
      norm = Math.hypot(...w)
      if (norm < 1e-12) break
      const next = w.map((x) => x / norm)

      // Converged when the direction stops moving.
      let delta = 0
      for (let a = 0; a < d; a++) delta += Math.abs(next[a] - v[a])
      v = next
      eigenvalue = norm
      if (delta < 1e-10) break
    }

    loadings.push(v)
    explained.push(totalVar === 0 ? 0 : eigenvalue / totalVar)

    // Deflate: remove this component's contribution so the next iteration
    // finds the next-largest direction.
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) {
        cov[a][b] -= eigenvalue * v[a] * v[b]
      }
    }
  }

  const cumulative: number[] = []
  let running = 0
  for (const e of explained) {
    running += e
    cumulative.push(running)
  }

  return { loadings, explained, cumulative, center, scale }
}

/** Projects samples onto the fitted components. */
export function pcaTransform(X: Matrix, p: PcaResult): Matrix {
  return X.map((row) => {
    const z = row.map((v, j) => (v - p.center[j]) / p.scale[j])
    return p.loadings.map((comp) => comp.reduce((s, w, j) => s + w * z[j], 0))
  })
}

// ---- ranking ---------------------------------------------------------------

/** Scores every feature by the chosen method, best first. */
export function rankFeatures(
  X: Matrix,
  y: number[],
  names: string[],
  method: SelectionMethod,
): Ranking[] {
  const d = X[0]?.length ?? 0
  const out: Ranking[] = []

  if (method === 'pca') {
    // A feature's PCA importance is its total weight across the leading
    // components, each weighted by that component's explained variance.
    const p = pca(X, Math.min(d, 8))
    for (let j = 0; j < d; j++) {
      let s = 0
      p.loadings.forEach((comp, c) => {
        s += Math.abs(comp[j]) * p.explained[c]
      })
      out.push({ index: j, name: names[j] ?? `f${j}`, score: s })
    }
  } else if (method === 'mutual-info') {
    for (let j = 0; j < d; j++) {
      out.push({ index: j, name: names[j] ?? `f${j}`, score: mutualInformation(column(X, j), y) })
    }
  } else if (method === 'anova') {
    for (let j = 0; j < d; j++) {
      out.push({ index: j, name: names[j] ?? `f${j}`, score: anovaF(column(X, j), y) })
    }
  } else {
    // Recursive elimination: repeatedly fit a linear model and drop the
    // weakest coefficient, so a feature's rank is when it survived to.
    return recursiveElimination(X, y, names)
  }

  return out.sort((a, b) => b.score - a.score)
}

/**
 * RFE against a logistic model. Rank is elimination order reversed, so the
 * last feature standing scores highest.
 */
function recursiveElimination(X: Matrix, y: number[], names: string[]): Ranking[] {
  const d = X[0]?.length ?? 0
  let active = Array.from({ length: d }, (_, j) => j)
  const order: number[] = []

  while (active.length > 1) {
    const sub = X.map((r) => active.map((j) => r[j]))
    const w = quickLogisticWeights(sub, y)
    // Weakest absolute coefficient goes first.
    let worst = 0
    for (let i = 1; i < w.length; i++) {
      if (Math.abs(w[i]) < Math.abs(w[worst])) worst = i
    }
    order.push(active[worst])
    active = active.filter((_, i) => i !== worst)
  }
  order.push(active[0])

  // order is worst-first; reverse for best-first.
  return order.reverse().map((j, rank) => ({
    index: j,
    name: names[j] ?? `f${j}`,
    score: (order.length - rank) / order.length,
  }))
}

/** Small logistic fit used only for RFE coefficients. */
function quickLogisticWeights(X: Matrix, y: number[], iters = 60): number[] {
  const d = X[0]?.length ?? 0
  const w = new Array(d).fill(0)
  const lr = 0.1

  for (let it = 0; it < iters; it++) {
    const grad = new Array(d).fill(0)
    for (let i = 0; i < X.length; i++) {
      let z = 0
      for (let j = 0; j < d; j++) z += w[j] * X[i][j]
      const p = 1 / (1 + Math.exp(-z))
      const err = p - y[i]
      for (let j = 0; j < d; j++) grad[j] += (err * X[i][j]) / X.length
    }
    for (let j = 0; j < d; j++) w[j] -= lr * grad[j]
  }
  return w
}

/** Drops features correlated above `threshold`, keeping the better-ranked one. */
export function dropCollinear(
  X: Matrix,
  ranked: Ranking[],
  threshold = 0.92,
): { kept: Ranking[]; dropped: { name: string; against: string; r: number }[] } {
  const kept: Ranking[] = []
  const dropped: { name: string; against: string; r: number }[] = []

  for (const cand of ranked) {
    let collides: { name: string; r: number } | null = null
    for (const k of kept) {
      const r = correlation(column(X, cand.index), column(X, k.index))
      if (Math.abs(r) > threshold) {
        collides = { name: k.name, r }
        break
      }
    }
    if (collides) {
      dropped.push({ name: cand.name, against: collides.name, r: collides.r })
    } else {
      kept.push(cand)
    }
  }
  return { kept, dropped }
}
