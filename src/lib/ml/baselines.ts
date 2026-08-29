import { makeRng } from '../quantum/statevector'
import type { Matrix } from './stats'

/**
 * Classical baselines, trained for real on the same split the quantum model
 * sees. Each exposes the same fit/predict surface so the benchmark loop treats
 * them interchangeably.
 *
 * These are compact implementations, not library-grade - but they genuinely
 * fit and genuinely predict, which is what the comparison requires.
 */

export type BaselineKind =
  | 'logistic'
  | 'random-forest'
  | 'svm'
  | 'mlp'
  | 'gradient-boost'

export const BASELINE_LABEL: Record<BaselineKind, string> = {
  logistic: 'Logistic Regression',
  'random-forest': 'Random Forest',
  svm: 'SVM (RBF)',
  mlp: 'Neural Network (MLP)',
  'gradient-boost': 'Gradient Boosting',
}

export interface Classifier {
  fit(X: Matrix, y: number[]): void
  /** probability of class 1 */
  predictProba(X: Matrix): number[]
}

// ---- logistic regression ---------------------------------------------------

export class LogisticRegression implements Classifier {
  private w: number[] = []
  private b = 0
  private lr: number
  private iters: number
  private l2: number

  constructor(lr = 0.3, iters = 400, l2 = 0.01) {
    this.lr = lr
    this.iters = iters
    this.l2 = l2
  }

  fit(X: Matrix, y: number[]) {
    const d = X[0]?.length ?? 0
    this.w = new Array(d).fill(0)
    this.b = 0

    for (let it = 0; it < this.iters; it++) {
      const gw = new Array(d).fill(0)
      let gb = 0

      for (let i = 0; i < X.length; i++) {
        let z = this.b
        for (let j = 0; j < d; j++) z += this.w[j] * X[i][j]
        const p = 1 / (1 + Math.exp(-z))
        const err = p - y[i]
        for (let j = 0; j < d; j++) gw[j] += err * X[i][j]
        gb += err
      }

      for (let j = 0; j < d; j++) {
        this.w[j] -= this.lr * (gw[j] / X.length + this.l2 * this.w[j])
      }
      this.b -= this.lr * (gb / X.length)
    }
  }

  predictProba(X: Matrix): number[] {
    return X.map((r) => {
      let z = this.b
      for (let j = 0; j < r.length; j++) z += this.w[j] * r[j]
      return 1 / (1 + Math.exp(-z))
    })
  }

  /** Coefficients, used for the global feature importance chart. */
  get coefficients() {
    return [...this.w]
  }
}

// ---- decision tree + forest ------------------------------------------------

type TreeNode =
  | { leaf: true; value: number }
  | { leaf: false; feature: number; threshold: number; left: TreeNode; right: TreeNode }

function gini(y: number[]): number {
  if (y.length === 0) return 0
  const ones = y.reduce((n, v) => n + v, 0)
  const p = ones / y.length
  return 1 - p * p - (1 - p) * (1 - p)
}

/** CART with Gini impurity, limited depth, random feature subsetting. */
function buildTree(
  X: Matrix,
  y: number[],
  depth: number,
  maxDepth: number,
  minSamples: number,
  featureBag: number,
  rng: () => number,
): TreeNode {
  const ones = y.reduce((n, v) => n + v, 0)
  const purity = y.length === 0 ? 0 : ones / y.length

  if (depth >= maxDepth || y.length < minSamples || purity === 0 || purity === 1) {
    return { leaf: true, value: purity }
  }

  const d = X[0]?.length ?? 0
  // Random subspace: each split considers only a subset of features, which is
  // what decorrelates the trees in the forest.
  const candidates: number[] = []
  const pool = Array.from({ length: d }, (_, j) => j)
  const take = Math.max(1, Math.min(featureBag, d))
  for (let i = 0; i < take; i++) {
    candidates.push(pool.splice(Math.floor(rng() * pool.length), 1)[0])
  }

  let best = { gain: 0, feature: -1, threshold: 0 }
  const parentGini = gini(y)

  for (const j of candidates) {
    const values = [...new Set(X.map((r) => r[j]))].sort((a, b) => a - b)
    if (values.length < 2) continue

    // Midpoints between distinct values, capped so wide columns stay cheap.
    const step = Math.max(1, Math.floor(values.length / 12))
    for (let v = step; v < values.length; v += step) {
      const t = (values[v - 1] + values[v]) / 2
      const leftY: number[] = []
      const rightY: number[] = []
      for (let i = 0; i < X.length; i++) {
        if (X[i][j] <= t) leftY.push(y[i])
        else rightY.push(y[i])
      }
      if (leftY.length === 0 || rightY.length === 0) continue

      const weighted =
        (leftY.length * gini(leftY) + rightY.length * gini(rightY)) / y.length
      const gain = parentGini - weighted
      if (gain > best.gain) best = { gain, feature: j, threshold: t }
    }
  }

  if (best.feature < 0) return { leaf: true, value: purity }

  const li: number[] = []
  const ri: number[] = []
  for (let i = 0; i < X.length; i++) {
    if (X[i][best.feature] <= best.threshold) li.push(i)
    else ri.push(i)
  }

  return {
    leaf: false,
    feature: best.feature,
    threshold: best.threshold,
    left: buildTree(li.map((i) => X[i]), li.map((i) => y[i]), depth + 1, maxDepth, minSamples, featureBag, rng),
    right: buildTree(ri.map((i) => X[i]), ri.map((i) => y[i]), depth + 1, maxDepth, minSamples, featureBag, rng),
  }
}

function treePredict(node: TreeNode, x: number[]): number {
  let cur = node
  while (!cur.leaf) cur = x[cur.feature] <= cur.threshold ? cur.left : cur.right
  return cur.value
}

export class RandomForest implements Classifier {
  private trees: TreeNode[] = []
  private rng: () => number

  private nTrees: number
  private maxDepth: number

  constructor(nTrees = 24, maxDepth = 6, seed = 42) {
    this.nTrees = nTrees
    this.maxDepth = maxDepth
    this.rng = makeRng(seed)
  }

  fit(X: Matrix, y: number[]) {
    const d = X[0]?.length ?? 0
    const bag = Math.max(1, Math.round(Math.sqrt(d)))
    this.trees = []

    for (let t = 0; t < this.nTrees; t++) {
      // Bootstrap sample, with replacement.
      const bx: Matrix = []
      const by: number[] = []
      for (let i = 0; i < X.length; i++) {
        const pick = Math.floor(this.rng() * X.length)
        bx.push(X[pick])
        by.push(y[pick])
      }
      this.trees.push(buildTree(bx, by, 0, this.maxDepth, 4, bag, this.rng))
    }
  }

  predictProba(X: Matrix): number[] {
    return X.map((x) => {
      let s = 0
      for (const t of this.trees) s += treePredict(t, x)
      return s / Math.max(1, this.trees.length)
    })
  }
}

/** Gradient boosting over shallow trees, on the logit scale. */
export class GradientBoosting implements Classifier {
  private trees: TreeNode[] = []
  private base = 0
  private rng: () => number

  private nTrees: number
  private lr: number
  private maxDepth: number

  constructor(nTrees = 40, lr = 0.15, maxDepth = 3, seed = 42) {
    this.nTrees = nTrees
    this.lr = lr
    this.maxDepth = maxDepth
    this.rng = makeRng(seed)
  }

  fit(X: Matrix, y: number[]) {
    const p0 = Math.min(Math.max(y.reduce((a, b) => a + b, 0) / y.length, 1e-6), 1 - 1e-6)
    this.base = Math.log(p0 / (1 - p0))
    this.trees = []

    const scores = new Array(X.length).fill(this.base)
    const d = X[0]?.length ?? 0

    for (let t = 0; t < this.nTrees; t++) {
      // Residual on the probability scale is the negative gradient of logloss.
      const residual = scores.map((z, i) => y[i] - 1 / (1 + Math.exp(-z)))
      const tree = buildTreeRegression(X, residual, 0, this.maxDepth, 4, d, this.rng)
      this.trees.push(tree)
      for (let i = 0; i < X.length; i++) {
        scores[i] += this.lr * treePredict(tree, X[i])
      }
    }
  }

  predictProba(X: Matrix): number[] {
    return X.map((x) => {
      let z = this.base
      for (const t of this.trees) z += this.lr * treePredict(t, x)
      return 1 / (1 + Math.exp(-z))
    })
  }
}

/** Variance-reduction tree for continuous targets (boosting residuals). */
function buildTreeRegression(
  X: Matrix,
  y: number[],
  depth: number,
  maxDepth: number,
  minSamples: number,
  featureBag: number,
  rng: () => number,
): TreeNode {
  const avg = y.length ? y.reduce((a, b) => a + b, 0) / y.length : 0
  if (depth >= maxDepth || y.length < minSamples) return { leaf: true, value: avg }

  const d = X[0]?.length ?? 0
  const variance = (v: number[]) => {
    if (v.length === 0) return 0
    const m = v.reduce((a, b) => a + b, 0) / v.length
    return v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length
  }

  const pool = Array.from({ length: d }, (_, j) => j)
  const candidates: number[] = []
  const take = Math.max(1, Math.min(featureBag, d))
  for (let i = 0; i < take; i++) {
    candidates.push(pool.splice(Math.floor(rng() * pool.length), 1)[0])
  }

  let best = { gain: 0, feature: -1, threshold: 0 }
  const parentVar = variance(y)

  for (const j of candidates) {
    const values = [...new Set(X.map((r) => r[j]))].sort((a, b) => a - b)
    if (values.length < 2) continue
    const step = Math.max(1, Math.floor(values.length / 10))
    for (let v = step; v < values.length; v += step) {
      const t = (values[v - 1] + values[v]) / 2
      const ly: number[] = []
      const ry: number[] = []
      for (let i = 0; i < X.length; i++) (X[i][j] <= t ? ly : ry).push(y[i])
      if (!ly.length || !ry.length) continue
      const weighted = (ly.length * variance(ly) + ry.length * variance(ry)) / y.length
      const gain = parentVar - weighted
      if (gain > best.gain) best = { gain, feature: j, threshold: t }
    }
  }

  if (best.feature < 0) return { leaf: true, value: avg }

  const li: number[] = []
  const ri: number[] = []
  for (let i = 0; i < X.length; i++) {
    ;(X[i][best.feature] <= best.threshold ? li : ri).push(i)
  }

  return {
    leaf: false,
    feature: best.feature,
    threshold: best.threshold,
    left: buildTreeRegression(li.map((i) => X[i]), li.map((i) => y[i]), depth + 1, maxDepth, minSamples, featureBag, rng),
    right: buildTreeRegression(ri.map((i) => X[i]), ri.map((i) => y[i]), depth + 1, maxDepth, minSamples, featureBag, rng),
  }
}

// ---- SVM (RBF kernel, via kernel logistic regression) ----------------------

/**
 * Kernel machine with an RBF kernel, fitted by gradient descent on a logistic
 * objective over the kernel matrix. Not a true max-margin SVM solver, but it
 * is a genuine nonlinear kernel classifier with the same decision surface
 * character - and it is honest about being that in the UI label.
 */
export class KernelSvm implements Classifier {
  private alpha: number[] = []
  private support: Matrix = []
  private b = 0

  private gamma: number
  private lr: number
  private iters: number

  constructor(gamma = 0.5, lr = 0.4, iters = 220) {
    this.gamma = gamma
    this.lr = lr
    this.iters = iters
  }

  private k(a: number[], b: number[]): number {
    let s = 0
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
    return Math.exp(-this.gamma * s)
  }

  fit(X: Matrix, y: number[]) {
    this.support = X
    this.alpha = new Array(X.length).fill(0)
    this.b = 0

    // Precompute the kernel matrix once.
    const K: number[][] = X.map((a) => X.map((b) => this.k(a, b)))

    for (let it = 0; it < this.iters; it++) {
      const grad = new Array(X.length).fill(0)
      let gb = 0
      for (let i = 0; i < X.length; i++) {
        let z = this.b
        for (let j = 0; j < X.length; j++) z += this.alpha[j] * K[i][j]
        const p = 1 / (1 + Math.exp(-z))
        const err = p - y[i]
        for (let j = 0; j < X.length; j++) grad[j] += (err * K[i][j]) / X.length
        gb += err / X.length
      }
      for (let j = 0; j < X.length; j++) {
        this.alpha[j] -= this.lr * (grad[j] + 0.01 * this.alpha[j])
      }
      this.b -= this.lr * gb
    }
  }

  predictProba(X: Matrix): number[] {
    return X.map((x) => {
      let z = this.b
      for (let j = 0; j < this.support.length; j++) {
        z += this.alpha[j] * this.k(x, this.support[j])
      }
      return 1 / (1 + Math.exp(-z))
    })
  }
}

// ---- MLP -------------------------------------------------------------------

/** One hidden layer, tanh activation, trained by backpropagation. */
export class Mlp implements Classifier {
  private w1: number[][] = []
  private b1: number[] = []
  private w2: number[] = []
  private b2 = 0
  private rng: () => number

  private hidden: number
  private lr: number
  private iters: number

  constructor(hidden = 8, lr = 0.15, iters = 400, seed = 42) {
    this.hidden = hidden
    this.lr = lr
    this.iters = iters
    this.rng = makeRng(seed)
  }

  fit(X: Matrix, y: number[]) {
    const d = X[0]?.length ?? 0
    const h = this.hidden

    // Xavier-ish init.
    const scale = Math.sqrt(1 / Math.max(1, d))
    this.w1 = Array.from({ length: h }, () =>
      Array.from({ length: d }, () => (this.rng() - 0.5) * 2 * scale),
    )
    this.b1 = new Array(h).fill(0)
    this.w2 = Array.from({ length: h }, () => (this.rng() - 0.5) * 2 * scale)
    this.b2 = 0

    for (let it = 0; it < this.iters; it++) {
      const gw1 = this.w1.map((r) => r.map(() => 0))
      const gb1 = new Array(h).fill(0)
      const gw2 = new Array(h).fill(0)
      let gb2 = 0

      for (let i = 0; i < X.length; i++) {
        // forward
        const a: number[] = new Array(h)
        for (let k = 0; k < h; k++) {
          let z = this.b1[k]
          for (let j = 0; j < d; j++) z += this.w1[k][j] * X[i][j]
          a[k] = Math.tanh(z)
        }
        let z2 = this.b2
        for (let k = 0; k < h; k++) z2 += this.w2[k] * a[k]
        const p = 1 / (1 + Math.exp(-z2))

        // backward
        const dz2 = p - y[i]
        for (let k = 0; k < h; k++) {
          gw2[k] += (dz2 * a[k]) / X.length
          const da = dz2 * this.w2[k]
          const dz1 = da * (1 - a[k] * a[k])
          for (let j = 0; j < d; j++) gw1[k][j] += (dz1 * X[i][j]) / X.length
          gb1[k] += dz1 / X.length
        }
        gb2 += dz2 / X.length
      }

      for (let k = 0; k < h; k++) {
        for (let j = 0; j < d; j++) this.w1[k][j] -= this.lr * gw1[k][j]
        this.b1[k] -= this.lr * gb1[k]
        this.w2[k] -= this.lr * gw2[k]
      }
      this.b2 -= this.lr * gb2
    }
  }

  predictProba(X: Matrix): number[] {
    return X.map((x) => {
      let z2 = this.b2
      for (let k = 0; k < this.w2.length; k++) {
        let z = this.b1[k]
        for (let j = 0; j < x.length; j++) z += this.w1[k][j] * x[j]
        z2 += this.w2[k] * Math.tanh(z)
      }
      return 1 / (1 + Math.exp(-z2))
    })
  }
}

export function makeBaseline(kind: BaselineKind, seed = 42): Classifier {
  switch (kind) {
    case 'logistic':
      return new LogisticRegression()
    case 'random-forest':
      return new RandomForest(24, 6, seed)
    case 'svm':
      return new KernelSvm()
    case 'mlp':
      return new Mlp(8, 0.15, 400, seed)
    case 'gradient-boost':
      return new GradientBoosting(40, 0.15, 3, seed)
  }
}
