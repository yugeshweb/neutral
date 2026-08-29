/**
 * Classification metrics, computed from actual predictions.
 *
 * Every figure the results screen shows comes from here, so a classical win is
 * reported exactly as readily as a quantum one.
 */

export type Confusion = { tp: number; fn: number; fp: number; tn: number }

export function confusion(yTrue: number[], yPred: number[]): Confusion {
  const c: Confusion = { tp: 0, fn: 0, fp: 0, tn: 0 }
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1) {
      if (yPred[i] === 1) c.tp++
      else c.fn++
    } else if (yPred[i] === 1) {
      c.fp++
    } else {
      c.tn++
    }
  }
  return c
}

export type Metrics = {
  accuracy: number
  sensitivity: number
  specificity: number
  precision: number
  f1: number
  rocAuc: number
  confusion: Confusion
}

const safeDiv = (a: number, b: number) => (b === 0 ? 0 : a / b)

/**
 * ROC curve by sweeping the decision threshold over every distinct score.
 * Returns points in increasing FPR order, with (0,0) and (1,1) anchored.
 */
export function rocCurve(
  yTrue: number[],
  scores: number[],
): { fpr: number; tpr: number; threshold: number }[] {
  const pairs = yTrue
    .map((t, i) => ({ t, s: scores[i] }))
    .sort((a, b) => b.s - a.s)

  const positives = yTrue.reduce((n, v) => n + (v === 1 ? 1 : 0), 0)
  const negatives = yTrue.length - positives

  const pts = [{ fpr: 0, tpr: 0, threshold: Infinity }]
  let tp = 0
  let fp = 0

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].t === 1) tp++
    else fp++
    // Only emit a point when the next score differs, so ties form one step.
    if (i === pairs.length - 1 || pairs[i].s !== pairs[i + 1].s) {
      pts.push({
        fpr: safeDiv(fp, negatives),
        tpr: safeDiv(tp, positives),
        threshold: pairs[i].s,
      })
    }
  }
  if (pts[pts.length - 1].fpr !== 1 || pts[pts.length - 1].tpr !== 1) {
    pts.push({ fpr: 1, tpr: 1, threshold: -Infinity })
  }
  return pts
}

/** AUC by trapezoidal integration of the ROC curve. */
export function auc(points: { fpr: number; tpr: number }[]): number {
  let area = 0
  for (let i = 1; i < points.length; i++) {
    area += ((points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr)) / 2
  }
  return area
}

export function evaluate(
  yTrue: number[],
  scores: number[],
  threshold = 0.5,
): Metrics {
  const yPred = scores.map((s) => (s >= threshold ? 1 : 0))
  const c = confusion(yTrue, yPred)

  const sensitivity = safeDiv(c.tp, c.tp + c.fn)
  const specificity = safeDiv(c.tn, c.tn + c.fp)
  const precision = safeDiv(c.tp, c.tp + c.fp)

  return {
    accuracy: safeDiv(c.tp + c.tn, yTrue.length),
    sensitivity,
    specificity,
    precision,
    f1: safeDiv(2 * precision * sensitivity, precision + sensitivity),
    rocAuc: auc(rocCurve(yTrue, scores)),
    confusion: c,
  }
}

/**
 * McNemar's test on paired predictions.
 *
 * Compares two models on the SAME samples, counting only the cases where they
 * disagree - which is the right test here, because both models see an
 * identical split. Uses the exact binomial for small discordant counts, where
 * the chi-square approximation is unreliable.
 */
export function mcnemar(
  yTrue: number[],
  predA: number[],
  predB: number[],
): { b: number; c: number; pValue: number; significant: boolean } {
  let b = 0 // A right, B wrong
  let c = 0 // A wrong, B right

  for (let i = 0; i < yTrue.length; i++) {
    const aOk = predA[i] === yTrue[i]
    const bOk = predB[i] === yTrue[i]
    if (aOk && !bOk) b++
    else if (!aOk && bOk) c++
  }

  const n = b + c
  if (n === 0) return { b, c, pValue: 1, significant: false }

  // Two-sided exact binomial with p = 0.5.
  const k = Math.min(b, c)
  let tail = 0
  for (let i = 0; i <= k; i++) tail += binomialPmf(n, i, 0.5)
  const pValue = Math.min(1, 2 * tail)

  return { b, c, pValue, significant: pValue < 0.05 }
}

function binomialPmf(n: number, k: number, p: number): number {
  return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))
}

function logChoose(n: number, k: number): number {
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1)
}

/** Lanczos approximation, accurate well past the counts used here. */
function logGamma(z: number): number {
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z)
  }
  z -= 1
  let x = 0.99999999999980993
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1)
  const t = z + g.length - 0.5
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}
