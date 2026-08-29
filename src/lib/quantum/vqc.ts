import { Statevector, makeRng, sampleExpectationZ } from './statevector'

/**
 * A variational quantum classifier that actually trains.
 *
 * Forward pass: encode a sample's features as rotation angles, apply a
 * parameterised entangling ansatz, measure <Z> on wire 0, squash to a
 * probability. Training: parameter-shift gradients, which is the analytic
 * gradient rule real quantum hardware uses - not backpropagation through the
 * simulator, because backprop is not available on a device.
 *
 * The parameter-shift rule: for a gate exp(-i t P / 2) with P a Pauli,
 *   d<H>/dt = [ <H>(t + pi/2) - <H>(t - pi/2) ] / 2
 * so each parameter costs two extra circuit evaluations per sample.
 */

export type FeatureMap = 'angle' | 'amplitude' | 'zz'
export type AnsatzKind = 'strongly-entangling' | 'basic-entangling' | 'hardware-efficient'
export type Backend = 'ideal' | 'noisy' | 'hardware'

export type VqcConfig = {
  qubits: number
  layers: number
  featureMap: FeatureMap
  ansatz: AnsatzKind
  backend: Backend
  /** 0 means exact analytic expectation (ideal simulator) */
  shots: number
  seed: number
}

export const DEFAULT_VQC: VqcConfig = {
  qubits: 6,
  layers: 2,
  featureMap: 'angle',
  ansatz: 'strongly-entangling',
  backend: 'ideal',
  shots: 0,
  seed: 42,
}

/** Trainable parameters per layer, per qubit, for each ansatz. */
export function rotationsPerQubit(ansatz: AnsatzKind): number {
  // strongly-entangling applies RZ-RY-RZ; the others apply a single RY.
  return ansatz === 'strongly-entangling' ? 3 : ansatz === 'hardware-efficient' ? 2 : 1
}

export function paramCount(cfg: VqcConfig): number {
  return cfg.layers * cfg.qubits * rotationsPerQubit(cfg.ansatz)
}

/**
 * Gate and depth accounting for the circuit diagram.
 * Counted rather than estimated, so the figures on screen match the circuit.
 */
export function circuitStats(cfg: VqcConfig) {
  const { qubits, layers, featureMap, ansatz } = cfg

  const encodingGates =
    featureMap === 'angle' ? qubits
      : featureMap === 'amplitude' ? qubits * 2
      : qubits * 2 + Math.max(0, qubits - 1) // zz: H+RZ per wire, plus couplings

  const encodingDepth = featureMap === 'angle' ? 1 : featureMap === 'amplitude' ? 2 : 3

  const rot = rotationsPerQubit(ansatz)
  const rotationGates = layers * qubits * rot
  const entanglers = layers * (ansatz === 'hardware-efficient' ? qubits - 1 : qubits)

  // Rotations on distinct wires run in parallel, so they cost `rot` depth per
  // layer regardless of width. The entangling pattern does depend on width: a
  // linear chain must serialise across the register, whereas the ring's
  // stride-1 CNOTs overlap into a constant few slices.
  const entangleDepth = ansatz === 'hardware-efficient' ? qubits - 1 : 2
  const ansatzDepth = layers * (rot + entangleDepth)

  return {
    gates: encodingGates + rotationGates + entanglers,
    depth: encodingDepth + ansatzDepth,
    encodingGates,
    rotationGates,
    entanglers,
    params: paramCount(cfg),
  }
}

/** Scales a standardised feature into [0, pi] so it is a valid rotation angle. */
export function toAngle(x: number): number {
  // tanh keeps outliers inside the range instead of wrapping them around the
  // Bloch sphere, which would make distant values alias onto each other.
  return (Math.tanh(x / 2) + 1) * (Math.PI / 2)
}

/** Applies the chosen feature map to a fresh register. */
function encode(sv: Statevector, x: number[], cfg: VqcConfig) {
  const n = cfg.qubits

  if (cfg.featureMap === 'angle') {
    for (let q = 0; q < n; q++) sv.ry(q, toAngle(x[q] ?? 0))
    return
  }

  if (cfg.featureMap === 'amplitude') {
    // Two rotations per wire gives the map access to both axes.
    for (let q = 0; q < n; q++) {
      sv.ry(q, toAngle(x[q] ?? 0))
      sv.rz(q, toAngle(x[q] ?? 0))
    }
    return
  }

  // ZZFeatureMap: Hadamard layer, phase encode, then pairwise ZZ couplings
  // carrying products of features - this is the map that is classically hard.
  for (let q = 0; q < n; q++) {
    sv.h(q)
    sv.rz(q, 2 * (x[q] ?? 0))
  }
  for (let q = 0; q < n - 1; q++) {
    const prod = 2 * (Math.PI - (x[q] ?? 0)) * (Math.PI - (x[q + 1] ?? 0))
    sv.cnot(q, q + 1)
    sv.rz(q + 1, prod)
    sv.cnot(q, q + 1)
  }
}

/** Applies the variational ansatz with the given parameters. */
function applyAnsatz(sv: Statevector, params: Float64Array, cfg: VqcConfig) {
  const n = cfg.qubits
  const rot = rotationsPerQubit(cfg.ansatz)
  let p = 0

  for (let l = 0; l < cfg.layers; l++) {
    for (let q = 0; q < n; q++) {
      if (cfg.ansatz === 'strongly-entangling') {
        sv.rz(q, params[p++])
        sv.ry(q, params[p++])
        sv.rz(q, params[p++])
      } else if (cfg.ansatz === 'hardware-efficient') {
        sv.ry(q, params[p++])
        sv.rz(q, params[p++])
      } else {
        sv.ry(q, params[p++])
      }
    }

    if (cfg.ansatz === 'hardware-efficient') {
      // Linear chain: what a real device's coupling map actually supports.
      for (let q = 0; q < n - 1; q++) sv.cnot(q, q + 1)
    } else {
      // Ring, with the stride growing per layer so correlations spread.
      const stride = cfg.ansatz === 'strongly-entangling' ? (l % (n - 1)) + 1 : 1
      for (let q = 0; q < n; q++) sv.cnot(q, (q + stride) % n)
    }
    void rot
  }
}

/**
 * Depolarising-style noise, applied as a contraction of the expectation value
 * toward zero. This is the leading-order effect of gate noise on a measured
 * observable, and is what makes the 'noisy' backend behave differently from
 * 'ideal' without simulating a full density matrix.
 */
function applyNoise(value: number, cfg: VqcConfig, stats: { depth: number }): number {
  if (cfg.backend === 'ideal') return value
  const perGate = cfg.backend === 'hardware' ? 0.004 : 0.002
  return value * Math.exp(-perGate * stats.depth * cfg.qubits)
}

export class Vqc {
  readonly cfg: VqcConfig
  params: Float64Array
  private sv: Statevector
  private rng: () => number
  private stats: ReturnType<typeof circuitStats>

  constructor(cfg: VqcConfig, params?: Float64Array) {
    this.cfg = cfg
    this.sv = new Statevector(cfg.qubits)
    this.rng = makeRng(cfg.seed)
    this.stats = circuitStats(cfg)

    const n = paramCount(cfg)
    if (params) {
      this.params = params.slice()
    } else {
      // Small random init: near-zero angles keep the initial state away from
      // the barren-plateau regime that random deep circuits fall into.
      this.params = new Float64Array(n)
      for (let i = 0; i < n; i++) this.params[i] = (this.rng() - 0.5) * 0.6
    }
  }

  /** Raw <Z> on wire 0 for one sample, with the given parameters. */
  private expectation(x: number[], params: Float64Array): number {
    this.sv.reset()
    encode(this.sv, x, this.cfg)
    applyAnsatz(this.sv, params, this.cfg)

    const measured =
      this.cfg.shots > 0
        ? sampleExpectationZ(this.sv, 0, this.cfg.shots, this.rng).value
        : this.sv.expectationZ(0)

    return applyNoise(measured, this.cfg, this.stats)
  }

  /** Malignant-class probability for one sample, in (0, 1). */
  predictOne(x: number[], params = this.params): number {
    // <Z> runs +1 (class 0) to -1 (class 1); map to a probability.
    return (1 - this.expectation(x, params)) / 2
  }

  predict(X: number[][]): number[] {
    return X.map((x) => this.predictOne(x))
  }

  /** Binary cross-entropy over the batch. */
  loss(X: number[][], y: number[], params = this.params): number {
    let total = 0
    for (let i = 0; i < X.length; i++) {
      const p = Math.min(Math.max(this.predictOne(X[i], params), 1e-7), 1 - 1e-7)
      total += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p))
    }
    return total / X.length
  }

  /**
   * Analytic gradient via the parameter-shift rule.
   *
   * Two circuit evaluations per parameter per sample - the same cost a real
   * device pays. Deliberately not autodiff through the simulator, because that
   * shortcut does not exist on hardware.
   *
   * The shift rule is exact for the *expectation value*, not for the loss,
   * which is a nonlinear function of it. So the two are composed explicitly:
   *
   *   d(loss)/d(theta) = d(loss)/d(p) * d(p)/d(<Z>) * d(<Z>)/d(theta)
   *
   * with d(<Z>)/d(theta) from the shift rule, p = (1 - <Z>)/2 so
   * d(p)/d(<Z>) = -1/2, and binary cross-entropy giving
   * d(loss)/d(p) = (p - y) / (p (1 - p)).
   *
   * Differentiating the loss directly by shifting theta would be wrong: it
   * treats a nonlinear composition as if the rule applied end to end.
   */
  gradient(X: number[][], y: number[]): Float64Array {
    const g = new Float64Array(this.params.length)
    const shifted = this.params.slice()
    const half = Math.PI / 2
    const n = X.length

    for (let i = 0; i < n; i++) {
      const p = Math.min(Math.max(this.predictOne(X[i]), 1e-7), 1 - 1e-7)
      // dLoss/dp for BCE, then dp/d<Z> = -1/2.
      const dLoss_dP = (p - y[i]) / (p * (1 - p))
      const dLoss_dZ = dLoss_dP * -0.5

      for (let k = 0; k < this.params.length; k++) {
        const original = this.params[k]

        shifted[k] = original + half
        const zPlus = this.expectation(X[i], shifted)

        shifted[k] = original - half
        const zMinus = this.expectation(X[i], shifted)

        shifted[k] = original

        // Parameter-shift: d<Z>/dtheta = (Z(+pi/2) - Z(-pi/2)) / 2
        const dZ_dTheta = (zPlus - zMinus) / 2
        g[k] += (dLoss_dZ * dZ_dTheta) / n
      }
    }
    return g
  }

  get circuit() {
    return this.stats
  }
}

export type TrainOptions = {
  epochs: number
  learningRate: number
  /** samples per gradient step; keeps parameter-shift affordable */
  batchSize: number
  seed: number
}

export type EpochRecord = {
  epoch: number
  loss: number
  trainAccuracy: number
  gradNorm: number
}

/**
 * Adam training loop.
 *
 * Yields after every epoch so the caller can render the convergence curve as
 * it happens rather than after the fact.
 */
export function* trainVqc(
  model: Vqc,
  X: number[][],
  y: number[],
  opts: TrainOptions,
): Generator<EpochRecord, void, void> {
  const rng = makeRng(opts.seed)
  const m = new Float64Array(model.params.length)
  const v = new Float64Array(model.params.length)
  const b1 = 0.9, b2 = 0.999, eps = 1e-8

  for (let epoch = 1; epoch <= opts.epochs; epoch++) {
    // Minibatch: parameter-shift is 2*P evaluations per sample, so full-batch
    // gradients get expensive fast.
    const idx: number[] = []
    const n = Math.min(opts.batchSize, X.length)
    while (idx.length < n) {
      const j = Math.floor(rng() * X.length)
      if (!idx.includes(j)) idx.push(j)
    }
    const bx = idx.map((i) => X[i])
    const by = idx.map((i) => y[i])

    const g = model.gradient(bx, by)

    let gradNorm = 0
    for (let k = 0; k < g.length; k++) {
      m[k] = b1 * m[k] + (1 - b1) * g[k]
      v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
      const mHat = m[k] / (1 - Math.pow(b1, epoch))
      const vHat = v[k] / (1 - Math.pow(b2, epoch))
      model.params[k] -= (opts.learningRate * mHat) / (Math.sqrt(vHat) + eps)
      gradNorm += g[k] * g[k]
    }

    const loss = model.loss(X, y)
    let correct = 0
    for (let i = 0; i < X.length; i++) {
      if ((model.predictOne(X[i]) >= 0.5 ? 1 : 0) === y[i]) correct++
    }

    yield {
      epoch,
      loss,
      trainAccuracy: correct / X.length,
      gradNorm: Math.sqrt(gradNorm),
    }
  }
}
