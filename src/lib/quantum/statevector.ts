/**
 * A real statevector simulator.
 *
 * This is not a mock. It holds 2^n complex amplitudes and applies gates by
 * actual matrix action on that vector. At 8 qubits that is 256 amplitudes -
 * trivial for a browser, and enough for the variational classifier this
 * platform trains.
 *
 * Amplitudes are stored as two parallel Float64Arrays (real, imaginary) rather
 * than an array of objects: it keeps the whole state in two contiguous buffers,
 * which matters once the optimiser is calling this thousands of times.
 *
 * Qubit 0 is the least significant bit, matching Qiskit's convention, so a
 * basis state index's bit k tells you the value of qubit k.
 */

export class Statevector {
  readonly n: number
  readonly size: number
  re: Float64Array
  im: Float64Array

  constructor(n: number) {
    if (n < 1 || n > 16) {
      throw new RangeError(`qubit count must be 1..16, got ${n}`)
    }
    this.n = n
    this.size = 1 << n
    this.re = new Float64Array(this.size)
    this.im = new Float64Array(this.size)
    // |00...0>
    this.re[0] = 1
  }

  /** Fresh |0...0>, reusing the existing buffers. */
  reset() {
    this.re.fill(0)
    this.im.fill(0)
    this.re[0] = 1
  }

  clone(): Statevector {
    const s = new Statevector(this.n)
    s.re.set(this.re)
    s.im.set(this.im)
    return s
  }

  /**
   * Applies an arbitrary single-qubit 2x2 unitary to `target`.
   *
   * Walks the state in blocks so each amplitude pair (|...0...>, |...1...>)
   * differing only at the target bit is visited exactly once.
   */
  apply1(
    target: number,
    m00r: number, m00i: number, m01r: number, m01i: number,
    m10r: number, m10i: number, m11r: number, m11i: number,
  ) {
    const stride = 1 << target
    const { re, im, size } = this

    for (let base = 0; base < size; base += stride << 1) {
      for (let off = 0; off < stride; off++) {
        const i0 = base + off
        const i1 = i0 + stride

        const a0r = re[i0], a0i = im[i0]
        const a1r = re[i1], a1i = im[i1]

        re[i0] = m00r * a0r - m00i * a0i + m01r * a1r - m01i * a1i
        im[i0] = m00r * a0i + m00i * a0r + m01r * a1i + m01i * a1r

        re[i1] = m10r * a0r - m10i * a0i + m11r * a1r - m11i * a1i
        im[i1] = m10r * a0i + m10i * a0r + m11r * a1i + m11i * a1r
      }
    }
  }

  /** RY(theta): real rotation about the Y axis. The encoding gate. */
  ry(target: number, theta: number) {
    const c = Math.cos(theta / 2)
    const s = Math.sin(theta / 2)
    this.apply1(target, c, 0, -s, 0, s, 0, c, 0)
  }

  /** RX(theta). */
  rx(target: number, theta: number) {
    const c = Math.cos(theta / 2)
    const s = Math.sin(theta / 2)
    this.apply1(target, c, 0, 0, -s, 0, -s, c, 0)
  }

  /** RZ(theta): diagonal, so it only needs a phase multiply. */
  rz(target: number, theta: number) {
    const c = Math.cos(theta / 2)
    const s = Math.sin(theta / 2)
    // diag(e^{-i t/2}, e^{+i t/2})
    this.apply1(target, c, -s, 0, 0, 0, 0, c, s)
  }

  /** Hadamard. */
  h(target: number) {
    const r = Math.SQRT1_2
    this.apply1(target, r, 0, r, 0, r, 0, -r, 0)
  }

  /** CNOT: swaps the amplitude pair wherever the control bit is 1. */
  cnot(control: number, target: number) {
    if (control === target) throw new Error('cnot control and target must differ')
    const { re, im, size } = this
    const cMask = 1 << control
    const tMask = 1 << target

    for (let i = 0; i < size; i++) {
      // Visit each pair once: only when control is set and target is clear.
      if ((i & cMask) !== 0 && (i & tMask) === 0) {
        const j = i | tMask
        const tr = re[i], ti = im[i]
        re[i] = re[j]; im[i] = im[j]
        re[j] = tr;    im[j] = ti
      }
    }
  }

  /** CZ: a sign flip on |11>, diagonal so no pairing needed. */
  cz(control: number, target: number) {
    const { re, im, size } = this
    const mask = (1 << control) | (1 << target)
    for (let i = 0; i < size; i++) {
      if ((i & mask) === mask) {
        re[i] = -re[i]
        im[i] = -im[i]
      }
    }
  }

  /** Probability of measuring |1> on `qubit`, summed over all basis states. */
  probOne(qubit: number): number {
    const { re, im, size } = this
    const mask = 1 << qubit
    let p = 0
    for (let i = 0; i < size; i++) {
      if ((i & mask) !== 0) p += re[i] * re[i] + im[i] * im[i]
    }
    return p
  }

  /**
   * Expectation value of PauliZ on `qubit`, in [-1, 1].
   * <Z> = P(0) - P(1) = 1 - 2*P(1).
   */
  expectationZ(qubit: number): number {
    return 1 - 2 * this.probOne(qubit)
  }

  /** Full probability distribution over basis states. */
  probabilities(): Float64Array {
    const { re, im, size } = this
    const p = new Float64Array(size)
    for (let i = 0; i < size; i++) p[i] = re[i] * re[i] + im[i] * im[i]
    return p
  }

  /** Total probability, for verifying unitarity in tests. */
  norm(): number {
    let s = 0
    for (let i = 0; i < this.size; i++) {
      s += this.re[i] * this.re[i] + this.im[i] * this.im[i]
    }
    return s
  }
}

/**
 * Finite-shot sampling of <Z>.
 *
 * A real device cannot read an exact expectation value - it samples the circuit
 * `shots` times and averages. Modelling that matters: it is the reason results
 * carry a standard error, and the reason shot count is a tunable on the model
 * builder rather than a cosmetic label.
 *
 * With `shots = 0` the exact analytic value is returned, which is what an ideal
 * simulator gives.
 */
export function sampleExpectationZ(
  sv: Statevector,
  qubit: number,
  shots: number,
  rng: () => number,
): { value: number; stderr: number } {
  const pOne = sv.probOne(qubit)

  if (shots <= 0) {
    return { value: 1 - 2 * pOne, stderr: 0 }
  }

  // Binomial draw over `shots` trials, done directly for small counts.
  let ones = 0
  for (let s = 0; s < shots; s++) {
    if (rng() < pOne) ones++
  }
  const pHat = ones / shots
  // sqrt(p(1-p)/N), doubled because <Z> = 1 - 2p scales the variance by 2.
  const stderr = 2 * Math.sqrt(Math.max(pHat * (1 - pHat), 1e-12) / shots)
  return { value: 1 - 2 * pHat, stderr }
}

/** Deterministic PRNG (mulberry32), so every run with a seed is reproducible. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
