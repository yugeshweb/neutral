import { useRef, useState, type ReactNode } from 'react'
import { LANE_COLOR } from '../lib/theme'

type Props = {
  /** the quantum term being explained */
  term: string
  /** plain-language explanation, assuming no prior knowledge */
  body: string
  children: ReactNode
}

/**
 * Explains a quantum term on hover or focus.
 *
 * Assumes the reader has never met the word. Positioned by fixed coordinates
 * from the trigger's rect so it escapes any overflow-hidden ancestor - the
 * panels here all clip, and a tooltip that gets cut in half is worse than none.
 */
export function Tooltip({ term, body, children }: Props) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ x: 0, y: 0, above: true })
  const ref = useRef<HTMLSpanElement>(null)

  const show = () => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    // Flip below when there is not enough room above.
    const above = r.top > 190
    setPos({
      x: Math.min(Math.max(r.left + r.width / 2, 150), window.innerWidth - 150),
      y: above ? r.top - 8 : r.bottom + 8,
      above,
    })
    setOpen(true)
  }

  return (
    <span
      ref={ref}
      className="relative inline-flex"
      onPointerEnter={show}
      onPointerLeave={() => setOpen(false)}
      onFocus={show}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      style={{ cursor: 'help', outline: 'none' }}
    >
      {children}

      {open && (
        <span
          role="tooltip"
          className="pointer-events-none fixed z-[100] w-[280px] rounded-[9px] p-3"
          style={{
            left: pos.x,
            top: pos.y,
            transform: `translate(-50%, ${pos.above ? '-100%' : '0'})`,
            background: '#1B1C20',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow:
              'inset 0 1px 0 rgba(255,255,255,0.07), 0 6px 24px rgba(0,0,0,0.75)',
          }}
        >
          <span
            className="mb-1 block font-mono text-[9.5px] tracking-[0.02em]"
            style={{ color: LANE_COLOR.quantum }}
          >
            {term}
          </span>
          <span className="block text-[11px] leading-relaxed text-ink-dim">{body}</span>
        </span>
      )}
    </span>
  )
}

/** The glossary. Every quantum term the UI uses appears here. */
export const GLOSSARY: Record<string, string> = {
  qubit:
    'The quantum equivalent of a bit. Unlike a bit it can hold a blend of 0 and 1 at once, so n qubits track 2^n values simultaneously. That is the potential advantage - and the cost, since simulating them doubles with every qubit added.',
  ansatz:
    'The trainable part of the circuit - a fixed pattern of rotation gates whose angles are the parameters being learned. German for "approach". Choosing one is like choosing a network architecture: it decides what the model can express.',
  'feature map':
    'How classical numbers get into the quantum computer. Each feature becomes a rotation angle applied to a qubit, so the data is written into the quantum state before any learning happens.',
  'angle encoding':
    'The simplest feature map: each feature value becomes one rotation angle on one qubit. Cheap, shallow, and needs exactly as many qubits as features.',
  'amplitude encoding':
    'Packs feature values into the amplitudes of the quantum state. More compact in principle, but the circuit that prepares it is deeper and harder to run on real hardware.',
  zzfeaturemap:
    'A feature map that also encodes products of feature pairs, creating correlations a classical model would have to be told about explicitly. Believed hard to simulate classically, which is where a genuine quantum advantage might come from.',
  shots:
    'A quantum computer cannot read an exact answer - it runs the circuit repeatedly and averages the outcomes. More shots means a more precise estimate but a slower run. Zero here means the exact value, which only a simulator can give.',
  'parameter shift':
    'The rule used to compute gradients on quantum hardware. Each parameter is evaluated twice, shifted forward and back by a quarter turn, and the difference gives the exact derivative. Backpropagation is not available on a real device.',
  'expectation value':
    'The average measurement outcome for an observable, between -1 and +1 here. It is the circuit output that gets turned into a class probability.',
  'barren plateau':
    'A failure mode where gradients vanish exponentially as circuits get wider or deeper, leaving the optimiser with no direction to move. It is the main reason this platform keeps qubit counts small.',
  entanglement:
    'A correlation between qubits with no classical equivalent - measuring one instantly constrains the other. Created here by CNOT gates, and it is what lets the circuit represent feature interactions.',
  vqc:
    'Variational Quantum Classifier. A hybrid model: a quantum circuit computes the prediction, while a classical optimiser adjusts its parameters. The practical design for current hardware.',
  'circuit depth':
    'How many layers of gates run in sequence. Depth matters because real qubits decohere - lose their quantum state - as time passes, so a deep circuit accumulates more error.',
  simulator:
    'Classical software that computes exactly what a quantum computer would do, by tracking every amplitude. Exact and noise-free, but the cost doubles per qubit, so it only works for small registers.',
}
