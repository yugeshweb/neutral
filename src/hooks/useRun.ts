import { useCallback, useRef, useState } from 'react'
import { loadDataset } from '../lib/ml/datasets'
import {
  DEFAULT_RUN,
  runPipeline,
  type RunConfig,
  type RunResult,
} from '../lib/ml/pipeline'
import type { EpochRecord } from '../lib/quantum/vqc'

export type RunPhase = 'idle' | 'running' | 'complete' | 'stopped'

export type LogLine = {
  id: number
  phase: string
  message: string
  timestamp: number
}

/**
 * Owns the run configuration and drives the real pipeline.
 *
 * The pipeline is a generator, so it is stepped from a timer rather than run
 * in one blocking loop - that keeps the UI responsive and lets the convergence
 * chart fill in as training proceeds, which is the whole point of showing it.
 */
export function useRun() {
  const [config, setConfig] = useState<RunConfig>(DEFAULT_RUN)
  const [phase, setPhase] = useState<RunPhase>('idle')
  const [logs, setLogs] = useState<LogLine[]>([])
  const [convergence, setConvergence] = useState<EpochRecord[]>([])
  const [result, setResult] = useState<RunResult | null>(null)
  const [progress, setProgress] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  const seq = useRef(0)
  const cancelled = useRef(false)
  const timer = useRef<number | null>(null)
  const startedAt = useRef(0)

  const patch = useCallback((p: Partial<RunConfig>) => {
    setConfig((c) => {
      const next = { ...c, ...p }
      // The qubit count is the feature count. Enforced here so the two can
      // never drift apart anywhere in the UI.
      next.vqc = { ...next.vqc, ...(p.vqc ?? {}), qubits: next.nFeatures }
      return next
    })
  }, [])

  const stop = useCallback(() => {
    cancelled.current = true
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
    setPhase((p) => (p === 'running' ? 'stopped' : p))
  }, [])

  const reset = useCallback(() => {
    stop()
    cancelled.current = false
    setPhase('idle')
    setLogs([])
    setConvergence([])
    setResult(null)
    setProgress(0)
    setElapsed(0)
  }, [stop])

  const start = useCallback(() => {
    cancelled.current = false
    seq.current = 0
    startedAt.current = Date.now()
    setPhase('running')
    setLogs([])
    setConvergence([])
    setResult(null)
    setProgress(0)
    setElapsed(0)

    const data = loadDataset(config.datasetId)
    const gen = runPipeline(data, { ...config, vqc: { ...config.vqc, qubits: config.nFeatures } })

    const pump = () => {
      if (cancelled.current) return

      // Drain a slice of work per frame, then yield to the browser so the
      // chart paints. Epoch steps are the expensive ones, so one at a time.
      const deadline = performance.now() + 24
      let done = false

      while (performance.now() < deadline) {
        const next = gen.next()
        if (next.done) {
          done = true
          break
        }
        const ev = next.value

        if (ev.phase === 'done') {
          setResult(ev.result)
          setProgress(1)
          setPhase('complete')
          setElapsed(Date.now() - startedAt.current)
          done = true
          break
        }

        if (ev.phase === 'quantum') {
          setConvergence((c) => [...c, ev.epoch])
          setProgress(ev.epoch.epoch / ev.total)
          // One epoch per frame: they are the slow step and the curve should
          // visibly advance rather than jumping.
          break
        }

        // Remaining variants all carry a `message`.
        const line = ev as { phase: string; message: string }
        setLogs((l) => [
          ...l,
          {
            id: seq.current++,
            phase: line.phase,
            message: line.message,
            timestamp: Date.now(),
          },
        ])
      }

      setElapsed(Date.now() - startedAt.current)
      if (!done) timer.current = window.setTimeout(pump, 0)
    }

    pump()
  }, [config])

  /**
   * Runs the pipeline synchronously and lands straight on the finished result.
   *
   * Simulator training is slow, and a live demo that stalls in front of judges
   * is fatal. This computes the same real numbers - nothing is pre-baked - but
   * without yielding to the browser between steps, so it completes in one go
   * and the Results screen is populated immediately.
   */
  const loadDemo = useCallback(() => {
    cancelled.current = true
    seq.current = 0
    const demo: RunConfig = {
      ...DEFAULT_RUN,
      nFeatures: 4,
      epochs: 12,
      baselines: ['logistic', 'random-forest', 'svm'],
    }
    setConfig(demo)
    setLogs([])
    setConvergence([])

    const started = Date.now()
    const data = loadDataset(demo.datasetId)
    const lines: LogLine[] = []
    const epochs: EpochRecord[] = []

    for (const ev of runPipeline(data, demo)) {
      if (ev.phase === 'done') {
        setResult(ev.result)
        setPhase('complete')
        setProgress(1)
        setElapsed(Date.now() - started)
      } else if (ev.phase === 'quantum') {
        epochs.push(ev.epoch)
      } else {
        const l = ev as { phase: string; message: string }
        lines.push({
          id: seq.current++,
          phase: l.phase,
          message: l.message,
          timestamp: Date.now(),
        })
      }
    }

    setLogs(lines)
    setConvergence(epochs)
    cancelled.current = false
  }, [])

  return {
    config,
    patch,
    setConfig,
    phase,
    logs,
    convergence,
    result,
    progress,
    elapsed,
    start,
    stop,
    reset,
    loadDemo,
  }
}
