import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PIPELINE_NODES } from '../lib/pipeline/graph'
import { createMockRunner } from '../lib/pipeline/runner'
import type { LogLevel, NodeStatus, PipelineEvent } from '../lib/pipeline/types'

export type NodeState = {
  status: NodeStatus
  progress: number
  metrics?: Record<string, number | string>
}

export type LogLine = {
  id: number
  nodeId: string
  level: LogLevel
  message: string
  timestamp: number
}

export type RunPhase = 'idle' | 'running' | 'stopped' | 'complete' | 'failed'

const initialStates = () =>
  Object.fromEntries(
    PIPELINE_NODES.map((n) => [n.id, { status: 'idle' as NodeStatus, progress: 0 }]),
  ) as Record<string, NodeState>

function levelFor(e: PipelineEvent): LogLevel {
  if (e.status === 'error') return 'error'
  if (e.status === 'done') return 'success'
  if (e.status === 'queued') return 'warn'
  // Scripted lines carry their own level in graph.ts; recover it by matching text.
  const spec = PIPELINE_NODES.find((n) => n.id === e.nodeId)
  const line = spec?.script.find((s) => s.message === e.message)
  return line?.level ?? 'info'
}

export function usePipeline(failAt?: string) {
  const [nodeStates, setNodeStates] = useState<Record<string, NodeState>>(initialStates)
  const [logs, setLogs] = useState<LogLine[]>([])
  const [phase, setPhase] = useState<RunPhase>('idle')
  const [elapsed, setElapsed] = useState(0)

  const logSeq = useRef(0)
  const startedAt = useRef<number | null>(null)

  // Recreated when failAt changes so the injected failure takes effect.
  const runner = useMemo(() => createMockRunner({ failAt }), [failAt])

  useEffect(() => {
    return runner.subscribe((e) => {
      setNodeStates((prev) => ({
        ...prev,
        [e.nodeId]: {
          status: e.status,
          progress: e.progress ?? prev[e.nodeId]?.progress ?? 0,
          metrics: e.metrics ?? prev[e.nodeId]?.metrics,
        },
      }))

      if (e.message) {
        setLogs((prev) => [
          ...prev,
          {
            id: logSeq.current++,
            nodeId: e.nodeId,
            level: levelFor(e),
            message: e.message as string,
            timestamp: e.timestamp,
          },
        ])
      }
    })
  }, [runner])

  // Derive the run phase from node states rather than tracking it separately,
  // so a real backend that reports terminal states gets the same behaviour.
  useEffect(() => {
    const all = Object.values(nodeStates)
    const anyError = all.some((s) => s.status === 'error')
    const anyActive = all.some((s) => s.status === 'running')
    const allSettled = all.every(
      (s) => s.status === 'done' || s.status === 'error' || s.status === 'queued',
    )

    setPhase((prev) => {
      if (prev === 'idle' && !anyActive) return 'idle'
      if (allSettled) return anyError ? 'failed' : 'complete'
      if (anyActive) return 'running'
      return prev
    })
  }, [nodeStates])

  // Elapsed timer, frozen once the run settles.
  useEffect(() => {
    if (phase !== 'running') return
    const id = setInterval(() => {
      if (startedAt.current) setElapsed(Date.now() - startedAt.current)
    }, 100)
    return () => clearInterval(id)
  }, [phase])

  const start = useCallback(() => {
    startedAt.current = Date.now()
    setElapsed(0)
    setPhase('running')
    runner.start()
  }, [runner])

  const stop = useCallback(() => {
    runner.stop()
    setPhase('stopped')
  }, [runner])

  const reset = useCallback(() => {
    runner.reset()
    setNodeStates(initialStates())
    setLogs([])
    setElapsed(0)
    startedAt.current = null
    setPhase('idle')
  }, [runner])

  return { nodeStates, logs, phase, elapsed, start, stop, reset }
}
