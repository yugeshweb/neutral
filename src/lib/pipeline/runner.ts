import { PIPELINE_NODES } from './graph'
import { trainModel, type QmlMetricKey, type QmlResult, type TrainRequest } from '../qmlApi'
import type { PipelineEvent, PipelineNodeSpec } from './types'

export interface PipelineRunner {
  subscribe(cb: (e: PipelineEvent) => void): () => void
  start(): void
  stop(): void
  reset(): void
}

function metric(result: QmlResult, modelName: string | undefined, key: QmlMetricKey) {
  const value = modelName ? result.models?.[modelName]?.metrics?.[key] : undefined
  return typeof value === 'number' ? value : null
}

function firstModel(result: QmlResult, quantum: boolean) {
  return Object.entries(result.models ?? {}).find(([, model]) =>
    quantum ? model.feature_space === 'quantum' : model.feature_space === 'classical',
  )?.[0]
}

function liveStageMetrics(result: QmlResult, nodeId: string): Record<string, number | string> {
  const dataset = result.dataset ?? {}
  const preprocessing = result.preprocessing ?? {}
  const execution = result.execution ?? {}
  const classical = firstModel(result, false)
  const quantum = firstModel(result, true)
  const models = Object.keys(result.models ?? {})

  switch (nodeId) {
    case 'ingest':
      return {
        rows: dataset.rows ?? 0,
        columns: dataset.features ?? 0,
        positive: dataset.positive_label ?? 'configured target',
      }
    case 'clean':
      return { imputation: 'training median', standardisation: 'training split only' }
    case 'features':
      return {
        selected: preprocessing.selected_features?.length ?? 0,
        reduction: preprocessing.reduction ?? execution.reduction ?? 'anova',
        qubits: preprocessing.qubits ?? 0,
      }
    case 'baseline':
      return {
        model: classical ?? 'not requested',
        balanced_accuracy: metric(result, classical, 'balanced_accuracy') ?? 'n/a',
      }
    case 'classical-infer':
      return {
        model: classical ?? 'not requested',
        rows: result.models?.[classical ?? '']?.resource?.test_rows ?? 0,
      }
    case 'encode':
      return {
        backend: execution.resolved_backend ?? execution.backend_mode ?? 'local',
        qubits: preprocessing.qubits ?? 0,
      }
    case 'vqc':
      return {
        model: quantum ?? 'not requested',
        balanced_accuracy: metric(result, quantum, 'balanced_accuracy') ?? 'n/a',
      }
    case 'measure':
      return {
        backend: execution.resolved_backend ?? execution.backend_mode ?? 'local',
        shots: execution.shots ?? 0,
      }
    case 'benchmark': {
      const classicalScore = metric(result, classical, 'balanced_accuracy')
      const quantumScore = metric(result, quantum, 'balanced_accuracy')
      return {
        models: models.length,
        balanced_delta:
          classicalScore !== null && quantumScore !== null
            ? Number((quantumScore - classicalScore).toFixed(4))
            : 'n/a',
      }
    }
    case 'explain':
      return { status: 'available on prediction request' }
    case 'results':
      return { models: models.length, dataset: dataset.name ?? 'configured dataset' }
    default:
      return {}
  }
}

/**
 * Runs the actual local QML service while preserving Neutral's pipeline graph.
 * The server returns one sealed result because the Python experiment is a
 * synchronous job; stage cards are marked complete only after that result has
 * been produced.
 */
export function createApiRunner(
  request: TrainRequest,
  onResult: (result: QmlResult) => void,
): PipelineRunner {
  let subscribers: ((e: PipelineEvent) => void)[] = []
  let controller: AbortController | null = null
  let running = false

  function emit(event: Omit<PipelineEvent, 'timestamp'>) {
    const next: PipelineEvent = { ...event, timestamp: Date.now() }
    for (const subscriber of subscribers) subscriber(next)
  }

  function publishResult(result: QmlResult) {
    for (const spec of PIPELINE_NODES) {
      emit({
        nodeId: spec.id,
        status: 'done',
        progress: 1,
        message: `server completed ${spec.label.toLowerCase()}`,
        metrics: liveStageMetrics(result, spec.id),
        ...(spec.id === 'results' ? { result } : {}),
      })
    }
    onResult(result)
  }

  async function run() {
    try {
      const result = await trainModel(request, controller?.signal)
      if (!running) return
      publishResult(result)
    } catch (error) {
      if (!running || (error instanceof DOMException && error.name === 'AbortError')) return
      const message = error instanceof Error ? error.message : 'training request failed'
      emit({ nodeId: 'ingest', status: 'error', progress: 1, message })
      for (const spec of PIPELINE_NODES.slice(1)) {
        emit({
          nodeId: spec.id,
          status: 'queued',
          progress: 0,
          message: 'blocked by training error',
        })
      }
    } finally {
      running = false
      controller = null
    }
  }

  return {
    subscribe(callback) {
      subscribers.push(callback)
      return () => {
        subscribers = subscribers.filter((subscriber) => subscriber !== callback)
      }
    },
    start() {
      if (running) return
      running = true
      controller = new AbortController()
      emit({
        nodeId: 'ingest',
        status: 'running',
        progress: 0,
        message: 'training request sent to local QML service',
      })
      void run()
    },
    stop() {
      running = false
      controller?.abort()
      controller = null
    },
    reset() {
      running = false
      controller?.abort()
      controller = null
      for (const spec of PIPELINE_NODES) {
        emit({ nodeId: spec.id, status: 'idle', progress: 0 })
      }
    },
  }
}

export type MockRunnerOptions = {
  /** playback multiplier; 2 runs the whole pipeline twice as fast */
  speed?: number
  /** force this node into `error`, leaving everything downstream blocked */
  failAt?: string
}

const TICK_MS = 90

type NodeRuntime = {
  spec: PipelineNodeSpec
  status: PipelineEvent['status']
  elapsed: number
  /** index of the next scripted log line to emit */
  cursor: number
}

/**
 * Drives the pipeline graph off a single interval, advancing every node whose
 * dependencies are satisfied. Because eligible nodes advance on the same tick,
 * the classical and quantum lanes genuinely progress in parallel rather than
 * being interleaved by a scheduler.
 */
export function createMockRunner(opts: MockRunnerOptions = {}): PipelineRunner {
  const speed = opts.speed ?? 1
  const failAt = opts.failAt

  let subscribers: ((e: PipelineEvent) => void)[] = []
  let timer: ReturnType<typeof setInterval> | null = null
  let nodes = seed()
  let running = false

  function seed(): Map<string, NodeRuntime> {
    return new Map(
      PIPELINE_NODES.map((spec) => [
        spec.id,
        { spec, status: 'idle' as PipelineEvent['status'], elapsed: 0, cursor: 0 },
      ]),
    )
  }

  function emit(e: Omit<PipelineEvent, 'timestamp'>) {
    const event: PipelineEvent = { ...e, timestamp: Date.now() }
    for (const cb of subscribers) cb(event)
  }

  function depsSatisfied(rt: NodeRuntime) {
    return rt.spec.deps.every((d) => nodes.get(d)?.status === 'done')
  }

  function depsFailed(rt: NodeRuntime) {
    return rt.spec.deps.some((d) => {
      const dep = nodes.get(d)
      return dep?.status === 'error' || dep?.status === 'queued'
    })
  }

  function tick() {
    let advanced = false

    for (const rt of nodes.values()) {
      if (rt.status === 'done' || rt.status === 'error') continue

      // Blocked: an upstream node failed or is itself blocked. `queued` is used
      // as the blocked marker so the UI can dim the whole downstream subtree.
      if (depsFailed(rt)) {
        if (rt.status !== 'queued') {
          rt.status = 'queued'
          emit({ nodeId: rt.spec.id, status: 'queued', message: 'blocked by upstream failure' })
        }
        continue
      }

      if (!depsSatisfied(rt)) continue

      if (rt.status === 'idle') {
        rt.status = 'running'
        emit({
          nodeId: rt.spec.id,
          status: 'running',
          progress: 0,
          message: `stage started: ${rt.spec.label.toLowerCase()}`,
        })
      }

      rt.elapsed += TICK_MS * speed
      const progress = Math.min(1, rt.elapsed / rt.spec.duration)
      advanced = true

      // Fire any scripted log lines this tick crossed over.
      while (rt.cursor < rt.spec.script.length && rt.spec.script[rt.cursor].at <= progress) {
        const line = rt.spec.script[rt.cursor]
        // The terminal `success` line is emitted with the done event instead.
        if (!(line.at >= 1 && progress >= 1)) {
          emit({ nodeId: rt.spec.id, status: 'running', progress, message: line.message })
        }
        rt.cursor++
      }

      if (progress >= 1) {
        if (failAt === rt.spec.id) {
          rt.status = 'error'
          emit({
            nodeId: rt.spec.id,
            status: 'error',
            progress: 1,
            message: `stage failed: ${rt.spec.label.toLowerCase()}`,
          })
        } else {
          rt.status = 'done'
          const last = rt.spec.script[rt.spec.script.length - 1]
          emit({
            nodeId: rt.spec.id,
            status: 'done',
            progress: 1,
            message: last?.message,
            metrics: rt.spec.metrics,
          })
        }
      } else {
        emit({ nodeId: rt.spec.id, status: 'running', progress })
      }
    }

    const settled = [...nodes.values()].every(
      (r) => r.status === 'done' || r.status === 'error' || r.status === 'queued',
    )
    if (settled && !advanced) stop()
  }

  function start() {
    if (running) return
    running = true
    timer = setInterval(tick, TICK_MS)
  }

  function stop() {
    running = false
    if (timer) clearInterval(timer)
    timer = null
  }

  function reset() {
    stop()
    nodes = seed()
    for (const spec of PIPELINE_NODES) {
      emit({ nodeId: spec.id, status: 'idle', progress: 0 })
    }
  }

  return {
    subscribe(cb) {
      subscribers.push(cb)
      return () => {
        subscribers = subscribers.filter((s) => s !== cb)
      }
    },
    start,
    stop,
    reset,
  }
}

/* ---------------------------------------------------------------------------
 * TODO: replace the mock with a live event stream.
 *
 * The UI layer only ever sees `PipelineRunner`, so swapping transports is a
 * one-line change in `usePipeline`: no component needs to be touched.
 *
 * EventSource (server-sent events):
 *
 *   export function createSSERunner(url: string): PipelineRunner {
 *     let es: EventSource | null = null
 *     let subscribers: ((e: PipelineEvent) => void)[] = []
 *
 *     return {
 *       subscribe(cb) {
 *         subscribers.push(cb)
 *         return () => { subscribers = subscribers.filter((s) => s !== cb) }
 *       },
 *       start() {
 *         es = new EventSource(`${url}/run`)
 *         es.onmessage = (msg) => {
 *           // The server is expected to send one PipelineEvent per message.
 *           const event = JSON.parse(msg.data) as PipelineEvent
 *           for (const cb of subscribers) cb(event)
 *         }
 *         es.onerror = () => {
 *           // Surface transport failure as a pipeline-level error event so the
 *           // existing error styling applies without special-casing.
 *           for (const cb of subscribers) {
 *             cb({ nodeId: 'ingest', status: 'error', message: 'stream disconnected', timestamp: Date.now() })
 *           }
 *         }
 *       },
 *       stop() { es?.close(); es = null },
 *       reset() {
 *         es?.close(); es = null
 *         void fetch(`${url}/reset`, { method: 'POST' })
 *         for (const spec of PIPELINE_NODES) {
 *           for (const cb of subscribers) {
 *             cb({ nodeId: spec.id, status: 'idle', progress: 0, timestamp: Date.now() })
 *           }
 *         }
 *       },
 *     }
 *   }
 *
 * WebSocket: identical shape, with `new WebSocket(url)` in start(), the same
 * JSON.parse in `onmessage`, `ws.close()` in stop(), and a
 * `ws.send(JSON.stringify({ type: 'reset' }))` in reset().
 *
 * Contract the backend must honour:
 *   - one PipelineEvent per message, `nodeId` matching an id in graph.ts
 *   - progress monotonically increasing within a stage, terminated by
 *     status 'done' (carrying `metrics`) or 'error'
 *   - blocked downstream nodes reported as status 'queued'
 * ------------------------------------------------------------------------- */
