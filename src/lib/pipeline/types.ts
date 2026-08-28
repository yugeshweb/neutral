export type NodeStatus = 'idle' | 'queued' | 'running' | 'done' | 'error'

export type LogLevel = 'info' | 'warn' | 'error' | 'success'

export type PipelineEvent = {
  nodeId: string
  status: NodeStatus
  progress?: number // 0..1
  message?: string
  metrics?: Record<string, number | string>
  timestamp: number
}

export type Lane = 'shared' | 'classical' | 'quantum'

export type PipelineNodeSpec = {
  id: string
  label: string
  subtitle: string
  lane: Lane
  /** ids that must report `done` before this node may start */
  deps: string[]
  /** wall-clock duration of the mocked stage, in ms at speed = 1 */
  duration: number
  /** mock configuration surfaced in the inspector drawer */
  config: Record<string, string>
  /** metrics emitted alongside the terminal `done` event */
  metrics: Record<string, number | string>
  /** stage-relative log lines, keyed by the progress fraction they fire at */
  script: { at: number; level: LogLevel; message: string }[]
}
