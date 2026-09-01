import { BENCHMARK_RESULTS, PIPELINE_NODES } from './pipeline/graph'
import type { LogLine, NodeState } from '../hooks/usePipeline'
import type { DatasetSummary } from './dataset'

/**
 * Builds the downloadable run report. Every metric here originates from the
 * mock runner, so the payload carries an explicit `disclaimer` and a
 * `synthetic: true` flag that survive into whatever consumes the file.
 */
export type RunReport = {
  schema: string
  generatedAt: string
  synthetic: true
  disclaimer: string
  dataset: {
    name: string
    source: 'bundled-sample' | 'user-upload'
    rows: number | null
    columns: number | null
  }
  run: {
    status: string
    elapsedMs: number
    stagesTotal: number
    stagesCompleted: number
  }
  stages: {
    id: string
    label: string
    lane: string
    status: string
    progress: number
    metrics: Record<string, number | string> | null
  }[]
  benchmark: typeof BENCHMARK_RESULTS
  log: { time: string; stage: string; level: string; message: string }[]
}

const DISCLAIMER =
  'All metrics in this report are placeholder values produced by a mock pipeline runner. ' +
  'No model was trained and no dataset was evaluated. Do not cite these figures as results.'

export function buildReport(args: {
  nodeStates: Record<string, NodeState>
  logs: LogLine[]
  phase: string
  elapsed: number
  datasetName: string
  upload: DatasetSummary | null
}): RunReport {
  const { nodeStates, logs, phase, elapsed, datasetName, upload } = args

  const stages = PIPELINE_NODES.map((spec) => {
    const st = nodeStates[spec.id]
    return {
      id: spec.id,
      label: spec.label,
      lane: spec.lane,
      status: st?.status ?? 'idle',
      progress: Number((st?.progress ?? 0).toFixed(3)),
      metrics: st?.metrics ?? null,
    }
  })

  return {
    schema: 'neutral.run-report/v1',
    generatedAt: new Date().toISOString(),
    synthetic: true,
    disclaimer: DISCLAIMER,
    dataset: {
      name: upload ? upload.name : datasetName,
      source: upload ? 'user-upload' : 'bundled-sample',
      rows: upload ? upload.rows : null,
      columns: upload ? upload.columns : null,
    },
    run: {
      status: phase,
      elapsedMs: elapsed,
      stagesTotal: stages.length,
      stagesCompleted: stages.filter((s) => s.status === 'done').length,
    },
    stages,
    benchmark: BENCHMARK_RESULTS,
    log: logs.map((l) => ({
      time: new Date(l.timestamp).toISOString(),
      stage: l.nodeId,
      level: l.level,
      message: l.message,
    })),
  }
}

/** Flat CSV of the comparison table, for spreadsheet users. */
export function buildMetricsCsv(): string {
  const { classical, quantum } = BENCHMARK_RESULTS
  const rows = [
    ['metric', 'classical', 'quantum', 'delta'],
    ...(['accuracy', 'rocAuc', 'sensitivity', 'specificity'] as const).map((k) => [
      k,
      classical[k].toFixed(3),
      quantum[k].toFixed(3),
      (quantum[k] - classical[k]).toFixed(3),
    ]),
  ]
  return [
    '# SYNTHETIC PLACEHOLDER DATA - no model was trained, do not cite',
    ...rows.map((r) => r.join(',')),
  ].join('\n')
}

/** Triggers a browser download for an in-memory string. */
export function downloadText(filename: string, mime: string, contents: string) {
  const blob = new Blob([contents], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Revoke on the next frame so the navigation has certainly started.
  requestAnimationFrame(() => URL.revokeObjectURL(url))
}
