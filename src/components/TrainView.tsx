import { ReactFlowProvider } from '@xyflow/react'
import { useEffect, useMemo, useState } from 'react'
import { usePipeline } from '../hooks/usePipeline'
import { useReducedMotion } from '../hooks/useReducedMotion'
import type { DatasetSummary } from '../lib/dataset'
import { buildMetricsCsv, buildReport, downloadText } from '../lib/export'
import { PIPELINE_NODES } from '../lib/pipeline/graph'
import { CanvasBackdrop } from './CanvasBackdrop'
import { Console } from './Console'
import { ImageViewer } from './ImageViewer'
import { InputPanel } from './InputPanel'
import { LaneLegend } from './LaneLegend'
import type { AppMode } from './LaunchScreen'
import { NodeDrawer } from './NodeDrawer'
import { OutputPanel } from './OutputPanel'
import { PipelineGraph } from './PipelineGraph'
import { ResultsPanel } from './ResultsPanel'
import { SourceConnector } from './SourceConnector'
import { TopBar } from './TopBar'

type Props = {
  onHome: () => void
  onMode: (m: AppMode) => void
  /** raised the first time a run reaches `complete` */
  onTrained: () => void
}

export function TrainView({ onHome, onMode, onTrained }: Props) {
  // Swap `usePipeline()` for a live-stream runner and nothing below changes.
  // Pass a node id (e.g. usePipeline('vqc')) to exercise the error path.
  const { nodeStates, logs, phase, elapsed, start, stop, reset } = usePipeline()

  const [dataset, setDataset] = useState('Wisconsin Breast Cancer')
  const [upload, setUpload] = useState<DatasetSummary | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [consoleOpen, setConsoleOpen] = useState(true)
  const [resultsOpen, setResultsOpen] = useState(false)
  const [viewerOpen, setViewerOpen] = useState(false)
  const reducedMotion = useReducedMotion()

  // Replacing or clearing an image upload must release its object URL.
  const replaceUpload = (next: DatasetSummary | null) => {
    setUpload((prev) => {
      if (prev?.objectUrl && prev.objectUrl !== next?.objectUrl) {
        URL.revokeObjectURL(prev.objectUrl)
      }
      return next
    })
    if (!next || next.kind !== 'image') setViewerOpen(false)
  }

  const drawerLogs = useMemo(
    () => (selected ? logs.filter((l) => l.nodeId === selected) : []),
    [logs, selected],
  )

  const stagesDone = useMemo(
    () => Object.values(nodeStates).filter((s) => s.status === 'done').length,
    [nodeStates],
  )

  // Clicking the results node opens the comparison panel instead of the drawer.
  const onSelect = (id: string | null) => {
    if (id === 'results') {
      setResultsOpen(true)
      setSelected(null)
      return
    }
    setSelected(id)
  }

  // Surface the comparison automatically once the run lands, and unlock the
  // predict / compare modes for the rest of the session.
  useEffect(() => {
    if (phase === 'complete') {
      setResultsOpen(true)
      onTrained()
    }
  }, [phase, onTrained])

  const handleReset = () => {
    reset()
    setSelected(null)
    setResultsOpen(false)
  }

  const handleDownload = (format: 'json' | 'csv') => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    if (format === 'csv') {
      downloadText(`netural-metrics-${stamp}.csv`, 'text/csv', buildMetricsCsv())
      return
    }
    const report = buildReport({
      nodeStates,
      logs,
      phase,
      elapsed,
      datasetName: dataset,
      upload,
    })
    downloadText(
      `netural-run-report-${stamp}.json`,
      'application/json',
      JSON.stringify(report, null, 2),
    )
  }

  return (
    <div className="flex h-full min-w-[1280px] flex-col bg-canvas">
      <TopBar
        phase={phase}
        elapsed={elapsed}
        dataset={dataset}
        uploadName={upload?.name ?? null}
        onDataset={setDataset}
        onStart={start}
        onStop={stop}
        onReset={handleReset}
        onHome={onHome}
        onMode={onMode}
      />

      <div className="flex min-h-0 flex-1">
        {/* left rail: data in */}
        <aside
          className="console-scroll relative z-20 w-[268px] shrink-0 overflow-y-auto p-3"
          style={{
            background: '#0E0F11',
            borderRight: '1px solid rgba(255,255,255,0.06)',
          }}
          aria-label="Input"
        >
          <InputPanel
            upload={upload}
            onUpload={replaceUpload}
            onView={() => setViewerOpen(true)}
            locked={phase === 'running'}
          />
        </aside>

        <main className="relative min-h-0 flex-1 overflow-hidden">
          <CanvasBackdrop />

          <div className="absolute inset-0 z-10">
            <ReactFlowProvider>
              <PipelineGraph
                nodeStates={nodeStates}
                selectedId={selected}
                onSelect={onSelect}
                reducedMotion={reducedMotion}
              />
            </ReactFlowProvider>
          </div>

          <SourceConnector
            status={nodeStates['ingest']?.status ?? 'idle'}
            reducedMotion={reducedMotion}
          />

          <LaneLegend />

          <NodeDrawer
            nodeId={selected}
            state={selected ? nodeStates[selected] : undefined}
            logs={drawerLogs}
            onClose={() => setSelected(null)}
          />

          <ResultsPanel open={resultsOpen} onClose={() => setResultsOpen(false)} />

          {viewerOpen && upload?.kind === 'image' && (
            <ImageViewer upload={upload} onClose={() => setViewerOpen(false)} />
          )}
        </main>

        {/* right rail: data out */}
        <aside
          className="console-scroll w-[268px] shrink-0 overflow-y-auto p-3"
          style={{
            background: '#0E0F11',
            borderLeft: '1px solid rgba(255,255,255,0.06)',
          }}
          aria-label="Output"
        >
          <OutputPanel
            phase={phase}
            stagesDone={stagesDone}
            stagesTotal={PIPELINE_NODES.length}
            onDownload={handleDownload}
            onOpenComparison={() => setResultsOpen(true)}
          />
        </aside>
      </div>

      <Console logs={logs} open={consoleOpen} onToggle={() => setConsoleOpen((o) => !o)} />
    </div>
  )
}
