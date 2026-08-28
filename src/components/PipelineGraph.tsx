import { ReactFlow, type Edge, type Node } from '@xyflow/react'
import { useMemo } from 'react'
import type { NodeState } from '../hooks/usePipeline'
import { PIPELINE_NODES } from '../lib/pipeline/graph'
import { LANE_COLOR } from '../lib/theme'
import { FlowEdge, type FlowEdgeData } from './FlowEdge'
import { StageNode, type StageNodeData } from './StageNode'

const nodeTypes = { stage: StageNode }
const edgeTypes = { flow: FlowEdge }

/** Hand-placed so the branch reads as two clean parallel lanes. */
const POSITIONS: Record<string, { x: number; y: number }> = {
  ingest: { x: 0, y: 208 },
  clean: { x: 262, y: 208 },
  features: { x: 524, y: 208 },
  baseline: { x: 806, y: 60 },
  'classical-infer': { x: 1068, y: 60 },
  encode: { x: 806, y: 356 },
  vqc: { x: 1068, y: 356 },
  measure: { x: 1330, y: 356 },
  benchmark: { x: 1612, y: 208 },
  explain: { x: 1874, y: 208 },
  results: { x: 2136, y: 208 },
}

type Props = {
  nodeStates: Record<string, NodeState>
  selectedId: string | null
  onSelect: (id: string | null) => void
  reducedMotion: boolean
}

export function PipelineGraph({ nodeStates, selectedId, onSelect, reducedMotion }: Props) {
  const nodes: Node[] = useMemo(
    () =>
      PIPELINE_NODES.map((spec) => {
        const st = nodeStates[spec.id]
        return {
          id: spec.id,
          type: 'stage',
          position: POSITIONS[spec.id],
          selected: selectedId === spec.id,
          data: {
            stageId: spec.id,
            label: spec.label,
            subtitle: spec.subtitle,
            lane: spec.lane,
            status: st?.status ?? 'idle',
            progress: st?.progress ?? 0,
            metrics: st?.metrics,
          } satisfies StageNodeData,
        }
      }),
    [nodeStates, selectedId],
  )

  const edges: Edge[] = useMemo(() => {
    const out: Edge[] = []

    for (const spec of PIPELINE_NODES) {
      for (const dep of spec.deps) {
        const source = nodeStates[dep]
        const target = nodeStates[spec.id]

        // An edge only animates once its SOURCE reports done. It settles to
        // 'complete' when the target has finished consuming it.
        let phase: FlowEdgeData['phase'] = 'idle'
        if (source?.status === 'done') {
          phase =
            target?.status === 'done' || target?.status === 'error' ? 'complete' : 'active'
        }

        // The edge takes the lane colour of whichever end is lane-specific,
        // so the branch colours reach back into the shared trunk.
        const specLane = PIPELINE_NODES.find((n) => n.id === dep)?.lane
        const lane = spec.lane !== 'shared' ? spec.lane : (specLane ?? 'shared')

        out.push({
          id: `${dep}->${spec.id}`,
          source: dep,
          target: spec.id,
          type: 'flow',
          data: {
            accent: LANE_COLOR[lane],
            phase,
            reducedMotion,
          } satisfies FlowEdgeData,
        })
      }
    }

    return out
  }, [nodeStates, reducedMotion])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeClick={(_, n) => onSelect(n.id)}
      onPaneClick={() => onSelect(null)}
      defaultViewport={{ x: 96, y: 8, zoom: 1 }}
      minZoom={0.45}
      maxZoom={1.4}
      translateExtent={[
        [-160, -140],
        [2560, 700],
      ]}
      proOptions={{ hideAttribution: true }}
      nodesFocusable
      nodesDraggable={false}
      nodesConnectable={false}
      edgesFocusable={false}
      panOnScroll
      selectionOnDrag={false}
    />
  )
}
