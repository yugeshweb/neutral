export type GradCamViewMode = 'hybrid' | 'heatmap' | 'contours'

interface Props {
  mode: GradCamViewMode
  onModeChange: (mode: GradCamViewMode) => void
  opacity: number
  onOpacityChange: (opacity: number) => void
  threshold: number
  onThresholdChange: (threshold: number) => void
  targetLayer?: string
  className?: string
}

export function GradCamControls({
  mode,
  onModeChange,
  opacity,
  onOpacityChange,
  threshold,
  onThresholdChange,
  targetLayer = 'backbone.layer4 (ResNet-50)',
  className = '',
}: Props) {
  return (
    <div
      className={`rounded-[8px] p-3 text-[11.5px] ${className}`}
      style={{
        background: '#0D0E10',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-white/5">


        {/* View Mode Toggle Buttons */}
        <div
          className="inline-flex rounded-[6px] p-0.5"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <button
            type="button"
            onClick={() => onModeChange('hybrid')}
            className={`cursor-pointer rounded-[4px] px-2.5 py-1 font-mono text-[10.5px] transition-colors ${
              mode === 'hybrid'
                ? 'bg-white/15 text-ink font-medium shadow-sm'
                : 'text-ink-faint hover:text-ink'
            }`}
          >
            Hybrid
          </button>
          <button
            type="button"
            onClick={() => onModeChange('heatmap')}
            className={`cursor-pointer rounded-[4px] px-2.5 py-1 font-mono text-[10.5px] transition-colors ${
              mode === 'heatmap'
                ? 'bg-white/15 text-ink font-medium shadow-sm'
                : 'text-ink-faint hover:text-ink'
            }`}
          >
            Heatmap
          </button>
          <button
            type="button"
            onClick={() => onModeChange('contours')}
            className={`cursor-pointer rounded-[4px] px-2.5 py-1 font-mono text-[10.5px] transition-colors ${
              mode === 'contours'
                ? 'bg-white/15 text-ink font-medium shadow-sm'
                : 'text-ink-faint hover:text-ink'
            }`}
          >
            Contours
          </button>
        </div>
      </div>

      {/* Sliders and Colormap scale */}
      {mode !== 'contours' && (
        <div className="mt-2.5 grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          {/* Opacity control */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between font-mono text-[10.5px] text-ink-faint">
              <span>Heatmap Opacity</span>
              <span>{Math.round(opacity * 100)}%</span>
            </div>
            <input
              type="range"
              min={0.2}
              max={1.0}
              step={0.05}
              value={opacity}
              onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#5FA88C]"
            />
          </div>

          {/* Saliency threshold cutoff */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between font-mono text-[10.5px] text-ink-faint">
              <span>ReLU Saliency Floor</span>
              <span>{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={0.4}
              step={0.02}
              value={threshold}
              onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#C08A3E]"
            />
          </div>
        </div>
      )}

      {/* Colorbar Legend */}
      <div className="mt-3 flex items-center gap-2 pt-2 border-t border-white/5">
        <span className="font-mono text-[10px] text-ink-faint shrink-0">0.0 (baseline)</span>
        <div
          className="h-2 flex-1 rounded-full overflow-hidden"
          style={{
            background:
              'linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000)',
            opacity: 0.85,
          }}
        />
        <span className="font-mono text-[10px] text-ink shrink-0 font-medium">1.0 (peak attention)</span>
      </div>
    </div>
  )
}
