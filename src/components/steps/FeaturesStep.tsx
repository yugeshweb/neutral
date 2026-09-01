import { useMemo } from 'react'
import { describeFeature, loadDataset } from '../../lib/ml/datasets'
import { componentComposition } from '../../lib/ml/explain'
import { dropCollinear, pca, rankFeatures } from '../../lib/ml/features'
import type { RunConfig } from '../../lib/ml/pipeline'
import { applyScaler, fitScaler, stratifiedSplit } from '../../lib/ml/stats'
import { makeRng } from '../../lib/quantum/statevector'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { ImportanceBars } from '../charts'
import { QubitBadge } from '../QubitBadge'
import { Tooltip } from '../Tooltip'
import { LiveChip, Panel, SectionLabel, Slider } from '../ui'

type Props = {
  config: RunConfig
  patch: (p: Partial<RunConfig>) => void
  locked: boolean
}

const METHOD_HELP: Record<string, string> = {
  pca: 'Principal Component Analysis: builds new axes that capture the most variance. Compact, but each axis is a blend of every original measurement, which is what makes explaining it hard.',
  'mutual-info': 'Mutual information: how much knowing a feature reduces uncertainty about the diagnosis. Catches nonlinear relationships that correlation misses.',
  anova: 'ANOVA F-test: how far apart the class means are, relative to the spread within each class. Fast, but only sees linear separation.',
  rfe: 'Recursive elimination: fits a model, drops the weakest feature, repeats. Slow but accounts for how features behave together.',
}

export function FeaturesStep({ config, patch, locked }: Props) {
  const data = useMemo(() => loadDataset(config.datasetId), [config.datasetId])

  const analysis = useMemo(() => {
    const rng = makeRng(config.seed)
    const split = stratifiedSplit(data.y, config.testFraction, rng)
    const Xtr = split.trainIdx.map((i) => data.X[i])
    const ytr = split.trainIdx.map((i) => data.y[i])
    const scaler = fitScaler(Xtr, config.scaler)
    const Str = applyScaler(Xtr, scaler)

    const ranked = rankFeatures(Str, ytr, data.featureNames, config.selection)
    const collinear = dropCollinear(Str, ranked, 0.92)
    const p = config.selection === 'pca' ? pca(Str, config.nFeatures, config.seed) : null

    return { ranked, collinear, pca: p, Str }
  }, [data, config.selection, config.nFeatures, config.seed, config.scaler, config.testFraction])

  const kept = analysis.ranked.slice(0, config.nFeatures)
  const dropped = analysis.ranked.slice(config.nFeatures)
  const varianceRetained = analysis.pca
    ? analysis.pca.cumulative[analysis.pca.cumulative.length - 1]
    : null

  return (
    <div className="grid grid-cols-[330px_1fr] gap-4">
      <div className="space-y-4">
        <Panel>
          <SectionLabel hint={{ term: config.selection, body: METHOD_HELP[config.selection] }}>
            selection method
          </SectionLabel>
          <div className="mt-2.5 grid grid-cols-2 gap-1.5">
            {(['mutual-info', 'anova', 'pca', 'rfe'] as const).map((m) => (
              <button
                key={m}
                type="button"
                disabled={locked}
                title={METHOD_HELP[m]}
                onClick={() => patch({ selection: m })}
                className="cursor-pointer rounded-[7px] px-2 py-2 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
                style={{
                  background: config.selection === m ? alpha(LANE_COLOR.quantum, 0.1) : '#0D0E10',
                  border: `1px solid ${config.selection === m ? alpha(LANE_COLOR.quantum, 0.35) : 'rgba(255,255,255,0.05)'}`,
                  boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
                }}
              >
                <span
                  className="font-mono text-[11.5px]"
                  style={{ color: config.selection === m ? '#E8E9EB' : '#9A9CA1' }}
                >
                  {m}
                </span>
              </button>
            ))}
          </div>
        </Panel>

        {/* the qubit link - the whole point of this screen */}
        <Panel>
          <div className="flex items-baseline justify-between">
            <SectionLabel>features to keep</SectionLabel>
            <LiveChip />
          </div>

          <div className="mt-3 flex items-baseline gap-2">
            <span className="font-mono text-[34px] font-medium leading-none tabular-nums text-ink">
              {config.nFeatures}
            </span>
            <span className="font-mono text-[13px] text-ink-faint">
              of {data.featureNames.length} features
            </span>
          </div>

          <div className="mt-3">
            <Slider
              id="nfeatures"
              min={2}
              max={Math.min(10, data.featureNames.length)}
              step={1}
              value={config.nFeatures}
              disabled={locked}
              onChange={(v) => patch({ nFeatures: v })}
            />
          </div>

          {/* the binding: features === qubits */}
          <div
            className="mt-3 rounded-[8px] p-3"
            style={{
              background: alpha(LANE_COLOR.quantum, 0.07),
              border: `1px solid ${alpha(LANE_COLOR.quantum, 0.26)}`,
            }}
          >
            <div className="flex items-center justify-center gap-2.5">
              <span className="font-mono text-[14.5px] tabular-nums text-ink">
                {config.nFeatures} features
              </span>
              <span className="font-mono text-[14.5px] text-ink-faint">=</span>
              <QubitBadge qubits={config.nFeatures} compact />
            </div>

            <div
              className="mt-2.5 flex items-center justify-between pt-2.5 font-mono text-[11px]"
              style={{ borderTop: `1px solid ${alpha(LANE_COLOR.quantum, 0.18)}` }}
            >
              <span className="text-ink-faint">simulated state</span>
              <span className="tabular-nums text-ink-dim">
                2^{config.nFeatures} = {(2 ** config.nFeatures).toLocaleString()} amplitudes
              </span>
            </div>
            <p className="mt-2 font-mono text-[11px] leading-relaxed text-ink-faint/85">
              One retained feature is one qubit. Every qubit added doubles the simulation
              cost and deepens the{' '}
              <Tooltip
                term="Barren plateau"
                body="A failure mode where gradients vanish exponentially as circuits get wider or deeper, leaving the optimiser with no direction to move. It is the main reason this platform keeps qubit counts small."
              >
                <span className="underline decoration-dotted underline-offset-2">
                  barren plateau
                </span>
              </Tooltip>{' '}
              risk, which is why this slider is the binding constraint on the whole pipeline.
            </p>
          </div>
        </Panel>

        {analysis.pca && (
          <Panel>
            <SectionLabel>variance retained</SectionLabel>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="font-mono text-[26px] tabular-nums text-ink">
                {((varianceRetained ?? 0) * 100).toFixed(1)}%
              </span>
              <span className="font-mono text-[11.5px] text-ink-faint">
                in {config.nFeatures} components
              </span>
            </div>
            <div className="mt-2 flex h-[5px] overflow-hidden rounded-full panel-well">
              <div
                style={{
                  width: `${(varianceRetained ?? 0) * 100}%`,
                  background: alpha(LANE_COLOR.quantum, 0.75),
                }}
              />
            </div>
            <div className="mt-2.5 space-y-1">
              {analysis.pca.explained.map((e, i) => (
                <div key={i} className="flex items-center justify-between font-mono text-[11px]">
                  <span className="text-ink-faint">PC{i + 1}</span>
                  <span className="tabular-nums text-ink-dim">{(e * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </Panel>
        )}
      </div>

      <div className="space-y-4">
        <Panel>
          <div className="mb-3 flex items-baseline justify-between">
            <div>
              <SectionLabel>ranked importance: all {data.featureNames.length} features</SectionLabel>
              <p className="mt-1 font-mono text-[11px] text-ink-faint">
                scored by {config.selection} on the training fold
              </p>
            </div>
            <QubitBadge qubits={config.nFeatures} compact />
          </div>

          <div className="console-scroll max-h-[300px] overflow-y-auto pr-1">
            <ImportanceBars
              items={analysis.ranked.map((r) => ({ name: r.name, score: r.score }))}
              keepCount={config.nFeatures}
            />
          </div>
        </Panel>

        <div className="grid grid-cols-2 gap-4">
          <Panel>
            <SectionLabel>kept: {kept.length}</SectionLabel>
            <div className="console-scroll mt-2.5 max-h-[170px] space-y-1.5 overflow-y-auto pr-1">
              {(analysis.pca
                ? Array.from({ length: config.nFeatures }, (_, i) => ({
                    name: `PC${i + 1}`,
                    detail: componentComposition(analysis.pca!, i, data.featureNames, 2)
                      .map((c) => c.name)
                      .join(' + '),
                  }))
                : kept.map((k) => ({
                    name: k.name,
                    detail: describeFeature(config.datasetId, k.name) ?? '',
                  }))
              ).map((f) => (
                <div key={f.name}>
                  <div className="flex items-center gap-1.5">
                    <span
                      className="h-[4px] w-[4px] shrink-0 rounded-full"
                      style={{ background: LANE_COLOR.quantum }}
                    />
                    <span className="font-mono text-[12px] text-ink-dim">{f.name}</span>
                  </div>
                  {f.detail && (
                    <p className="ml-[10px] font-mono text-[11px] leading-relaxed text-ink-faint/75">
                      {f.detail.slice(0, 62)}
                      {f.detail.length > 62 ? '…' : ''}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </Panel>

          <Panel>
            <SectionLabel>dropped: {dropped.length}</SectionLabel>
            <div className="console-scroll mt-2.5 max-h-[170px] space-y-1 overflow-y-auto pr-1">
              {dropped.map((d) => (
                <div key={d.name} className="flex items-baseline justify-between gap-2">
                  <span className="min-w-0 truncate font-mono text-[11.5px] text-ink-faint">
                    {d.name}
                  </span>
                  <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-faint/60">
                    {d.score.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>

            {analysis.collinear.dropped.length > 0 && (
              <div
                className="mt-2.5 pt-2.5"
                style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
              >
                <div className="font-mono text-[11px] text-ink-faint/70">
                  collinear pairs found (|r| &gt; 0.92)
                </div>
                {analysis.collinear.dropped.slice(0, 3).map((c) => (
                  <div key={c.name} className="mt-1 font-mono text-[11px] text-ink-faint">
                    {c.name} ≈ {c.against}{' '}
                    <span style={{ color: LANE_COLOR.classical }}>r={c.r.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}
