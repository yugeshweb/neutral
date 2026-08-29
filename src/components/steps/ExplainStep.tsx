import { useMemo, useState } from 'react'
import { describeFeature, loadDataset } from '../../lib/ml/datasets'
import {
  columnMeans,
  componentComposition,
  localAttribution,
  mapComponentsToFeatures,
  permutationImportance,
  riskBand,
} from '../../lib/ml/explain'
import { pca, pcaTransform, rankFeatures } from '../../lib/ml/features'
import type { RunResult } from '../../lib/ml/pipeline'
import { applyScaler, fitScaler, stratifiedSplit } from '../../lib/ml/stats'
import { makeRng } from '../../lib/quantum/statevector'
import { Vqc, trainVqc } from '../../lib/quantum/vqc'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { DivergingBars, ImportanceBars } from '../charts'
import { QubitBadge } from '../QubitBadge'
import { LiveChip, Panel, SectionLabel, Well } from '../ui'

type Props = {
  result: RunResult | null
}

const BAND_COLOR = { low: '#5FA88C', moderate: '#C08A3E', high: '#A3543D' }

export function ExplainStep({ result }: Props) {
  const [patientIdx, setPatientIdx] = useState(0)

  const analysis = useMemo(() => {
    if (!result) return null

    const data = loadDataset(result.config.datasetId)
    const cfg = result.config
    const rng = makeRng(cfg.seed)
    const split = stratifiedSplit(data.y, cfg.testFraction, rng)

    const Xtr = split.trainIdx.map((i) => data.X[i])
    const ytr = split.trainIdx.map((i) => data.y[i])
    const Xte = split.testIdx.map((i) => data.X[i])
    const scaler = fitScaler(Xtr, cfg.scaler)
    const Str = applyScaler(Xtr, scaler)
    const Ste = applyScaler(Xte, scaler)

    let Ftr: number[][]
    let Fte: number[][]
    let pcaResult = null
    let modelFeatureNames: string[]
    let keptIdx: number[] = []

    if (cfg.selection === 'pca') {
      pcaResult = pca(Str, cfg.nFeatures, cfg.seed)
      Ftr = pcaTransform(Str, pcaResult)
      Fte = pcaTransform(Ste, pcaResult)
      modelFeatureNames = Array.from({ length: cfg.nFeatures }, (_, i) => `PC${i + 1}`)
    } else {
      const ranked = rankFeatures(Str, ytr, data.featureNames, cfg.selection)
      keptIdx = ranked.slice(0, cfg.nFeatures).map((r) => r.index)
      Ftr = Str.map((r) => keptIdx.map((j) => r[j]))
      Fte = Ste.map((r) => keptIdx.map((j) => r[j]))
      modelFeatureNames = ranked.slice(0, cfg.nFeatures).map((r) => r.name)
    }

    // Retrain the VQC to the same state. The optimiser is fully deterministic
    // given the seed, so this reproduces the model the results came from.
    const model = new Vqc({ ...cfg.vqc, qubits: cfg.nFeatures, seed: cfg.seed })
    for (const _ of trainVqc(model, Ftr, ytr, {
      epochs: cfg.epochs,
      learningRate: cfg.learningRate,
      batchSize: cfg.batchSize,
      seed: cfg.seed,
    })) {
      void _
    }

    const baseline = columnMeans(Ftr)

    // Global importance, model-agnostic.
    const global = permutationImportance(
      (X) => model.predict(X),
      Fte.slice(0, 40),
      split.testIdx.slice(0, 40).map((i) => data.y[i]),
      modelFeatureNames,
      makeRng(cfg.seed),
      2,
    )

    return {
      data,
      cfg,
      model,
      Fte,
      Xte,
      baseline,
      modelFeatureNames,
      pcaResult,
      keptIdx,
      global,
      testIdx: split.testIdx,
    }
  }, [result])

  if (!result || !analysis) {
    return (
      <Panel>
        <div className="grid place-items-center py-14">
          <p className="font-mono text-[11px] text-ink-faint">
            no model to explain yet — run the pipeline first
          </p>
        </div>
      </Panel>
    )
  }

  const { data, cfg, model, Fte, Xte, baseline, modelFeatureNames, pcaResult, global } = analysis
  const idx = Math.min(patientIdx, Fte.length - 1)
  const probability = model.predictOne(Fte[idx])
  const band = riskBand(probability)
  const trueLabel = analysis.testIdx[idx] !== undefined ? data.y[analysis.testIdx[idx]] : 0

  // Local attribution in whatever space the model saw.
  const localRaw = localAttribution(
    (x) => model.predictOne(x),
    Fte[idx],
    baseline,
    modelFeatureNames,
  )

  // The hard part: push component attributions back onto clinical features.
  const clinical = pcaResult
    ? mapComponentsToFeatures(
        localRaw,
        pcaResult,
        data.featureNames,
        Xte[idx],
        (n) => describeFeature(cfg.datasetId, n),
      )
    : localRaw.map((a) => ({
        ...a,
        description: describeFeature(cfg.datasetId, a.name),
      }))

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[1fr_320px] gap-4">
        <Panel>
          <div className="mb-3 flex items-baseline justify-between">
            <div>
              <SectionLabel>global feature importance</SectionLabel>
              <p className="mt-1 font-mono text-[9px] text-ink-faint">
                permutation importance on the hybrid model — how much shuffling each input
                degrades performance
              </p>
            </div>
            <LiveChip label="measured" />
          </div>

          <ImportanceBars items={global.map((g) => ({ name: g.name, score: g.value, kept: true }))} />

          <p className="mt-3 font-mono text-[9px] leading-relaxed text-ink-faint/85">
            Model-agnostic: the same procedure runs identically on the quantum circuit and any
            classical baseline, so the numbers are comparable across lanes.
          </p>
        </Panel>

        <Panel>
          <SectionLabel>select a patient</SectionLabel>
          <div className="mt-2.5">
            <input
              type="range"
              min={0}
              max={Fte.length - 1}
              value={idx}
              onChange={(e) => setPatientIdx(Number(e.target.value))}
              className="feature-slider w-full cursor-pointer"
              aria-label="Patient index"
            />
            <div className="mt-1 flex justify-between font-mono text-[9px] text-ink-faint">
              <span>record {idx}</span>
              <span>{Fte.length} in holdout</span>
            </div>
          </div>

          <Well className="mt-3 p-3">
            <div className="flex items-baseline gap-2">
              <span
                className="font-mono text-[26px] font-medium leading-none tabular-nums"
                style={{ color: BAND_COLOR[band] }}
              >
                {(probability * 100).toFixed(1)}
              </span>
              <span className="font-mono text-[10px] text-ink-faint">% risk</span>
            </div>

            <div
              className="mt-2 inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[9px]"
              style={{
                color: BAND_COLOR[band],
                background: alpha(BAND_COLOR[band], 0.1),
                border: `1px solid ${alpha(BAND_COLOR[band], 0.28)}`,
              }}
            >
              <span className="h-[4px] w-[4px] rounded-full" style={{ background: BAND_COLOR[band] }} />
              {band} risk
            </div>

            <div
              className="mt-2.5 flex justify-between pt-2 font-mono text-[9px]"
              style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
            >
              <span className="text-ink-faint">actual label</span>
              <span style={{ color: trueLabel === 1 ? '#A3543D' : '#5FA88C' }}>
                {trueLabel === 1 ? data.positiveLabel : data.negativeLabel}
              </span>
            </div>
            <div className="flex justify-between font-mono text-[9px]">
              <span className="text-ink-faint">prediction</span>
              <span className="text-ink-dim">
                {probability >= 0.5 ? data.positiveLabel : data.negativeLabel}
              </span>
            </div>
          </Well>

          <QubitBadge qubits={cfg.nFeatures} compact />
        </Panel>
      </div>

      {/* the mapping - the part that shows medical seriousness */}
      <Panel>
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <SectionLabel>
              why this patient — contributions in clinical terms
            </SectionLabel>
            <p className="mt-1 font-mono text-[9px] leading-relaxed text-ink-faint">
              {pcaResult
                ? 'Attributions computed in component space, then pushed back through the PCA rotation onto the original measurements.'
                : 'Attribution by occlusion: each feature is replaced with the population mean and the prediction re-run.'}
            </p>
          </div>
          <LiveChip label="measured" />
        </div>

        {pcaResult && (
          <div className="mb-4 grid grid-cols-2 gap-4">
            <Well className="p-3">
              <div className="mb-2 font-mono text-[9px] text-ink-faint">
                1. raw component attribution — not clinically usable
              </div>
              {localRaw.slice(0, 4).map((a) => (
                <div key={a.name} className="flex justify-between py-[2px] font-mono text-[9.5px]">
                  <span className="text-ink-faint">{a.name}</span>
                  <span className="tabular-nums text-ink-dim">
                    {a.value >= 0 ? '+' : ''}
                    {a.value.toFixed(4)}
                  </span>
                </div>
              ))}
              <p className="mt-2 font-mono text-[8.5px] leading-relaxed text-ink-faint/70">
                &quot;PC1 contributed +0.4&quot; tells a clinician nothing — each component is a
                blend of every measurement.
              </p>
            </Well>

            <Well className="p-3">
              <div className="mb-2 font-mono text-[9px] text-ink-faint">
                2. what PC1 is actually made of
              </div>
              {componentComposition(pcaResult, 0, data.featureNames, 4).map((c) => (
                <div key={c.name} className="flex justify-between py-[2px] font-mono text-[9.5px]">
                  <span className="min-w-0 truncate text-ink-faint">{c.name}</span>
                  <span className="shrink-0 tabular-nums text-ink-dim">
                    {c.loading >= 0 ? '+' : ''}
                    {c.loading.toFixed(3)}
                  </span>
                </div>
              ))}
              <p className="mt-2 font-mono text-[8.5px] leading-relaxed text-ink-faint/70">
                Each component&apos;s attribution distributes over the originals in proportion to
                these loadings.
              </p>
            </Well>
          </div>
        )}

        <div
          className="rounded-[8px] p-3.5"
          style={{
            background: alpha(LANE_COLOR.quantum, 0.05),
            border: `1px solid ${alpha(LANE_COLOR.quantum, 0.2)}`,
          }}
        >
          <div className="mb-3 font-mono text-[9.5px]" style={{ color: LANE_COLOR.quantum }}>
            {pcaResult ? '3. mapped back to real measurements' : 'feature contributions'}
          </div>

          <DivergingBars
            positiveLabel={`toward ${data.positiveLabel}`}
            negativeLabel={`toward ${data.negativeLabel}`}
            items={clinical.slice(0, 8).map((a) => ({
              name: a.name,
              value: a.value,
              detail:
                (a.raw !== undefined ? `measured ${a.raw.toFixed(2)} · ` : '') +
                (a.description ?? '').slice(0, 68),
            }))}
          />
        </div>

        <p className="mt-3 font-mono text-[9px] leading-relaxed text-ink-faint/80">
          This is decision support, not a diagnosis. The platform is a research prototype and
          has not been clinically validated.
        </p>
      </Panel>
    </div>
  )
}
