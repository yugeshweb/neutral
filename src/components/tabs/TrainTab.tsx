import { useEffect, useMemo, useRef, useState } from 'react'
import {
  getDiseasePipeline,
  saveTrainedPipeline,
  type TrainedPipelineArtifact,
} from '../../lib/diseaseRegistry'
import { INTAKE_DISEASE_IDS } from '../../lib/intakeSpec'
import { CONDITION_VECTOR, VecLesion } from '../vectors'
import {
  CUSTOM_DATASET_ID,
  hasCustomDataset,
  loadDataset,
  registerCustomDataset,
} from '../../lib/ml/datasets'
import { ingest } from '../../lib/ingest'
import { NotImplementedError } from '../../lib/ingest/types'
import { convertUpload, suggestLabelColumn, previewColumns } from '../../lib/ml/customDataset'
import { pca, pcaTransform, rankFeatures, type SelectionMethod } from '../../lib/ml/features'
import { rocCurve } from '../../lib/ml/metrics'
import {
  DEFAULT_RUN,
  runPipeline,
  type RunConfig,
  type RunProgress,
  type RunResult,
} from '../../lib/ml/pipeline'
import {
  applyScaler,
  fitScaler,
  imputeMissing,
  type ImputeStrategy,
  type Scaler,
} from '../../lib/ml/stats'
import type { EpochRecord } from '../../lib/quantum/vqc'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { InfoDot } from '../InfoDot'
import { InputBuilder } from '../InputBuilder'
import { kindOf, type InputKind, type InputRow } from '../../lib/inputKinds'
import { ConvergenceChart } from '../charts'
import { ScatterPlot } from '../charts/ScatterPlot'
import { IconArrowLeft, IconArrowRight, IconCheck, IconPulse } from '../icons'

/*
 * The flow, in the order the user works through it: describe the inputs, pick
 * the condition they belong to, choose the feature treatment, then train.
 * One card at a time, each advanced by its own Next button.
 */
type Step = 'inputs' | 'disease' | 'features' | 'training' | 'results'

export function TrainTab({
  onNavigateToPredict,
}: {
  onNavigateToPredict?: (diseaseId: string) => void
}) {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('breast-cancer')
  const [activeStep, setActiveStep] = useState<Step>('inputs')

  /*
   * The declared inputs. Starts as one EHR row; the user adds more and picks a
   * type per row. Only `ehr` rows become a trainable matrix - the imaging
   * kinds are recorded as reference sources and say so.
   */
  const [inputRows, setInputRows] = useState<InputRow[]>([
    { id: 'in-1', kind: 'ehr', fileName: null, rows: null, note: null },
  ])
  const nextInputId = useRef(2)
  const [uploadFileName, setUploadFileName] = useState<string | null>(null)
  const [is3DScatter, setIs3DScatter] = useState(false)

  // Training execution states
  const [trainingLogs, setTrainingLogs] = useState<{ phase: string; message: string; timestamp: number }[]>([])
  const [convergenceData, setConvergenceData] = useState<EpochRecord[]>([])
  const [currentProgress, setCurrentProgress] = useState(0)
  const [trainingResult, setTrainingResult] = useState<RunResult | null>(null)

  const disease = getDiseasePipeline(selectedDiseaseId)

  // Ingestion configuration
  const [imputeStrategy, setImputeStrategy] = useState<ImputeStrategy>('median')
  const [scalerType, setScalerType] = useState<Scaler['kind']>('standard')
  const [reductionMethod, setReductionMethod] = useState<SelectionMethod>('mutual-info')
  const [nFeatures, setNFeatures] = useState<number>(disease.defaultQubits)
  const [epochs, setEpochs] = useState<number>(24)
  const cancelTrainingRef = useRef(false)

  // Update qubit count when disease changes
  useEffect(() => {
    setNFeatures(disease.defaultQubits)
  }, [disease.defaultQubits])

  // Load active dataset for the selected disease
  const dataset = useMemo(() => {
    try {
      if (uploadFileName && hasCustomDataset()) {
        return loadDataset(CUSTOM_DATASET_ID)
      }
      return loadDataset(selectedDiseaseId)
    } catch {
      return loadDataset('breast-cancer')
    }
  }, [selectedDiseaseId, uploadFileName])

  // Compute Feature Preview & Dimensionality Reduction
  const previewData = useMemo(() => {
    try {
      const imputed = imputeMissing(dataset.X, dataset.y, imputeStrategy)
      const scaler = fitScaler(imputed.X, scalerType)
      const scaledX = applyScaler(imputed.X, scaler)

      // Dimensionality reduction projection for visualization
      const pcaRes = pca(scaledX, 3, 42)
      const proj = pcaTransform(scaledX, pcaRes)

      const points = proj.map((row, i) => ({
        x: row[0],
        y: row[1],
        z: row[2] ?? 0,
        label: imputed.y[i],
        id: i,
      }))

      const ranked = rankFeatures(scaledX, imputed.y, dataset.featureNames, reductionMethod)

      /*
       * The split the run will use. Computed the way `stratifiedSplit`
       * computes it - per class, `max(1, round(n * fraction))`, summed -
       * rather than rounding the total, because those two disagree whenever
       * the per-class rounds go the same way. It runs on the imputed labels,
       * so dropping rows under the `drop` strategy is reflected too.
       */
      const perClass = new Map<number, number>()
      for (const label of imputed.y) perClass.set(label, (perClass.get(label) ?? 0) + 1)
      let testCount = 0
      for (const n of perClass.values()) {
        testCount += Math.max(1, Math.round(n * DEFAULT_RUN.testFraction))
      }

      return {
        imputedRows: imputed.X.length,
        filledCells: imputed.filled.reduce((a, b) => a + b, 0),
        droppedRows: imputed.rowsDropped,
        pcaResult: pcaRes,
        points,
        ranked,
        scaler,
        testCount,
        trainCount: imputed.y.length - testCount,
      }
    } catch (e) {
      return null
    }
  }, [dataset, imputeStrategy, scalerType, reductionMethod])

  // ---- the declared inputs ------------------------------------------------

  const addInput = () => {
    const id = `in-${nextInputId.current++}`
    setInputRows((prev) => [
      ...prev,
      { id, kind: 'ehr', fileName: null, rows: null, note: null },
    ])
  }

  const removeInput = (id: string) =>
    setInputRows((prev) => prev.filter((r) => r.id !== id))

  // Changing the type invalidates whatever was loaded under the old one.
  const setInputKind = (id: string, kind: InputKind) =>
    setInputRows((prev) =>
      prev.map((r) =>
        r.id === id ? { ...r, kind, fileName: null, rows: null, note: null } : r,
      ),
    )

  /**
   * The `ehr` kind goes through the same `ingest()` entry point used by the
   * Predict tab - it auto-detects CSV, FHIR, HL7 or PDF from the file itself
   * and runs the matching adapter, so a cohort standardized one way here is
   * standardized the identical way there. The imaging kinds have no feature
   * extraction behind them yet and are kept as reference sources.
   */
  const handleRowFile = async (id: string, file: File) => {
    const row = inputRows.find((r) => r.id === id)
    if (!row) return

    if (!kindOf(row.kind).trains) {
      setInputRows((prev) =>
        prev.map((r) =>
          r.id === id
            ? { ...r, fileName: file.name, rows: null, note: 'not trained on' }
            : r,
        ),
      )
      return
    }

    try {
      const parsed = await ingest(file)
      const summary = parsed.dataset
      const previews = previewColumns(summary)
      const labelCol =
        suggestLabelColumn(previews) || summary.headers[summary.headers.length - 1]
      const converted = convertUpload(summary, labelCol)
      registerCustomDataset(converted.dataset, file.name)
      setUploadFileName(file.name)
      setInputRows((prev) =>
        prev.map((r) =>
          r.id === id
            ? { ...r, fileName: file.name, rows: summary.rows, note: null }
            : r,
        ),
      )
    } catch (err) {
      const note =
        err instanceof NotImplementedError
          ? 'not implemented'
          : err instanceof Error
            ? err.message
            : 'failed'
      setInputRows((prev) =>
        prev.map((r) => (r.id === id ? { ...r, fileName: file.name, rows: null, note } : r)),
      )
    }
  }



  // Execute Live Training
  const handleStartTraining = () => {
    setActiveStep('training')
    setTrainingLogs([])
    setConvergenceData([])
    setCurrentProgress(0)
    setTrainingResult(null)
    cancelTrainingRef.current = false

    const runCfg: RunConfig = {
      ...DEFAULT_RUN,
      datasetId: uploadFileName ? CUSTOM_DATASET_ID : selectedDiseaseId,
      nFeatures,
      impute: imputeStrategy,
      scaler: scalerType,
      selection: reductionMethod,
      epochs,
      vqc: {
        ...DEFAULT_RUN.vqc,
        qubits: nFeatures,
      },
    }

    const gen = runPipeline(dataset, runCfg)

    const pump = () => {
      if (cancelTrainingRef.current) {
        return
      }

      const deadline = performance.now() + 30
      let finished = false

      while (performance.now() < deadline) {
        const next = gen.next()
        if (next.done) {
          finished = true
          break
        }

        const ev = next.value as RunProgress

        if (ev.phase === 'done') {
          setTrainingResult(ev.result)
          setCurrentProgress(1)
          setActiveStep('results')

          // Auto-persist fitted pipeline for Tab 3 Predict
          const qModel = ev.result.models.find((m) => m.kind === 'quantum')
          const cModel = ev.result.models.find((m) => m.kind === 'classical')

          if (qModel && cModel) {
            const fitted = ev.result.fitted
            const artifact: TrainedPipelineArtifact = {
              diseaseId: selectedDiseaseId,
              trainedAt: new Date().toISOString(),
              datasetName: dataset.name,
              rows: dataset.X.length,
              featureNames: dataset.featureNames,
              keptFeatures: ev.result.keptFeatures,
              // The scaler the run actually fitted on its training fold - not
              // the preview's, which is fitted on the full matrix for display.
              scaler: fitted.scaler,
              keptIndices: fitted.keptIndices,
              pca: fitted.pca,
              imputeValues: fitted.imputeValues,
              baselineVector: fitted.baselineVector,
              quantumWeights: fitted.quantumWeights,
              quantumConfig: fitted.quantumConfig,
              classicalMetrics: {
                accuracy: cModel.metrics.accuracy,
                precision: cModel.metrics.precision,
                sensitivity: cModel.metrics.sensitivity,
                specificity: cModel.metrics.specificity,
                f1: cModel.metrics.f1,
                rocAuc: cModel.metrics.rocAuc,
                trainingTime: `${(cModel.trainMs / 1000).toFixed(2)}s`,
                inferenceTime: `${cModel.inferenceMs.toFixed(1)}ms`,
              },
              quantumMetrics: {
                accuracy: qModel.metrics.accuracy,
                precision: qModel.metrics.precision,
                sensitivity: qModel.metrics.sensitivity,
                specificity: qModel.metrics.specificity,
                f1: qModel.metrics.f1,
                rocAuc: qModel.metrics.rocAuc,
                trainingTime: `${(qModel.trainMs / 1000).toFixed(2)}s`,
                inferenceTime: `${qModel.inferenceMs.toFixed(1)}ms`,
              },
              classicalModelName: cModel.label,
              quantumModelName: qModel.label,
              confusionMatrixClassical: {
                tp: cModel.metrics.confusion.tp,
                fn: cModel.metrics.confusion.fn,
                tn: cModel.metrics.confusion.tn,
                fp: cModel.metrics.confusion.fp,
              },
              confusionMatrixQuantum: {
                tp: qModel.metrics.confusion.tp,
                fn: qModel.metrics.confusion.fn,
                tn: qModel.metrics.confusion.tn,
                fp: qModel.metrics.confusion.fp,
              },
              rocPointsClassical: rocCurve(ev.result.yTest, cModel.scores),
              rocPointsQuantum: rocCurve(ev.result.yTest, qModel.scores),
            }
            saveTrainedPipeline(artifact)
          }

          finished = true
          break
        }

        if (ev.phase === 'quantum') {
          setConvergenceData((c) => [...c, ev.epoch])
          setCurrentProgress(ev.epoch.epoch / ev.total)
          break
        }

        const msgLine = ev as { phase: string; message: string }
        setTrainingLogs((l) => [
          ...l,
          { phase: msgLine.phase, message: msgLine.message, timestamp: Date.now() },
        ])
      }

      if (!finished) {
        window.requestAnimationFrame(pump)
      }
    }

    pump()
  }

  const benchmarkQuantum = disease.quantumModel

  const trainedClassical = trainingResult?.models.find((m) => m.kind === 'classical')
  const trainedQuantum = trainingResult?.models.find((m) => m.kind === 'quantum')



  return (
    <div className="console-scroll canvas-grid h-full overflow-y-auto overflow-x-hidden">
      <div className="screen">
        {/* Step Navigation Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div className="flex items-center gap-2">
            <h1 className="text-[19px] font-medium text-ink">Train</h1>
            <InfoDot label="About this screen">
              Ingest a cohort or your own CSV, then train a classical baseline and a
              variational quantum classifier on the same split. The fitted model is
              saved and becomes the one the Predict tab scores with.
            </InfoDot>
          </div>

          {/* Progress marker: read-only, so the flow is advanced by the Next
              buttons rather than by jumping between arbitrary steps. */}
          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            {(['inputs', 'disease', 'features', 'training'] as const).map((k, i) => {
              const order: Step[] = ['inputs', 'disease', 'features', 'training', 'results']
              const at = order.indexOf(activeStep)
              const done = i < at
              const on = i === at || (k === 'training' && activeStep === 'results')
              return (
                <span key={k} className="flex items-center gap-1.5">
                  <span
                    className="grid h-5 w-5 place-items-center rounded-full"
                    style={{
                      background: on || done ? alpha(LANE_COLOR.quantum, 0.16) : 'transparent',
                      border: `1px solid ${on || done ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.12)'}`,
                      color: on || done ? LANE_COLOR.quantum : '#6A6C72',
                    }}
                  >
                    {i + 1}
                  </span>
                  {i < 3 && <span className="h-px w-4 bg-white/10" />}
                </span>
              )
            })}
          </div>
        </div>

        {/* STEP 1: the inputs, built up row by row. */}
        {activeStep === 'inputs' && (
          <div className="panel-raised rounded-panel panel-pad flow-step">
            <div className="flex items-center justify-center gap-2">
              <h2 className="text-[14.5px] font-medium text-ink">Inputs</h2>
              <InfoDot label="About these inputs">
                Add a row per source and pick its type. Only a CSV table is parsed
                into the numeric matrix the pipeline trains on; the other types are
                recorded as reference sources and marked as not trained on.
              </InfoDot>
            </div>

            <div className="flow-body console-scroll mt-4 overflow-y-auto">
              <InputBuilder
                rows={inputRows}
                onAdd={addInput}
                onRemove={removeInput}
                onKind={setInputKind}
                onFile={handleRowFile}
              />
            </div>

            <div className="mt-4 flex shrink-0 items-center justify-between border-t border-white/5 pt-4">
              <span className="font-mono text-[11px] text-ink-faint">
                {uploadFileName && `training on ${dataset.name}`}
              </span>
              <button
                type="button"
                onClick={() => setActiveStep('disease')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink hover:text-white"
              >
                Next
                <IconArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* STEP 2: the condition. */}
        {activeStep === 'disease' && (
          <div className="panel-raised rounded-panel panel-pad flow-step">
            <div className="flex items-center justify-center gap-2">
              <h2 className="text-[14.5px] font-medium text-ink">Condition</h2>
              <InfoDot label="About routing">
                The selected condition determines the preprocessing transformers, the
                feature-extraction recipe, and the quantum circuit ansatz.
              </InfoDot>
            </div>

            {/* The same four conditions the Predict tab offers, so a model
                trained here always has an intake to be used through.

                Four across from 640px up, not 1024: at two columns these are
                two rows of squares, which at 220px each needs 452px of body
                and overflows a 768px-tall window. Four across is one row of
                220px, which fits everywhere the flow-step does. Below 640 it
                falls to two columns, where `.flow-step` drops its min-height
                and the page is allowed to scroll.

                `flex + items-center` on the wrapper centres the row inside
                whatever height `.flow-step` reserves: the cards are capped at
                220px, so on a tall step they would otherwise sit pinned to
                the top with empty space left below them. */}
            <div className="flow-body mt-4 flex items-center">
              <div className="grid w-full grid-cols-2 gap-3 sm:grid-cols-4">
              {INTAKE_DISEASE_IDS.map((id) => {
                const d = getDiseasePipeline(id)
                const active = d.id === selectedDiseaseId
                const Vec = CONDITION_VECTOR[id] ?? VecLesion
                return (
                  <button
                    key={d.id}
                    type="button"
                    data-pressed={active}
                    onClick={() => setSelectedDiseaseId(d.id)}
                    className="tile mx-auto flex aspect-square w-full max-w-[220px] cursor-pointer flex-col p-3 text-left"
                    style={{ ['--tile-accent' as string]: LANE_COLOR.quantum }}
                  >
                    {/* The plate: same treatment as the opening screen's
                        tiles, with the condition's drawing as the focal
                        element rather than a corner glyph. Not `aspect-square`
                        here - the card itself is already square, so a second
                        square plate inside it would leave no room for the
                        text below. `flex-1` gives it whatever the card has
                        left after the text block. */}
                    <div className="tile-art relative min-h-0 w-full flex-1">
                      <Vec size={56} accent={LANE_COLOR.quantum} />
                      <span className="engraved absolute left-2 top-1.5 font-mono text-[10.5px]">
                        {d.categoryLabel.split(' / ')[0]}
                      </span>
                    </div>

                    <div className="mt-2.5 min-w-0 shrink-0">
                      {/*
                       * `shortName`, not `name`: the full descriptive title
                       * ("Heart Disease Clinical Risk (Cleveland Cohort)")
                       * wraps to a different number of lines per condition, which
                       * on a card whose plate is `flex-1` inside a fixed
                       * `aspect-square` handed shorter names extra plate
                       * height - the images came out visibly uneven. Every
                       * short name is two words and fits one line, so every
                       * text block, and so every plate, is now the same size.
                       * The full name is still one hover away on the title.
                       */}
                      <div
                        className="truncate text-[13.5px] font-medium leading-snug"
                        title={d.name}
                        style={{ color: active ? '#E8E9EB' : '#9A9CA1' }}
                      >
                        {d.shortName}
                      </div>
                      <div className="tile-muted mt-1 font-mono text-[10.5px]">
                        {d.totalSamples.toLocaleString()} samples · {d.defaultQubits} qubits
                      </div>
                    </div>
                  </button>
                )
              })}
              </div>
            </div>

            <div className="mt-4 flex shrink-0 items-center justify-between border-t border-white/5 pt-4">
              <button
                type="button"
                onClick={() => setActiveStep('inputs')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink-dim hover:text-ink"
              >
                <IconArrowLeft className="h-3.5 w-3.5" />
                Back
              </button>
              <button
                type="button"
                onClick={() => setActiveStep('features')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink hover:text-white"
              >
                Next
                <IconArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* STEP 2: PREPROCESSING & FEATURE PREVIEW */}
        {activeStep === 'features' && previewData && (
          <div className="space-y-4">
            {/*
             * Preprocessing.
             *
             * Was four "stage" cards, each carrying three labels for a single
             * control - "Stage 1 · Imputation", "Missing Cell Handler" and
             * "Strategy:" all naming the same dropdown - plus a fourth card
             * with no control at all, only numbers derived from the qubit
             * slider further down. What is left is the three choices that
             * actually change the run, each labelled once, with the one
             * derived figure that matters shown beside it.
             */}
            <div className="panel-raised rounded-panel panel-pad">
              <div className="mb-3 flex items-center justify-center gap-2">
                <h2 className="text-[14.5px] font-medium text-ink">Preprocessing</h2>
                <InfoDot label="About these stages">
                  Applied in order: missing cells are filled, features are scaled,
                  then reduced to the qubit count. Every transform is fitted on the
                  training fold only, so nothing from the test split leaks into the
                  model.
                </InfoDot>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="panel-well rounded-[8px] well-pad">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12.5px] text-ink">Impute</span>
                    <select
                      value={imputeStrategy}
                      onChange={(e) => setImputeStrategy(e.target.value as ImputeStrategy)}
                      className="select font-mono text-[11px]"
                    >
                      <option value="median">Median</option>
                      <option value="mean">Mean</option>
                      <option value="drop">Drop</option>
                    </select>
                  </div>
                  <div className="engraved mt-2 font-mono text-[11px]">
                    {previewData.droppedRows > 0
                      ? `${previewData.droppedRows} rows dropped`
                      : `${previewData.filledCells} cells filled`}
                  </div>
                </div>

                <div className="panel-well rounded-[8px] well-pad">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12.5px] text-ink">Scale</span>
                    <select
                      value={scalerType}
                      onChange={(e) => setScalerType(e.target.value as Scaler['kind'])}
                      className="select font-mono text-[11px]"
                    >
                      <option value="standard">Standard (z-score)</option>
                      <option value="robust">Robust (IQR)</option>
                      <option value="minmax">Min-Max (0..1)</option>
                    </select>
                  </div>
                  <div className="engraved mt-2 font-mono text-[11px]">
                    fitted on train fold only
                  </div>
                </div>

                <div className="panel-well rounded-[8px] well-pad">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12.5px] text-ink">Reduce</span>
                    <select
                      value={reductionMethod}
                      onChange={(e) => setReductionMethod(e.target.value as SelectionMethod)}
                      className="select font-mono text-[11px]"
                    >
                      <option value="mutual-info">Mutual Info</option>
                      <option value="f-score">ANOVA F-test</option>
                      <option value="pca">PCA Projection</option>
                    </select>
                  </div>
                  <div className="engraved mt-2 font-mono text-[11px]">
                    {dataset.featureNames.length} dims → {nFeatures}
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Space Visualization: 2D/3D Scatter Plot */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* Scatter Plot Projection (8 cols) */}
              <div
                className="lg:col-span-8 panel-raised rounded-panel panel-pad"
              >
                {/* One header row rather than two lines plus a subtitle: the
                    explanatory sentence moved into the info dot, which is what
                    keeps this panel short. */}
                <div className="mb-2.5 flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <h3 className="truncate text-[14.5px] font-medium text-ink">
                      Feature space separability
                    </h3>
                    <InfoDot label="About this projection">
                      Class clustering in the projected component space, shown before
                      committing to a training run. Separation here suggests the
                      reduced features carry the signal; overlap suggests they do not.
                    </InfoDot>
                  </div>

                  <div className="flex shrink-0 items-center gap-1 font-mono text-[11.5px]">
                    <button
                      type="button"
                      onClick={() => setIs3DScatter(false)}
                      data-pressed={!is3DScatter}
                      className="key cursor-pointer rounded-[6px] px-2 py-1 text-ink-faint hover:text-ink data-[pressed=true]:text-ink"
                    >
                      2D
                    </button>
                    <button
                      type="button"
                      onClick={() => setIs3DScatter(true)}
                      data-pressed={is3DScatter}
                      className="key cursor-pointer rounded-[6px] px-2 py-1 text-ink-faint hover:text-ink data-[pressed=true]:text-ink"
                    >
                      3D
                    </button>
                  </div>
                </div>

                <ScatterPlot
                  points={previewData.points}
                  positiveLabel={dataset.positiveLabel}
                  negativeLabel={dataset.negativeLabel}
                  height={210}
                  is3d={is3DScatter}
                />
              </div>

              {/* Training Parameters & Trigger (4 cols) */}
              <div
                className="lg:col-span-4 panel-raised rounded-panel panel-pad flex flex-col justify-between"
              >
                <div>
                  <h3 className="mb-3 text-[14.5px] font-medium text-ink">Run settings</h3>

                  <div className="space-y-3 font-mono text-[12.5px]">
                    <div>
                      <label className="mb-1 block text-ink-faint">
                        Qubits / features: <span className="text-ink">{nFeatures}</span>
                      </label>
                      <input
                        type="range"
                        min={4}
                        max={8}
                        step={1}
                        value={nFeatures}
                        aria-label="Qubits / features"
                        onChange={(e) => setNFeatures(Number(e.target.value))}
                        className="w-full cursor-pointer accent-[#3E8C9E]"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-ink-faint">
                        Epochs: <span className="text-ink">{epochs}</span>
                      </label>
                      <input
                        type="range"
                        min={12}
                        max={48}
                        step={6}
                        value={epochs}
                        aria-label="VQC epochs"
                        onChange={(e) => setEpochs(Number(e.target.value))}
                        className="w-full cursor-pointer accent-[#3E8C9E]"
                      />
                    </div>

                    {/*
                     * The features the selector actually picked. This is the
                     * one thing a feature step exists to answer and it was
                     * not shown anywhere: the panel instead carried three
                     * static lines of architecture trivia that never changed,
                     * one of which ("Gradient Boosted Trees") did not even
                     * match the baselines the pipeline runs.
                     */}
                    <div className="border-t border-white/5 pt-2.5">
                      <div className="engraved mb-1.5 text-[11px]">
                        selected features
                      </div>
                      <ol className="space-y-1 text-[11.5px] text-ink-dim">
                        {previewData.ranked.slice(0, nFeatures).map((f, i) => (
                          <li key={f.index} className="flex items-baseline gap-2">
                            <span className="w-3 shrink-0 text-right text-ink-faint">
                              {i + 1}
                            </span>
                            <span className="min-w-0 flex-1 truncate" title={f.name}>
                              {f.name}
                            </span>
                            <span
                              className="shrink-0 tabular-nums"
                              style={{ color: LANE_COLOR.quantum }}
                            >
                              {f.score.toFixed(2)}
                            </span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </div>
                </div>

              </div>
            </div>

            <div className="mt-4 flex shrink-0 items-center justify-between gap-3 border-t border-white/5 pt-4">
              <button
                type="button"
                onClick={() => setActiveStep('disease')}
                className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink-dim hover:text-ink"
              >
                <IconArrowLeft className="h-3.5 w-3.5" />
                Back
              </button>

              {/* What the run will actually do, stated before it starts. The
                  split was not shown anywhere, so there was no way to know
                  how many cases the reported metrics would be measured on. */}
              <div className="flex items-center gap-3">
                <span className="hidden font-mono text-[11px] text-ink-faint sm:inline">
                  {previewData.trainCount} train / {previewData.testCount} test · both
                  lanes, same split
                </span>
                <button
                  type="button"
                  onClick={handleStartTraining}
                  className="key flex cursor-pointer items-center gap-2 rounded-[6px] px-4 py-2 text-[13px] text-ink hover:text-white"
                >
                  <IconPulse className="h-3.5 w-3.5" />
                  Train
                </button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 3: LIVE TRAINING PROGRESS */}
        {activeStep === 'training' && (
          <div
            className="panel-raised rounded-panel panel-pad space-y-4"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="h-3 w-3 rounded-full animate-ping" style={{ background: LANE_COLOR.quantum }} />
                <h2 className="text-[16px] font-medium text-ink">
                  Training Classical & Quantum Models in Parallel...
                </h2>
              </div>
              <span className="font-mono text-[13px] text-ink-faint">
                {Math.round(currentProgress * 100)}% Complete
              </span>
            </div>

            {/* Progress Bar */}
            <div className="h-2 w-full rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-150"
                style={{
                  width: `${currentProgress * 100}%`,
                  background: `linear-gradient(90deg, ${LANE_COLOR.classical}, ${LANE_COLOR.quantum})`,
                }}
              />
            </div>

            {/* Convergence Chart & Live Logs */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-7 panel-well rounded-[8px] well-pad">
                <div className="mb-2 flex items-center justify-between font-mono text-[12px]">
                  <span className="text-ink-dim">VQC Quantum Loss & Accuracy Convergence</span>
                  <span className="text-ink-faint">Live Adam Optimization</span>
                </div>
                <ConvergenceChart points={convergenceData} height={180} />
              </div>

              <div className="lg:col-span-5 panel-well rounded-[8px] well-pad font-mono text-[11.5px] flex flex-col justify-between">
                <div>
                  <div className="text-ink-faint text-[11px] mb-2">Live Execution Log</div>
                  <div className="space-y-1.5 max-h-[160px] overflow-y-auto pr-1 console-scroll">
                    {trainingLogs.map((log, i) => (
                      <div key={i} className="text-ink-dim flex items-start gap-2">
                        <span className="text-ink-faint shrink-0">[{log.phase}]</span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                    {convergenceData.length > 0 && (
                      <div className="text-[#3E8C9E]">
                        Quantum Epoch {convergenceData[convergenceData.length - 1].epoch} / {epochs} · Loss:{' '}
                        {convergenceData[convergenceData.length - 1].loss.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
                <div className="pt-2 text-[11px] text-ink-faint border-t border-white/5">
                  Statevector simulator
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: POST-TRAINING RESULTS & BENCHMARK COMPARISON */}
        {activeStep === 'results' && trainingResult && trainedClassical && trainedQuantum && (
          <div className="space-y-4">
            {/* Completion Header */}
            <div
              className="panel-raised rounded-panel panel-pad flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <span className="h-6 w-6 grid place-items-center rounded-full bg-[#3E8C9E]/20 text-[#3E8C9E]">
                  <IconCheck className="h-3.5 w-3.5" />
                </span>
                <h2 className="text-[15.5px] font-medium text-ink">Training complete</h2>
                <InfoDot label="What was saved">
                  The fitted scaler, feature selection and trained circuit parameters
                  were saved. The Predict tab reloads them and scores with this exact
                  model.
                </InfoDot>
              </div>

              {onNavigateToPredict && (
                <button
                  type="button"
                  onClick={() => onNavigateToPredict(selectedDiseaseId)}
                  className="flex items-center gap-1.5 rounded-[6px] px-3 py-1.5 font-mono text-[12.5px] font-medium text-black cursor-pointer transition-transform hover:scale-105"
                  style={{ background: LANE_COLOR.quantum }}
                >
                  Test in Predict Tab <IconArrowRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Unified Evaluation Metrics of the Newly Trained Models */}
            <div
              className="panel-raised rounded-panel panel-pad"
            >
              <h3 className="font-mono text-[13px] font-medium text-ink-faint mb-3">
                Newly Trained Run Metrics (Holdout Evaluation)
              </h3>

              <div className="console-scroll overflow-x-auto">
                <table className="w-full text-left font-mono text-[13px]">
                  <thead>
                    <tr className="border-b border-white/10 text-[11.5px] text-ink-faint">
                      <th className="py-2 font-normal">Metric</th>
                      <th className="py-2 text-right font-normal">Trained Classical</th>
                      <th className="py-2 text-right font-normal">Trained Quantum (VQC)</th>
                      <th className="py-2 text-right font-normal">Delta (Q - C)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {[
                      { label: 'Accuracy', c: trainedClassical.metrics.accuracy, q: trainedQuantum.metrics.accuracy },
                      { label: 'Precision', c: trainedClassical.metrics.precision, q: trainedQuantum.metrics.precision },
                      { label: 'Sensitivity / Recall', c: trainedClassical.metrics.sensitivity, q: trainedQuantum.metrics.sensitivity },
                      { label: 'Specificity', c: trainedClassical.metrics.specificity, q: trainedQuantum.metrics.specificity },
                      { label: 'F1 Score', c: trainedClassical.metrics.f1, q: trainedQuantum.metrics.f1 },
                      { label: 'AUC-ROC', c: trainedClassical.metrics.rocAuc, q: trainedQuantum.metrics.rocAuc },
                    ].map((row) => {
                      const delta = row.q - row.c
                      return (
                        <tr key={row.label}>
                          <td className="py-2 text-ink font-medium">{row.label}</td>
                          <td className="py-2 text-right text-ink-dim tabular-nums">{row.c.toFixed(3)}</td>
                          <td className="py-2 text-right text-ink font-medium tabular-nums" style={{ color: delta >= 0 ? LANE_COLOR.quantum : '#E8E9EB' }}>
                            {row.q.toFixed(3)}
                          </td>
                          <td className="py-2 text-right tabular-nums">
                            <span
                              className="rounded px-1.5 py-0.5 text-[12px]"
                              style={{
                                background: delta > 0 ? alpha('#3E8C9E', 0.15) : delta < 0 ? alpha('#A3543D', 0.15) : 'transparent',
                                color: delta > 0 ? '#3E8C9E' : delta < 0 ? '#A3543D' : '#6A6C72',
                              }}
                            >
                              {delta >= 0 ? '+' : ''}
                              {delta.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrices & Comparison vs Platform Stored Benchmark */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* Confusion Matrices (6 cols) */}
              <div
                className="lg:col-span-6 panel-raised rounded-panel panel-pad"
              >
                <h3 className="font-mono text-[13px] font-medium text-ink-faint mb-3">
                  Confusion Matrices (Trained Models)
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  {/* Classical Matrix */}
                  <div className="panel-well rounded-[8px] well-pad">
                    <div className="font-mono text-[11.5px] text-ink-faint mb-2" style={{ color: LANE_COLOR.classical }}>
                      Classical ({trainedClassical.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1 font-mono text-[13px] text-center">
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[15.5px]">{trainedClassical.metrics.confusion.tp}</div>
                        <div className="text-[10.5px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[15.5px]">{trainedClassical.metrics.confusion.fp}</div>
                        <div className="text-[10.5px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[15.5px]">{trainedClassical.metrics.confusion.fn}</div>
                        <div className="text-[10.5px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[15.5px]">{trainedClassical.metrics.confusion.tn}</div>
                        <div className="text-[10.5px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>

                  {/* Quantum Matrix */}
                  <div className="panel-well rounded-[8px] well-pad">
                    <div className="font-mono text-[11.5px] text-ink-faint mb-2" style={{ color: LANE_COLOR.quantum }}>
                      Quantum ({trainedQuantum.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1 font-mono text-[13px] text-center">
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[15.5px]">{trainedQuantum.metrics.confusion.tp}</div>
                        <div className="text-[10.5px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[15.5px]">{trainedQuantum.metrics.confusion.fp}</div>
                        <div className="text-[10.5px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[15.5px]">{trainedQuantum.metrics.confusion.fn}</div>
                        <div className="text-[10.5px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[15.5px]">{trainedQuantum.metrics.confusion.tn}</div>
                        <div className="text-[10.5px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direct Comparison: Newly Trained vs Platform Stored Benchmark (6 cols) */}
              <div
                className="lg:col-span-6 panel-raised rounded-panel panel-pad"
              >
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-mono text-[13px] font-medium text-ink-faint">
                    This run vs stored benchmark
                  </h3>
                  <InfoDot label="About this comparison">
                    The stored benchmark is a previously recorded result for{' '}
                    {disease.name}, shown for reference only. It is not a paired test,
                    a difference here is not evidence of a real gain.
                  </InfoDot>
                </div>

                <div className="space-y-3 font-mono text-[12.5px]">
                  <div className="panel-well rounded-[8px] well-pad space-y-2">
                    <div className="flex justify-between items-center text-ink-dim">
                      <span>Quantum Accuracy:</span>
                      <span>
                        <span className="text-ink font-medium">{trainedQuantum.metrics.accuracy.toFixed(3)}</span>{' '}
                        (Benchmark: {benchmarkQuantum.metrics.accuracy.toFixed(3)})
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-ink-dim">
                      <span>Quantum Sensitivity:</span>
                      <span>
                        <span className="text-ink font-medium">{trainedQuantum.metrics.sensitivity.toFixed(3)}</span>{' '}
                        (Benchmark: {benchmarkQuantum.metrics.sensitivity.toFixed(3)})
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-ink-dim">
                      <span>Quantum AUC-ROC:</span>
                      <span>
                        <span className="text-ink font-medium">{trainedQuantum.metrics.rocAuc.toFixed(3)}</span>{' '}
                        (Benchmark: {benchmarkQuantum.metrics.rocAuc.toFixed(3)})
                      </span>
                    </div>
                  </div>

                </div>
              </div>
            </div>

            {/* ROC Curves for Newly Trained Models */}
          </div>
        )}
      </div>
    </div>
  )
}
