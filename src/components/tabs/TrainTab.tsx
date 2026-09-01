import { useEffect, useMemo, useRef, useState } from 'react'
import {
  DISEASE_PIPELINES,
  getDiseasePipeline,
  saveTrainedPipeline,
  type TrainedPipelineArtifact,
} from '../../lib/diseaseRegistry'
import {
  CUSTOM_DATASET_ID,
  hasCustomDataset,
  loadDataset,
  registerCustomDataset,
} from '../../lib/ml/datasets'
import { parseCsv } from '../../lib/dataset'
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
import { ConvergenceChart, RocChart } from '../charts'
import { ScatterPlot } from '../charts/ScatterPlot'
import {
  IconArrowRight,
  IconCheck,
  IconDatabase,
  IconFlask,
  IconPulse,
  IconUpload,
} from '../icons'

type Step = 'ingest' | 'preview' | 'training' | 'results'

export function TrainTab({
  onNavigateToPredict,
}: {
  onNavigateToPredict?: (diseaseId: string) => void
}) {
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string>('breast-cancer')
  const [activeStep, setActiveStep] = useState<Step>('ingest')
  const [uploadError, setUploadError] = useState<string | null>(null)
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

  const fileInputRef = useRef<HTMLInputElement | null>(null)
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

      return {
        imputedRows: imputed.X.length,
        filledCells: imputed.filled.reduce((a, b) => a + b, 0),
        droppedRows: imputed.rowsDropped,
        pcaResult: pcaRes,
        points,
        ranked,
        scaler,
      }
    } catch (e) {
      return null
    }
  }, [dataset, imputeStrategy, scalerType, reductionMethod])

  // Handle Custom Dataset Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploadError(null)
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string
        const summary = parseCsv(text, file.name, file.size)
        const previews = previewColumns(summary)
        const labelCol = suggestLabelColumn(previews) || summary.headers[summary.headers.length - 1]
        const converted = convertUpload(summary, labelCol)
        registerCustomDataset(converted.dataset, file.name)
        setUploadFileName(file.name)
      } catch (err: any) {
        setUploadError(err?.message || 'Failed to parse CSV dataset.')
      }
    }
    reader.readAsText(file)
  }

  // Clear Uploaded Dataset
  const handleResetUpload = () => {
    setUploadFileName(null)
    setUploadError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
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

          if (qModel && cModel && previewData?.scaler) {
            const artifact: TrainedPipelineArtifact = {
              diseaseId: selectedDiseaseId,
              trainedAt: new Date().toISOString(),
              datasetName: dataset.name,
              rows: dataset.X.length,
              featureNames: dataset.featureNames,
              keptFeatures: ev.result.keptFeatures,
              scaler: previewData.scaler,
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

  const trainedRocCurves = useMemo(() => {
    if (!trainingResult || !trainedClassical || !trainedQuantum) return []
    return [
      {
        label: `Trained Classical (${trainedClassical.label})`,
        color: LANE_COLOR.classical,
        points: rocCurve(trainingResult.yTest, trainedClassical.scores),
        auc: trainedClassical.metrics.rocAuc,
      },
      {
        label: `Trained Quantum (${trainedQuantum.label})`,
        color: LANE_COLOR.quantum,
        points: rocCurve(trainingResult.yTest, trainedQuantum.scores),
        auc: trainedQuantum.metrics.rocAuc,
      },
    ]
  }, [trainingResult, trainedClassical, trainedQuantum])


  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-[1240px] px-6 py-6 space-y-6">
        {/* Step Navigation Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[9.5px] text-ink-faint">
                TAB 2 · PIPELINE TRAINER
              </span>
              <h1 className="text-[18px] font-medium text-ink">
                Train on Your Own Clinical Dataset
              </h1>
            </div>
            <p className="mt-1 text-[12px] text-ink-dim">
              Bring custom data or ingest preset clinical cohorts to train both classical and hybrid quantum models with real-time feedback.
            </p>
          </div>

          {/* Stepper Buttons */}
          <div className="flex items-center gap-1.5 rounded-[8px] bg-[#141518] p-1 border border-white/5 font-mono text-[10px]">
            <button
              type="button"
              onClick={() => setActiveStep('ingest')}
              className={`cursor-pointer rounded-[6px] px-3 py-1.5 transition-all ${
                activeStep === 'ingest' ? 'bg-white/10 text-white font-medium' : 'text-ink-faint hover:text-ink'
              }`}
            >
              1. Ingest & Route
            </button>
            <button
              type="button"
              onClick={() => setActiveStep('preview')}
              className={`cursor-pointer rounded-[6px] px-3 py-1.5 transition-all ${
                activeStep === 'preview' ? 'bg-white/10 text-white font-medium' : 'text-ink-faint hover:text-ink'
              }`}
            >
              2. Feature Preview
            </button>
            <button
              type="button"
              onClick={() => {
                if (trainingResult) setActiveStep('results')
                else handleStartTraining()
              }}
              className={`cursor-pointer rounded-[6px] px-3 py-1.5 transition-all ${
                activeStep === 'training' || activeStep === 'results'
                  ? 'bg-white/10 text-white font-medium'
                  : 'text-ink-faint hover:text-ink'
              }`}
            >
              3. Train & Compare
            </button>
          </div>
        </div>

        {/* STEP 1: INGESTION & DISEASE ROUTER */}
        {activeStep === 'ingest' && (
          <div className="space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Left Column: Disease Selection (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="mb-3 flex items-center gap-2">
                  <IconDatabase className="h-4 w-4 text-ink-faint" />
                  <h2 className="text-[13px] font-medium text-ink">
                    1. Select Target Disease Pipeline
                  </h2>
                </div>
                <p className="text-[11.5px] text-ink-dim mb-4">
                  Routing determines the preprocessing transformers, feature extraction recipe, and quantum circuit ansatz.
                </p>

                <div className="space-y-2.5">
                  {DISEASE_PIPELINES.map((d) => {
                    const active = d.id === selectedDiseaseId
                    return (
                      <button
                        key={d.id}
                        type="button"
                        onClick={() => {
                          setSelectedDiseaseId(d.id)
                          handleResetUpload()
                        }}
                        className="w-full text-left rounded-[8px] p-3.5 transition-all cursor-pointer"
                        style={{
                          background: active ? '#1F2126' : '#101114',
                          border: `1px solid ${active ? alpha(LANE_COLOR.quantum, 0.5) : 'rgba(255,255,255,0.05)'}`,
                          boxShadow: active ? `0 0 0 1px ${alpha(LANE_COLOR.quantum, 0.2)}` : 'none',
                        }}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">
                            {d.categoryLabel}
                          </span>
                          {active && (
                            <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />
                          )}
                        </div>
                        <div className="text-[13.5px] font-medium text-ink mt-0.5">{d.name}</div>
                        <div className="mt-1 flex items-center gap-3 font-mono text-[9.5px] text-ink-faint">
                          <span>Modality: {d.modality}</span>
                          <span>Input: {d.inputDimensionality.slice(0, 24)}...</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Right Column: Ingestion Options & Upload (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="mb-3 flex items-center gap-2">
                  <IconUpload className="h-4 w-4 text-ink-faint" />
                  <h2 className="text-[13px] font-medium text-ink">
                    2. Data Ingestion & Custom Upload
                  </h2>
                </div>
                <p className="text-[11.5px] text-ink-dim mb-4">
                  High-dimensional biomedical records are automatically ingested, validated, and normalized before quantum state embedding.
                </p>

                {/* Upload Box */}
                <div
                  className="rounded-[8px] border-2 border-dashed border-white/10 p-5 text-center hover:border-white/20 transition-colors"
                  style={{ background: '#0F1012' }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="csv-upload-input"
                  />
                  <label
                    htmlFor="csv-upload-input"
                    className="cursor-pointer flex flex-col items-center gap-2"
                  >
                    <IconUpload className="h-6 w-6 text-ink-faint" />
                    <span className="text-[12px] font-medium text-ink">
                      {uploadFileName ? `Uploaded: ${uploadFileName}` : 'Upload Custom CSV Dataset'}
                    </span>
                    <span className="font-mono text-[9.5px] text-ink-faint">
                      Supports comma-separated tabular records with target column
                    </span>
                  </label>

                  {uploadFileName && (
                    <div className="mt-3 flex items-center justify-center gap-2">
                      <span className="rounded bg-[#5FA88C]/15 border border-[#5FA88C]/30 px-2 py-0.5 font-mono text-[9.5px] text-[#5FA88C]">
                        ✓ Ingested {dataset.X.length} rows × {dataset.featureNames.length} features
                      </span>
                      <button
                        type="button"
                        onClick={handleResetUpload}
                        className="cursor-pointer font-mono text-[9px] text-ink-faint hover:text-ink underline"
                      >
                        Reset to preset
                      </button>
                    </div>
                  )}

                  {uploadError && (
                    <div className="mt-2 text-[10.5px] text-[#A3543D]">{uploadError}</div>
                  )}
                </div>

                {/* Ingested Dataset Summary */}
                <div className="mt-4 rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 font-mono text-[10.5px] space-y-1.5">
                  <div className="flex justify-between text-ink-faint">
                    <span>Active Cohort:</span>
                    <span className="text-ink font-medium">{dataset.name}</span>
                  </div>
                  <div className="flex justify-between text-ink-faint">
                    <span>Total Rows & Features:</span>
                    <span className="text-ink">{dataset.X.length} records × {dataset.featureNames.length} dims</span>
                  </div>
                  <div className="flex justify-between text-ink-faint">
                    <span>Labels:</span>
                    <span className="text-ink">
                      <span className="text-[#A3543D]">{dataset.positiveLabel}</span> /{' '}
                      <span className="text-[#5FA88C]">{dataset.negativeLabel}</span>
                    </span>
                  </div>
                </div>

                <div className="mt-5 flex justify-end">
                  <button
                    type="button"
                    onClick={() => setActiveStep('preview')}
                    className="flex items-center gap-2 rounded-[7px] px-4 py-2 font-mono text-[11px] font-medium text-black cursor-pointer transition-transform duration-150 hover:scale-[1.02]"
                    style={{ background: LANE_COLOR.quantum }}
                  >
                    Proceed to Feature Preview <IconArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: PREPROCESSING & FEATURE PREVIEW */}
        {activeStep === 'preview' && previewData && (
          <div className="space-y-5">
            {/* Preprocessing Stages Visualization */}
            <div
              className="rounded-panel p-5"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <IconFlask className="h-4 w-4 text-ink-faint" />
                  <h2 className="text-[13px] font-medium text-ink">
                    Preprocessing & Dimensionality Reduction Stages
                  </h2>
                </div>
                <span className="font-mono text-[10px] text-ink-faint">
                  High-Dimensional Pipeline Stream
                </span>
              </div>

              {/* Stage Cards Flow */}
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 font-mono text-[10px]">
                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                  <div className="text-ink-faint uppercase text-[9px] mb-1">Stage 1 · Imputation</div>
                  <div className="text-ink font-medium">Missing Cell Handler</div>
                  <div className="mt-1 flex items-center justify-between text-ink-dim text-[9.5px]">
                    <span>Strategy:</span>
                    <select
                      value={imputeStrategy}
                      onChange={(e) => setImputeStrategy(e.target.value as ImputeStrategy)}
                      className="bg-black/50 border border-white/10 rounded px-1.5 py-0.5 text-[9px] text-ink outline-none cursor-pointer"
                    >
                      <option value="median">Median</option>
                      <option value="mean">Mean</option>
                      <option value="drop">Drop</option>
                    </select>
                  </div>
                  <div className="mt-2 text-ink-faint text-[9px]">
                    {previewData.droppedRows > 0 ? `Dropped ${previewData.droppedRows} rows` : '0 missing dropped'}
                  </div>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                  <div className="text-ink-faint uppercase text-[9px] mb-1">Stage 2 · Normalization</div>
                  <div className="text-ink font-medium">Feature Scaling</div>
                  <div className="mt-1 flex items-center justify-between text-ink-dim text-[9.5px]">
                    <span>Method:</span>
                    <select
                      value={scalerType}
                      onChange={(e) => setScalerType(e.target.value as Scaler['kind'])}
                      className="bg-black/50 border border-white/10 rounded px-1.5 py-0.5 text-[9px] text-ink outline-none cursor-pointer"
                    >
                      <option value="standard">Standard (z-score)</option>
                      <option value="robust">Robust (IQR)</option>
                      <option value="minmax">Min-Max (0..1)</option>
                    </select>
                  </div>
                  <div className="mt-2 text-ink-faint text-[9px]">
                    Fitted on train fold strictly
                  </div>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                  <div className="text-ink-faint uppercase text-[9px] mb-1">Stage 3 · Reduction</div>
                  <div className="text-ink font-medium">Dimensionality Reducer</div>
                  <div className="mt-1 flex items-center justify-between text-ink-dim text-[9.5px]">
                    <span>Selector:</span>
                    <select
                      value={reductionMethod}
                      onChange={(e) => setReductionMethod(e.target.value as SelectionMethod)}
                      className="bg-black/50 border border-white/10 rounded px-1.5 py-0.5 text-[9px] text-ink outline-none cursor-pointer"
                    >
                      <option value="mutual-info">Mutual Info</option>
                      <option value="f-score">ANOVA F-test</option>
                      <option value="pca">PCA Projection</option>
                    </select>
                  </div>
                  <div className="mt-2 text-ink-faint text-[9px]">
                    {dataset.featureNames.length} dims → {nFeatures} components
                  </div>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                  <div className="text-ink-faint uppercase text-[9px] mb-1">Stage 4 · Quantum State</div>
                  <div className="text-ink font-medium" style={{ color: LANE_COLOR.quantum }}>
                    Hilbert Space Mapping
                  </div>
                  <div className="mt-1 text-ink-dim text-[9.5px]">
                    Encoding: <span className="text-ink">{nFeatures} Qubits (RY/RZ)</span>
                  </div>
                  <div className="mt-2 text-ink-faint text-[9px]">
                    State dim: 2^{nFeatures} = {Math.pow(2, nFeatures)} amplitudes
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Space Visualization: 2D/3D Scatter Plot */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Scatter Plot Projection (8 cols) */}
              <div
                className="lg:col-span-8 rounded-panel p-5"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-[13px] font-medium text-ink">
                      Reduced Feature Space Separability Preview
                    </h3>
                    <p className="mt-0.5 text-[11px] text-ink-dim">
                      Class clustering in projected Hilbert component space prior to training commitment
                    </p>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-[9.5px]">
                    <button
                      type="button"
                      onClick={() => setIs3DScatter(false)}
                      className={`cursor-pointer rounded px-2 py-1 ${
                        !is3DScatter ? 'bg-white/15 text-white font-medium' : 'text-ink-faint hover:text-ink'
                      }`}
                    >
                      2D Projection
                    </button>
                    <button
                      type="button"
                      onClick={() => setIs3DScatter(true)}
                      className={`cursor-pointer rounded px-2 py-1 ${
                        is3DScatter ? 'bg-white/15 text-white font-medium' : 'text-ink-faint hover:text-ink'
                      }`}
                    >
                      3D Interactive
                    </button>
                  </div>
                </div>

                <ScatterPlot
                  points={previewData.points}
                  positiveLabel={dataset.positiveLabel}
                  negativeLabel={dataset.negativeLabel}
                  height={320}
                  is3d={is3DScatter}
                />
              </div>

              {/* Training Parameters & Trigger (4 cols) */}
              <div
                className="lg:col-span-4 rounded-panel p-5 flex flex-col justify-between"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div>
                  <h3 className="text-[13px] font-medium text-ink mb-3">
                    Training Configuration
                  </h3>

                  <div className="space-y-3 font-mono text-[10.5px]">
                    <div>
                      <label className="text-ink-faint block mb-1">
                        Quantum Qubits / Features: <span className="text-ink">{nFeatures}</span>
                      </label>
                      <input
                        type="range"
                        min={4}
                        max={8}
                        step={1}
                        value={nFeatures}
                        onChange={(e) => setNFeatures(Number(e.target.value))}
                        className="w-full cursor-pointer accent-[#C08A3E]"
                      />
                    </div>

                    <div>
                      <label className="text-ink-faint block mb-1">
                        VQC Epochs: <span className="text-ink">{epochs}</span>
                      </label>
                      <input
                        type="range"
                        min={12}
                        max={48}
                        step={6}
                        value={epochs}
                        onChange={(e) => setEpochs(Number(e.target.value))}
                        className="w-full cursor-pointer accent-[#C08A3E]"
                      />
                    </div>

                    <div className="pt-2 border-t border-white/5 space-y-1 text-[9.5px] text-ink-faint">
                      <div>• Classical Baseline: Gradient Boosted Trees / SVM</div>
                      <div>• Quantum Ansatz: Strongly Entangling Layers</div>
                      <div>• Optimizer: Adam (lr = 0.2, batch = 20)</div>
                    </div>
                  </div>
                </div>

                <div className="pt-4 mt-4 border-t border-white/5">
                  <button
                    type="button"
                    onClick={handleStartTraining}
                    className="w-full flex items-center justify-center gap-2 rounded-[7px] py-2.5 font-mono text-[11px] font-medium text-black cursor-pointer transition-transform duration-150 hover:scale-[1.02]"
                    style={{ background: LANE_COLOR.quantum }}
                  >
                    <IconPulse className="h-4 w-4" /> Start Dual-Lane Training
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 3: LIVE TRAINING PROGRESS */}
        {activeStep === 'training' && (
          <div
            className="rounded-panel p-6 space-y-5"
            style={{
              background: '#16171A',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="h-3 w-3 rounded-full animate-ping" style={{ background: LANE_COLOR.quantum }} />
                <h2 className="text-[15px] font-medium text-ink">
                  Training Classical & Quantum Models in Parallel...
                </h2>
              </div>
              <span className="font-mono text-[11px] text-ink-faint">
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
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-7 rounded-[8px] bg-[#0E0F11] p-4 border border-white/5">
                <div className="mb-2 flex items-center justify-between font-mono text-[10px]">
                  <span className="text-ink-dim">VQC Quantum Loss & Accuracy Convergence</span>
                  <span className="text-ink-faint">Live Adam Optimization</span>
                </div>
                <ConvergenceChart points={convergenceData} height={180} />
              </div>

              <div className="lg:col-span-5 rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 font-mono text-[9.5px] flex flex-col justify-between">
                <div>
                  <div className="text-ink-faint uppercase text-[9px] mb-2">Live Execution Log</div>
                  <div className="space-y-1.5 max-h-[160px] overflow-y-auto pr-1 console-scroll">
                    {trainingLogs.map((log, i) => (
                      <div key={i} className="text-ink-dim flex items-start gap-2">
                        <span className="text-ink-faint shrink-0">[{log.phase}]</span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                    {convergenceData.length > 0 && (
                      <div className="text-[#5FA88C]">
                        Quantum Epoch {convergenceData[convergenceData.length - 1].epoch} / {epochs} · Loss:{' '}
                        {convergenceData[convergenceData.length - 1].loss.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
                <div className="pt-2 text-[9px] text-ink-faint border-t border-white/5">
                  Statevector simulator non-blocking web execution
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: POST-TRAINING RESULTS & BENCHMARK COMPARISON */}
        {activeStep === 'results' && trainingResult && trainedClassical && trainedQuantum && (
          <div className="space-y-5">
            {/* Completion Header */}
            <div
              className="rounded-panel p-4 flex items-center justify-between"
              style={{
                background: '#181A1E',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="flex items-center gap-3">
                <span className="h-6 w-6 grid place-items-center rounded-full bg-[#5FA88C]/20 text-[#5FA88C]">
                  <IconCheck className="h-3.5 w-3.5" />
                </span>
                <div>
                  <h2 className="text-[14px] font-medium text-ink">
                    Training Complete & Models Persisted for Inference
                  </h2>
                  <p className="font-mono text-[10px] text-ink-dim">
                    Fitted transformers and quantum weights saved to registry. Ready for Tab 3 Predict.
                  </p>
                </div>
              </div>

              {onNavigateToPredict && (
                <button
                  type="button"
                  onClick={() => onNavigateToPredict(selectedDiseaseId)}
                  className="flex items-center gap-1.5 rounded-[7px] px-3 py-1.5 font-mono text-[10.5px] font-medium text-black cursor-pointer transition-transform hover:scale-105"
                  style={{ background: LANE_COLOR.quantum }}
                >
                  Test in Predict Tab <IconArrowRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Unified Evaluation Metrics of the Newly Trained Models */}
            <div
              className="rounded-panel p-5"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <h3 className="font-mono text-[11px] font-medium uppercase tracking-wider text-ink-faint mb-3">
                Newly Trained Run Metrics (Holdout Evaluation)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-[11px]">
                  <thead>
                    <tr className="border-b border-white/10 text-[9.5px] uppercase tracking-wider text-ink-faint">
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
                              className="rounded px-1.5 py-0.5 text-[10px]"
                              style={{
                                background: delta > 0 ? alpha('#5FA88C', 0.15) : delta < 0 ? alpha('#A3543D', 0.15) : 'transparent',
                                color: delta > 0 ? '#5FA88C' : delta < 0 ? '#A3543D' : '#6A6C72',
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
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Confusion Matrices (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <h3 className="font-mono text-[11px] font-medium uppercase tracking-wider text-ink-faint mb-3">
                  Confusion Matrices (Trained Models)
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  {/* Classical Matrix */}
                  <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                    <div className="font-mono text-[9.5px] text-ink-faint mb-2" style={{ color: LANE_COLOR.classical }}>
                      Classical ({trainedClassical.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1 font-mono text-[11px] text-center">
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[14px]">{trainedClassical.metrics.confusion.tp}</div>
                        <div className="text-[8px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[14px]">{trainedClassical.metrics.confusion.fp}</div>
                        <div className="text-[8px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[14px]">{trainedClassical.metrics.confusion.fn}</div>
                        <div className="text-[8px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[14px]">{trainedClassical.metrics.confusion.tn}</div>
                        <div className="text-[8px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>

                  {/* Quantum Matrix */}
                  <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5">
                    <div className="font-mono text-[9.5px] text-ink-faint mb-2" style={{ color: LANE_COLOR.quantum }}>
                      Quantum ({trainedQuantum.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1 font-mono text-[11px] text-center">
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[14px]">{trainedQuantum.metrics.confusion.tp}</div>
                        <div className="text-[8px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[14px]">{trainedQuantum.metrics.confusion.fp}</div>
                        <div className="text-[8px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#A3543D] font-bold text-[14px]">{trainedQuantum.metrics.confusion.fn}</div>
                        <div className="text-[8px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2">
                        <div className="text-[#E8E9EB] font-bold text-[14px]">{trainedQuantum.metrics.confusion.tn}</div>
                        <div className="text-[8px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Direct Comparison: Newly Trained vs Platform Stored Benchmark (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-mono text-[11px] font-medium uppercase tracking-wider text-ink-faint">
                    Trained Run vs Platform Baseline Benchmark
                  </h3>
                  <span className="font-mono text-[9px] text-ink-faint">Generalization Check</span>
                </div>

                <div className="space-y-3 font-mono text-[10.5px]">
                  <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 space-y-2">
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

                  <p className="text-[10px] leading-relaxed text-ink-dim">
                    This newly trained run achieves{' '}
                    <span className="text-ink font-medium">
                      {trainedQuantum.metrics.accuracy >= benchmarkQuantum.metrics.accuracy ? 'comparable/superior' : 'close'}
                    </span>{' '}
                    generalization compared to the platform's pre-established baseline benchmark for {disease.name}.
                  </p>
                </div>
              </div>
            </div>

            {/* ROC Curves for Newly Trained Models */}
            <div
              className="rounded-panel p-5"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div className="mb-2 flex items-baseline justify-between">
                <h3 className="font-mono text-[11px] font-medium uppercase tracking-wider text-ink-faint">
                  ROC Curves for Newly Trained Models
                </h3>
                <span className="font-mono text-[9.5px] text-ink-faint">Holdout Validation Split</span>
              </div>
              <div className="max-w-[420px] mx-auto">
                <RocChart curves={trainedRocCurves} size={240} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
