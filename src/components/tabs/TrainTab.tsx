import { useMemo, useRef, useState } from 'react'
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
import { pca, pcaTransform, rankFeatures } from '../../lib/ml/features'
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
} from '../../lib/ml/stats'
import type { EpochRecord } from '../../lib/quantum/vqc'
import { LANE_COLOR, alpha } from '../../lib/theme'
import { ConvergenceChart, RocChart } from '../charts'
import { ScatterPlot } from '../charts/ScatterPlot'
import {
  IconArrowRight,
  IconCheck,
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

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const cancelTrainingRef = useRef(false)

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
      const imputed = imputeMissing(dataset.X, dataset.y, 'median')
      const scaler = fitScaler(imputed.X, 'standard')
      const scaledX = applyScaler(imputed.X, scaler)

      // Dimensionality reduction projection for 2D/3D visualization
      const pcaRes = pca(scaledX, 3, 42)
      const proj = pcaTransform(scaledX, pcaRes)

      const points = proj.map((row, i) => ({
        x: row[0],
        y: row[1],
        z: row[2] ?? 0,
        label: imputed.y[i],
        id: i,
      }))

      const ranked = rankFeatures(scaledX, imputed.y, dataset.featureNames, 'mutual-info')

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
  }, [dataset])

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
      nFeatures: disease.defaultQubits,
      impute: 'median',
      scaler: 'standard',
      selection: 'mutual-info',
      epochs: 24,
      vqc: {
        ...DEFAULT_RUN.vqc,
        qubits: disease.defaultQubits,
      },
    }

    const gen = runPipeline(dataset, runCfg)

    const pump = () => {
      if (cancelTrainingRef.current) return

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

  // Required Data Format description based on disease
  const expectedDataFormat = useMemo(() => {
    if (selectedDiseaseId === 'breast-cancer') {
      return {
        format: 'Tabular CSV (Fine Needle Aspirate / Morphological measurements)',
        columns: '30 nuclear features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, concavity_mean, concave_points_mean, symmetry_mean...',
        target: 'Binary Diagnosis Target (1 = Malignant, 0 = Benign)',
      }
    }
    if (selectedDiseaseId === 'brain-seizure') {
      return {
        format: 'EEG Biosignal Feature CSV (Spectral band powers & wavelet coefficients)',
        columns: '10 rhythmic features: delta_power, theta_power, alpha_power, beta_power, gamma_power, spectral_entropy, hjorth_mobility, sample_entropy...',
        target: 'Binary Seizure Target (1 = Seizure Detected, 0 = Normal EEG Baseline)',
      }
    }
    return {
      format: 'Clinical & Hemodynamic CSV (Diagnostic cardiac measurements)',
      columns: '10 clinical attributes: age, resting_bp, cholesterol, max_heart_rate, st_depression, num_vessels, chest_pain_type, exercise_angina...',
      target: 'Binary Risk Target (1 = High Risk Stenosis/CAD, 0 = Normal / Low Risk)',
    }
  }, [selectedDiseaseId])

  return (
    <div className="console-scroll h-full overflow-y-auto bg-canvas">
      <div className="mx-auto w-full max-w-[1400px] px-6 lg:px-10 py-7 space-y-6">
        {/* Step Navigation Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div>
            <h1 className="text-[18px] font-medium text-ink">
              Train Models on Custom Dataset
            </h1>
            <p className="mt-0.5 text-[12px] text-ink-dim">
              Ingest biomedical data, preview preprocessing steps in code, and evaluate classical and hybrid quantum pipelines.
            </p>
          </div>

          <div className="flex items-center gap-1.5 rounded-[8px] bg-[#141518] p-1 border border-white/5 font-mono text-[11px]">
            <button
              type="button"
              onClick={() => setActiveStep('ingest')}
              className={`cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all ${
                activeStep === 'ingest' ? 'bg-white/12 text-white font-medium' : 'text-ink-faint hover:text-ink'
              }`}
            >
              1. Ingest Data
            </button>
            <button
              type="button"
              onClick={() => setActiveStep('preview')}
              className={`cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all ${
                activeStep === 'preview' ? 'bg-white/12 text-white font-medium' : 'text-ink-faint hover:text-ink'
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
              className={`cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all ${
                activeStep === 'training' || activeStep === 'results'
                  ? 'bg-white/12 text-white font-medium'
                  : 'text-ink-faint hover:text-ink'
              }`}
            >
              3. Training & Benchmark
            </button>
          </div>
        </div>

        {/* STEP 1: INGESTION & DISEASE SELECTION */}
        {activeStep === 'ingest' && (
          <div className="space-y-5">
            {/* Target Disease Selector */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
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
                    className="w-full text-left rounded-[10px] p-4 transition-all cursor-pointer"
                    style={{
                      background: active ? '#1E2025' : '#141518',
                      border: `1px solid ${active ? alpha(LANE_COLOR.quantum, 0.55) : 'rgba(255,255,255,0.06)'}`,
                      boxShadow: active ? `0 0 0 1px ${alpha(LANE_COLOR.quantum, 0.25)}` : 'none',
                    }}
                  >
                    <div className="flex items-center justify-between font-mono text-[10px] uppercase text-ink-faint">
                      <span>{d.categoryLabel}</span>
                      {active && <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />}
                    </div>
                    <div className="text-[14.5px] font-medium text-ink mt-1">{d.name}</div>
                    <div className="text-[11px] text-ink-dim mt-0.5">{d.modality}</div>
                  </button>
                )
              })}
            </div>

            {/* Expected Data Specification for Selected Disease */}
            <div
              className="rounded-panel p-5 space-y-3 font-mono text-[11px]"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="text-ink-faint uppercase text-[9.5px] tracking-wider">
                Expected Input Data Format for {disease.name}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-ink-dim">
                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 space-y-1">
                  <span className="text-ink font-medium">Input Modality:</span>
                  <p className="text-[10.5px]">{expectedDataFormat.format}</p>
                </div>
                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 space-y-1">
                  <span className="text-ink font-medium">Required Features:</span>
                  <p className="text-[10.5px] leading-relaxed">{expectedDataFormat.columns}</p>
                </div>
                <div className="rounded-[8px] bg-[#0E0F11] p-3 border border-white/5 space-y-1">
                  <span className="text-ink font-medium">Target Classification:</span>
                  <p className="text-[10.5px]">{expectedDataFormat.target}</p>
                </div>
              </div>
            </div>

            {/* Upload Area */}
            <div
              className="rounded-panel p-6 space-y-4"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-[14px] font-medium text-ink">Upload Dataset or Use Preset Cohort</h3>
                  <p className="text-[11.5px] text-ink-dim">Upload your CSV file matching the expected format above</p>
                </div>
                {uploadFileName && (
                  <button
                    type="button"
                    onClick={handleResetUpload}
                    className="cursor-pointer font-mono text-[10px] text-ink-faint hover:text-ink underline"
                  >
                    Reset to default dataset
                  </button>
                )}
              </div>

              <div
                className="rounded-[10px] border-2 border-dashed border-white/10 p-6 text-center hover:border-white/25 transition-colors"
                style={{ background: '#0E0F11' }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="train-csv-upload"
                />
                <label htmlFor="train-csv-upload" className="cursor-pointer flex flex-col items-center gap-2">
                  <IconUpload className="h-7 w-7 text-ink-faint" />
                  <span className="text-[13px] font-medium text-ink">
                    {uploadFileName ? `Uploaded: ${uploadFileName}` : 'Click to Upload Custom CSV Dataset'}
                  </span>
                  <span className="font-mono text-[10px] text-ink-faint">
                    CSV with header row and corresponding feature columns
                  </span>
                </label>
                {uploadError && <div className="mt-2.5 font-mono text-[11px] text-[#A3543D]">{uploadError}</div>}
              </div>

              {/* Active Ingested Cohort Status */}
              <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 font-mono text-[11px] flex flex-wrap items-center justify-between gap-3">
                <div>
                  <span className="text-ink-faint">Active Cohort: </span>
                  <span className="text-ink font-medium">{dataset.name}</span>
                </div>
                <div>
                  <span className="text-ink-faint">Dimensions: </span>
                  <span className="text-ink">{dataset.X.length} rows × {dataset.featureNames.length} features</span>
                </div>
                <div>
                  <span className="text-ink-faint">Classes: </span>
                  <span className="text-[#A3543D] font-medium">{dataset.positiveLabel}</span> /{' '}
                  <span className="text-[#5FA88C] font-medium">{dataset.negativeLabel}</span>
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  type="button"
                  onClick={() => setActiveStep('preview')}
                  className="flex items-center gap-2 rounded-[7px] px-5 py-2 font-mono text-[12px] font-medium text-black cursor-pointer transition-transform hover:scale-[1.02]"
                  style={{ background: LANE_COLOR.quantum }}
                >
                  Proceed to Feature Preview <IconArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: PREPROCESSING IN CODE & FEATURE PREVIEW */}
        {activeStep === 'preview' && previewData && (
          <div className="space-y-5">
            {/* Explanation of how Preprocessing in Code Works for this Data */}
            <div
              className="rounded-panel p-5 space-y-4"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div>
                  <h2 className="text-[15px] font-medium text-ink">
                    Code Preprocessing Pipeline for {disease.name}
                  </h2>
                  <p className="text-[12px] text-ink-dim">
                    Deterministic pipeline executing on the active {dataset.X.length} records before quantum state compilation
                  </p>
                </div>
                <span className="font-mono text-[11px] text-ink-dim">
                  Status: Pipeline Fitted & Ready
                </span>
              </div>

              {/* 4 Pipeline Stages in Points & Explanation */}
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-3.5 font-mono text-[11px]">
                <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-1.5">
                  <div className="text-ink-faint uppercase text-[9.5px]">Stage 1 · Imputation</div>
                  <div className="text-ink font-medium">Missing Value Handler</div>
                  <p className="text-[10.5px] text-ink-dim leading-relaxed">
                    Column-median replacement fitted strictly on training partition. {previewData.droppedRows > 0 ? `Dropped ${previewData.droppedRows} rows.` : '0 missing gaps.'}
                  </p>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-1.5">
                  <div className="text-ink-faint uppercase text-[9.5px]">Stage 2 · Scaling</div>
                  <div className="text-ink font-medium">Standard Normalization</div>
                  <p className="text-[10.5px] text-ink-dim leading-relaxed">
                    Z-score scaling ($\mu=0, \sigma=1$) to prevent high-magnitude features from dominating quantum rotations.
                  </p>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-1.5">
                  <div className="text-ink-faint uppercase text-[9.5px]">Stage 3 · Reduction</div>
                  <div className="text-ink font-medium">Dimensionality Reducer</div>
                  <p className="text-[10.5px] text-ink-dim leading-relaxed">
                    {dataset.featureNames.length} input features reduced to {disease.defaultQubits} principal components via Mutual Information selection.
                  </p>
                </div>

                <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5 space-y-1.5">
                  <div className="text-ink-faint uppercase text-[9.5px]" style={{ color: LANE_COLOR.quantum }}>
                    Stage 4 · Quantum State
                  </div>
                  <div className="text-ink font-medium" style={{ color: LANE_COLOR.quantum }}>
                    Hilbert Space Embedding
                  </div>
                  <p className="text-[10.5px] text-ink-dim leading-relaxed">
                    RY/RZ angle embedding onto {disease.defaultQubits} entangled qubits ($2^{disease.defaultQubits} = {Math.pow(2, disease.defaultQubits)}$ state amplitudes).
                  </p>
                </div>
              </div>
            </div>

            {/* Feature Preview: 2D/3D Scatter Plot */}
            <div
              className="rounded-panel p-5 space-y-4"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-[14px] font-medium text-ink">
                    Reduced Feature Space Class Separability
                  </h3>
                  <p className="text-[11.5px] text-ink-dim">
                    Samples projected onto top components colored by clinical class label
                  </p>
                </div>

                <div className="flex items-center gap-2 font-mono text-[10px]">
                  <button
                    type="button"
                    onClick={() => setIs3DScatter(false)}
                    className={`cursor-pointer rounded px-2.5 py-1 ${
                      !is3DScatter ? 'bg-white/15 text-white font-medium' : 'text-ink-faint hover:text-ink'
                    }`}
                  >
                    2D Projection
                  </button>
                  <button
                    type="button"
                    onClick={() => setIs3DScatter(true)}
                    className={`cursor-pointer rounded px-2.5 py-1 ${
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

              <div className="flex justify-end pt-3 border-t border-white/5">
                <button
                  type="button"
                  onClick={handleStartTraining}
                  className="flex items-center gap-2 rounded-[7px] px-6 py-2.5 font-mono text-[12px] font-medium text-black cursor-pointer transition-transform hover:scale-[1.02]"
                  style={{ background: LANE_COLOR.quantum }}
                >
                  <IconPulse className="h-4 w-4" /> Start Training Dual-Lane Models
                </button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 3: LIVE TRAINING EXECUTION */}
        {activeStep === 'training' && (
          <div
            className="rounded-panel p-6 space-y-5"
            style={{
              background: '#16171A',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="h-3 w-3 rounded-full animate-ping" style={{ background: LANE_COLOR.quantum }} />
                <h2 className="text-[16px] font-medium text-ink">
                  Training Classical & Quantum Models in Parallel...
                </h2>
              </div>
              <span className="font-mono text-[12px] text-ink-faint">
                {Math.round(currentProgress * 100)}% Complete
              </span>
            </div>

            <div className="h-2.5 w-full rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-150"
                style={{
                  width: `${currentProgress * 100}%`,
                  background: `linear-gradient(90deg, ${LANE_COLOR.classical}, ${LANE_COLOR.quantum})`,
                }}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-7 rounded-[8px] bg-[#0E0F11] p-4 border border-white/5">
                <div className="mb-2 flex items-center justify-between font-mono text-[11px]">
                  <span className="text-ink-dim">VQC Quantum Loss Convergence</span>
                  <span className="text-ink-faint">Adam Optimization</span>
                </div>
                <ConvergenceChart points={convergenceData} height={190} />
              </div>

              <div className="lg:col-span-5 rounded-[8px] bg-[#0E0F11] p-4 border border-white/5 font-mono text-[10px] flex flex-col justify-between">
                <div>
                  <div className="text-ink-faint uppercase text-[9.5px] mb-2">Execution Stages</div>
                  <div className="space-y-1.5 max-h-[170px] overflow-y-auto pr-1 console-scroll">
                    {trainingLogs.map((log, i) => (
                      <div key={i} className="text-ink-dim flex items-start gap-2">
                        <span className="text-ink-faint shrink-0">[{log.phase}]</span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                    {convergenceData.length > 0 && (
                      <div className="text-[#5FA88C]">
                        Epoch {convergenceData[convergenceData.length - 1].epoch} / 24 · Loss:{' '}
                        {convergenceData[convergenceData.length - 1].loss.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
                <div className="pt-2 text-[9.5px] text-ink-faint border-t border-white/5">
                  Running non-blocking simulation
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: RESULTS & BENCHMARK COMPARISON */}
        {activeStep === 'results' && trainingResult && trainedClassical && trainedQuantum && (
          <div className="space-y-6">
            {/* Persistence Header */}
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
                    Training Completed & Models Persisted to Registry
                  </h2>
                  <p className="font-mono text-[10.5px] text-ink-dim">
                    Saved fitted transformers and quantum parameters are now available for inference in Tab 3 Predict.
                  </p>
                </div>
              </div>

              {onNavigateToPredict && (
                <button
                  type="button"
                  onClick={() => onNavigateToPredict(selectedDiseaseId)}
                  className="flex items-center gap-1.5 rounded-[7px] px-3.5 py-1.5 font-mono text-[11px] font-medium text-black cursor-pointer transition-transform hover:scale-105"
                  style={{ background: LANE_COLOR.quantum }}
                >
                  Open in Predict Tab <IconArrowRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Direct Comparison: Newly Trained Models vs Stored Platform Benchmark */}
            <div
              className="rounded-panel p-5 space-y-4"
              style={{
                background: '#16171A',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <h3 className="font-mono text-[12px] font-medium uppercase tracking-wider text-ink-faint">
                Comparison: Newly Trained Run vs Platform Stored Benchmark
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-[12px]">
                  <thead>
                    <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-ink-faint">
                      <th className="py-2.5 font-medium">Metric</th>
                      <th className="py-2.5 text-right font-medium">Trained Classical</th>
                      <th className="py-2.5 text-right font-medium">Trained Quantum</th>
                      <th className="py-2.5 text-right font-medium">Stored Benchmark (Quantum)</th>
                      <th className="py-2.5 text-right font-medium">Generalization Delta</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {[
                      {
                        label: 'Accuracy',
                        tc: trainedClassical.metrics.accuracy,
                        tq: trainedQuantum.metrics.accuracy,
                        bq: benchmarkQuantum.metrics.accuracy,
                      },
                      {
                        label: 'Precision',
                        tc: trainedClassical.metrics.precision,
                        tq: trainedQuantum.metrics.precision,
                        bq: benchmarkQuantum.metrics.precision,
                      },
                      {
                        label: 'Sensitivity / Recall',
                        tc: trainedClassical.metrics.sensitivity,
                        tq: trainedQuantum.metrics.sensitivity,
                        bq: benchmarkQuantum.metrics.sensitivity,
                      },
                      {
                        label: 'Specificity',
                        tc: trainedClassical.metrics.specificity,
                        tq: trainedQuantum.metrics.specificity,
                        bq: benchmarkQuantum.metrics.specificity,
                      },
                      {
                        label: 'F1 Score',
                        tc: trainedClassical.metrics.f1,
                        tq: trainedQuantum.metrics.f1,
                        bq: benchmarkQuantum.metrics.f1,
                      },
                      {
                        label: 'AUC-ROC',
                        tc: trainedClassical.metrics.rocAuc,
                        tq: trainedQuantum.metrics.rocAuc,
                        bq: benchmarkQuantum.metrics.rocAuc,
                      },
                    ].map((row) => {
                      const delta = row.tq - row.bq
                      return (
                        <tr key={row.label} className="hover:bg-white/[0.02]">
                          <td className="py-3 font-medium text-ink">{row.label}</td>
                          <td className="py-3 text-right text-ink-dim tabular-nums">{row.tc.toFixed(3)}</td>
                          <td className="py-3 text-right font-medium text-ink tabular-nums" style={{ color: LANE_COLOR.quantum }}>
                            {row.tq.toFixed(3)}
                          </td>
                          <td className="py-3 text-right text-ink-dim tabular-nums">{row.bq.toFixed(3)}</td>
                          <td className="py-3 text-right tabular-nums">
                            <span
                              className="rounded px-2 py-0.5 text-[10.5px] font-medium"
                              style={{
                                background: delta >= 0 ? alpha('#5FA88C', 0.15) : alpha('#A3543D', 0.15),
                                color: delta >= 0 ? '#5FA88C' : '#A3543D',
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

            {/* Confusion Matrices & ROC Curves for Newly Trained Models */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Confusion Matrices (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5 space-y-3"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.07)',
                }}
              >
                <h3 className="font-mono text-[12px] font-medium uppercase tracking-wider text-ink-faint">
                  Confusion Matrices (Trained Models)
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  {/* Classical Matrix */}
                  <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5">
                    <div className="font-mono text-[10px] text-ink-faint mb-2" style={{ color: LANE_COLOR.classical }}>
                      Classical ({trainedClassical.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1.5 font-mono text-[12px] text-center">
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#E8E9EB] font-bold text-[15px]">{trainedClassical.metrics.confusion.tp}</div>
                        <div className="text-[9px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#A3543D] font-bold text-[15px]">{trainedClassical.metrics.confusion.fp}</div>
                        <div className="text-[9px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#A3543D] font-bold text-[15px]">{trainedClassical.metrics.confusion.fn}</div>
                        <div className="text-[9px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#E8E9EB] font-bold text-[15px]">{trainedClassical.metrics.confusion.tn}</div>
                        <div className="text-[9px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>

                  {/* Quantum Matrix */}
                  <div className="rounded-[8px] bg-[#0E0F11] p-3.5 border border-white/5">
                    <div className="font-mono text-[10px] text-ink-faint mb-2" style={{ color: LANE_COLOR.quantum }}>
                      Quantum ({trainedQuantum.label})
                    </div>
                    <div className="grid grid-cols-2 gap-1.5 font-mono text-[12px] text-center">
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#E8E9EB] font-bold text-[15px]">{trainedQuantum.metrics.confusion.tp}</div>
                        <div className="text-[9px] text-ink-faint">TP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#A3543D] font-bold text-[15px]">{trainedQuantum.metrics.confusion.fp}</div>
                        <div className="text-[9px] text-ink-faint">FP</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#A3543D] font-bold text-[15px]">{trainedQuantum.metrics.confusion.fn}</div>
                        <div className="text-[9px] text-ink-faint">FN</div>
                      </div>
                      <div className="rounded bg-white/5 p-2.5">
                        <div className="text-[#E8E9EB] font-bold text-[15px]">{trainedQuantum.metrics.confusion.tn}</div>
                        <div className="text-[9px] text-ink-faint">TN</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* ROC Curves (6 cols) */}
              <div
                className="lg:col-span-6 rounded-panel p-5 flex flex-col justify-between"
                style={{
                  background: '#16171A',
                  border: '1px solid rgba(255,255,255,0.07)',
                }}
              >
                <div>
                  <div className="mb-2 flex items-baseline justify-between">
                    <h3 className="font-mono text-[12px] font-medium uppercase tracking-wider text-ink-faint">
                      ROC Curves (Newly Trained Models)
                    </h3>
                    <span className="font-mono text-[10px] text-ink-faint">Holdout Split</span>
                  </div>
                  <div className="py-2 max-w-[340px] mx-auto">
                    <RocChart curves={trainedRocCurves} size={230} />
                  </div>
                </div>

                <div className="pt-2 border-t border-white/5 font-mono text-[10.5px] text-ink-dim flex justify-between">
                  <span>Trained Classical AUC: {trainedClassical.metrics.rocAuc.toFixed(3)}</span>
                  <span style={{ color: LANE_COLOR.quantum }}>Trained Quantum AUC: {trainedQuantum.metrics.rocAuc.toFixed(3)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
