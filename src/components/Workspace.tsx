import { useMemo, useState } from 'react'
import { useRun } from '../hooks/useRun'
import { LANE_COLOR } from '../lib/theme'
import { IconArrowLeft, IconArrowRight } from './icons'
import { QubitBadge } from './QubitBadge'
import { Stepper, STEPS, type StepId } from './Stepper'
import { DataStep } from './steps/DataStep'
import { ExplainStep } from './steps/ExplainStep'
import { FeaturesStep } from './steps/FeaturesStep'
import { ModelStep } from './steps/ModelStep'
import { PreprocessStep } from './steps/PreprocessStep'
import { ResultsStep } from './steps/ResultsStep'
import { TrainStep } from './steps/TrainStep'
import { Wordmark } from './Wordmark'

const TITLE: Record<StepId, { title: string; blurb: string }> = {
  data: { title: 'Data', blurb: 'Choose a dataset, inspect its health, set the split and seed' },
  preprocess: { title: 'Preprocessing', blurb: 'Handle missing values, scale features, address class imbalance' },
  features: { title: 'Feature selection', blurb: 'Rank features and choose how many to keep — this sets the qubit count' },
  model: { title: 'Model builder', blurb: 'Configure the circuit and pick the classical baselines to beat' },
  train: { title: 'Training', blurb: 'Run both lanes together and watch the cost function converge' },
  results: { title: 'Results', blurb: 'Compare every model on the same holdout split' },
  explain: { title: 'Explainability', blurb: 'What drove the model, globally and for one patient' },
}

type Props = {
  onHome: () => void
  initialStep?: StepId
}

export function Workspace({ onHome, initialStep = 'data' }: Props) {
  const [step, setStep] = useState<StepId>(initialStep)
  const run = useRun()

  const locked = run.phase === 'running'
  const hasResult = run.result !== null

  const done = useMemo(() => {
    const s = new Set<StepId>()
    const order: StepId[] = ['data', 'preprocess', 'features', 'model']
    const idx = order.indexOf(step)
    // Configuration steps count as done once passed.
    order.slice(0, idx).forEach((id) => s.add(id))
    if (run.phase === 'complete') {
      order.forEach((id) => s.add(id))
      s.add('train')
    }
    if (hasResult) s.add('results')
    return s
  }, [step, run.phase, hasResult])

  const blocked = useMemo(() => {
    const b: Partial<Record<StepId, string>> = {}
    if (!hasResult) {
      b.results = 'Run the pipeline first'
      b.explain = 'Run the pipeline first'
    }
    return b
  }, [hasResult])

  const idx = STEPS.findIndex((s) => s.id === step)
  const prev = idx > 0 ? STEPS[idx - 1] : null
  const next = idx < STEPS.length - 1 ? STEPS[idx + 1] : null
  const nextBlocked = next ? blocked[next.id] : undefined

  // Offered on Results/Explain when there is no run yet - landing there
  // directly (e.g. from the Predict or Compare launch cards) must not strand
  // the user on an empty screen with no visible way to produce something.
  // Guarded against a run already in flight: the stepper allows sitting on
  // Results/Explain mid-run (they are blocked only by the absence of a
  // result, not by run.phase), so this button is reachable while training is
  // already running and must not start a second one on top of it.
  const startTrainingFromHere = () => {
    setStep('train')
    if (!locked) run.start()
  }

  return (
    <div className="flex h-full min-w-[1100px] flex-col bg-canvas">
      <header
        className="flex h-14 shrink-0 items-center gap-3 px-4"
        style={{
          background: '#111214',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
        }}
      >
        <button
          type="button"
          onClick={onHome}
          className="flex cursor-pointer items-center gap-2 rounded-[8px] px-2 py-1.5 text-ink-faint transition-colors duration-150 hover:text-ink"
          aria-label="Back to menu"
        >
          <IconArrowLeft className="h-3.5 w-3.5" />
          <Wordmark size={14} />
        </button>

        <div className="h-5 w-px" style={{ background: 'rgba(255,255,255,0.07)' }} />

        <div>
          <div className="text-[13px] text-ink">{TITLE[step].title}</div>
          <div className="font-mono text-[9px] text-ink-faint">{TITLE[step].blurb}</div>
        </div>

        <div className="flex-1" />

        {/* qubit count, visible on every screen after feature selection */}
        {idx >= 2 && <QubitBadge qubits={run.config.nFeatures} compact />}
      </header>

      <Stepper current={step} done={done} blocked={blocked} onGo={setStep} />

      <div className="console-scroll min-h-0 flex-1 overflow-y-auto p-4">
        {step === 'data' && (
          <DataStep config={run.config} patch={run.patch} locked={locked} />
        )}
        {step === 'preprocess' && (
          <PreprocessStep config={run.config} patch={run.patch} locked={locked} />
        )}
        {step === 'features' && (
          <FeaturesStep config={run.config} patch={run.patch} locked={locked} />
        )}
        {step === 'model' && (
          <ModelStep config={run.config} patch={run.patch} locked={locked} />
        )}
        {step === 'train' && (
          <TrainStep
            config={run.config}
            phase={run.phase}
            logs={run.logs}
            convergence={run.convergence}
            progress={run.progress}
            elapsed={run.elapsed}
            onStart={run.start}
            onStop={run.stop}
            onReset={run.reset}
            onLoadDemo={run.loadDemo}
          />
        )}
        {step === 'results' && (
          <ResultsStep
            result={run.result}
            running={locked}
            onStartTraining={startTrainingFromHere}
          />
        )}
        {step === 'explain' && (
          <ExplainStep
            result={run.result}
            running={locked}
            onStartTraining={startTrainingFromHere}
          />
        )}
      </div>

      {/* step navigation */}
      <footer
        className="flex h-12 shrink-0 items-center gap-3 px-4"
        style={{
          background: '#0E0F11',
          borderTop: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        {prev ? (
          <button
            type="button"
            onClick={() => setStep(prev.id)}
            className="flex cursor-pointer items-center gap-1.5 rounded-[7px] px-2.5 py-1.5 font-mono text-[10px] text-ink-faint transition-colors duration-150 hover:text-ink"
          >
            <IconArrowLeft className="h-3 w-3" />
            {prev.label}
          </button>
        ) : (
          <span />
        )}

        <div className="flex-1" />

        <span className="font-mono text-[9px] text-ink-faint">
          step {idx + 1} of {STEPS.length}
        </span>

        {next && (
          <button
            type="button"
            disabled={Boolean(nextBlocked)}
            title={nextBlocked}
            onClick={() => setStep(next.id)}
            className="flex cursor-pointer items-center gap-1.5 rounded-[7px] px-3 py-1.5 font-mono text-[10px] transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
            style={{
              background: nextBlocked ? 'transparent' : 'rgba(255,255,255,0.05)',
              color: nextBlocked ? '#6A6C72' : LANE_COLOR.quantum,
            }}
          >
            {next.label}
            <IconArrowRight className="h-3 w-3" />
          </button>
        )}
      </footer>
    </div>
  )
}
