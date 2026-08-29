import { useState } from 'react'
import { ConditionCatalog } from './components/ConditionCatalog'
import { EhrValidationView } from './components/EhrValidationView'
import { LaunchScreen, type AppMode } from './components/LaunchScreen'
import type { StepId } from './components/Stepper'
import { Workspace } from './components/Workspace'

/**
 * Top-level router.
 *
 * `null` is the launch screen. Train, Predict and Compare drop into the
 * workspace at the step they correspond to, and the stepper takes over from
 * there - so those three are shortcuts into one pipeline rather than three
 * separate apps. Conditions opens the clinical registry served by the backend.
 *
 * `/ehr-validation` is a deliberately isolated surface: it exercises the EHR
 * cohort contract against the backend without touching the pipeline UI, so a
 * validation demo cannot disturb a run in progress.
 */
const ENTRY_STEP: Record<'train' | 'predict' | 'compare', StepId> = {
  train: 'data',
  predict: 'explain',
  compare: 'results',
}

export default function App() {
  const [mode, setMode] = useState<AppMode | null>(null)

  // Standalone route, checked before anything else renders.
  if (typeof window !== 'undefined' && window.location.pathname === '/ehr-validation') {
    return <EhrValidationView />
  }

  if (mode === null) {
    return <LaunchScreen onSelect={setMode} />
  }

  if (mode === 'conditions') {
    return <ConditionCatalog onHome={() => setMode(null)} />
  }

  return <Workspace onHome={() => setMode(null)} initialStep={ENTRY_STEP[mode]} />
}
