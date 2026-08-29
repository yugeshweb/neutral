import { useState } from 'react'
import { LaunchScreen, type AppMode } from './components/LaunchScreen'
import type { StepId } from './components/Stepper'
import { Workspace } from './components/Workspace'

/**
 * Top-level router. `null` is the launch screen; each card drops into the
 * workspace at the step it corresponds to, and the stepper takes over from
 * there - so the three entry points are shortcuts into one pipeline rather
 * than three separate apps.
 */
const ENTRY_STEP: Record<AppMode, StepId> = {
  train: 'data',
  predict: 'explain',
  compare: 'results',
}

export default function App() {
  const [mode, setMode] = useState<AppMode | null>(null)

  if (mode === null) {
    return <LaunchScreen onSelect={setMode} />
  }

  return <Workspace onHome={() => setMode(null)} initialStep={ENTRY_STEP[mode]} />
}
