import { useCallback, useState } from 'react'
import { ConditionCatalog } from './components/ConditionCatalog'
import { CompareView } from './components/CompareView'
import { LaunchScreen, type AppMode } from './components/LaunchScreen'
import { ModeBar } from './components/ModeBar'
import { PredictView } from './components/PredictView'
import { TrainView } from './components/TrainView'

/**
 * Top-level router. `null` is the launch screen; the modes are the
 * platform's entry points - train the model, score a case, compare against the
 * classical baseline, or browse the condition catalog.
 *
 * TrainView is kept mounted once visited so a run in flight is not destroyed by
 * a trip to another mode; the other views are cheap enough to remount.
 */
export default function App() {
  const [mode, setMode] = useState<AppMode | null>(null)
  const [visitedTrain, setVisitedTrain] = useState(false)
  const [trained, setTrained] = useState(false)

  const onTrained = useCallback(() => setTrained(true), [])

  const goHome = () => setMode(null)

  const goMode = (m: AppMode) => {
    if (m === 'train') setVisitedTrain(true)
    setMode(m)
  }

  return (
    <>
      {mode === null && <LaunchScreen onSelect={goMode} />}

      {/* kept mounted across mode switches so an in-flight run survives */}
      {visitedTrain && (
        <div className={mode === 'train' ? 'h-full' : 'hidden'}>
          <TrainView onHome={goHome} onTrained={onTrained} />
        </div>
      )}

      {(mode === 'predict' || mode === 'compare') && (
        <div className="flex h-full min-w-[1080px] flex-col bg-canvas">
          <ModeBar mode={mode} onHome={goHome} />
          <div className="min-h-0 flex-1">
            {mode === 'predict' ? (
              <PredictView trained={trained} />
            ) : (
              <CompareView trained={trained} />
            )}
          </div>
        </div>
      )}

      {mode === 'conditions' && (
        <div className="h-full">
          <ConditionCatalog />
        </div>
      )}
    </>
  )
}
