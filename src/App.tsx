import { useState, type ReactElement } from 'react'
import { EhrValidationView } from './components/EhrValidationView'
import { BenchmarkTab } from './components/tabs/BenchmarkTab'
import { TrainTab } from './components/tabs/TrainTab'
import { PredictTab } from './components/tabs/PredictTab'
import { Wordmark } from './components/Wordmark'
import { LANE_COLOR } from './lib/theme'
import { IconBars, IconFlask, IconPulse } from './components/icons'

export type TabId = 'benchmark' | 'train' | 'predict'

const TABS: { id: TabId; label: string; Icon: (p: { className?: string }) => ReactElement }[] = [
  { id: 'benchmark', label: '1. Benchmark', Icon: IconBars },
  { id: 'train', label: '2. Train', Icon: IconFlask },
  { id: 'predict', label: '3. Predict', Icon: IconPulse },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('benchmark')
  const [targetDiseaseId, setTargetDiseaseId] = useState<string>('breast-cancer')

  // Standalone route for validation if requested directly
  if (typeof window !== 'undefined' && window.location.pathname === '/ehr-validation') {
    return <EhrValidationView />
  }

  const handleNavigateToPredict = (diseaseId: string) => {
    setTargetDiseaseId(diseaseId)
    setActiveTab('predict')
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-canvas text-ink">
      {/* Top Application Header Bar */}
      {/*
       * The nav is centred against the viewport, not against the space left
       * over by its neighbours: the wordmark's tagline is far wider than the
       * status tag, so a plain justify-between would sit the tabs off-centre.
       * Taking the nav out of flow and centring it absolutely keeps it fixed
       * regardless of what either side grows to.
       */}
      <header
        className="relative z-30 flex h-16 shrink-0 items-center justify-between border-b px-6"
        style={{
          background: '#111214',
          borderColor: 'var(--border)',
        }}
      >
        <div className="flex items-center gap-2">
          <Wordmark size={18} />
          <span className="hidden sm:inline-block h-3.5 w-px bg-white/10" />
          <span className="hidden xl:inline-block font-mono text-[12px] text-ink-faint">
            Hybrid Quantum-Classical Disease Detection Platform
          </span>
        </div>

        {/* Segmented control: the group is a plain track, so the keys inside
            supply the only borders and nothing double-lines. */}
        <nav className="absolute left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-[8px] bg-black/25 p-1 font-mono text-[13px]">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setActiveTab(id)}
              data-pressed={activeTab === id}
              aria-current={activeTab === id ? 'page' : undefined}
              className="key flex cursor-pointer items-center justify-center gap-1.5 rounded-[6px] px-2.5 py-1.5 text-ink-faint hover:text-ink data-[pressed=true]:text-ink sm:w-[128px] sm:px-0"
            >
              <Icon className="h-3.5 w-3.5 shrink-0" />
              {/* Below sm the labels would collide with the wordmark, so the
                  keys fall back to icons with the name kept for assistive tech. */}
              <span className="hidden sm:inline">{label}</span>
              <span className="sr-only sm:hidden">{label}</span>
            </button>
          ))}
        </nav>

        {/* Hardware Status Tag */}
        <div className="hidden md:flex items-center gap-3 font-mono text-[12px] text-ink-faint">
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: LANE_COLOR.quantum }} />
            Qiskit VQC Active
          </span>
        </div>
      </header>

      {/* Main Tab Surface */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {activeTab === 'benchmark' && <BenchmarkTab />}
        {activeTab === 'train' && <TrainTab onNavigateToPredict={handleNavigateToPredict} />}
        {activeTab === 'predict' && <PredictTab initialDiseaseId={targetDiseaseId} />}
      </main>
    </div>
  )
}
