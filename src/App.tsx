import { useState } from 'react'
import { EhrValidationView } from './components/EhrValidationView'
import { BenchmarkTab } from './components/tabs/BenchmarkTab'
import { TrainTab } from './components/tabs/TrainTab'
import { PredictTab } from './components/tabs/PredictTab'
import { Wordmark } from './components/Wordmark'
import { LANE_COLOR, alpha } from './lib/theme'
import { IconBars, IconFlask, IconPulse } from './components/icons'

export type TabId = 'benchmark' | 'train' | 'predict'

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
      <header
        className="flex h-14 shrink-0 items-center justify-between px-6 z-30"
        style={{
          background: '#111214',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
        }}
      >
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Wordmark size={18} />
            <span className="hidden sm:inline-block h-3.5 w-px bg-white/10" />
            <span className="hidden sm:inline-block font-mono text-[10px] text-ink-faint">
              Hybrid Quantum-Classical Disease Detection Platform
            </span>
          </div>
        </div>

        {/* The 3 Core Platform Tabs in exact required order */}
        <nav className="flex items-center gap-1 rounded-[8px] bg-[#17181B] p-1 border border-white/5 font-mono text-[11px]">
          <button
            type="button"
            onClick={() => setActiveTab('benchmark')}
            className="flex items-center gap-1.5 cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all"
            style={{
              background: activeTab === 'benchmark' ? 'rgba(255,255,255,0.1)' : 'transparent',
              color: activeTab === 'benchmark' ? '#E8E9EB' : '#9A9CA1',
              fontWeight: activeTab === 'benchmark' ? 500 : 400,
              boxShadow: activeTab === 'benchmark' ? '0 1px 4px rgba(0,0,0,0.5)' : 'none',
            }}
          >
            <IconBars className="h-3.5 w-3.5" />
            <span>1. Benchmark</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('train')}
            className="flex items-center gap-1.5 cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all"
            style={{
              background: activeTab === 'train' ? alpha(LANE_COLOR.classical, 0.18) : 'transparent',
              color: activeTab === 'train' ? '#E8E9EB' : '#9A9CA1',
              fontWeight: activeTab === 'train' ? 500 : 400,
              boxShadow: activeTab === 'train' ? '0 1px 4px rgba(0,0,0,0.5)' : 'none',
            }}
          >
            <IconFlask className="h-3.5 w-3.5" />
            <span>2. Train</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('predict')}
            className="flex items-center gap-1.5 cursor-pointer rounded-[6px] px-3.5 py-1.5 transition-all"
            style={{
              background: activeTab === 'predict' ? alpha(LANE_COLOR.quantum, 0.22) : 'transparent',
              color: activeTab === 'predict' ? '#E8E9EB' : '#9A9CA1',
              fontWeight: activeTab === 'predict' ? 500 : 400,
              boxShadow: activeTab === 'predict' ? '0 1px 4px rgba(0,0,0,0.5)' : 'none',
            }}
          >
            <IconPulse className="h-3.5 w-3.5" />
            <span>3. Predict</span>
          </button>
        </nav>

        {/* Hardware Status Tag */}
        <div className="hidden md:flex items-center gap-3 font-mono text-[10px] text-ink-faint">
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
