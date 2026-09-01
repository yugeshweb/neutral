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
        className="flex h-16 shrink-0 items-center justify-between px-6 lg:px-10 z-30"
        style={{
          background: '#111214',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
        }}
      >
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <Wordmark size={20} />
          <span className="hidden md:inline-block h-4 w-px bg-white/10" />
          <span className="hidden md:inline-block font-mono text-[11px] text-ink-faint">
            Hybrid Quantum-Classical Disease Detection
          </span>
        </div>

        {/* Centered Navigation Tabs */}
        <nav className="flex items-center justify-center gap-2 rounded-[10px] bg-[#17181B] p-1.5 border border-white/8 shadow-inner font-mono text-[13px]">
          <button
            type="button"
            onClick={() => setActiveTab('benchmark')}
            className="flex items-center gap-2 cursor-pointer rounded-[7px] px-5 py-2 transition-all duration-150"
            style={{
              background: activeTab === 'benchmark' ? 'rgba(255,255,255,0.12)' : 'transparent',
              color: activeTab === 'benchmark' ? '#FFFFFF' : '#9A9CA1',
              fontWeight: activeTab === 'benchmark' ? 600 : 400,
              boxShadow: activeTab === 'benchmark' ? '0 2px 8px rgba(0,0,0,0.6)' : 'none',
            }}
          >
            <IconBars className="h-4 w-4" />
            <span>1. Benchmark</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('train')}
            className="flex items-center gap-2 cursor-pointer rounded-[7px] px-5 py-2 transition-all duration-150"
            style={{
              background: activeTab === 'train' ? alpha(LANE_COLOR.classical, 0.22) : 'transparent',
              color: activeTab === 'train' ? '#FFFFFF' : '#9A9CA1',
              fontWeight: activeTab === 'train' ? 600 : 400,
              boxShadow: activeTab === 'train' ? '0 2px 8px rgba(0,0,0,0.6)' : 'none',
            }}
          >
            <IconFlask className="h-4 w-4" />
            <span>2. Train</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('predict')}
            className="flex items-center gap-2 cursor-pointer rounded-[7px] px-5 py-2 transition-all duration-150"
            style={{
              background: activeTab === 'predict' ? alpha(LANE_COLOR.quantum, 0.28) : 'transparent',
              color: activeTab === 'predict' ? '#FFFFFF' : '#9A9CA1',
              fontWeight: activeTab === 'predict' ? 600 : 400,
              boxShadow: activeTab === 'predict' ? '0 2px 8px rgba(0,0,0,0.6)' : 'none',
            }}
          >
            <IconPulse className="h-4 w-4" />
            <span>3. Predict</span>
          </button>
        </nav>

        {/* Right Status Badge */}
        <div className="hidden sm:flex items-center gap-2.5 font-mono text-[11px] text-ink-dim rounded-[6px] bg-white/5 px-3 py-1.5 border border-white/5">
          <span className="h-2 w-2 rounded-full" style={{ background: LANE_COLOR.quantum }} />
          <span>Qiskit VQC Active</span>
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
