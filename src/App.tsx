import { useEffect, useState, type ReactElement } from 'react'
import { EhrValidationView } from './components/EhrValidationView'
import { HomeScreen } from './components/HomeScreen'
import { BenchmarkTab } from './components/tabs/BenchmarkTab'
import { TrainTab } from './components/tabs/TrainTab'
import { PredictTab } from './components/tabs/PredictTab'
import { Wordmark } from './components/Wordmark'
import { IconBars, IconFlask, IconPulse } from './components/icons'

export type TabId = 'train' | 'predict' | 'benchmark'

/** The opening screen. Not a tab: it has no key in the nav. */
type Route = TabId | 'home'

const TABS: { id: TabId; label: string; Icon: (p: { className?: string }) => ReactElement }[] = [
  { id: 'train', label: '1. Train', Icon: IconFlask },
  { id: 'predict', label: '2. Predict', Icon: IconPulse },
  { id: 'benchmark', label: '3. Benchmark', Icon: IconBars },
]

const TAB_IDS = TABS.map((t) => t.id)

/**
 * Routing, without a router.
 *
 * Each tab owns a path, so a screen can be linked to and the back button
 * works. The app is four views deep with no nested or parameterised routes,
 * which is not enough to justify pulling in a routing library - `pushState`
 * plus a `popstate` listener covers it.
 *
 * "/" is the opening screen, and anything unrecognised resolves to it rather
 * than 404ing, since every path under this origin is served the same SPA entry
 * point.
 */
function routeFromPath(pathname: string): Route {
  const seg = pathname.replace(/^\/+|\/+$/g, '')
  return (TAB_IDS as string[]).includes(seg) ? (seg as TabId) : 'home'
}

function pathFor(route: Route): string {
  return route === 'home' ? '/' : `/${route}`
}

export default function App() {
  const [route, setRoute] = useState<Route>(() =>
    typeof window === 'undefined' ? 'home' : routeFromPath(window.location.pathname),
  )
  const [targetDiseaseId, setTargetDiseaseId] = useState<string>('breast-cancer')

  // Back and forward move between screens rather than leaving the app.
  useEffect(() => {
    const onPop = () => setRoute(routeFromPath(window.location.pathname))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  /*
   * Normalise the address bar on first load, so an unrecognised path shows "/"
   * rather than leaving the URL disagreeing with the view. `replaceState`
   * rather than `pushState`: this is a correction, not a navigation, and should
   * not add a history entry the back button has to step through.
   */
  useEffect(() => {
    const path = pathFor(route)
    if (window.location.pathname !== path) {
      window.history.replaceState(null, '', path)
    }
    // Deliberately first-render only; later changes go through `go`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const go = (id: Route) => {
    setRoute(id)
    const path = pathFor(id)
    if (window.location.pathname !== path) {
      window.history.pushState(null, '', path)
    }
  }

  // Standalone route for validation if requested directly
  if (typeof window !== 'undefined' && window.location.pathname === '/ehr-validation') {
    return <EhrValidationView />
  }

  const handleNavigateToPredict = (diseaseId: string) => {
    setTargetDiseaseId(diseaseId)
    go('predict')
  }

  return (
    // `w-full`, not `w-screen`: 100vw includes the vertical scrollbar's width,
    // so on any platform that reserves space for one the shell ends up wider
    // than the usable viewport and the page scrolls sideways by that amount.
    <div className="flex h-screen w-full flex-col overflow-hidden bg-canvas text-ink">
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
          {/* The wordmark is the way back to the opening screen, which is the
              convention everywhere else and saves spending a nav key on it. */}
          <a
            href="/"
            aria-label="Home"
            onClick={(e) => {
              if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
              e.preventDefault()
              go('home')
            }}
            className="flex cursor-pointer items-center no-underline"
          >
            <Wordmark size={18} />
          </a>
        </div>

        {/* Segmented control: the group is a plain track, so the keys inside
            supply the only borders and nothing double-lines. */}
        <nav className="absolute left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-[8px] bg-black/25 p-1 font-mono text-[13px]">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => go(id)}
              data-pressed={route === id}
              aria-current={route === id ? 'page' : undefined}
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
      </header>

      {/* Main Tab Surface */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {route === 'home' && <HomeScreen onOpen={go} />}
        {route === 'benchmark' && <BenchmarkTab />}
        {route === 'train' && <TrainTab onNavigateToPredict={handleNavigateToPredict} />}
        {route === 'predict' && <PredictTab initialDiseaseId={targetDiseaseId} />}
      </main>
    </div>
  )
}
