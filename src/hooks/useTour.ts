import { useCallback, useEffect, useState } from 'react'

export type TourStop = {
  /** matches a data-tour="..." attribute on the element to spotlight */
  target: string
  title: string
  body: string
  /** where the callout sits relative to the spotlighted element */
  placement?: 'top' | 'bottom'
}

const STORAGE_PREFIX = 'netural.tour.seen.'

function hasSeen(id: string): boolean {
  try {
    return localStorage.getItem(STORAGE_PREFIX + id) === '1'
  } catch {
    // Private browsing or storage disabled - treat every visit as first.
    return false
  }
}

function markSeen(id: string) {
  try {
    localStorage.setItem(STORAGE_PREFIX + id, '1')
  } catch {
    /* nothing to persist to, tour just replays next time - not worth surfacing */
  }
}

/**
 * Drives one spotlight tour. `autoStartOnce` fires it the first time this id
 * is seen in this browser (tracked in localStorage) and never again
 * automatically after that - `start()` stays available for a manual replay
 * regardless, e.g. from a "replay tour" control.
 */
export function useTour(id: string, stops: TourStop[], autoStartOnce = true) {
  const [index, setIndex] = useState(0)
  const [active, setActive] = useState(false)

  useEffect(() => {
    if (autoStartOnce && !hasSeen(id)) {
      setIndex(0)
      setActive(true)
    }
    // Only ever auto-fires once per id, on mount - not on every autoStartOnce
    // or id identity change, which would replay on unrelated re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = useCallback(() => {
    setIndex(0)
    setActive(true)
  }, [])

  const close = useCallback(() => {
    setActive(false)
    markSeen(id)
  }, [id])

  const next = useCallback(() => {
    setIndex((i) => {
      if (i + 1 >= stops.length) {
        setActive(false)
        markSeen(id)
        return i
      }
      return i + 1
    })
  }, [id, stops.length])

  const prev = useCallback(() => {
    setIndex((i) => Math.max(0, i - 1))
  }, [])

  return {
    active,
    index,
    stop: stops[index],
    total: stops.length,
    isLast: index === stops.length - 1,
    isFirst: index === 0,
    start,
    close,
    next,
    prev,
  }
}
