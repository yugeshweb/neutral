import { useEffect, useState } from 'react'

/**
 * Tracks prefers-reduced-motion so JS-driven motion (the SVG travelling dot,
 * which CSS cannot suppress) can be dropped alongside the CSS animations.
 */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  return reduced
}
