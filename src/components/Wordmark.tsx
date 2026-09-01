import { LANE_COLOR } from '../lib/theme'

/**
 * The Neutral wordmark.
 *
 * Lettering only: set lowercase, with a quantum-to-classical gradient running
 * left to right in the same two hues the pipeline lanes use - so the logo
 * states what the platform does rather than decorating it.
 *
 * Geist throughout; no display face is loaded for one word. `size` scales the
 * whole lockup from a single number.
 */
export function Wordmark({ size = 38 }: { size?: number }) {
  return (
    <span
      className="inline-block font-semibold"
      style={{
        fontSize: size,
        lineHeight: 1,
        // Set lowercase, so the tracking is tightened: the wide letter-spacing
        // a capitalised lockup needs reads as gappy on lowercase letterforms.
        letterSpacing: '-0.01em',
        backgroundImage: `linear-gradient(100deg, ${LANE_COLOR.quantum} 0%, #E8E9EB 48%, ${LANE_COLOR.classical} 100%)`,
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        WebkitTextFillColor: 'transparent',
      }}
    >
      Neutral
    </span>
  )
}
