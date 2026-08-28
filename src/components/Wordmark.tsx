import { LANE_COLOR } from '../lib/theme'

/**
 * The Netural wordmark.
 *
 * Lettering only: wide tracking, and a quantum-to-classical gradient running
 * left to right in the same two hues the pipeline lanes use - so the logo
 * states what the platform does rather than decorating it.
 *
 * Inter throughout; no display face is loaded for one word. `size` scales the
 * whole lockup from a single number.
 */
export function Wordmark({ size = 38 }: { size?: number }) {
  return (
    <span
      className="inline-block font-semibold"
      style={{
        fontSize: size,
        lineHeight: 1,
        letterSpacing: '0.18em',
        // Trailing tracking pushes the block left; nudge it back to centre.
        paddingLeft: '0.18em',
        backgroundImage: `linear-gradient(100deg, ${LANE_COLOR.quantum} 0%, #E8E9EB 48%, ${LANE_COLOR.classical} 100%)`,
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        WebkitTextFillColor: 'transparent',
      }}
    >
      NETURAL
    </span>
  )
}
