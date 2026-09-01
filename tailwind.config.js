/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#0B0B0D',
        panel: { DEFAULT: '#17181B', deep: '#111214', well: '#0D0E10' },
        line: 'rgba(255,255,255,0.06)',
        ink: { DEFAULT: '#E8E9EB', dim: '#9A9CA1', faint: '#6A6C72' },
        lane: {
          neutral: '#8A8F98',
          classical: '#C08A3E',
          quantum: '#3E8C9E',
          error: '#A3543D',
        },
      },
      fontFamily: {
        // Geist for prose and labels; JetBrains Mono for anything numeric, so
        // metrics stay column-aligned without the text face going technical.
        sans: ['Geist', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderRadius: { panel: '8px' },
    },
  },
  plugins: [],
}
