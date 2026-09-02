import type { Lane, NodeStatus } from './pipeline/types'

/** The entire palette. Four hues, nothing else. */
export const LANE_COLOR: Record<Lane, string> = {
  shared: '#8A8F98',
  classical: '#C08A3E',
  quantum: '#3E8C9E',
}

export const ERROR_COLOR = '#A3543D'

/**
 * Selection green.
 *
 * Used only to mark the active choice, and only on a border. Deliberately not
 * ERROR_COLOR and not the `done` lamp green, so a selected card reads as
 * chosen rather than as finished or failed.
 */
export const SELECT_COLOR = '#4FA86A'

export const LANE_LABEL: Record<Lane, string> = {
  shared: 'shared',
  classical: 'classical',
  quantum: 'quantum',
}

/**
 * Lamp colour per status. `done` reuses the quantum hue lightened rather than
 * introducing a fifth colour; `running` reuses the classical amber.
 */
export const LAMP_COLOR: Record<NodeStatus, string> = {
  idle: '#3A3C42',
  queued: '#4A4136',
  running: '#C08A3E',
  done: '#5FA88C',
  error: '#A3543D',
}

export const STATUS_LABEL: Record<NodeStatus, string> = {
  idle: 'idle',
  queued: 'blocked',
  running: 'running',
  done: 'done',
  error: 'error',
}

export function laneColor(lane: Lane, status: NodeStatus) {
  return status === 'error' ? ERROR_COLOR : LANE_COLOR[lane]
}

/** rgba() from a hex triple, for the low-alpha washes used across the panels. */
export function alpha(hex: string, a: number) {
  const v = hex.replace('#', '')
  const n = parseInt(v, 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`
}

export function formatElapsed(ms: number) {
  const total = Math.floor(ms / 100)
  const tenths = total % 10
  const seconds = Math.floor(total / 10) % 60
  const minutes = Math.floor(total / 600)
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`
}

export function formatClock(ts: number) {
  const d = new Date(ts)
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0'))
    .join(':')
}
