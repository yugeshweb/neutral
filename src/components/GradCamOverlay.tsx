import { useEffect, useRef } from 'react'
import type { Finding } from '../lib/findings'

interface Props {
  findings: Finding[]
  matrix?: number[][]
  opacity?: number
  threshold?: number
  colormap?: 'jet' | 'turbo' | 'inferno'
  className?: string
  width?: number
  height?: number
  onHoverActivation?: (val: number | null) => void
}

/**
 * Converts a normalized saliency score [0..1] to an RGBA tuple using
 * standard radiological Jet/Turbo color mapping.
 */
function saliencyToRgba(
  val: number,
  opacity: number,
  threshold: number,
  colormap: 'jet' | 'turbo' | 'inferno' = 'jet'
): [number, number, number, number] {
  if (val < threshold) {
    return [0, 0, 0, 0]
  }

  // Normalized above threshold
  const t = Math.min(1, Math.max(0, (val - threshold) / (1 - threshold)))
  const alpha = Math.floor(opacity * (0.3 + 0.7 * Math.pow(t, 0.8)) * 255)

  if (colormap === 'inferno') {
    // Inferno-like medical heat ramp: black -> purple -> orange -> yellow
    const r = Math.floor(Math.min(255, t * 290))
    const g = Math.floor(Math.max(0, Math.min(255, (t - 0.3) * 350)))
    const b = Math.floor(Math.max(0, Math.min(255, (t < 0.5 ? t * 2 : 1 - (t - 0.5) * 2) * 200)))
    return [r, g, b, alpha]
  }

  // Standard Jet colormap: Deep Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
  let r = 0
  let g = 0
  let b = 0

  if (t < 0.125) {
    r = 0
    g = 0
    b = Math.floor(128 + t * 8 * 127)
  } else if (t < 0.375) {
    const f = (t - 0.125) / 0.25
    r = 0
    g = Math.floor(f * 255)
    b = 255
  } else if (t < 0.625) {
    const f = (t - 0.375) / 0.25
    r = Math.floor(f * 255)
    g = 255
    b = Math.floor((1 - f) * 255)
  } else if (t < 0.875) {
    const f = (t - 0.625) / 0.25
    r = 255
    g = Math.floor((1 - f * 0.5) * 255)
    b = 0
  } else {
    const f = (t - 0.875) / 0.125
    r = Math.floor(255 - f * 40)
    g = Math.floor((1 - f) * 128)
    b = 0
  }

  return [r, g, b, alpha]
}

/**
 * Grad-CAM Saliency Map Overlay
 *
 * Renders an authentic gradient-weighted class activation heatmap over medical scans.
 * When an actual model API returns a 2D activation matrix, it renders that directly via
 * bilinear upsampling. In demo mode, it synthesizes the Grad-CAM field using multi-scale
 * Gaussian kernel density centered on localized findings.
 */
export function GradCamOverlay({
  findings,
  matrix,
  opacity = 0.65,
  threshold = 0.12,
  colormap = 'jet',
  className = '',
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    if (width === 0 || height === 0) return

    // Offscreen buffer at lower resolution for smooth bilinear interpolation
    const gridW = matrix ? matrix[0]?.length ?? 28 : 36
    const gridH = matrix ? matrix.length : 36

    const offscreen = document.createElement('canvas')
    offscreen.width = gridW
    offscreen.height = gridH
    const offCtx = offscreen.getContext('2d')
    if (!offCtx) return

    const imgData = offCtx.createImageData(gridW, gridH)
    const data = imgData.data

    // Build raw activation grid
    const rawGrid: number[][] = []

    if (matrix && matrix.length > 0) {
      // Use genuine model-provided 2D Grad-CAM matrix
      for (let y = 0; y < gridH; y++) {
        rawGrid[y] = []
        for (let x = 0; x < gridW; x++) {
          rawGrid[y][x] = matrix[y]?.[x] ?? 0
        }
      }
    } else {
      // Synthesize realistic Grad-CAM activation field based on findings
      for (let y = 0; y < gridH; y++) {
        rawGrid[y] = []
        const ny = y / (gridH - 1)
        for (let x = 0; x < gridW; x++) {
          const nx = x / (gridW - 1)
          let sumActivation = 0

          for (const f of findings) {
            const dx = nx - f.x
            const dy = ny - f.y
            const distSq = dx * dx + dy * dy
            // Spread is proportional to marker radius
            const sigma = f.r * 0.95
            const weight = f.severity === 'high' ? 1.0 : f.severity === 'moderate' ? 0.78 : 0.5
            const gaussian = Math.exp(-distSq / (2 * sigma * sigma))
            sumActivation += gaussian * weight * f.confidence
          }

          // Subtle background baseline noise
          const ambient = 0.03 * Math.sin(nx * 5) * Math.cos(ny * 5)
          rawGrid[y][x] = Math.max(0, Math.min(1, sumActivation + ambient))
        }
      }
    }

    // Paint into offscreen image data
    let idx = 0
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const val = rawGrid[y][x]
        const [r, g, b, a] = saliencyToRgba(val, opacity, threshold, colormap)
        data[idx] = r
        data[idx + 1] = g
        data[idx + 2] = b
        data[idx + 3] = a
        idx += 4
      }
    }

    offCtx.putImageData(imgData, 0, 0)

    // Render scaled up onto main canvas with bilinear smoothing
    ctx.clearRect(0, 0, width, height)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(offscreen, 0, 0, width, height)
  }, [findings, matrix, opacity, threshold, colormap])

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={480}
      className={`pointer-events-none absolute inset-0 h-full w-full ${className}`}
      style={{
        mixBlendMode: 'screen',
      }}
    />
  )
}
