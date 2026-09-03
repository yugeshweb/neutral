import type { Finding } from './findings'

export interface IsoContour {
  level: number
  label: string
  pathD: string
  dash?: string
  opacity: number
}

export interface CaliperMeasurement {
  x1: number
  y1: number
  x2: number
  y2: number
  label: string
}

export interface FindingMorphology {
  angle: number
  axisRatio: number
  sigmaU: number
  sigmaV: number
  h1: number
  h2: number
  h3: number
  p1: number
  p2: number
  p3: number
  lesionPath: string
  isoCurves: IsoContour[]
  calipers: {
    major: CaliperMeasurement
    minor: CaliperMeasurement
  }
}

/** 32-bit FNV-1a hash for deterministic morphology derived from finding attributes */
function hashFinding(f: Finding): number {
  const str = `${f.id}:${f.label}:${f.severity}:${f.x.toFixed(3)}:${f.y.toFixed(3)}`
  let h = 2166136261
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

interface Point {
  x: number
  y: number
}

/**
 * Converts an array of points into a smooth closed SVG cubic Bézier path
 * using Catmull-Rom spline interpolation.
 */
function pointsToClosedBezierPath(points: Point[]): string {
  const n = points.length
  if (n < 3) return ''

  let d = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`

  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n]
    const p1 = points[i]
    const p2 = points[(i + 1) % n]
    const p3 = points[(i + 2) % n]

    const cp1x = p1.x + (p2.x - p0.x) / 6
    const cp1y = p1.y + (p2.y - p0.y) / 6
    const cp2x = p2.x - (p3.x - p1.x) / 6
    const cp2y = p2.y - (p3.y - p1.y) / 6

    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`
  }

  d += ' Z'
  return d
}

// Cache of computed morphologies by finding ID to ensure zero GC pressure during rendering
const morphologyCache = new Map<string, FindingMorphology>()

/**
 * Returns deterministic clinical morphology for a finding.
 *
 * Provides orientation angle, eccentricity, harmonic spiculation factors,
 * closed SVG contour paths, and clinical measurement calipers.
 */
export function getFindingMorphology(f: Finding): FindingMorphology {
  const cacheKey = `${f.id}-${f.x}-${f.y}-${f.r}-${f.severity}`
  const cached = morphologyCache.get(cacheKey)
  if (cached) return cached

  const h = hashFinding(f)

  // Lesion orientation angle in radians (0 to PI)
  const angle = ((h % 180) * Math.PI) / 180

  // Anisotropic eccentricity based on severity
  // Malignant / high severity lesions exhibit stronger directional elongation & spiculation
  let axisRatio = 1.15
  let h1 = 0.05
  let h2 = 0.0
  let h3 = 0.0
  const p1 = (h % 314) / 100
  const p2 = ((h >> 3) % 314) / 100
  const p3 = ((h >> 6) % 314) / 100

  if (f.severity === 'high') {
    // Spiculated / infiltrative carcinoma morphology (BI-RADS 5 / WHO CNS)
    axisRatio = 1.38 + ((h >> 2) % 24) / 100
    h1 = 0.15 // 3-lobed macro-lobulation
    h2 = 0.10 // 5-lobed stellate spiculation
    h3 = 0.06 // 7-lobed micro-infiltrative notches
  } else if (f.severity === 'moderate') {
    // Pleomorphic cluster / microcalcification envelope
    axisRatio = 1.22 + ((h >> 2) % 18) / 100
    h1 = 0.11 // 4-lobed cluster shape
    h2 = 0.07 // 2-lobed indentation
    h3 = 0.02
  } else {
    // Circumscribed / benign or low suspicion
    axisRatio = 1.10 + ((h >> 2) % 12) / 100
    h1 = 0.05
    h2 = 0.02
    h3 = 0.0
  }

  const sigmaU = f.r * 0.92 * axisRatio
  const sigmaV = f.r * 0.92

  // Modulation function for angular margin variation
  const modulation = (psi: number): number => {
    return 1 + h1 * Math.cos(3 * psi + p1) + h2 * Math.cos(5 * psi + p2) + h3 * Math.sin(2 * psi + p3)
  }

  // Generate perimeter points at a given radius scale
  const sampleContourPoints = (scale: number): Point[] => {
    const samples = 36
    const points: Point[] = []

    for (let k = 0; k < samples; k++) {
      const phi = (k / samples) * 2 * Math.PI
      const psi = phi - angle
      // Elliptical base radius
      const rBase = f.r / Math.sqrt(Math.cos(psi) ** 2 + axisRatio ** 2 * Math.sin(psi) ** 2)
      const rad = rBase * scale * modulation(psi)

      points.push({
        x: (f.x + rad * Math.cos(phi)) * 1000,
        y: (f.y + rad * Math.sin(phi)) * 1000,
      })
    }

    return points
  }

  // Primary lesion boundary (at scale 1.0)
  const lesionPoints = sampleContourPoints(1.0)
  const lesionPath = pointsToClosedBezierPath(lesionPoints)

  // Multi-level Iso-activation contours (Grad-CAM topographic elevation curves)
  // 85% core, 60% diagnostic margin, 40% infiltrative margin, 20% penumbra
  const isoCurves: IsoContour[] = [
    {
      level: 0.85,
      label: '85% Core',
      pathD: pointsToClosedBezierPath(sampleContourPoints(0.48)),
      opacity: 0.95,
    },
    {
      level: 0.60,
      label: '60% Margin',
      pathD: pointsToClosedBezierPath(sampleContourPoints(0.82)),
      opacity: 0.75,
    },
    {
      level: 0.40,
      label: '40% Infiltrative',
      pathD: pointsToClosedBezierPath(sampleContourPoints(1.15)),
      dash: '4 3',
      opacity: 0.55,
    },
    {
      level: 0.20,
      label: '20% Penumbra',
      pathD: pointsToClosedBezierPath(sampleContourPoints(1.48)),
      dash: '2 3',
      opacity: 0.35,
    },
  ]

  // Dimension measurement calipers along primary axes
  const majorRad = f.r * axisRatio * modulation(0) * 1.05
  const minorRad = f.r * modulation(Math.PI / 2) * 1.05

  const majorLengthMm = (2 * f.r * 155 * axisRatio).toFixed(1)
  const minorLengthMm = (2 * f.r * 155).toFixed(1)

  const calipers = {
    major: {
      x1: (f.x - majorRad * Math.cos(angle)) * 1000,
      y1: (f.y - majorRad * Math.sin(angle)) * 1000,
      x2: (f.x + majorRad * Math.cos(angle)) * 1000,
      y2: (f.y + majorRad * Math.sin(angle)) * 1000,
      label: `${majorLengthMm} mm`,
    },
    minor: {
      x1: (f.x - minorRad * Math.cos(angle + Math.PI / 2)) * 1000,
      y1: (f.y - minorRad * Math.sin(angle + Math.PI / 2)) * 1000,
      x2: (f.x + minorRad * Math.cos(angle + Math.PI / 2)) * 1000,
      y2: (f.y + minorRad * Math.sin(angle + Math.PI / 2)) * 1000,
      label: `${minorLengthMm} mm`,
    },
  }

  const result: FindingMorphology = {
    angle,
    axisRatio,
    sigmaU,
    sigmaV,
    h1,
    h2,
    h3,
    p1,
    p2,
    p3,
    lesionPath,
    isoCurves,
    calipers,
  }

  morphologyCache.set(cacheKey, result)
  return result
}
