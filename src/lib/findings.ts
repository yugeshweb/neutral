/**
 * Mock region-of-interest findings for an uploaded image.
 *
 * These are NOT produced by any detector. Coordinates are generated from a
 * hash of the file name so a given image always yields the same markers -
 * stable enough to demo, obviously synthetic on inspection.
 *
 * TODO: when a real segmentation model exists, replace `deriveFindings` with a
 * call that returns the same `Finding[]` shape. The overlay component reads
 * only this interface, so nothing in the UI needs to change.
 */

export type Finding = {
  id: string
  /** centre position as a fraction of image width/height, 0..1 */
  x: number
  y: number
  /** marker radius as a fraction of the smaller image edge */
  r: number
  label: string
  severity: 'low' | 'moderate' | 'high'
  /** placeholder confidence, 0..1 */
  confidence: number
  notes: string[]
  metrics: Record<string, string>
}

/** Deterministic 32-bit hash, so the same file always yields the same markers. */
function hash(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function rng(seed: number) {
  let s = seed || 1
  return () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) % 10000) / 10000
  }
}

type LabelSpec = { label: string; severity: Finding['severity'] }

/**
 * Marker vocabularies per condition.
 *
 * A mammographic term like "clustered microcalcification" is meaningless over
 * an EEG trace, so each condition has its own set. The markers are still
 * placed from a filename hash rather than detected; this only keeps the words
 * on screen plausible for the image being shown.
 */
const LABELS_BY_CONDITION: Record<string, LabelSpec[]> = {
  'breast-cancer': [
    { label: 'Irregular mass margin', severity: 'high' },
    { label: 'Clustered microcalcification', severity: 'moderate' },
    { label: 'Focal asymmetry', severity: 'moderate' },
    { label: 'Architectural distortion', severity: 'low' },
  ],
  // Presented as brain tumour, so the vocabulary is lesion terms rather than
  // the EEG waveform terms this entry used when it was labelled seizure.
  'brain-seizure': [
    { label: 'Enhancing lesion core', severity: 'high' },
    { label: 'Peritumoral edema', severity: 'moderate' },
    { label: 'Midline shift', severity: 'moderate' },
    { label: 'Focal signal abnormality', severity: 'low' },
  ],
  'heart-disease': [
    { label: 'ST-segment depression', severity: 'high' },
    { label: 'T-wave inversion', severity: 'moderate' },
    { label: 'Q-wave abnormality', severity: 'moderate' },
    { label: 'Baseline wander', severity: 'low' },
  ],
  alzheimers: [
    { label: 'Hippocampal atrophy', severity: 'high' },
    { label: 'Ventricular enlargement', severity: 'moderate' },
    { label: 'Cortical thinning', severity: 'moderate' },
    { label: 'White-matter hyperintensity', severity: 'low' },
  ],
}

const LABELS = LABELS_BY_CONDITION['breast-cancer']

const NOTES: Record<string, string[]> = {
  'Irregular mass margin': [
    'spiculated boundary, low circularity',
    'texture entropy above local baseline',
    'flagged for radiologist review',
  ],
  'Clustered microcalcification': [
    'high-frequency cluster, 6 points',
    'mean spacing below 1.2 mm',
    'morphology heterogeneous',
  ],
  'Focal asymmetry': [
    'density differs from contralateral region',
    'no discrete mass boundary detected',
  ],
  'Architectural distortion': [
    'radiating pattern without central mass',
    'low confidence, benign mimics common',
  ],
}

export function deriveFindings(fileName: string, conditionId?: string): Finding[] {
  const next = rng(hash(fileName))
  const count = 2 + Math.floor(next() * 2) // 2 or 3 markers
  const labels = (conditionId && LABELS_BY_CONDITION[conditionId]) || LABELS

  const out: Finding[] = []
  for (let i = 0; i < count; i++) {
    const spec = labels[Math.floor(next() * labels.length)] ?? labels[0]
    // Bias toward the centre: on a scan the subject occupies the middle, so
    // markers placed at the edges would visibly sit on empty background.
    const angle = next() * Math.PI * 2
    const radius = 0.08 + next() * 0.22
    out.push({
      id: `roi-${i + 1}`,
      x: 0.5 + Math.cos(angle) * radius,
      y: 0.5 + Math.sin(angle) * radius * 0.9,
      r: 0.06 + next() * 0.04,
      label: spec.label,
      severity: spec.severity,
      confidence: Number((0.62 + next() * 0.33).toFixed(2)),
      // Only the mammographic labels carry notes; the others show none.
      notes: NOTES[spec.label] ?? [],
      metrics: {
        area_px: String(400 + Math.floor(next() * 2600)),
        eccentricity: (0.3 + next() * 0.6).toFixed(2),
        mean_intensity: (80 + next() * 140).toFixed(1),
      },
    })
  }
  return out
}

export const SEVERITY_COLOR: Record<Finding['severity'], string> = {
  low: '#8A8F98',
  moderate: '#C08A3E',
  high: '#A3543D',
}
