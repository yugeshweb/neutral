import { splitRow, type DatasetSummary } from '../dataset'
import type { Dataset } from './pipeline'

/**
 * Turns an uploaded file's parsed summary into a trainable Dataset.
 *
 * The ingest adapters (CSV, FHIR, HL7 v2) all converge on the same
 * DatasetSummary shape with a canonical CSV in `content` - this is the one
 * place that shape becomes the numeric matrix the pipeline actually trains
 * on. Everything upstream of this function is format-specific; everything
 * downstream never learns which format the data came from.
 */

export type ColumnPreview = {
  name: string
  /** fraction of sampled values that parse as a finite number, 0..1 */
  numericFraction: number
  distinctValues: number
  sample: string[]
}

export class CustomDatasetError extends Error {}

/**
 * True when a column's values are consecutive integers (1,2,3,...) or a
 * shared text prefix plus consecutive integers (p1,p2,p3,...) - the two
 * shapes a spreadsheet's own row number or a generated patient id actually
 * takes. This is a much more specific signal than "mostly unique values",
 * which any continuous clinical measurement satisfies too.
 */
function isSequential(values: string[]): boolean {
  const numbers = values.map((v) => {
    const m = /^\D*?(\d+)$/.exec(v.trim())
    return m ? Number(m[1]) : NaN
  })
  if (numbers.some(Number.isNaN)) return false

  const sorted = [...numbers].sort((a, b) => a - b)
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] !== sorted[i - 1] + 1) return false
  }
  return true
}

/** Previews every column so the caller can offer a sensible label-column default. */
export function previewColumns(summary: DatasetSummary): ColumnPreview[] {
  if (!summary.content) {
    throw new CustomDatasetError('this file has no parsed content to preview')
  }
  const lines = summary.content.split('\n').filter((l) => l.length > 0)
  const rows = lines.slice(1).map(splitRow)

  return summary.headers.map((name, j) => {
    const values = rows.map((r) => r[j] ?? '')
    const numeric = values.filter((v) => v !== '' && Number.isFinite(Number(v)))
    return {
      name,
      numericFraction: values.length ? numeric.length / values.length : 0,
      distinctValues: new Set(values).size,
      sample: values.slice(0, 3),
    }
  })
}

/**
 * A column is a plausible label if it takes exactly two distinct values -
 * the whole pipeline (metrics, McNemar, the VQC's single output qubit) is
 * built for binary classification, so anything else cannot be trained here.
 */
export function suggestLabelColumn(columns: ColumnPreview[]): string | null {
  const binary = columns.filter((c) => c.distinctValues === 2)
  if (binary.length === 0) return null
  // Prefer a name that reads as a label over an incidentally-binary feature.
  const named = binary.find((c) => /label|target|class|diagnos|outcome|status|stroke|disease/i.test(c.name))
  return (named ?? binary[binary.length - 1]).name
}

export type ConvertedUpload = {
  dataset: Dataset
  /** column name -> why it was excluded, so the choice is not a black box */
  droppedColumns: { name: string; reason: string }[]
}

export function convertUpload(
  summary: DatasetSummary,
  labelColumn: string,
  positiveLabel?: string,
  negativeLabel?: string,
): ConvertedUpload {
  if (!summary.content) {
    throw new CustomDatasetError('this file has no parsed content to train on')
  }

  const lines = summary.content.split('\n').filter((l) => l.length > 0)
  if (lines.length < 2) {
    throw new CustomDatasetError('file has a header but no data rows')
  }
  const rows = lines.slice(1).map(splitRow)

  const labelIdx = summary.headers.indexOf(labelColumn)
  if (labelIdx < 0) {
    throw new CustomDatasetError(`column "${labelColumn}" not found`)
  }

  const rawLabels = rows.map((r) => r[labelIdx] ?? '')
  const distinctLabels = [...new Set(rawLabels)].filter((v) => v !== '')
  if (distinctLabels.length !== 2) {
    throw new CustomDatasetError(
      `the label column must have exactly two distinct values to train a classifier, found ${distinctLabels.length} (${distinctLabels.slice(0, 5).join(', ')}${distinctLabels.length > 5 ? '…' : ''})`,
    )
  }
  // Numeric-looking labels sort so "1" reads as positive, matching every
  // built-in dataset's convention; otherwise keep first-seen order.
  const sorted = distinctLabels.every((v) => Number.isFinite(Number(v)))
    ? [...distinctLabels].sort((a, b) => Number(a) - Number(b))
    : distinctLabels
  const posValue = sorted[1]
  const y = rawLabels.map((v) => (v === posValue ? 1 : 0))

  // Every other numeric-enough column becomes a feature. Non-numeric or
  // entirely-empty columns are dropped rather than guessed at - a silently
  // invented feature would be worse than a smaller feature set.
  //
  // A column is also dropped when it looks like a row identifier - by name
  // (patient_id, MRN, ...) or by being a sequential index (1,2,3,... or
  // p1,p2,p3,...). Training on row identity teaches nothing generalisable,
  // and if it happens to correlate with row order it can look spuriously
  // predictive. Uniqueness alone is NOT the signal: a genuinely continuous
  // measurement (age, blood pressure) is often almost all-distinct in a small
  // uploaded sample too, and dropping those would gut the feature set for
  // exactly the datasets this matters most for.
  const candidateIdx = summary.headers
    .map((_, j) => j)
    .filter((j) => j !== labelIdx)

  const droppedColumns: { name: string; reason: string }[] = []
  const featureIdx: number[] = []
  for (const j of candidateIdx) {
    const values = rows.map((r) => r[j] ?? '')
    const numeric = values.filter((v) => v !== '' && Number.isFinite(Number(v)))
    const distinct = new Set(values).size
    const nameLooksLikeId = /^(patient[_ ]?id|mrn|record[_ ]?(id|number)|row[_ ]?(id|number)|id)$/i.test(
      summary.headers[j],
    )
    const isSequentialIndex = distinct === rows.length && isSequential(values)

    if (nameLooksLikeId || isSequentialIndex) {
      droppedColumns.push({ name: summary.headers[j], reason: 'looks like a row identifier' })
    } else if (numeric.length >= values.length * 0.5) {
      featureIdx.push(j)
    } else {
      droppedColumns.push({ name: summary.headers[j], reason: 'not numeric' })
    }
  }

  if (featureIdx.length === 0) {
    throw new CustomDatasetError(
      'no numeric feature columns found - every column besides the label was non-numeric or empty',
    )
  }

  // Column-mean imputation for the numeric columns' own gaps, same policy
  // the built-in datasets' preprocessing step offers as a default.
  const means = featureIdx.map((j) => {
    const nums = rows.map((r) => Number(r[j])).filter(Number.isFinite)
    return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0
  })

  const X = rows.map((r) =>
    featureIdx.map((j, k) => {
      const v = Number(r[j])
      return Number.isFinite(v) ? v : means[k]
    }),
  )

  return {
    dataset: {
      id: 'custom',
      name: summary.name,
      featureNames: featureIdx.map((j) => summary.headers[j]),
      X,
      y,
      positiveLabel: positiveLabel?.trim() || `${labelColumn} = ${posValue}`,
      negativeLabel: negativeLabel?.trim() || `${labelColumn} = ${sorted[0]}`,
    },
    droppedColumns,
  }
}
