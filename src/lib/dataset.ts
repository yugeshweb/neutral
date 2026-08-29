/**
 * Real CSV parsing for uploaded files. The summary is genuine - row and column
 * counts come from the actual file - and the original text is retained so the
 * local QML service can train on the same file after the user confirms it.
 */

export type UploadKind = 'csv' | 'image'

export type DatasetSummary = {
  kind: UploadKind
  name: string
  sizeBytes: number
  rows: number
  columns: number
  headers: string[]
  /** first few parsed rows, for the preview table */
  preview: string[][]
  /** non-fatal problems worth surfacing to the user */
  warnings: string[]
  /** original CSV text; retained only for the local training request */
  content: string | null
  /** object URL for image uploads; null for CSV. Revoke when replaced. */
  objectUrl: string | null
  /** natural pixel dimensions, images only */
  imageSize: { w: number; h: number } | null
}

export const IMAGE_TYPES = /\.(png|jpe?g|webp|bmp)$/i

export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024

/** Split one CSV line, honouring double-quoted fields containing commas. */
function splitRow(line: string): string[] {
  const out: string[] = []
  let field = ''
  let quoted = false

  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          field += '"'
          i++
        } else {
          quoted = false
        }
      } else {
        field += c
      }
    } else if (c === '"') {
      quoted = true
    } else if (c === ',') {
      out.push(field)
      field = ''
    } else {
      field += c
    }
  }
  out.push(field)
  return out.map((f) => f.trim())
}

export function parseCsv(text: string, name: string, sizeBytes: number): DatasetSummary {
  const warnings: string[] = []

  const lines = text
    .split(/\r\n|\n|\r/)
    .filter((l) => l.trim().length > 0)

  if (lines.length === 0) {
    throw new Error('file is empty')
  }

  const headers = splitRow(lines[0])
  if (headers.length < 2) {
    throw new Error('expected a comma-separated header row with 2 or more columns')
  }

  const body = lines.slice(1)
  if (body.length === 0) {
    throw new Error('file has a header but no data rows')
  }

  // Flag ragged rows rather than rejecting: real exports are often imperfect.
  const ragged = body.filter((l) => splitRow(l).length !== headers.length).length
  if (ragged > 0) {
    warnings.push(`${ragged} row(s) do not match the header column count`)
  }

  const blankHeaders = headers.filter((h) => h.length === 0).length
  if (blankHeaders > 0) {
    warnings.push(`${blankHeaders} column(s) have an empty header`)
  }

  return {
    kind: 'csv',
    name,
    sizeBytes,
    rows: body.length,
    columns: headers.length,
    headers,
    preview: body.slice(0, 4).map(splitRow),
    warnings,
    content: text,
    objectUrl: null,
    imageSize: null,
  }
}

/** Loads an image upload and reads its true pixel dimensions. */
export function loadImage(file: File): Promise<DatasetSummary> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      resolve({
        kind: 'image',
        name: file.name,
        sizeBytes: file.size,
        rows: 0,
        columns: 0,
        headers: [],
        preview: [],
        warnings: [],
        content: null,
        objectUrl: url,
        imageSize: { w: img.naturalWidth, h: img.naturalHeight },
      })
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('could not decode image'))
    }
    img.src = url
  })
}

export function formatBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}
