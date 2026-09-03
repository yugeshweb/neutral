import * as pdfjsLib from 'pdfjs-dist'
// Vite's `?url` import hands back the built worker's final asset URL, which is
// what pdf.js needs to run parsing off the main thread. Pinned to the
// installed pdfjs-dist version, not fetched from a CDN - this has to work
// offline, the same guarantee every other adapter in this directory gives.
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import { createWorker } from 'tesseract.js'
import { csvCell, type DatasetSummary } from '../dataset'
import type { IngestAdapter, IngestResult, SchemaField } from './types'

/**
 * PDF report adapter.
 *
 * A real parse, not a mock: it extracts the actual text layer of the PDF
 * (pdf.js - the same engine Firefox's built-in viewer uses) and reads
 * labelled values off it - "Troponin I: 0.02 ng/mL", "BMI 27.4", "MMSE Score:
 * 24/30" - the way a lab report, discharge summary or radiology impression
 * actually presents data. One PDF is one patient/case: unlike a FHIR bundle
 * or an HL7 feed, a clinical PDF report is not a cohort export.
 *
 * A PDF with no text layer (a scanned fax, a photographed report) falls back
 * to OCR: each page is rendered to a canvas via pdf.js and read with
 * Tesseract.js (a real WASM port of the Tesseract OCR engine, not a mock),
 * then the same label matching runs on the OCR'd text. This is genuinely
 * lower-confidence than a native text layer - OCR misreads digits, and every
 * OCR'd result says so in its notes rather than presenting it as equally
 * reliable.
 *
 * Scope, stated plainly: the OCR path recognizes English, upright, reasonably
 * clean scans - it does not deskew, does not handle handwriting, and a
 * heavily degraded fax may OCR to nothing usable, same as a text PDF using
 * terminology outside `FIELD_ALIASES` below. Tesseract.js needs its language
 * model, `eng.traineddata` (~4 MB), from the network on first use - the one
 * part of this adapter that is not fully offline like the rest of this
 * directory; the model is cached by the browser after that.
 */

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

/**
 * Canonical column name -> label variants a report might use for it, checked
 * in order. Column names match `fhir.ts`'s `KNOWN_LOINC` values where the
 * same measurement appears there, so a cohort assembled from FHIR bundles
 * and one assembled from PDF reports land on the same column names.
 */
const FIELD_ALIASES: Record<string, string[]> = {
  age: ['Age'],
  bmi: ['BMI', 'Body Mass Index'],
  glucose: ['Fasting Glucose', 'Blood Glucose', 'Glucose'],
  hba1c: ['HbA1c', 'Hemoglobin A1c', 'A1C'],
  cholesterol_total: ['Total Cholesterol', 'Cholesterol'],
  hdl_cholesterol: ['HDL Cholesterol', 'HDL'],
  ldl_cholesterol: ['LDL Cholesterol', 'LDL'],
  triglycerides: ['Triglycerides'],
  systolic_bp: ['Systolic BP', 'Systolic Blood Pressure', 'SBP'],
  diastolic_bp: ['Diastolic BP', 'Diastolic Blood Pressure', 'DBP'],
  heart_rate: ['Resting Heart Rate', 'Heart Rate', 'Pulse'],
  max_heart_rate: ['Max Heart Rate', 'Peak Heart Rate', 'Maximum Heart Rate Achieved'],
  troponin: ['Troponin I', 'Troponin T', 'hs-cTnT', 'hs-cTnI', 'Trop-T', 'Troponin'],
  creatinine: ['Creatinine'],
  egfr: ['eGFR', 'GFR'],
  crp_high_sensitivity: ['hs-CRP', 'CRP', 'C-Reactive Protein'],
  mmse: ['MMSE Score', 'MMSE', 'Mini-Mental State Exam', 'Mini Mental State Examination'],
  body_weight: ['Weight'],
  body_height: ['Height'],
  st_depression: ['ST Depression', 'ST-Segment Depression'],
}

const ID_ALIASES = ['Patient ID', 'MRN', 'Medical Record Number', 'Accession Number']

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** First numeric value following a label, e.g. "Troponin I: 0.02 ng/mL" -> 0.02. */
function findNumeric(text: string, aliases: string[]): number | null {
  for (const alias of aliases) {
    const pattern = new RegExp(`${escapeRegex(alias)}\\s*[:\\-]?\\s*([0-9]+\\.?[0-9]*)`, 'i')
    const match = text.match(pattern)
    if (match) {
      const value = Number(match[1])
      if (Number.isFinite(value)) return value
    }
  }
  return null
}

function findId(text: string): string | null {
  for (const alias of ID_ALIASES) {
    const pattern = new RegExp(`${escapeRegex(alias)}\\s*[:\\-]?\\s*([A-Za-z0-9-]+)`, 'i')
    const match = text.match(pattern)
    if (match) return match[1]
  }
  return null
}

/** Extracts the full text layer, page by page, space-joined within a page. */
async function extractText(
  doc: pdfjsLib.PDFDocumentProxy,
): Promise<{ text: string; pages: number }> {
  const pageTexts: string[] = []
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i)
    const content = await page.getTextContent()
    pageTexts.push(content.items.map((it) => ('str' in it ? it.str : '')).join(' '))
  }
  return { text: pageTexts.join('\n'), pages: doc.numPages }
}

/** Rasterises one page at a resolution OCR reads reliably. */
async function renderPageToCanvas(
  page: pdfjsLib.PDFPageProxy,
  scale = 2.5,
): Promise<HTMLCanvasElement> {
  const viewport = page.getViewport({ scale })
  const canvas = document.createElement('canvas')
  canvas.width = Math.ceil(viewport.width)
  canvas.height = Math.ceil(viewport.height)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 2D context unavailable')
  await page.render({ canvasContext: ctx, viewport }).promise
  return canvas
}

/**
 * OCRs every page of a text-less PDF. One worker for the whole document -
 * spinning one up per page would repeat the language-model load each time.
 */
async function ocrDocument(doc: pdfjsLib.PDFDocumentProxy): Promise<string> {
  const worker = await createWorker('eng')
  try {
    const pageTexts: string[] = []
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i)
      const canvas = await renderPageToCanvas(page)
      const { data } = await worker.recognize(canvas)
      pageTexts.push(data.text)
    }
    return pageTexts.join('\n')
  } finally {
    await worker.terminate()
  }
}

export async function parsePdfReport(
  file: File,
): Promise<IngestResult> {
  let doc: pdfjsLib.PDFDocumentProxy
  try {
    doc = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise
  } catch (err) {
    throw new Error(
      `could not open this PDF (${err instanceof Error ? err.message : 'parse error'})`,
    )
  }

  let text: string
  let pages: number
  let ocrUsed = false
  try {
    const result = await extractText(doc)
    pages = result.pages
    text = result.text
    if (text.trim().length === 0) {
      // No embedded text layer - a scanned or photographed page. Fall back
      // to OCR rather than failing outright.
      ocrUsed = true
      text = await ocrDocument(doc)
    }
  } catch (err) {
    throw new Error(
      `could not read this PDF (${err instanceof Error ? err.message : 'parse error'})`,
    )
  }

  if (text.trim().length === 0) {
    throw new Error(
      `read ${pages} page(s) via OCR but recognized no text at all - the scan may be too degraded, rotated, or blank for Tesseract to read`,
    )
  }

  const matched: [string, number][] = []
  for (const [column, aliases] of Object.entries(FIELD_ALIASES)) {
    const value = findNumeric(text, aliases)
    if (value !== null) matched.push([column, value])
  }

  // A few columns are the exact same measurement one of the platform's own
  // trained models names differently - emitted as an extra column, not a
  // rename, so a match against either naming convention still finds it.
  // Only for measurements that genuinely are the same reading; a resting and
  // a peak-exercise heart rate are not the same value, so no alias for that.
  const aliasOut = (from: string, to: string) => {
    const hit = matched.find(([c]) => c === from)
    if (hit && !matched.some(([c]) => c === to)) matched.push([to, hit[1]])
  }
  aliasOut('systolic_bp', 'resting_bp')
  aliasOut('cholesterol_total', 'cholesterol')

  if (matched.length === 0) {
    const via = ocrUsed ? 'via OCR' : 'of embedded text'
    throw new Error(
      `read ${pages} page(s) ${via} (${text.length} characters), but none of the recognized labels (${Object.values(FIELD_ALIASES).flat().slice(0, 6).join(', ')}, ...) were found - ${ocrUsed ? 'the scan may be too degraded to read accurately, or its' : "this report's"} terminology is not in the recognized vocabulary yet`,
    )
  }

  const id = findId(text) ?? file.name.replace(/\.pdf$/i, '')
  const headers = ['patient_id', ...matched.map(([c]) => c)]
  const row = [id, ...matched.map(([, v]) => String(v))]

  const schema: SchemaField[] = headers.map((h) => ({
    name: h,
    type: h === 'patient_id' ? 'identifier' : 'numeric',
    present: 1,
  }))

  const notes = [
    ocrUsed
      ? `no embedded text layer - OCR'd ${pages} page(s) with Tesseract (${text.length} characters recognized)`
      : `read ${pages} page(s) of embedded text (${text.length} characters)`,
    `matched ${matched.length} of ${Object.keys(FIELD_ALIASES).length} recognized labels: ${matched.map(([c]) => c).join(', ')}`,
    'one PDF is one case - this produced a single row, not a cohort',
  ]

  const warnings: string[] = [
    'a single-row upload cannot be split into train/test - use this in the Predict tab, or combine several parsed reports into a CSV for training',
  ]
  if (ocrUsed) {
    warnings.push(
      'values came from OCR, not a native text layer - digits are the most common OCR misread, so double-check anything this scores before relying on it',
    )
  }
  if (matched.length < 3) {
    warnings.push('fewer than 3 fields recognized - this report may use terminology outside the recognized vocabulary')
  }

  const dataset: DatasetSummary = {
    kind: 'csv',
    name: file.name,
    sizeBytes: file.size,
    rows: 1,
    columns: headers.length,
    headers,
    preview: [row],
    warnings,
    content: [headers, row].map((r) => r.map(csvCell).join(',')).join('\n'),
    objectUrl: null,
    imageSize: null,
  }

  return { dataset, schema, notes }
}

export const pdfAdapter: IngestAdapter = {
  format: 'pdf',
  label: 'PDF report (labs / discharge / radiology)',
  system: 'EHR / EMR',
  extensions: ['.pdf'],
  status: 'implemented',
  description:
    'A lab report, discharge summary or radiology impression exported as PDF - including a scanned or photographed one, OCR’d with Tesseract when there is no text layer to read directly. Matches labelled values against a recognized vocabulary (troponin, BMI, MMSE, blood pressure, and more) - one PDF is one case, not a cohort.',
  vocabularies: [],
  async parse(file: File) {
    return parsePdfReport(file)
  },
}
