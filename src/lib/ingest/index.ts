import { IMAGE_TYPES, loadImage, parseCsv } from '../dataset'
import { fhirAdapter } from './fhir'
import { hl7Adapter } from './hl7'
import { pdfAdapter } from './pdf'
import {
  NotImplementedError,
  type IngestAdapter,
  type IngestResult,
  type SchemaField,
  type SourceFormat,
} from './types'

export * from './types'
export { parseFhirBundle } from './fhir'
export { parseHl7Feed } from './hl7'
export { parsePdfReport } from './pdf'

/** Numeric-looking columns are the ones the model can actually consume. */
function inferSchema(headers: string[], preview: string[][]): SchemaField[] {
  return headers.map((h, i) => {
    const values = preview.map((r) => r[i]).filter((v) => v !== undefined && v !== '')
    const numeric =
      values.length > 0 && values.every((v) => Number.isFinite(Number(v)))
    const lower = h.toLowerCase()
    const isLabel = ['label', 'target', 'diagnosis', 'class', 'outcome'].includes(lower)
    const isId = lower.includes('id') && !numeric

    return {
      name: h,
      type: isLabel ? 'label' : isId ? 'identifier' : numeric ? 'numeric' : 'categorical',
      present: values.length,
    }
  })
}

const csvAdapter: IngestAdapter = {
  format: 'csv',
  label: 'CSV / Excel export',
  system: 'Manual export',
  extensions: ['.csv'],
  status: 'implemented',
  description:
    'The manual export path - what a clinician gets out of the hospital system by hand. Quoted fields, ragged rows and blank headers are all handled.',
  vocabularies: [],
  async parse(file: File): Promise<IngestResult> {
    const dataset = parseCsv(await file.text(), file.name, file.size)
    return {
      dataset,
      schema: inferSchema(dataset.headers, dataset.preview),
      notes: [
        `parsed ${dataset.rows} rows x ${dataset.columns} columns`,
        'column types inferred from the first rows',
      ],
    }
  },
}

const dicomAdapter: IngestAdapter = {
  format: 'dicom',
  label: 'DICOM study',
  system: 'PACS',
  extensions: ['.dcm'],
  status: 'not-implemented',
  description:
    'Imaging from a PACS server. Each file is pixel data plus a long metadata header covering patient, modality and acquisition settings.',
  vocabularies: ['SNOMED'],
  async parse() {
    throw new NotImplementedError(
      'dicom',
      'A DICOM tag reader for the tabular metadata fields. The pixel data needs a CNN feature extractor first, which is a separate model outside this pipeline.',
    )
  },
}

const vcfAdapter: IngestAdapter = {
  format: 'vcf',
  label: 'VCF / expression matrix',
  system: 'Sequencing',
  extensions: ['.vcf', '.tsv'],
  status: 'not-implemented',
  description:
    'Genomic variants from a sequencing run, or a processed expression matrix. Wide and sparse - thousands of features against far fewer patients.',
  vocabularies: [],
  async parse() {
    throw new NotImplementedError(
      'vcf',
      'A header/INFO parser, transposition so patients become rows, and aggressive feature selection before the matrix is usable at this width.',
    )
  },
}

/** Registry, in the order the source picker lists them. */
export const ADAPTERS: IngestAdapter[] = [
  csvAdapter,
  fhirAdapter,
  hl7Adapter,
  pdfAdapter,
  dicomAdapter,
  vcfAdapter,
]

export const ADAPTER_BY_FORMAT = new Map(ADAPTERS.map((a) => [a.format, a]))

/** Everything the file picker should offer, images included. */
export const ACCEPT_ATTR = [
  ...ADAPTERS.flatMap((a) => a.extensions),
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.bmp',
].join(',')

/**
 * Picks an adapter from the filename, then confirms by content where the
 * extension is ambiguous - `.json` is only FHIR if it holds a Bundle, and
 * `.txt` could be anything.
 */
export function adapterFor(file: File): IngestAdapter | null {
  const name = file.name.toLowerCase()
  const match = ADAPTERS.find((a) => a.extensions.some((e) => name.endsWith(e)))
  return match ?? null
}

/**
 * Single entry point for the input panel. Images bypass the adapter layer -
 * they are not tabular and feed the viewer, not the matrix.
 */
export async function ingest(file: File): Promise<IngestResult> {
  if (IMAGE_TYPES.test(file.name)) {
    const dataset = await loadImage(file)
    return {
      dataset,
      schema: [],
      notes: ['image loaded for display; pixel data needs a CNN before this pipeline'],
    }
  }

  const adapter = adapterFor(file)
  if (!adapter) {
    throw new Error(
      `unsupported file type - expected one of ${ADAPTERS.flatMap((a) => a.extensions).join(', ')} or an image`,
    )
  }

  return adapter.parse(file)
}

export function formatLabel(f: SourceFormat) {
  return ADAPTER_BY_FORMAT.get(f)?.label ?? f
}
