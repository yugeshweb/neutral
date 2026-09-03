import type { DatasetSummary } from '../dataset'

/**
 * The ingestion adapter layer.
 *
 * Hospital data almost never arrives as a CSV. It comes out of an EHR as an
 * HL7 v2 feed or a FHIR bundle, out of a PACS as DICOM, out of a LIS keyed by
 * LOINC codes, or out of a sequencing run as VCF. Each of those is a different
 * parse, but the model behind them is not: rows are patients, columns are
 * numeric features, one column is the label.
 *
 * So every adapter implements the same interface and returns the same shape.
 * The pipeline downstream never learns which format it came from.
 */

export type SourceFormat = 'csv' | 'fhir' | 'hl7v2' | 'pdf' | 'dicom' | 'vcf'

export type AdapterStatus = 'implemented' | 'not-implemented'

/** Where the data physically comes from, for the source list in the UI. */
export type SourceSystem =
  | 'EHR / EMR'
  | 'PACS'
  | 'LIS'
  | 'Sequencing'
  | 'Manual export'

/**
 * A description of the columns an adapter produced, so the user can confirm the
 * right thing landed before training on it.
 */
export type SchemaField = {
  name: string
  /** the vocabulary this field came from, when it came from one */
  coding?: { system: 'LOINC' | 'ICD-10' | 'RxNorm' | 'SNOMED'; code: string }
  type: 'numeric' | 'categorical' | 'identifier' | 'label'
  /** non-null values found, over total rows */
  present: number
}

export type IngestResult = {
  dataset: DatasetSummary
  schema: SchemaField[]
  /** what the adapter did, surfaced so the transform is not a black box */
  notes: string[]
}

/**
 * One method, one output shape. Implement this and the format is supported.
 *
 * `parse` throws `NotImplementedError` for stubbed formats rather than
 * returning an empty result - a format that silently yields nothing is worse
 * than one that says it is not built yet.
 */
export interface IngestAdapter {
  format: SourceFormat
  label: string
  /** system this format typically comes out of */
  system: SourceSystem
  /** file extensions accepted, for the file picker */
  extensions: string[]
  status: AdapterStatus
  /** one line on what this format is, for the source picker */
  description: string
  /** the coding systems a reader will meet in this format */
  vocabularies: string[]
  parse(file: File): Promise<IngestResult>
}

export class NotImplementedError extends Error {
  readonly format: SourceFormat
  /** what building it would actually involve, so the gap is legible */
  readonly requires: string

  constructor(format: SourceFormat, requires: string) {
    super(`${format.toUpperCase()} ingestion is not implemented`)
    this.name = 'NotImplementedError'
    this.format = format
    this.requires = requires
  }
}
