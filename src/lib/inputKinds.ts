/**
 * The input types the Train tab's builder offers.
 *
 * Kept out of the component file so the module exports components only, which
 * is what React Fast Refresh needs to swap a component without remounting.
 *
 * Two groupings, on purpose:
 *
 * - EHR/clinical data is ONE pipeline (`ehr`), not four. CSV, FHIR, HL7 and
 *   PDF all converge on the same `DatasetSummary` shape through `ingest()`'s
 *   own format auto-detection (by extension, confirmed by content) - the
 *   pipeline downstream never learns which format a row started as, so
 *   making the user pick the format up front was a distinction without a
 *   difference. One row, one file, the right adapter runs automatically.
 * - Imaging is one pipeline PER MODALITY, not one generic "image" bucket.
 *   An MRI, a CT, a mammogram and a histopathology slide are different
 *   studies from different systems, and lumping them into one dropdown
 *   entry implied they were interchangeable. None of them are analysed yet
 *   (pixel data needs a modality-specific CNN this platform does not build),
 *   but which study a row is stays visible rather than being flattened to
 *   "image."
 *
 * `trains` is the honest field: only `ehr` is actually parsed into the
 * numeric matrix the pipeline fits on. Every imaging kind is accepted and
 * recorded as a reference source, and the row says so rather than being
 * silently dropped.
 */

export type InputKind =
  | 'ehr'
  | 'mri'
  | 'ct'
  | 'mammogram'
  | 'histology'
  | 'ecg'
  | 'angiogram'
  | 'eeg'

export type InputKindSpec = {
  value: InputKind
  label: string
  accept: string
  trains: boolean
}

export const INPUT_KINDS: InputKindSpec[] = [
  {
    value: 'ehr',
    label: 'EHR / clinical data (CSV, FHIR, HL7, PDF)',
    accept: '.csv,.json,.hl7,.txt,.pdf',
    trains: true,
  },
  {
    value: 'mri',
    label: 'MRI',
    accept: '.png,.jpg,.jpeg,.webp,.dcm,.nii,.nii.gz',
    trains: false,
  },
  {
    value: 'ct',
    label: 'CT',
    accept: '.png,.jpg,.jpeg,.webp,.dcm,.nii,.nii.gz',
    trains: false,
  },
  {
    value: 'mammogram',
    label: 'Mammogram',
    accept: '.png,.jpg,.jpeg,.webp,.dcm',
    trains: false,
  },
  {
    value: 'histology',
    label: 'Histopathology slide',
    accept: '.png,.jpg,.jpeg,.webp,.dcm,.svs,.tif',
    trains: false,
  },
  {
    value: 'ecg',
    label: 'ECG waveform',
    accept: '.png,.jpg,.jpeg,.webp,.xml,.scp,.dcm',
    trains: false,
  },
  {
    value: 'angiogram',
    label: 'Angiogram',
    accept: '.png,.jpg,.jpeg,.webp,.dcm',
    trains: false,
  },
  {
    value: 'eeg',
    label: 'EEG recording',
    accept: '.png,.jpg,.jpeg,.webp,.edf,.bdf',
    trains: false,
  },
]

export type InputRow = {
  id: string
  kind: InputKind
  fileName: string | null
  rows: number | null
  note: string | null
}

export function kindOf(kind: InputKind): InputKindSpec {
  return INPUT_KINDS.find((k) => k.value === kind) ?? INPUT_KINDS[0]
}
