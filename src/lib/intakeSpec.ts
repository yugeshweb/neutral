import type { SourceFormat } from './ingest/types'

/**
 * What each condition actually accepts as input.
 *
 * The problem statement asks for a platform that ingests real biomedical data,
 * and real biomedical data is not one format. A seizure model reads an EEG
 * signal; a breast-cancer model reads morphometric measurements that arrive as
 * a table or an EHR extract; an Alzheimer's model wants both clinical scores
 * and a volumetric MRI. So the intake fields are declared per condition rather
 * than showing the same four boxes everywhere, which would misrepresent what
 * the model behind each one can use.
 *
 * `wired` is the honest bit. Only CSV, FHIR R4 and HL7 v2 have working parsers
 * in `lib/ingest`. Everything else is declared here because it is genuinely
 * part of the intended pipeline, and the UI says plainly that it is not built
 * rather than accepting a file and inventing a result.
 */

export type IntakeField = {
  id: string
  /** what the clinician would call it */
  label: string
  /** the source system it comes out of */
  system: string
  /** file extensions the picker accepts */
  accept: string
  /** true when a parser exists behind it */
  wired: boolean
  /** the adapter to run, for wired fields */
  format?: SourceFormat
  /**
   * What this input is. A plain sentence for most fields; an array for the
   * few whose hint joins two distinct facts with a comma (FHIR's coding
   * systems, HL7's message and segment types) - those read as two lines
   * rather than one comma-spliced sentence.
   */
  hint: string | string[]
  /** for unwired fields: what building it would take */
  requires?: string
  /**
   * True for fields that take a picture. A standard image (PNG/JPEG) is
   * accepted and displayed with the region overlay, so the imaging path can be
   * demonstrated end to end. The clinical formats beside it (DICOM, NIfTI,
   * whole-slide) still have no reader, and no detector runs on the pixels
   * either - `imaging` governs display, never analysis.
   */
  imaging?: boolean
}

export type ConditionIntake = {
  diseaseId: string
  fields: IntakeField[]
}

const CLINICAL_TABLE: IntakeField = {
  id: 'table',
  label: 'Clinical measurements',
  system: 'Manual export',
  accept: '.csv',
  wired: true,
  format: 'csv',
  hint: 'One row per patient, one column per measurement.',
}

const FHIR_BUNDLE: IntakeField = {
  id: 'fhir',
  label: 'FHIR R4 bundle',
  system: 'EHR / EMR',
  accept: '.json',
  wired: true,
  format: 'fhir',
  hint: ['Observations keyed by LOINC,', 'conditions by ICD-10.'],
}

const HL7_FEED: IntakeField = {
  id: 'hl7',
  label: 'HL7 v2 message feed',
  system: 'EHR / EMR',
  accept: '.hl7,.txt',
  wired: true,
  format: 'hl7v2',
  hint: ['ORU^R01 result messages,', 'OBX segments keyed by LOINC.'],
}

export const CONDITION_INTAKE: ConditionIntake[] = [
  {
    diseaseId: 'breast-cancer',
    fields: [
      { ...CLINICAL_TABLE, hint: 'Nuclear morphometry per lesion, one row per case.' },
      FHIR_BUNDLE,
      HL7_FEED,
      {
        id: 'histology',
        label: 'Histopathology slide',
        system: 'PACS',
        accept: '.png,.jpg,.jpeg,.webp,.dcm,.svs,.tif',
        wired: true,
        imaging: true,
        requires:
          'A whole-slide image reader plus a CNN feature extractor to turn pixels into the morphometric features this model consumes. That extractor is a separate model outside this pipeline.',
        hint: 'Digitised FNA or whole-slide image.',
      },
    ],
  },
  {
    diseaseId: 'brain-seizure',
    fields: [
      {
        id: 'mri-brain',
        label: 'Brain MRI',
        system: 'PACS',
        accept: '.png,.jpg,.jpeg,.webp,.dcm,.nii,.nii.gz',
        wired: true,
        imaging: true,
        requires:
          'A DICOM or NIfTI reader and a tumour segmentation network. The model behind this condition reads EEG-derived features, not voxels, so an image is displayed but never scored.',
        hint: 'T1 or FLAIR volume, displayed for review.',
      },
      {
        id: 'eeg',
        label: 'EEG recording',
        system: 'Neurophysiology',
        accept: '.png,.jpg,.jpeg,.webp,.edf,.bdf',
        wired: true,
        imaging: true,
        requires:
          'An EDF/BDF reader, window segmentation, and the band-power and non-linear feature extraction that turns a raw multi-channel trace into the per-window features this model reads.',
        hint: 'Raw multi-channel trace, European Data Format.',
      },
      {
        ...CLINICAL_TABLE,
        label: 'Extracted EEG features',
        hint: 'Pre-computed band powers and entropy, one row per window.',
      },
      HL7_FEED,
    ],
  },
  {
    diseaseId: 'heart-disease',
    fields: [
      { ...CLINICAL_TABLE, hint: 'Hemodynamic and exercise ECG attributes per patient.' },
      FHIR_BUNDLE,
      HL7_FEED,
      {
        id: 'ecg',
        label: 'ECG waveform',
        system: 'Cardiology',
        accept: '.png,.jpg,.jpeg,.webp,.xml,.scp,.dcm',
        wired: true,
        imaging: true,
        requires:
          'An SCP-ECG or DICOM waveform reader plus interval and morphology extraction, to derive the ST and rhythm features this model expects from a raw trace.',
        hint: 'Twelve-lead resting or stress trace.',
      },
    ],
  },
  {
    diseaseId: 'alzheimers',
    fields: [
      { ...CLINICAL_TABLE, hint: 'MMSE, education, and volumetric measures per visit.' },
      FHIR_BUNDLE,
      {
        id: 'mri',
        label: 'Structural MRI',
        system: 'PACS',
        accept: '.png,.jpg,.jpeg,.webp,.dcm,.nii,.nii.gz',
        wired: true,
        imaging: true,
        requires:
          'A DICOM or NIfTI reader and a volumetric segmentation step to produce normalised whole-brain volume and intracranial volume, which this model currently expects pre-computed.',
        hint: 'T1-weighted volume for brain volumetry.',
      },
      HL7_FEED,
    ],
  },
  {
    diseaseId: 'stroke-risk',
    fields: [
      { ...CLINICAL_TABLE, hint: 'Structured cardiovascular risk factors, one row per patient.' },
      FHIR_BUNDLE,
      HL7_FEED,
    ],
  },
  {
    diseaseId: 'parkinsons',
    fields: [
      { ...CLINICAL_TABLE, hint: 'Sustained-phonation acoustic measures, one row per recording.' },
      FHIR_BUNDLE,
      {
        id: 'voice',
        label: 'Voice recording',
        system: 'Neurophysiology',
        accept: '.wav,.mp3,.flac',
        wired: false,
        requires:
          'An audio reader plus the acoustic feature extraction (jitter, shimmer, HNR, and the non-linear dynamics measures) that turns a raw sustained-vowel recording into the per-row features this model reads.',
        hint: 'Raw sustained-vowel phonation, not yet parsed.',
      },
      HL7_FEED,
    ],
  },
]

export function intakeFor(diseaseId: string): IntakeField[] {
  return CONDITION_INTAKE.find((c) => c.diseaseId === diseaseId)?.fields ?? []
}

/** The conditions Train and Predict offer, in display order. */
export const INTAKE_DISEASE_IDS = CONDITION_INTAKE.map((c) => c.diseaseId)
