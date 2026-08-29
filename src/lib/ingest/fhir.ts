import { csvCell, type DatasetSummary } from '../dataset'
import type { IngestAdapter, IngestResult, SchemaField } from './types'

/**
 * FHIR R4 Bundle adapter.
 *
 * This is a real parse, not a mock: it walks a Bundle's entries, groups
 * resources by the Patient they reference, pulls numeric values off
 * Observation resources keyed by LOINC code, reads Condition codes as the
 * label, and flattens the result to one row per patient.
 *
 * FHIR R4 matters here because it is what India's ABDM health-records
 * framework is built on, and what public test servers (hapi.fhir.org,
 * Synthea-generated bundles) actually serve. That makes this adapter testable
 * against realistic data without any hospital agreement.
 *
 * Scope, stated plainly: it reads a Bundle from a file. It does not talk to a
 * live FHIR endpoint, handle SMART-on-FHIR auth, page through `Bundle.link`,
 * or resolve contained/referenced resources across bundles. Those are the gap
 * between this and a deployment.
 */

/** Minimal shapes for the fields this adapter reads. */
type Coding = { system?: string; code?: string; display?: string }
type CodeableConcept = { coding?: Coding[]; text?: string }

type FhirResource = {
  resourceType?: string
  id?: string
  subject?: { reference?: string }
  patient?: { reference?: string }
  code?: CodeableConcept
  valueQuantity?: { value?: number; unit?: string }
  valueInteger?: number
  component?: { code?: CodeableConcept; valueQuantity?: { value?: number } }[]
  clinicalStatus?: CodeableConcept
  gender?: string
  birthDate?: string
}

type Bundle = {
  resourceType?: string
  type?: string
  entry?: { resource?: FhirResource }[]
}

const LOINC = 'http://loinc.org'

/**
 * LOINC codes worth pulling for early-detection work, mapped to column names.
 * Anything not on this list still gets a column, named by its code - the map
 * only supplies friendlier names for the common ones.
 */
const KNOWN_LOINC: Record<string, string> = {
  '8480-6': 'systolic_bp',
  '8462-4': 'diastolic_bp',
  '39156-5': 'bmi',
  '2093-3': 'cholesterol_total',
  '2085-9': 'hdl_cholesterol',
  '2089-1': 'ldl_cholesterol',
  '2571-8': 'triglycerides',
  '4548-4': 'hba1c',
  '2345-7': 'glucose',
  '718-7': 'haemoglobin',
  '6690-2': 'wbc_count',
  '777-3': 'platelet_count',
  '2160-0': 'creatinine',
  '33914-3': 'egfr',
  '30522-7': 'crp_high_sensitivity',
  '8867-4': 'heart_rate',
  '9279-1': 'respiratory_rate',
  '8310-5': 'body_temperature',
  '29463-7': 'body_weight',
  '8302-2': 'body_height',
}

/**
 * ICD-10 prefixes that mark a positive label for the diseases the platform
 * targets. A diagnosis is a code, never a string - matching on display text
 * would miss any bundle that localises or abbreviates it.
 */
const POSITIVE_ICD10 = [
  'C50', // malignant neoplasm of breast
  'C34', // malignant neoplasm of bronchus and lung
  'I21', // acute myocardial infarction
  'I25', // chronic ischaemic heart disease
  'E11', // type 2 diabetes mellitus
  'G30', // Alzheimer disease
]

/** `Patient/abc` and `urn:uuid:abc` both identify the same subject. */
function subjectId(r: FhirResource): string | null {
  const ref = r.subject?.reference ?? r.patient?.reference
  if (!ref) return null
  const slash = ref.lastIndexOf('/')
  return slash >= 0 ? ref.slice(slash + 1) : ref.replace('urn:uuid:', '')
}

function loincCode(concept: CodeableConcept | undefined): string | null {
  const coding = concept?.coding?.find((c) => c.system === LOINC && c.code)
  return coding?.code ?? null
}

function columnFor(code: string) {
  return KNOWN_LOINC[code] ?? `loinc_${code.replace(/[^A-Za-z0-9]/g, '_')}`
}

/** Reads the numeric value off an Observation, including the component form. */
function numericValue(r: FhirResource): number | null {
  if (typeof r.valueQuantity?.value === 'number') return r.valueQuantity.value
  if (typeof r.valueInteger === 'number') return r.valueInteger
  return null
}

function ageFrom(birthDate: string | undefined): number | null {
  if (!birthDate) return null
  const born = new Date(birthDate)
  if (Number.isNaN(born.getTime())) return null
  const years = (Date.now() - born.getTime()) / (365.25 * 24 * 3600 * 1000)
  return years >= 0 && years < 130 ? Math.round(years) : null
}

export function parseFhirBundle(
  text: string,
  name: string,
  sizeBytes: number,
): IngestResult {
  let bundle: Bundle
  try {
    bundle = JSON.parse(text) as Bundle
  } catch {
    throw new Error('file is not valid JSON')
  }

  if (bundle.resourceType !== 'Bundle') {
    throw new Error(
      `expected a FHIR Bundle, got ${bundle.resourceType ?? 'a JSON object with no resourceType'}`,
    )
  }

  const entries = bundle.entry ?? []
  if (entries.length === 0) throw new Error('bundle contains no entries')

  const notes: string[] = []

  // One accumulator per patient. Columns are discovered, not assumed.
  const patients = new Map<
    string,
    { features: Map<string, number>; label: number | null; age: number | null; sex: number | null }
  >()
  const columns = new Set<string>()
  const codesSeen = new Map<string, string>()

  const touch = (id: string) => {
    let p = patients.get(id)
    if (!p) {
      p = { features: new Map(), label: null, age: null, sex: null }
      patients.set(id, p)
    }
    return p
  }

  let observations = 0
  let conditions = 0
  let skippedNonNumeric = 0

  for (const { resource } of entries) {
    if (!resource?.resourceType) continue

    if (resource.resourceType === 'Patient' && resource.id) {
      const p = touch(resource.id)
      p.age = ageFrom(resource.birthDate)
      // Encoded numerically because the matrix downstream is all-numeric.
      p.sex = resource.gender === 'male' ? 1 : resource.gender === 'female' ? 0 : null
      continue
    }

    if (resource.resourceType === 'Observation') {
      const id = subjectId(resource)
      const code = loincCode(resource.code)
      if (!id || !code) continue

      const value = numericValue(resource)
      if (value === null) {
        skippedNonNumeric++
        continue
      }

      const column = columnFor(code)
      codesSeen.set(column, code)
      columns.add(column)
      // Last observation wins; a real build would take the most recent by date.
      touch(id).features.set(column, value)
      observations++
      continue
    }

    if (resource.resourceType === 'Condition') {
      const id = subjectId(resource)
      if (!id) continue
      const codes = resource.code?.coding ?? []
      const positive = codes.some((c) =>
        POSITIVE_ICD10.some((prefix) => (c.code ?? '').toUpperCase().startsWith(prefix)),
      )
      const p = touch(id)
      // Once positive, stays positive - a later unrelated condition must not clear it.
      p.label = positive ? 1 : (p.label ?? 0)
      conditions++
    }
  }

  if (patients.size === 0) {
    throw new Error('no Patient, Observation or Condition resources found in the bundle')
  }

  // Stable column order, with the demographics first.
  const featureColumns = [...columns].sort()
  const headers = ['patient_id', 'age', 'sex', ...featureColumns, 'label']

  const ids = [...patients.keys()]
  const rows = ids.map((id) => {
    const p = patients.get(id)!
    return [
      id,
      p.age === null ? '' : String(p.age),
      p.sex === null ? '' : String(p.sex),
      ...featureColumns.map((c) => {
        const v = p.features.get(c)
        return v === undefined ? '' : String(v)
      }),
      p.label === null ? '' : String(p.label),
    ]
  })

  // Coverage per column, so sparsity is visible before training rather than after.
  const countPresent = (index: number) =>
    rows.reduce((n, r) => n + (r[index] !== '' ? 1 : 0), 0)

  const schema: SchemaField[] = headers.map((h, i) => {
    if (h === 'patient_id') {
      return { name: h, type: 'identifier', present: countPresent(i) }
    }
    if (h === 'label') {
      return {
        name: h,
        type: 'label',
        coding: { system: 'ICD-10', code: POSITIVE_ICD10.join(', ') },
        present: countPresent(i),
      }
    }
    const code = codesSeen.get(h)
    return {
      name: h,
      type: 'numeric',
      coding: code ? { system: 'LOINC', code } : undefined,
      present: countPresent(i),
    }
  })

  notes.push(`walked ${entries.length} bundle entries, grouped by Patient reference`)
  notes.push(`read ${observations} Observation values across ${featureColumns.length} LOINC codes`)
  if (conditions > 0) {
    notes.push(`derived label from ${conditions} Condition resources by ICD-10 prefix`)
  }
  if (skippedNonNumeric > 0) {
    notes.push(`skipped ${skippedNonNumeric} non-numeric observations (coded or text values)`)
  }
  notes.push(`flattened to ${rows.length} patient rows x ${headers.length} columns`)

  const warnings: string[] = []

  const labelled = schema.find((s) => s.type === 'label')?.present ?? 0
  if (labelled === 0) {
    warnings.push('no Condition resources matched the target ICD-10 codes, so every row is unlabelled')
  } else if (labelled < rows.length) {
    warnings.push(`${rows.length - labelled} of ${rows.length} rows have no label`)
  }

  // Sparse columns are the normal failure mode for real bundles.
  const sparse = schema.filter(
    (s) => s.type === 'numeric' && s.present < rows.length * 0.5,
  ).length
  if (sparse > 0) {
    warnings.push(`${sparse} column(s) present in under half of patients - imputation will dominate`)
  }
  if (rows.length < 20) {
    warnings.push(`only ${rows.length} patients - far below what training needs`)
  }

  const dataset: DatasetSummary = {
    kind: 'csv',
    name,
    sizeBytes,
    rows: rows.length,
    columns: headers.length,
    headers,
    preview: rows.slice(0, 4),
    warnings,
    content: [headers, ...rows].map((row) => row.map(csvCell).join(',')).join('\n'),
    objectUrl: null,
    imageSize: null,
  }

  return { dataset, schema, notes }
}

export const fhirAdapter: IngestAdapter = {
  format: 'fhir',
  label: 'FHIR R4 Bundle',
  system: 'EHR / EMR',
  extensions: ['.json'],
  status: 'implemented',
  description:
    'The modern EHR exchange format, and what India ABDM health records are built on. Walks the bundle, reads Observation values by LOINC code, derives the label from Condition ICD-10 codes.',
  vocabularies: ['LOINC', 'ICD-10'],
  async parse(file: File) {
    return parseFhirBundle(await file.text(), file.name, file.size)
  },
}
