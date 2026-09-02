/**
 * The input types the Train tab's builder offers.
 *
 * Kept out of the component file so the module exports components only, which
 * is what React Fast Refresh needs to swap a component without remounting.
 *
 * `trains` is the honest field: only a CSV table is parsed into the numeric
 * matrix the pipeline fits on. The rest are accepted and recorded as reference
 * sources, and the row says so rather than being silently dropped.
 */

export type InputKind = 'csv' | 'fhir' | 'hl7' | 'image' | 'signal'

export type InputKindSpec = {
  value: InputKind
  label: string
  accept: string
  trains: boolean
}

export const INPUT_KINDS: InputKindSpec[] = [
  { value: 'csv', label: 'CSV table', accept: '.csv', trains: true },
  { value: 'fhir', label: 'FHIR R4 bundle', accept: '.json', trains: false },
  { value: 'hl7', label: 'HL7 v2 feed', accept: '.hl7,.txt', trains: false },
  { value: 'image', label: 'Image / scan', accept: '.png,.jpg,.jpeg,.webp', trains: false },
  { value: 'signal', label: 'Signal trace', accept: '.edf,.bdf,.csv', trains: false },
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
