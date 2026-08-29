import { csvCell, type DatasetSummary } from '../dataset'
import type { IngestAdapter, IngestResult, SchemaField } from './types'

/**
 * HL7 v2 message feed adapter.
 *
 * This is a real parse, not a mock: it splits a feed into individual ORU/ADT
 * messages, reads each message's segments, pulls the patient identifier from
 * PID-3, and reads OBX segments as observations - OBX-3 names the
 * measurement (LOINC-coded when the sending system supplies it), OBX-5 holds
 * the value, OBX-2 says whether that value is numeric.
 *
 * HL7 v2 is the older, pipe-delimited interface, and it is still the most
 * widely deployed one in hospitals - more common in practice than FHIR,
 * which is why it is worth a real parser rather than leaving it stubbed.
 *
 * Scope, stated plainly: it parses `\r`/`\n`-delimited ER7 text with the
 * default HL7 delimiters (`|^~\&`). It does not handle custom delimiter sets,
 * MLLP framing, HL7 v2 escape sequences beyond the common four, or segments
 * this adapter does not read (ORC, NTE, and imaging-specific segments are
 * skipped, not misread). Those are the gap between this and a live interface
 * engine.
 */

type Segment = string[]

const FIELD_SEP = '|'

/** One HL7 message, decomposed into segments and each segment into fields. */
function parseMessage(raw: string): Segment[] {
  return raw
    .split(/\r\n|\r|\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.split(FIELD_SEP))
}

/**
 * Splits a feed into messages. Real feeds concatenate multiple ER7 messages,
 * each starting with its own MSH segment - so a new MSH marks a new message
 * rather than a new segment of the current one.
 */
function splitMessages(text: string): string[] {
  const lines = text.split(/\r\n|\r|\n/)
  const messages: string[][] = []
  let current: string[] = []

  for (const line of lines) {
    if (line.startsWith('MSH')) {
      if (current.length > 0) messages.push(current)
      current = [line]
    } else if (line.trim().length > 0) {
      current.push(line)
    }
  }
  if (current.length > 0) messages.push(current)

  return messages.map((m) => m.join('\n'))
}

/** HL7 escape sequences: \F\ pipe, \S\ component, \T\ subcomponent, \R\ repeat. */
function unescape(value: string): string {
  return value
    .replaceAll('\\F\\', '|')
    .replaceAll('\\S\\', '^')
    .replaceAll('\\T\\', '&')
    .replaceAll('\\R\\', '~')
    .replaceAll('\\E\\', '\\')
}

function field(seg: Segment, index: number): string {
  return seg[index] !== undefined ? unescape(seg[index]) : ''
}

/** PID-3 (patient identifier list) is component-separated: id^checkDigit^... */
function patientId(pid: Segment): string | null {
  const raw = field(pid, 3)
  if (!raw) return null
  return raw.split('~')[0].split('^')[0] || null
}

/** OBX-3 (observation identifier) is CE-typed: code^text^codingSystem. */
function observationCode(obx: Segment): { code: string; label: string; loinc: string | null } {
  const raw = field(obx, 3)
  const [code, text, system] = raw.split('^')
  const isLoinc = (system ?? '').toUpperCase().includes('LN') || (system ?? '').toUpperCase() === 'LOINC'
  return { code: code || raw, label: text || code || raw, loinc: isLoinc ? code : null }
}

function columnFor(code: { code: string; label: string }): string {
  const base = code.label || code.code
  return base
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || `obs_${code.code}`
}

function ageFrom(pid: Segment): number | null {
  // PID-7 is the DOB (YYYYMMDD or longer timestamp forms).
  const raw = field(pid, 7)
  const digits = raw.slice(0, 8)
  if (!/^\d{8}$/.test(digits)) return null
  const year = Number(digits.slice(0, 4))
  const month = Number(digits.slice(4, 6)) - 1
  const day = Number(digits.slice(6, 8))
  const born = new Date(year, month, day)
  if (Number.isNaN(born.getTime())) return null
  const years = (Date.now() - born.getTime()) / (365.25 * 24 * 3600 * 1000)
  return years >= 0 && years < 130 ? Math.round(years) : null
}

function sexFrom(pid: Segment): number | null {
  // PID-8: administrative sex, HL7 table 0001.
  const v = field(pid, 8).toUpperCase()
  if (v === 'M') return 1
  if (v === 'F') return 0
  return null
}

export function parseHl7Feed(text: string, name: string, sizeBytes: number): IngestResult {
  const notes: string[] = []
  const messages = splitMessages(text)

  if (messages.length === 0) {
    throw new Error('no HL7 messages found - expected one or more segments starting with MSH')
  }

  const patients = new Map<
    string,
    { observations: Map<string, number>; age: number | null; sex: number | null }
  >()
  const columns = new Set<string>()
  const codesSeen = new Map<string, { loinc: string | null }>()

  let observations = 0
  let skippedNonNumeric = 0
  let messagesWithoutPid = 0
  let messageTypeCounts = new Map<string, number>()

  for (const raw of messages) {
    const segments = parseMessage(raw)
    const msh = segments.find((s) => s[0] === 'MSH')
    const pid = segments.find((s) => s[0] === 'PID')

    if (msh) {
      // MSH-9 is the message type (e.g. ORU^R01), but MSH-1 is the field
      // separator itself and is not a split-produced field - every field
      // number after it lands one position earlier in the split array than
      // its HL7 field number would suggest, so MSH-9 sits at array index 8.
      const type = field(msh, 8).split('^')[0] || 'unknown'
      messageTypeCounts.set(type, (messageTypeCounts.get(type) ?? 0) + 1)
    }

    if (!pid) {
      messagesWithoutPid++
      continue
    }
    const id = patientId(pid)
    if (!id) {
      messagesWithoutPid++
      continue
    }

    let record = patients.get(id)
    if (!record) {
      record = { observations: new Map(), age: null, sex: null }
      patients.set(id, record)
    }
    // A later message's demographics overwrite an earlier one's, matching
    // how an interface engine treats PID as the current patient state.
    record.age = ageFrom(pid) ?? record.age
    record.sex = sexFrom(pid) ?? record.sex

    for (const obx of segments) {
      if (obx[0] !== 'OBX') continue

      const valueType = field(obx, 2).toUpperCase() // NM, ST, CE, ...
      const rawValue = field(obx, 5)
      const code = observationCode(obx)

      if (valueType !== 'NM' && !/^-?\d+(\.\d+)?$/.test(rawValue)) {
        skippedNonNumeric++
        continue
      }
      const value = Number(rawValue)
      if (!Number.isFinite(value)) {
        skippedNonNumeric++
        continue
      }

      const column = columnFor(code)
      columns.add(column)
      codesSeen.set(column, { loinc: code.loinc })
      record.observations.set(column, value)
      observations++
    }
  }

  if (patients.size === 0) {
    throw new Error('no PID segments with a resolvable patient identifier found in this feed')
  }

  const featureColumns = [...columns].sort()
  const headers = ['patient_id', 'age', 'sex', ...featureColumns]

  const ids = [...patients.keys()]
  const rows = ids.map((id) => {
    const p = patients.get(id)!
    return [
      id,
      p.age === null ? '' : String(p.age),
      p.sex === null ? '' : String(p.sex),
      ...featureColumns.map((c) => {
        const v = p.observations.get(c)
        return v === undefined ? '' : String(v)
      }),
    ]
  })

  const countPresent = (index: number) => rows.reduce((n, r) => n + (r[index] !== '' ? 1 : 0), 0)

  const schema: SchemaField[] = headers.map((h, i) => {
    if (h === 'patient_id') return { name: h, type: 'identifier', present: countPresent(i) }
    const meta = codesSeen.get(h)
    return {
      name: h,
      type: 'numeric',
      coding: meta?.loinc ? { system: 'LOINC', code: meta.loinc } : undefined,
      present: countPresent(i),
    }
  })

  notes.push(`split feed into ${messages.length} HL7 v2 messages`)
  if (messageTypeCounts.size > 0) {
    const summary = [...messageTypeCounts].map(([t, n]) => `${t}:${n}`).join(', ')
    notes.push(`message types: ${summary}`)
  }
  notes.push(`read ${observations} OBX observations across ${featureColumns.length} distinct fields`)
  if (skippedNonNumeric > 0) {
    notes.push(`skipped ${skippedNonNumeric} non-numeric OBX values (ST/CE/free text)`)
  }
  if (messagesWithoutPid > 0) {
    notes.push(`${messagesWithoutPid} message(s) had no resolvable PID-3 and were dropped`)
  }
  notes.push(`grouped into ${rows.length} patient rows x ${headers.length} columns`)

  const warnings: string[] = []
  // age and sex come from PID, not OBX, so they were never eligible for LOINC
  // coding - only the observation columns belong in this comparison.
  const observationFields = schema.filter((s) => featureColumns.includes(s.name))
  const loincCoded = observationFields.filter((s) => s.coding?.system === 'LOINC').length
  if (observationFields.length > 0 && loincCoded < observationFields.length) {
    warnings.push(
      `${observationFields.length - loincCoded} of ${observationFields.length} observation field(s) had no LOINC coding system in OBX-3 and were named from local text`,
    )
  }
  const sparse = schema.filter((s) => s.type === 'numeric' && s.present < rows.length * 0.5).length
  if (sparse > 0) {
    warnings.push(`${sparse} field(s) present in under half of patients - imputation will dominate`)
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

export const hl7Adapter: IngestAdapter = {
  format: 'hl7v2',
  label: 'HL7 v2 message feed',
  system: 'EHR / EMR',
  extensions: ['.hl7', '.txt'],
  status: 'implemented',
  description:
    'The older pipe-delimited feed, still the most widely deployed interface in hospitals. Segments carry the data; OBX fields hold observation values.',
  vocabularies: ['LOINC'],
  async parse(file: File) {
    return parseHl7Feed(await file.text(), file.name, file.size)
  },
}
