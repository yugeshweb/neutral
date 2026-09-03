"""FHIR R4 Bundle adapter - a Python port of `src/lib/ingest/fhir.ts`.

Same real parse, same scope statement: walks a Bundle's entries, groups
resources by the Patient they reference, reads numeric Observation values
keyed by LOINC code, derives a binary label from Condition ICD-10 codes,
flattens to one row per patient. Reads a Bundle from a file - no live FHIR
endpoint, no SMART-on-FHIR auth, no `Bundle.link` paging.
"""

from __future__ import annotations

import json
from typing import Any

LOINC = "http://loinc.org"

# Kept identical to `KNOWN_LOINC` in fhir.ts, so a cohort assembled from a
# FHIR bundle lands on the same column names in Python as in the browser.
KNOWN_LOINC: dict[str, str] = {
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "39156-5": "bmi",
    "2093-3": "cholesterol_total",
    "2085-9": "hdl_cholesterol",
    "2089-1": "ldl_cholesterol",
    "2571-8": "triglycerides",
    "4548-4": "hba1c",
    "2345-7": "glucose",
    "718-7": "haemoglobin",
    "6690-2": "wbc_count",
    "777-3": "platelet_count",
    "2160-0": "creatinine",
    "33914-3": "egfr",
    "30522-7": "crp_high_sensitivity",
    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",
    "8310-5": "body_temperature",
    "29463-7": "body_weight",
    "8302-2": "body_height",
}

POSITIVE_ICD10 = ["C50", "C34", "I21", "I25", "E11", "G30"]


def _subject_id(resource: dict[str, Any]) -> str | None:
    ref = (resource.get("subject") or {}).get("reference") or (resource.get("patient") or {}).get("reference")
    if not ref:
        return None
    if "/" in ref:
        return ref.rsplit("/", 1)[1]
    return ref.replace("urn:uuid:", "")


def _loinc_code(concept: dict[str, Any] | None) -> str | None:
    for coding in (concept or {}).get("coding", []) or []:
        if coding.get("system") == LOINC and coding.get("code"):
            return coding["code"]
    return None


def _column_for(code: str) -> str:
    return KNOWN_LOINC.get(code, f"loinc_{''.join(c if c.isalnum() else '_' for c in code)}")


def _numeric_value(resource: dict[str, Any]) -> float | None:
    vq = resource.get("valueQuantity")
    if isinstance(vq, dict) and isinstance(vq.get("value"), (int, float)):
        return float(vq["value"])
    vi = resource.get("valueInteger")
    if isinstance(vi, (int, float)):
        return float(vi)
    return None


def _age_from(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    from datetime import date

    try:
        parts = [int(p) for p in birth_date.split("-")[:3]]
        while len(parts) < 3:
            parts.append(1)
        born = date(parts[0], parts[1], parts[2])
    except (ValueError, IndexError):
        return None
    years = (date.today() - born).days / 365.25
    return round(years) if 0 <= years < 130 else None


def parse_fhir_bundle(text: str) -> tuple[list[str], list[list[str]], list[str]]:
    """Returns (headers, rows, notes). Raises ValueError for a malformed or
    non-Bundle file."""

    try:
        bundle = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("file is not valid JSON") from exc

    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"expected a FHIR Bundle, got {bundle.get('resourceType') or 'a JSON object with no resourceType'}")

    entries = bundle.get("entry") or []
    if not entries:
        raise ValueError("bundle contains no entries")

    patients: dict[str, dict[str, Any]] = {}
    columns: set[str] = set()

    def touch(pid: str) -> dict[str, Any]:
        return patients.setdefault(pid, {"features": {}, "label": None, "age": None, "sex": None})

    observations = 0
    conditions = 0

    for entry in entries:
        resource = entry.get("resource") or {}
        rtype = resource.get("resourceType")
        if not rtype:
            continue

        if rtype == "Patient" and resource.get("id"):
            p = touch(resource["id"])
            p["age"] = _age_from(resource.get("birthDate"))
            gender = resource.get("gender")
            p["sex"] = 1.0 if gender == "male" else 0.0 if gender == "female" else None
            continue

        if rtype == "Observation":
            pid = _subject_id(resource)
            code = _loinc_code(resource.get("code"))
            if not pid or not code:
                continue
            value = _numeric_value(resource)
            if value is None:
                continue
            column = _column_for(code)
            columns.add(column)
            touch(pid)["features"][column] = value
            observations += 1
            continue

        if rtype == "Condition":
            pid = _subject_id(resource)
            if not pid:
                continue
            codes = (resource.get("code") or {}).get("coding") or []
            positive = any(
                (c.get("code") or "").upper().startswith(prefix)
                for c in codes
                for prefix in POSITIVE_ICD10
            )
            p = touch(pid)
            p["label"] = 1 if positive else (p["label"] or 0)
            conditions += 1

    if not patients:
        raise ValueError("no Patient, Observation or Condition resources found in the bundle")

    feature_columns = sorted(columns)
    headers = ["patient_id", "age", "sex", *feature_columns, "label"]
    rows = []
    for pid, p in patients.items():
        row = [
            pid,
            "" if p["age"] is None else str(p["age"]),
            "" if p["sex"] is None else str(int(p["sex"])),
            *("" if p["features"].get(c) is None else str(p["features"][c]) for c in feature_columns),
            "" if p["label"] is None else str(p["label"]),
        ]
        rows.append(row)

    notes = [
        f"walked {len(entries)} bundle entries, grouped by Patient reference",
        f"read {observations} Observation values across {len(feature_columns)} LOINC codes",
    ]
    if conditions:
        notes.append(f"derived label from {conditions} Condition resources by ICD-10 prefix")
    notes.append(f"flattened to {len(rows)} patient rows x {len(headers)} columns")

    return headers, rows, notes
