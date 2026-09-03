"""HL7 v2 message feed adapter - a Python port of `src/lib/ingest/hl7.ts`.

Splits a feed into ER7 messages (a new MSH starts a new message), reads
PID-3 as the patient id, PID-7/PID-8 as age/sex, and OBX segments as
observations (OBX-3 names the measurement, OBX-5 holds the value, OBX-2
says whether it's numeric). Default `|^~\\&` delimiters only, same stated
scope as the TypeScript adapter - no MLLP framing, no custom delimiter sets.
"""

from __future__ import annotations

import re

FIELD_SEP = "|"


def _unescape(value: str) -> str:
    return (
        value.replace("\\F\\", "|")
        .replace("\\S\\", "^")
        .replace("\\T\\", "&")
        .replace("\\R\\", "~")
        .replace("\\E\\", "\\")
    )


def _field(seg: list[str], index: int) -> str:
    return _unescape(seg[index]) if index < len(seg) else ""


def _split_messages(text: str) -> list[list[list[str]]]:
    lines = re.split(r"\r\n|\r|\n", text)
    messages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("MSH"):
            if current:
                messages.append(current)
            current = [line]
        elif line.strip():
            current.append(line)
    if current:
        messages.append(current)
    return [[seg_line.split(FIELD_SEP) for seg_line in msg] for msg in messages]


def _patient_id(pid: list[str]) -> str | None:
    raw = _field(pid, 3)
    if not raw:
        return None
    return raw.split("~")[0].split("^")[0] or None


def _observation_code(obx: list[str]) -> tuple[str, str, str | None]:
    raw = _field(obx, 3)
    parts = raw.split("^")
    code = parts[0] if parts else raw
    text = parts[1] if len(parts) > 1 else ""
    system = parts[2] if len(parts) > 2 else ""
    is_loinc = system.upper() in ("LN", "LOINC")
    return code or raw, text or code or raw, (code if is_loinc else None)


def _column_for(code: str, label: str) -> str:
    base = label or code
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    return slug or f"obs_{code}"


def _age_from(pid: list[str]) -> int | None:
    raw = _field(pid, 7)[:8]
    if not re.match(r"^\d{8}$", raw):
        return None
    from datetime import date

    try:
        born = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None
    years = (date.today() - born).days / 365.25
    return round(years) if 0 <= years < 130 else None


def _sex_from(pid: list[str]) -> int | None:
    v = _field(pid, 8).upper()
    return 1 if v == "M" else 0 if v == "F" else None


def parse_hl7_feed(text: str) -> tuple[list[str], list[list[str]], list[str]]:
    """Returns (headers, rows, notes). Raises ValueError if no messages or
    no resolvable patient ids are found."""

    messages = _split_messages(text)
    if not messages:
        raise ValueError("no HL7 messages found - expected one or more segments starting with MSH")

    patients: dict[str, dict] = {}
    columns: set[str] = set()
    observations = 0

    for segments in messages:
        pid = next((s for s in segments if s and s[0] == "PID"), None)
        if not pid:
            continue
        patient_id = _patient_id(pid)
        if not patient_id:
            continue

        record = patients.setdefault(patient_id, {"observations": {}, "age": None, "sex": None})
        record["age"] = _age_from(pid) or record["age"]
        record["sex"] = _sex_from(pid) if _sex_from(pid) is not None else record["sex"]

        for obx in segments:
            if not obx or obx[0] != "OBX":
                continue
            value_type = _field(obx, 2).upper()
            raw_value = _field(obx, 5)
            code, label, _loinc = _observation_code(obx)
            if value_type != "NM" and not re.match(r"^-?\d+(\.\d+)?$", raw_value):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            column = _column_for(code, label)
            columns.add(column)
            record["observations"][column] = value
            observations += 1

    if not patients:
        raise ValueError("no PID segments with a resolvable patient identifier found in this feed")

    feature_columns = sorted(columns)
    headers = ["patient_id", "age", "sex", *feature_columns]
    rows = []
    for pid, rec in patients.items():
        row = [
            pid,
            "" if rec["age"] is None else str(rec["age"]),
            "" if rec["sex"] is None else str(rec["sex"]),
            *("" if rec["observations"].get(c) is None else str(rec["observations"][c]) for c in feature_columns),
        ]
        rows.append(row)

    notes = [
        f"split feed into {len(messages)} HL7 v2 messages",
        f"read {observations} OBX observations across {len(feature_columns)} distinct fields",
        f"grouped into {len(rows)} patient rows x {len(headers)} columns",
    ]

    return headers, rows, notes
