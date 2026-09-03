"""DICOM adapter: metadata tags only, stated plainly.

A DICOM file is a header (patient/study/series metadata, standard tags) plus
pixel data. This adapter reads the header - real, structured values, not a
guess - and explicitly does NOT analyse the pixel data: turning an MRI/CT
volume into the kind of features a disease's model actually trains on
(radiomics texture statistics, a CNN's learned embedding) is a separate
model this platform does not build, the same boundary the frontend's DICOM
adapter (`src/lib/ingest/index.ts`) draws. For most DICOM files this means
few or none of a disease's `required_fields` will be found here - age and
sex are commonly present as header tags; troponin, MMSE, chest-pain-type
are not, because they were never something a scanner's header records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Union

DicomSource = Union[str, Path, bytes, BinaryIO]

# DICOM tag keyword -> canonical column name (see `aliases.py`). Only
# de-identified, non-PHI-bearing demographic/technical tags - never
# PatientName, PatientID as a raw value, etc.
TAG_TO_COLUMN: dict[str, str] = {
    "PatientAge": "age",
    "PatientSex": "sex",
    "PatientWeight": "body_weight",
    "PatientSize": "body_height",  # DICOM stores height in meters
}

# Descriptive tags surfaced as notes/metadata, not as model features - they
# say what the study IS, not a clinical measurement.
DESCRIPTIVE_TAGS = ["Modality", "StudyDescription", "SeriesDescription", "BodyPartExamined"]


def _read_bytes(source: DicomSource) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "read"):
        data = source.read()
        return data if isinstance(data, bytes) else data.encode("utf-8")
    raise ValueError(f"unsupported DICOM source type: {type(source).__name__}")


def extract_dicom_fields(source: DicomSource) -> dict[str, Any]:
    """Returns {"fields": {canonical_name: value}, "descriptive": {tag: value},
    "notes": [...]}. Never raises for a file that simply has few tags - an
    empty `fields` dict is a legitimate, honestly-reported result; raises
    only if the file cannot be read as DICOM at all."""

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("pydicom is not installed - cannot read DICOM files") from exc

    import io

    try:
        ds = pydicom.dcmread(io.BytesIO(_read_bytes(source)), force=False)
    except Exception as exc:
        raise ValueError(f"could not read this file as DICOM: {exc}") from exc

    fields: dict[str, float] = {}
    for tag, column in TAG_TO_COLUMN.items():
        raw = getattr(ds, tag, None)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        # PatientAge is DICOM's own "nnnY"/"nnnM" age-string format (PS3.5).
        if tag == "PatientAge":
            digits = "".join(c for c in text if c.isdigit())
            if digits:
                fields[column] = float(digits)
            continue
        if tag == "PatientSex":
            fields[column] = 1.0 if text.upper().startswith("M") else 0.0
            continue
        try:
            fields[column] = float(text)
        except ValueError:
            continue

    descriptive: dict[str, str] = {}
    for tag in DESCRIPTIVE_TAGS:
        raw = getattr(ds, tag, None)
        if raw is not None and str(raw).strip():
            descriptive[tag] = str(raw).strip()

    notes = [
        f"read {len(fields)} demographic/technical field(s) from the DICOM header: "
        f"{', '.join(fields) or '(none)'}",
        "pixel data was NOT analysed - radiomics/CNN feature extraction from the "
        "image itself is a separate model this pipeline does not build",
    ]
    if descriptive.get("Modality"):
        notes.append(f"modality: {descriptive['Modality']}")

    return {"fields": fields, "descriptive": descriptive, "notes": notes}
