"""Canonical column name -> label variants a report/tag might use for it.

Shared vocabulary between the PDF and DICOM adapters (and, ported the other
direction, the frontend's `src/lib/ingest/pdf.ts`) so a measurement is named
the same way regardless of which format it arrived in. A disease's
`required_fields` (from its profile/model contract) are matched against
these canonical names, not the raw label text a report happens to use.
"""

from __future__ import annotations

import re

FIELD_ALIASES: dict[str, list[str]] = {
    "age": ["Age", "PatientAge"],
    "sex": ["Sex", "Gender", "PatientSex"],
    "bmi": ["BMI", "Body Mass Index"],
    "glucose": ["Fasting Glucose", "Blood Glucose", "Glucose"],
    "hba1c": ["HbA1c", "Hemoglobin A1c", "A1C"],
    "cholesterol_total": ["Total Cholesterol", "Cholesterol"],
    "hdl_cholesterol": ["HDL Cholesterol", "HDL"],
    "ldl_cholesterol": ["LDL Cholesterol", "LDL"],
    "triglycerides": ["Triglycerides"],
    "systolic_bp": ["Systolic BP", "Systolic Blood Pressure", "SBP"],
    "diastolic_bp": ["Diastolic BP", "Diastolic Blood Pressure", "DBP"],
    "heart_rate": ["Resting Heart Rate", "Heart Rate", "Pulse"],
    "max_heart_rate": ["Max Heart Rate", "Peak Heart Rate", "Maximum Heart Rate Achieved"],
    "troponin": ["Troponin I", "Troponin T", "hs-cTnT", "hs-cTnI", "Trop-T", "Troponin"],
    "creatinine": ["Creatinine"],
    "egfr": ["eGFR", "GFR"],
    "crp_high_sensitivity": ["hs-CRP", "CRP", "C-Reactive Protein"],
    "mmse": ["MMSE Score", "MMSE", "Mini-Mental State Exam", "Mini Mental State Examination"],
    "body_weight": ["Weight"],
    "body_height": ["Height"],
    "st_depression": ["ST Depression", "ST-Segment Depression"],
    # Same measurement, a different disease's own column name for it - kept
    # as extra canonical keys (not a rename) so a report matches whichever
    # disease's own naming convention it's being checked against.
    "resting_bp": ["Resting BP", "Systolic BP", "Systolic Blood Pressure"],
    "cholesterol": ["Cholesterol", "Total Cholesterol"],
    # The UCI Cleveland cohort's own abbreviated column names
    # (`heart_disease_clinical.json`'s `required_fields`) - a different
    # naming convention again from the two above.
    "trestbps": ["Resting BP", "Systolic BP", "Systolic Blood Pressure"],
    "chol": ["Cholesterol", "Total Cholesterol"],
    "thalach": ["Max Heart Rate", "Peak Heart Rate", "Maximum Heart Rate Achieved"],
}

ID_ALIASES = ["Patient ID", "MRN", "Medical Record Number", "Accession Number"]


def _escape(s: str) -> str:
    return re.escape(s)


def find_numeric(text: str, aliases: list[str]) -> float | None:
    """First numeric value following a label, e.g. "Troponin I: 0.02 ng/mL" -> 0.02."""

    for alias in aliases:
        match = re.search(rf"{_escape(alias)}\s*[:\-]?\s*([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def find_id(text: str) -> str | None:
    for alias in ID_ALIASES:
        match = re.search(rf"{_escape(alias)}\s*[:\-]?\s*([A-Za-z0-9-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_fields_from_text(text: str) -> dict[str, float]:
    """Runs every canonical field's aliases against free text, first match wins
    per field. Used by the PDF adapter on its extracted text layer/OCR output."""

    matched: dict[str, float] = {}
    for column, aliases in FIELD_ALIASES.items():
        value = find_numeric(text, aliases)
        if value is not None:
            matched[column] = value
    return matched
