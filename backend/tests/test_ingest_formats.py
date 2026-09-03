"""Tests for `qhealth_qml.ingest` and its wiring into `standardize()`: format
detection, and each adapter (CSV/FHIR/HL7/PDF/DICOM) actually producing
correctly-shaped, correctly-valued output usable as model input.

Fixtures live in tests/fixtures/ingest/ - real files, not hand-typed text:
a FHIR bundle and HL7 feed already used elsewhere in this project, a PDF
generated with reportlab (a genuine text layer, not a mock), and a DICOM
file generated with pydicom (genuine DICOM tag structure, not a stub).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml import standardize as std
from qhealth_qml.ingest import sniff_format, UnrecognizedFormatError
from qhealth_qml.ingest.dicom_adapter import extract_dicom_fields
from qhealth_qml.ingest.pdf_adapter import OcrEngineMissingError, extract_pdf_text

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
MANUAL_CHECK = Path(__file__).parent / "manual_check"

def _tesseract_available() -> bool:
    """Same lookup `pdf_adapter._locate_tesseract_binary` does - PATH first,
    then the common Windows install location a silent installer often
    skips adding to PATH - so this test file's skip/run decisions agree
    with what `extract_pdf_text()` will actually do."""

    if shutil.which("tesseract"):
        return True
    from qhealth_qml.ingest.pdf_adapter import _WINDOWS_FALLBACK_PATHS

    return any(Path(p).exists() for p in _WINDOWS_FALLBACK_PATHS)


HAS_TESSERACT = _tesseract_available()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_sniff_format_by_extension():
    assert sniff_format("a,b\n1,2\n", "cohort.csv") == "csv"
    assert sniff_format((FIXTURES / "sample-fhir-bundle.json").read_bytes(), "bundle.json") == "fhir"
    assert sniff_format((FIXTURES / "sample-hl7-feed.hl7").read_bytes(), "feed.hl7") == "hl7"
    assert sniff_format((FIXTURES / "sample-cardiology-report.pdf").read_bytes(), "report.pdf") == "pdf"
    assert sniff_format((FIXTURES / "sample-mri-brain.dcm").read_bytes(), "scan.dcm") == "dicom"


def test_sniff_format_by_content_without_extension():
    # No filename hint at all - falls back to content sniffing.
    fhir_bytes = (FIXTURES / "sample-fhir-bundle.json").read_bytes()
    assert sniff_format(fhir_bytes, "upload") == "fhir"
    dicom_bytes = (FIXTURES / "sample-mri-brain.dcm").read_bytes()
    assert sniff_format(dicom_bytes, "upload") == "dicom"
    pdf_bytes = (FIXTURES / "sample-cardiology-report.pdf").read_bytes()
    assert sniff_format(pdf_bytes, "upload") == "pdf"


def test_sniff_format_rejects_nonsense():
    with pytest.raises(UnrecognizedFormatError):
        sniff_format(b"\xff\xfe\x00\x01 not anything", "mystery.bin")


# ---------------------------------------------------------------------------
# PDF: native text layer
# ---------------------------------------------------------------------------


def test_pdf_native_text_extraction_reads_real_content():
    text, pages, ocr_used = extract_pdf_text(FIXTURES / "sample-cardiology-report.pdf")
    assert pages == 1
    assert ocr_used is False
    assert "Troponin I" in text
    assert "0.09" in text


def test_standardize_pdf_predict_against_heart_disease(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "heart_disease_sample.csv", "heart-disease", schema_store=store)
    fitted = store.load("heart-disease")

    X, y = std.standardize(FIXTURES / "sample-cardiology-report.pdf", "heart-disease", schema_store=store)

    assert y is None
    assert X.shape == (1, len(fitted.feature_names))
    row = dict(zip(fitted.feature_names, X[0]))
    # From the report's actual text: "Age: 61", "Systolic BP: 148",
    # "Total Cholesterol: 238" - verified against the source PDF, not
    # assumed.
    assert row["age"] == 61.0
    assert row["trestbps"] == 148.0
    assert row["chol"] == 238.0
    # Exam/interpretation findings a lab report genuinely doesn't contain.
    assert np.isnan(row["cp"])
    assert np.isnan(row["thal"])


def test_standardize_pdf_predict_matches_zero_fields_raises(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "heart_disease_sample.csv", "heart-disease", schema_store=store)

    # A PDF whose only recognizable label ("MMSE") isn't a heart-disease
    # field at all - zero matches, not a partial one.
    import io

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 700, "MMSE Score: 28")
    c.save()
    buf.seek(0)

    with pytest.raises(std.SchemaMismatchError):
        std.standardize(buf, "heart-disease", schema_store=store, filename="irrelevant.pdf")


# ---------------------------------------------------------------------------
# PDF: scanned (OCR)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract OCR engine not installed on this machine")
def test_standardize_scanned_pdf_predict_via_ocr(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "alzheimers_sample.csv", "alzheimers", schema_store=store)
    fitted = store.load("alzheimers")

    X, y = std.standardize(
        FIXTURES / "sample-scanned-alzheimers-report.pdf", "alzheimers", schema_store=store
    )
    assert y is None
    assert X.shape == (1, len(fitted.feature_names))
    row = dict(zip(fitted.feature_names, X[0]))
    assert row["Age"] == 83.0
    assert row["MMSE"] == 19.0


@pytest.mark.skipif(HAS_TESSERACT, reason="only demonstrates the missing-engine error path")
def test_scanned_pdf_without_tesseract_raises_clear_error(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "alzheimers_sample.csv", "alzheimers", schema_store=store)

    with pytest.raises(std.UnsupportedFormatError, match="Tesseract"):
        std.standardize(
            FIXTURES / "sample-scanned-alzheimers-report.pdf", "alzheimers", schema_store=store
        )


def test_extract_pdf_text_raises_ocr_engine_missing_directly():
    if HAS_TESSERACT:
        pytest.skip("Tesseract is installed in this environment")
    with pytest.raises(OcrEngineMissingError):
        extract_pdf_text(FIXTURES / "sample-scanned-alzheimers-report.pdf")


# ---------------------------------------------------------------------------
# DICOM: header metadata only
# ---------------------------------------------------------------------------


def test_extract_dicom_fields_reads_real_tags():
    result = extract_dicom_fields(FIXTURES / "sample-mri-brain.dcm")
    assert result["fields"]["age"] == 79.0
    assert result["fields"]["sex"] == 0.0  # 'F'
    assert result["descriptive"]["Modality"] == "MR"
    assert any("pixel data was NOT analysed" in n for n in result["notes"])


def test_standardize_dicom_predict_against_alzheimers(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "alzheimers_sample.csv", "alzheimers", schema_store=store)
    fitted = store.load("alzheimers")

    X, y = std.standardize(FIXTURES / "sample-mri-brain.dcm", "alzheimers", schema_store=store)

    assert y is None
    assert X.shape == (1, len(fitted.feature_names))
    row = dict(zip(fitted.feature_names, X[0]))
    assert row["Age"] == 79.0
    # Not in a DICOM header - cognitive-test scores and radiomics volumetrics
    # need a real assessment/segmentation step this pipeline doesn't run.
    assert np.isnan(row["MMSE"])
    assert np.isnan(row["eTIV"])


def test_dicom_predict_before_any_training_upload_raises_not_fitted(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    with pytest.raises(std.SchemaNotFittedError):
        std.standardize(FIXTURES / "sample-mri-brain.dcm", "alzheimers", schema_store=store)


# ---------------------------------------------------------------------------
# FHIR and HL7: multi-row sources reuse the existing CSV pipeline unchanged
# ---------------------------------------------------------------------------


def test_fhir_bundle_parses_to_real_patient_rows():
    # The FHIR adapter's own contract, independent of any disease: a real
    # 60-patient Synthea-style bundle -> one row per patient, generic
    # LOINC-derived column names, a label column derived from Condition
    # ICD-10 codes.
    from qhealth_qml.ingest.fhir_adapter import parse_fhir_bundle

    text = (FIXTURES / "sample-fhir-bundle.json").read_text(encoding="utf-8-sig")
    headers, rows, notes = parse_fhir_bundle(text)
    assert headers[:3] == ["patient_id", "age", "sex"]
    assert "label" in headers
    assert len(rows) == 60


def test_standardize_fhir_bundle_predicts_against_heart_disease(tmp_path):
    # A FHIR bundle's label column is always the generic ICD-10-derived
    # "label" - it never happens to match a specific disease's own
    # `target_column` name ("heart_disease", "diagnosis", ...), so a bundle
    # is realistically always a PREDICT-time source, same posture as HL7.
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "heart_disease_sample.csv", "heart-disease", schema_store=store)
    fitted = store.load("heart-disease")

    X, y = std.standardize(FIXTURES / "sample-fhir-bundle.json", "heart-disease", schema_store=store)
    assert y is None
    assert X.shape == (60, len(fitted.feature_names))
    age_index = fitted.feature_names.index("age")
    assert not np.all(np.isnan(X[:, age_index]))


def test_standardize_hl7_feed_predicts_against_heart_disease(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "heart_disease_sample.csv", "heart-disease", schema_store=store)
    fitted = store.load("heart-disease")

    X, y = std.standardize(FIXTURES / "sample-hl7-feed.hl7", "heart-disease", schema_store=store)
    assert y is None
    assert X.shape == (6, len(fitted.feature_names))
    # Only 'age' overlaps between the HL7 feed's lab-derived columns and
    # heart-disease's own UCI column names - real, honest partial coverage.
    age_index = fitted.feature_names.index("age")
    assert not np.isnan(X[0][age_index])


# ---------------------------------------------------------------------------
# Plain images (mammogram, MRI, CT, histopathology, ECG, angiogram, EEG as
# PNG/JPEG - NOT a DICOM export): every modality is identified and reported
# on individually, and every one of them honestly extracts nothing, since a
# plain image carries no header tags (unlike DICOM) and pixel analysis is
# out of scope.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected_modality",
    [
        ("sample-mammogram.png", "mammogram"),
        ("sample-ecg-waveform.png", "ECG waveform"),
    ],
)
def test_sniff_and_reject_plain_images_by_modality(filename, expected_modality):
    path = FIXTURES / filename
    assert sniff_format(path, filename) == "image"

    from qhealth_qml import ingest as ingest_pkg

    parsed = ingest_pkg.ingest(path, filename)
    assert parsed.format == "image"
    assert parsed.fields == {}
    assert any(expected_modality in note for note in parsed.notes)


def test_standardize_plain_image_raises_unsupported_format_not_schema_not_fitted(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(MANUAL_CHECK / "heart_disease_sample.csv", "heart-disease", schema_store=store)

    # Even with a fitted schema already on hand, a plain image is rejected
    # for what it fundamentally is - not "no layout yet".
    with pytest.raises(std.UnsupportedFormatError, match="mammogram"):
        std.standardize(FIXTURES / "sample-mammogram.png", "breast-cancer", schema_store=store)


def test_standardize_plain_image_before_any_fit_still_names_the_real_reason(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    # No prior labeled upload for this disease at all - the image-specific
    # rejection still fires first, not a "not fitted yet" error, since a
    # plain image would never contribute a field regardless.
    with pytest.raises(std.UnsupportedFormatError, match="no structured fields"):
        std.standardize(FIXTURES / "sample-mammogram.png", "breast-cancer", schema_store=store)
