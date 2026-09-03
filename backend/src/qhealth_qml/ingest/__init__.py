"""Format detection and dispatch for the standardizer.

One job: turn "whatever file this is" into either
  (a) canonical CSV text - for CSV, FHIR and HL7, which are naturally
      multi-row - so `standardize.py`'s existing, tested CSV pipeline
      (`_parse_csv_rows`, `_resolve_feature_columns`, `_standardize_labeled`,
      `_standardize_unlabeled`) runs on it completely unchanged, or
  (b) a single-case extracted-fields dict - for PDF and DICOM, which are one
      patient/case, not a cohort - matched against a disease's
      `required_fields` via `ingest.aliases`.

Mirrors `src/lib/ingest/index.ts`'s adapter-registry shape on the frontend,
so the two sides describe the same set of formats the same way.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Union

from .aliases import extract_fields_from_text, find_id
from .dicom_adapter import extract_dicom_fields
from .fhir_adapter import parse_fhir_bundle
from .hl7_adapter import parse_hl7_feed
from .pdf_adapter import OcrEngineMissingError, PdfTextExtractionError, extract_pdf_text

SourceFormat = Literal["csv", "fhir", "hl7", "pdf", "dicom", "image"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

RawFile = Union[str, Path, bytes, BinaryIO]


class UnrecognizedFormatError(Exception):
    """Neither the extension nor the content matched any known adapter."""


def _peek_bytes(raw_file: RawFile, filename: str | None) -> tuple[bytes, str]:
    """Returns (content_bytes, resolved_filename)."""

    if isinstance(raw_file, Path):
        if not raw_file.exists():
            raise UnrecognizedFormatError(f"path does not exist: {raw_file}")
        return raw_file.read_bytes(), filename or raw_file.name
    if isinstance(raw_file, str):
        # A multi-line string is content, never a path - constructing a Path
        # from a large text blob risks an OS "filename too long" error on
        # some platforms before `.exists()` even gets a chance to say no.
        maybe_path = Path(raw_file) if "\n" not in raw_file else None
        if maybe_path is not None and maybe_path.exists():
            return maybe_path.read_bytes(), filename or maybe_path.name
        return raw_file.encode("utf-8"), filename or "upload"
    if isinstance(raw_file, bytes):
        return raw_file, filename or "upload"
    if hasattr(raw_file, "read"):
        data = raw_file.read()
        data = data if isinstance(data, bytes) else data.encode("utf-8")
        return data, filename or getattr(raw_file, "name", "upload")
    raise UnrecognizedFormatError(f"unsupported input type: {type(raw_file).__name__}")


def sniff_format(raw_file: RawFile, filename: str | None = None) -> SourceFormat:
    """Extension first, content as a tiebreaker - a `.json` is only FHIR if
    it actually holds a Bundle; anything with no text-decodable content and
    a DICOM-shaped preamble is DICOM."""

    content, name = _peek_bytes(raw_file, filename)
    return _sniff(content, name)


def _sniff(content: bytes, name: str) -> SourceFormat:
    lower = name.lower()

    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".dcm", ".dicom")):
        return "dicom"
    if lower.endswith(".hl7"):
        return "hl7"
    if lower.endswith(IMAGE_EXTENSIONS):
        return "image"

    # DICOM files carry a 128-byte preamble then the literal bytes "DICM".
    if content[128:132] == b"DICM":
        return "dicom"
    if content[:5] == b"%PDF-":
        return "pdf"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if content[:3] == b"\xff\xd8\xff":  # JPEG
        return "image"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image"
    if content[:2] == b"BM":
        return "image"

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UnrecognizedFormatError(
            f"could not determine the format of {name!r} - not CSV, FHIR JSON, "
            "HL7, PDF, DICOM or a common image type by extension or content"
        ) from None

    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("resourceType") == "Bundle":
            return "fhir"
        raise UnrecognizedFormatError(
            f"{name!r} is JSON but not a FHIR Bundle (no resourceType: 'Bundle')"
        )
    if stripped.startswith("MSH|"):
        return "hl7"
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "," in first_line:
        return "csv"

    raise UnrecognizedFormatError(
        f"could not determine the format of {name!r} - not CSV, FHIR JSON, "
        "HL7, PDF, DICOM or a common image type by extension or content"
    )


@dataclass
class CanonicalTable:
    """Multi-row output (CSV/FHIR/HL7): ready to hand to the existing CSV
    pipeline unchanged."""

    format: SourceFormat
    csv_text: str
    notes: list[str] = field(default_factory=list)


@dataclass
class SingleCaseFields:
    """Single-row output (PDF/DICOM/image): a best-effort field extraction
    for one patient/case, matched via alias against whatever the caller's
    disease schema requires. `fields` is legitimately empty for a plain
    image - see `format == "image"` - not a bug, a plain PNG/JPEG carries no
    structured tags at all, unlike DICOM's header."""

    format: SourceFormat
    case_id: str
    fields: dict[str, float]
    notes: list[str] = field(default_factory=list)
    ocr_used: bool = False


def _rows_to_csv(headers: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def ingest(raw_file: RawFile, filename: str | None = None) -> CanonicalTable | SingleCaseFields:
    """Single entry point. Detects format, parses, and returns either a
    `CanonicalTable` (CSV/FHIR/HL7) or `SingleCaseFields` (PDF/DICOM)."""

    content, name = _peek_bytes(raw_file, filename)
    fmt = _sniff(content, name)

    if fmt == "csv":
        return CanonicalTable(format="csv", csv_text=content.decode("utf-8-sig"))

    if fmt == "fhir":
        headers, rows, notes = parse_fhir_bundle(content.decode("utf-8-sig"))
        return CanonicalTable(format="fhir", csv_text=_rows_to_csv(headers, rows), notes=notes)

    if fmt == "hl7":
        headers, rows, notes = parse_hl7_feed(content.decode("utf-8-sig"))
        return CanonicalTable(format="hl7", csv_text=_rows_to_csv(headers, rows), notes=notes)

    if fmt == "pdf":
        text, pages, ocr_used = extract_pdf_text(content)
        matched = extract_fields_from_text(text)
        case_id = find_id(text) or Path(name).stem
        notes = [
            (f"OCR'd {pages} page(s) with Tesseract" if ocr_used else f"read {pages} page(s) of embedded text")
            + f" ({len(text)} characters)",
            f"matched {len(matched)} recognized label(s): {', '.join(matched) or '(none)'}",
        ]
        return SingleCaseFields(format="pdf", case_id=case_id, fields=matched, notes=notes, ocr_used=ocr_used)

    if fmt == "dicom":
        result = extract_dicom_fields(content)
        case_id = Path(name).stem
        return SingleCaseFields(format="dicom", case_id=case_id, fields=result["fields"], notes=result["notes"])

    if fmt == "image":
        # Every imaging modality is one pipeline, same distinction the
        # frontend's `inputKinds.ts` makes - "which study is this" stays
        # visible even though none of them are analysed. A plain PNG/JPEG
        # (unlike DICOM) carries no header tags at all, so `fields` is
        # genuinely empty, not under-extracted.
        modality = _guess_image_modality(name)
        return SingleCaseFields(
            format="image",
            case_id=Path(name).stem,
            fields={},
            notes=[
                f"recognized as a {modality} image" if modality else "recognized as an image",
                "no structured fields are extractable from a plain PNG/JPEG/WEBP/BMP - "
                "it carries no header tags (unlike DICOM) and pixel data is not analysed "
                "(mammography CAD, radiomics, histopathology or ECG-waveform feature "
                "extraction, and CNN-based detection generally, are separate models this "
                "pipeline does not build)",
                "if this came from a scanner/PACS, its original DICOM export would at "
                "least carry header tags (age, sex, modality) - see the dicom adapter",
            ],
        )

    raise UnrecognizedFormatError(f"unhandled format: {fmt}")


_MODALITY_HINTS: dict[str, tuple[str, ...]] = {
    "mammogram": ("mammo",),
    "MRI": ("mri",),
    "CT": ("-ct-", "_ct_", "ct-head", "ct_head", "ct-scan"),
    "histopathology slide": ("histo", "biopsy", "slide"),
    "ECG waveform": ("ecg", "ekg"),
    "angiogram": ("angio",),
    "EEG recording": ("eeg",),
}


def _guess_image_modality(filename: str) -> str | None:
    """Filename-only heuristic, purely for a more specific note - never
    changes what gets extracted (always nothing, for a plain image)."""

    lower = filename.lower()
    for modality, hints in _MODALITY_HINTS.items():
        if any(hint in lower for hint in hints):
            return modality
    return None
