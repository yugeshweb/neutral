"""PDF report adapter (Python side).

Mirrors `src/lib/ingest/pdf.ts` exactly in approach: read the PDF's native
text layer first (pypdf); if there is none (a scanned or photographed
report), render each page to an image (PyMuPDF - pure Python, no poppler
binary needed) and OCR it. One PDF is one patient/case, same as the
TypeScript adapter - a clinical PDF report is not a cohort export.

Scope, stated plainly: OCR here needs the Tesseract *engine* installed on
this machine (`pytesseract` is only a wrapper around the `tesseract`
executable) - unlike the frontend's Tesseract.js, which bundles its own
WASM runtime and needs nothing preinstalled. Looked for on PATH first, then
the common Windows install location (silent/unattended installers often
skip the "add to PATH" step) - if truly not found anywhere,
`extract_pdf_text()` raises a clear, typed error naming exactly that,
rather than a cryptic subprocess failure.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

from pypdf import PdfReader


class PdfTextExtractionError(Exception):
    """Raised when neither a native text layer nor OCR could produce text."""


class OcrEngineMissingError(PdfTextExtractionError):
    """The PDF has no text layer and the Tesseract OCR engine is not
    installed/discoverable on this machine - a real, stated environment
    dependency, not a silent failure."""


PdfSource = Union[str, Path, bytes, BinaryIO]


def _read_bytes(source: PdfSource) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "read"):
        data = source.read()
        return data if isinstance(data, bytes) else data.encode("utf-8")
    raise PdfTextExtractionError(f"unsupported PDF source type: {type(source).__name__}")


# Windows installers (the UB-Mannheim Tesseract build in particular, run
# silently/unattended) commonly land the binary here without adding it to
# PATH - `pytesseract` otherwise has no way to find it. Checked only as a
# fallback, after PATH itself; never overrides a `tesseract` the user has
# already put on PATH deliberately.
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _locate_tesseract_binary(pytesseract_module) -> None:
    import shutil

    if shutil.which(pytesseract_module.pytesseract.tesseract_cmd or "tesseract"):
        return
    for candidate in _WINDOWS_FALLBACK_PATHS:
        if Path(candidate).exists():
            pytesseract_module.pytesseract.tesseract_cmd = candidate
            return
    # Nothing found anywhere - leave tesseract_cmd as-is and let
    # `get_tesseract_version()` fail with its own clear error.


def _native_text(pdf_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages), len(pages)


def _ocr_text(pdf_bytes: bytes, n_pages: int) -> str:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise PdfTextExtractionError(
            "PyMuPDF is not installed - cannot rasterize pages for OCR"
        ) from exc

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise PdfTextExtractionError(
            "pytesseract/Pillow are not installed - cannot OCR a scanned PDF"
        ) from exc

    _locate_tesseract_binary(pytesseract)

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OcrEngineMissingError(
            "this PDF has no native text layer (it looks scanned/photographed), "
            "and the Tesseract OCR engine is not installed or not on PATH on this "
            "machine, so it cannot be read here. Either install Tesseract "
            "(https://github.com/tesseract-ocr/tesseract) and rerun, or use the "
            "browser upload path, which OCRs scanned PDFs via a bundled WASM "
            "engine and needs nothing preinstalled."
        ) from exc

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    texts: list[str] = []
    for page in doc:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        texts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n".join(texts)


def extract_pdf_text(source: PdfSource) -> tuple[str, int, bool]:
    """Returns (text, page_count, ocr_used)."""

    pdf_bytes = _read_bytes(source)
    text, n_pages = _native_text(pdf_bytes)
    if text.strip():
        return text, n_pages, False

    text = _ocr_text(pdf_bytes, n_pages)
    if not text.strip():
        raise PdfTextExtractionError(
            f"read {n_pages} page(s) via OCR but recognized no text at all - "
            "the scan may be too degraded, rotated, or blank"
        )
    return text, n_pages, True
