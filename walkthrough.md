# Walkthrough: Standardizer & Ingestion Pipeline Integration

## Overview
We have integrated the newly pulled standardizer pipeline (`qhealth_qml.standardize`), format adapters (CSV, FHIR R4, HL7 v2, PDF with OCR, DICOM, EDF EEG, WFDB ECG, NIfTI 3D MRI, DICOM CT, Image 2D), and extractors into the 8-step training backend orchestration layer. All cross-component contracts, exception handling, and model registrations have been verified.

---

## Changes Made & Integration Points

### 1. Ingestion & Preprocessing Pipeline Unification
- **[`backend/src/qhealth_qml/pipeline/__init__.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/pipeline/__init__.py)**:
  - Unified exports from both the new ingestion pipeline (`Pipeline`, `Recipe`, `SourceSpec`, `AdapterRegistry`, `types`, etc.) and the 8-step training orchestration layer (`run_training_pipeline`, `register_disease_models`, `disease_registry`, `LeakageSafePreprocessor`, `app`, etc.).

### 2. Standardizer Contract & Aliasing
- **[`backend/src/qhealth_qml/standardize.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/standardize.py)**:
  - Added alias mapping `_DISEASE_ALIASES = {"glioma": "brain-tumor"}` so both `"glioma"` and `"brain-tumor"` resolve to the mpMRI radiomics schema cleanly while preserving the exact 7-disease registry format in `list_supported_diseases()`.
- **[`backend/src/qhealth_qml/pipeline/exceptions.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/pipeline/exceptions.py)**:
  - Re-exported exception classes directly from `qhealth_qml.standardize` (`StandardizationError`, `UnknownDiseaseError`, `UnsupportedFormatError`, `EmptyDatasetError`, `SchemaMismatchError`, `SchemaNotFittedError`) so `isinstance()` checks and FastAPI handlers operate on identical classes.

### 3. Training Dispatcher & API Layer
- **[`backend/src/qhealth_qml/pipeline/dispatcher.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/pipeline/dispatcher.py)**:
  - Seamlessly invokes `qhealth_qml.standardize.standardize(raw_file, disease_id)` on incoming datasets.
  - Automatically captures and forwards model-level metadata (e.g. quantum circuit topology, trees, custom tags) in `result["models"][model_type]["metadata"]`.
- **[`backend/src/qhealth_qml/pipeline/api.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/pipeline/api.py)**:
  - Hardened exception handlers using safe attribute access (`getattr(exc, "message", str(exc))` and `getattr(exc, "missing_fields", None)`), ensuring accurate HTTP status codes:
    * 404 for `UnknownDiseaseError`
    * 400 for `EmptyDatasetError` / `UnsupportedFormatError`
    * 422 for `SchemaMismatchError` (with `missing_fields`)
    * 409 for `SchemaNotFittedError`

### 4. Optional Dependency Handling & Fixture Support
- **[`backend/src/qhealth_qml/ingest/pdf_adapter.py`](file:///c:/Users/praty/Downloads/SIH2K26/neutral/backend/src/qhealth_qml/ingest/pdf_adapter.py)**:
  - Gracefully handles native text extraction via `pypdf` and OCR via `pymupdf` + `pytesseract`.
- Installed necessary signal and imaging libraries: `pypdf`, `pydicom`, `wfdb`, `mne`, `nibabel`, `pymupdf`, `pytesseract`.

---

## Verification Results

### Automated Test Suite
Ran full test suite covering standardizer, adapters, extractors, metamorphic testing, leakage prevention, API layer, and model dispatchers:
```bash
python -m pytest backend/tests/ -v
```
**Result**: `285 passed, 1 skipped in 39.54s (100% passing)`

### Frontend TypeScript Verification
```bash
npx tsc --noEmit
```
**Result**: Passed with zero type errors.
