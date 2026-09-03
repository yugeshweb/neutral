# Standardizer contract

What the data-ingestion/preprocessing module (`qhealth_qml.standardize`)
guarantees to the training backend, and what it needs from it in return.

## The function

```python
from qhealth_qml.standardize import standardize

X, y = standardize(raw_file, disease_id)
```

- `raw_file`: a path, raw text/bytes, or a file-like object (an
  uploaded-file stream). Format is auto-detected — CSV, FHIR R4 Bundle,
  HL7 v2 feed, PDF report (native text or scanned/OCR'd), or DICOM (header
  metadata) — via `qhealth_qml.ingest`, which mirrors the frontend's
  `src/lib/ingest/` adapters format-for-format (Python port, not a shared
  runtime — the two languages can't share code, but the vocabulary/label
  aliases and the LOINC/ICD-10 maps are kept identical on purpose). See
  "Which formats actually work" below for what each one can and can't do.
- `disease_id`: one of `"heart-disease"`, `"breast-cancer"`, `"stroke"`,
  `"brain-tumor"`, `"seizure"`, `"alzheimers"`, `"parkinsons"` — call
  `list_supported_diseases()` rather than hardcoding this list, it's read
  from the registry. Each entry also carries a `status`
  (`"trained_model_registered"` — a full audited `ModelDefinition` exists in
  the platform registry — vs `"no_trained_model_yet"` — the raw-column
  contract exists but no model has been trained/evaluated against it) and
  `clinical_context` (`typical_workup`, `this_pipeline_covers`, `gap` — what
  real diagnostic modality this pipeline's data comes from, and honestly,
  what it doesn't cover yet).

  Two different provenances, both real, worth knowing apart:
  - `stroke`, `brain-tumor`, `seizure`, `alzheimers`, `parkinsons` are 5 of
    the 6 conditions in this repo's neuro-conditions research program (see
    `specs/001-neurological-conditions/`) — audited profiles, registered
    models, some (`alzheimers`, `seizure`) already promoted
    `operational_reference` on real evaluation numbers. **`brain-tumor`** is
    that program's glioma MGMT-methylation-status pipeline specifically — a
    molecular-characterization proxy, not general tumor detection from a
    CT/MRI + confirming biopsy; see its `notes`. The 6th condition, **P2
    ICH, is deliberately not registered here** — it's `not available`
    upstream too, no lawful dataset was ever found for it, so there is
    nothing to standardize.
  - `heart-disease` and `breast-cancer` are this platform's own tabular
    demo pipelines (UCI Cleveland cohort; WDBC biopsy features), unrelated
    to the neuro-conditions program. **`heart-disease` is also not the same
    thing as the separate raw-PTB-XL-ECG-waveform MI/abnormal-ECG detection
    effort** documented elsewhere in this repo
    (`backend/run_raw_hybrid.py`) — that one takes raw 12-lead `.hea`/`.dat`
    WFDB waveforms through a CNN encoder, a completely different input
    shape none of `standardize()`'s adapters read, and it has not produced
    a result yet. If/when that pipeline needs its own standardizer, it
    belongs in a new module next to `raw_hybrid.py`, not bolted onto this
    one — the two have nothing in common except the word "heart."
- Returns `X: np.ndarray` shaped `(n_rows, n_features)`, `float64`, NaN for
  missing values (not imputed — imputation must be fit on the train fold
  only, so it happens downstream in your `PreprocessingPipeline.fit`, not
  here).
- Returns `y: np.ndarray` of `0`/`1` ints, shape `(n_rows,)`, when the
  uploaded file contains that disease's label column (a training/benchmark
  upload). Returns `y = None` when it doesn't (a predict upload — the
  standardizer never fabricates a label).

## Which formats actually work, and what "standardized" means for each

Two shapes of source, handled differently on purpose:

| Format | Shape | Train (has a label)? | Predict | Column matching |
|---|---|---|---|---|
| CSV | cohort (many rows) | Yes — `required_fields` must ALL be present, or it's rejected loudly. A hand-prepared training file with gaps is a real data-quality problem worth surfacing. | Yes | Exact column name |
| FHIR R4 Bundle | cohort (many rows) | No in practice — the label column is always the generic ICD-10-derived `"label"`, which never happens to match a specific disease's own `target_column` name | Yes, partial-match: NaN-fills whatever raw column the bundle didn't carry; only rejects if the disease's schema matched *nothing at all* | Exact column name (LOINC-derived names, e.g. `systolic_bp`, `cholesterol_total`) |
| HL7 v2 feed | cohort (many rows) | No — same reason as FHIR, no label at all | Yes, same partial-match rule as FHIR | Exact column name (OBX-3-derived) |
| PDF report | single case (1 row) | Never — one report is one patient, nothing to split into train/test | Yes, partial-match: matched via `ingest.aliases.FIELD_ALIASES` (canonical name -> label variants like `"Troponin I"`, `"MMSE Score"`) against the disease's already-fitted feature names, case-insensitively | Alias, not exact text |
| DICOM | single case (1 row) | Never | Yes, partial-match, same as PDF but from header tags (`PatientAge`, `PatientSex`, `PatientWeight`) — **pixel data is never analysed**; radiomics/CNN feature extraction from the image itself is a separate model this pipeline does not build | Alias, from a small DICOM-tag table |
| Plain image (PNG/JPEG/WEBP/BMP — mammogram, MRI, CT, histopathology slide, ECG waveform, angiogram, EEG recording as a picture, not a DICOM export) | single case (1 row) | Never | **Always rejected**, on purpose — recognized and named by modality (the filename-guessed one, e.g. "recognized as a mammogram image") for a clear error, but a plain image carries no header tags at all (unlike DICOM) and pixel data is never analysed, so there is nothing to extract, ever, regardless of disease | none — always `UnsupportedFormatError` |

Practical implication: **PDF, DICOM, FHIR and HL7 all need a fitted column
layout to exist first** (a labeled CSV upload, or a real trained artifact —
see below) — they only ever fill in *some* of it, never establish it from
scratch. Run a CSV upload for the disease you want to test one of these
against before trying it. Scanned PDFs additionally need the Tesseract OCR
engine installed on the machine running this — see `ingest/pdf_adapter.py`
for the exact error if it's missing. A plain image needs none of that setup
— it's rejected immediately and consistently, not because nothing has been
fitted yet, but because a picture with no header tags has nothing this
module could ever extract from it. **If a real imaging modality (mammogram,
MRI, CT, histopathology slide, ECG waveform, angiogram, EEG) needs to
actually contribute to a prediction, upload its DICOM export instead of a
plain picture** — DICOM at least carries header tags (age, sex, modality);
turning the pixels themselves into features needs a CNN/radiomics model
this pipeline does not build, the same boundary the frontend draws.

## Errors

Every failure raises a subclass of `StandardizationError`, never a bare
exception — safe to catch at the API layer and turn into a 4xx with
`.disease_id` and the message:

| Exception | When |
|---|---|
| `UnknownDiseaseError` | `disease_id` isn't registered |
| `UnsupportedFormatError` | file isn't parseable as CSV (or isn't UTF-8) |
| `EmptyDatasetError` | header parsed, zero data rows |
| `SchemaMismatchError` | required columns missing (`.missing_fields` lists them) |
| `SchemaNotFittedError` | a predict upload arrived before any training upload ever fit this disease's column layout |

## Why predict can fail with `SchemaNotFittedError`

A labeled upload's categorical columns (e.g. stroke's `gender`,
`work_type`) get one-hot encoded, and *which* one-hot columns exist depends
on which category values were actually seen in that file. An unlabeled
predict batch might be one row — nowhere near enough to safely re-derive
that layout on its own. So `standardize()` never re-derives it: the first
labeled call for a disease fits and persists the column layout
(`backend/runtime/standardizer_schemas/<disease_id>.schema.json`, gitignored
like every other runtime artifact), and every later unlabeled call for that
disease reuses it. **Practical implication for you:** run Benchmark/Train
(a labeled upload) for a disease before Predict is exercised against it in
the same environment — or, once real model artifacts exist, swap
`SchemaStore` for reading the layout out of the trained artifact itself
(`artifact.preprocessor.feature_names` / `categorical_columns`, already
persisted per model in `experiment.save_model_artifact` — see
`execution._run_tabular_inference` for the existing single-row version of
this exact reuse).

## What happens when your teammate pushes a real trained model

Nothing needs to change here, and I proved it rather than just claiming it
(`test_predict_prefers_a_real_trained_artifact_over_the_schema_cache` in
`tests/test_standardize.py` builds a real `SavedModelArtifact` via the
actual training engine and confirms `standardize()` picks it up
automatically). Mechanism: for a registry-backed disease, the model
contract already declares exactly where its trained artifact will live —
`runtime/models/<model_id>.joblib`, relative to the repo root, the same
path `execution.py` writes to and reads from. Every predict-mode call now
checks that exact path first:

- **Artifact exists** → its own fitted layout
  (`artifact.preprocessor.feature_names` +
  `artifact.dataset["provenance"]["categorical_columns"]`) is used, in
  preference to this module's standalone `SchemaStore` cache.
- **No artifact yet** (today's state, for every disease) → falls back to
  `SchemaStore` exactly as described above.
- **Artifact exists but won't load** (corrupt file, incompatible
  `schema_version`) → raises `StandardizationError` loudly. A silent
  fallback to a different, possibly-wrong column layout would be worse than
  an error here.

So: the day a real `stroke-clinical-risk-tabular.joblib` (or any other
registry-backed model) lands at that path, Predict-tab-style inference
against it starts working with zero changes to this module, the frontend,
or the training backend — the three sides already agree on the path (the
registry contract) and the shape (`SavedModelArtifact`).

## What's *not* this module's job

Split, imputation-fitting, scaling-fitting, dimensionality-reduction-fitting,
angle-scaling, training, evaluation, persistence of model weights. That's
all `experiment.PreprocessingPipeline` (already built, tested, and already
what a saved model artifact bundles) plus whatever orchestrates it — the
box right after this one in the flow. `standardize()` only guarantees `X`'s
columns are the same disease's same columns, same order, every time.

## Adding a disease

Two ways in, matching whether a real model exists yet:

- **A trained/audited model already exists** (has an entry under
  `backend/src/qhealth_qml/platform/registry_data/models/`, the file
  `load_registry()` actually validates): add a `backend/profiles/<name>.json`
  and reference both files as `(profile_filename, model_filename)` in
  `standardize._DISEASE_SOURCES`. `required_fields` comes from the model
  contract's `input_contract.required_fields` — leave it `[]` there if the
  raw input is already a feature-extraction script's output (no fixed raw
  whitelist), as with brain-tumor's radiomics features.
- **No model yet** (this is where heart-disease and breast-cancer are
  today): put `required_fields`, `display_name`, `status:
  "no_trained_model_yet"`, and `clinical_context` directly in the profile
  JSON instead, and pair it with `None` in `_DISEASE_SOURCES` — e.g.
  `"heart-disease": ("heart_disease_clinical.json", None)`. Deliberately
  does **not** touch `platform/registry_data/models/`: a bare-bones file
  there would make `load_registry()` — which every entry in that directory
  must satisfy, 7 validation rules including a non-empty evaluation-record
  list — either reject it or, worse, need fabricated evaluation numbers to
  pass. Add the real `ModelDefinition` there once a model is actually
  trained and evaluated, then move the disease to the first path above.

Nothing about a disease's schema is hardcoded in `standardize.py` itself —
it's read from the profile (and, once one exists, the model contract), so
there is exactly one place a disease's contract lives, not two that can
drift apart.
