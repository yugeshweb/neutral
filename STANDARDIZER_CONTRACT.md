# Standardizer contract

What the data-ingestion/preprocessing module (`qhealth_qml.standardize`)
guarantees to the training backend, and what it needs from it in return.

## The function

```python
from qhealth_qml.standardize import standardize

X, y = standardize(raw_file, disease_id)
```

- `raw_file`: a path, raw CSV text, CSV bytes, or a file-like object (an
  uploaded-file stream). CSV only for now — DICOM/PDF/raw-image ingestion is
  a documented stub, not silently unsupported (matches the frontend's
  DICOM/VCF adapters).
- `disease_id`: one of `"stroke"`, `"glioma"`, `"seizure"`, `"alzheimers"`,
  `"parkinsons"` — call `list_supported_diseases()` rather than hardcoding
  this list, it's read from the registry.
- Returns `X: np.ndarray` shaped `(n_rows, n_features)`, `float64`, NaN for
  missing values (not imputed — imputation must be fit on the train fold
  only, so it happens downstream in your `PreprocessingPipeline.fit`, not
  here).
- Returns `y: np.ndarray` of `0`/`1` ints, shape `(n_rows,)`, when the
  uploaded file contains that disease's label column (a training/benchmark
  upload). Returns `y = None` when it doesn't (a predict upload — the
  standardizer never fabricates a label).

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

## What's *not* this module's job

Split, imputation-fitting, scaling-fitting, dimensionality-reduction-fitting,
angle-scaling, training, evaluation, persistence of model weights. That's
all `experiment.PreprocessingPipeline` (already built, tested, and already
what a saved model artifact bundles) plus whatever orchestrates it — the
box right after this one in the flow. `standardize()` only guarantees `X`'s
columns are the same disease's same columns, same order, every time.

## Adding a disease

Add a `backend/profiles/<name>.json` and a model contract under
`backend/src/qhealth_qml/platform/registry_data/models/<model_id>.json`
(`required_fields` there is the standardizer's raw-column whitelist —
leave it `[]` if the raw input is already a feature-extraction script's
output, like seizure/glioma), then one line in
`standardize._DISEASE_SOURCES`. Nothing about a disease's schema is
hardcoded in `standardize.py` itself — it's read from those two files, so
there is exactly one place a disease's contract lives, not two that can
drift apart.
