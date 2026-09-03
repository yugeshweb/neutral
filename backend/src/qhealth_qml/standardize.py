"""Standardizer pipeline: raw upload + disease_id -> (X, y).

This module is the boundary between "whatever a user uploaded" and the
canonical numeric matrix every disease's training/inference lane consumes.
It corresponds to box C ("Standardizer Pipeline") in the platform flow:

    Upload + disease_id -> [API layer] -> [Standardizer (this module)]
        -> canonical (X, y) -> [split, leakage-free fit/scale/reduce]
        -> classical + quantum lanes -> metrics -> persisted artifacts

Deliberately NOT this module's job (owned by the training backend, downstream):
  - train/test splitting
  - imputation/scaling/dimensionality-reduction *fitting* (must be fit on the
    train fold only, so it cannot live in a function that doesn't know about
    a split — see `experiment.PreprocessingPipeline` for that step)
  - model training, evaluation, persistence of model weights

What this module *does* guarantee: given the same disease_id, the feature
matrix has the same column order and width every time, whether the call is
standardizing a labeled training file or an unlabeled predict-time upload.
That guarantee is the single thing that has to hold for the standardizer and
the training backend to never disagree — see STANDARDIZER_CONTRACT.md.

Public interface
-----------------
    standardize(raw_file, disease_id) -> (X, y)
        X: np.ndarray, shape (n_rows, n_features), float64.
        y: np.ndarray of 0/1 ints (shape (n_rows,)) when `raw_file` contains
           the disease's label column (training/benchmark upload), else None
           (predict upload — inference only).

    list_supported_diseases() -> list[dict]
        For a disease selector/router: id, display name, modality, the raw
        columns a file must contain, and free-text notes.

    get_disease_schema(disease_id) -> DiseaseSchema
        Full schema for one disease, for anything that wants to introspect
        expected columns before uploading (e.g. a "what does this pipeline
        need" panel).

Every disease this module knows about is backed by an existing, audited
profile under backend/profiles/ and a model contract under
backend/src/qhealth_qml/platform/registry_data/models/ — nothing here is
invented; the registry below only *points at* those files rather than
duplicating their contents, so there is exactly one place each disease's
raw-column contract can drift from what the trained model actually expects.

Forward compatibility with a real trained model, once one exists
------------------------------------------------------------------
For a registry-backed disease, a model contract already declares exactly
where its trained artifact will live (`runtime/models/<model_id>.joblib`,
relative to the repo root — the same path `execution.py`'s training/scoring
code writes to and reads from). `standardize()`'s predict path checks that
exact path on every call: if a real `SavedModelArtifact` is sitting there,
its *own* fitted column layout (`artifact.preprocessor.feature_names` +
`artifact.dataset["provenance"]["categorical_columns"]`) is used, in
preference to this module's own standalone `SchemaStore` cache. Nothing
needs to change in this module, the training backend, or the frontend for
that handoff to happen — the moment a teammate's training run drops a real
artifact at that path, predict-time standardization starts reading it
automatically. See `_load_artifact_schema()` and STANDARDIZER_CONTRACT.md.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO, Union

import numpy as np

from . import ingest as _ingest_pkg
from .experiment import encode_raw_row, load_csv_dataset, load_model_artifact
from .ingest import CanonicalTable, SingleCaseFields, UnrecognizedFormatError
from .ingest.pdf_adapter import OcrEngineMissingError, PdfTextExtractionError

# --------------------------------------------------------------------------
# Errors. Every failure mode raises one of these, never a bare ValueError -
# a caller (an HTTP layer, a CLI, a test) can catch `StandardizationError`
# and always get a `.disease_id` and a human-readable message.
# --------------------------------------------------------------------------


class StandardizationError(Exception):
    """Base class for every error this module raises."""

    def __init__(self, message: str, *, disease_id: str | None = None):
        super().__init__(message)
        self.disease_id = disease_id


class UnknownDiseaseError(StandardizationError):
    """`disease_id` is not in the registry."""


class UnsupportedFormatError(StandardizationError):
    """`raw_file` could not be read as a CSV at all."""


class EmptyDatasetError(StandardizationError):
    """The file parsed, but contained zero data rows."""


class SchemaMismatchError(StandardizationError):
    """The file is missing columns the disease's schema requires."""

    def __init__(self, message: str, *, disease_id: str, missing_fields: list[str]):
        super().__init__(message, disease_id=disease_id)
        self.missing_fields = missing_fields


class SchemaNotFittedError(StandardizationError):
    """An unlabeled (predict) upload arrived before any labeled upload ever
    established this disease's canonical column layout.

    Standardizing an unlabeled file has to reproduce the exact column order
    and one-hot expansion a labeled file produced earlier (fit once, applied
    consistently) - see STANDARDIZER_CONTRACT.md. There is nothing to
    reproduce yet.
    """


# --------------------------------------------------------------------------
# Disease registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiseaseSchema:
    disease_id: str
    display_name: str
    target_column: str
    positive_label: str
    id_column: str | None
    group_column: str | None
    required_fields: tuple[str, ...]
    """Raw columns a file must contain for this disease, in the order the
    model contract declares them. Empty tuple means "no fixed whitelist":
    every column in the file other than target/id/group is a feature (used
    for diseases whose raw input is already a feature-extraction script's
    output, e.g. EEG window features or radiomics - see `notes`)."""
    modality: str
    reduction: str
    profile_path: str
    model_id: str | None
    """None for a disease with no entry yet under platform/registry_data/models/
    - i.e. no model has been trained or evaluated for it. `status` says so
    explicitly rather than leaving that implicit."""
    status: str = "trained_model_registered"
    """"trained_model_registered" (a full, audited ModelDefinition exists in
    the platform registry - `load_registry()` will find it) or
    "no_trained_model_yet" (only the raw-column contract exists so far;
    standardizing still works, there's just nothing downstream to train
    against yet)."""
    clinical_context: dict[str, Any] = field(default_factory=dict)
    """typical_workup / this_pipeline_covers / gap, straight from the
    profile, for a disease-selector UI to show honestly what this pipeline
    does and doesn't cover."""
    notes: str = ""


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_DIR = REPO_ROOT / "backend" / "profiles"
MODELS_DIR = Path(__file__).resolve().parent / "platform" / "registry_data" / "models"

# disease_id -> profile filename, optionally paired with a model contract
# filename under platform/registry_data/models/.
#
# A disease with a model contract is "registry-backed": required_fields
# comes from that audited, platform-registry-validated file (the same one
# `load_registry()` reads), and adding one here requires that file to be a
# fully valid ModelDefinition (registry.py's 7 validation rules) - don't
# add a bare-bones file there just to satisfy this module, it'll break
# `load_registry()` for every other caller.
#
# A disease with no model contract yet is "profile-embedded": its
# `required_fields`/`display_name`/`clinical_context` live directly in the
# profile JSON instead (harmless extra keys `load_early_detection_profile`
# ignores). That's the honest state for a disease nobody has trained or
# evaluated a model for yet - see each profile's "status" field.
_DISEASE_SOURCES: dict[str, tuple[str, str | None]] = {
    # heart-disease and breast-cancer are this platform's own tabular demo
    # pipelines (Cleveland UCI cohort; WDBC biopsy features respectively) -
    # neither corresponds to a condition in the neuro-conditions research
    # program below. See STANDARDIZER_CONTRACT.md for what each does and
    # does not cover, including how "heart-disease" here differs from the
    # separate raw-PTB-XL-ECG-waveform MI/abnormal-ECG detection effort
    # elsewhere in this repo (backend/run_raw_hybrid.py).
    "heart-disease": ("heart_disease_clinical.json", None),
    "breast-cancer": ("breast_cancer_wdbc.json", None),
    # The six neuro-conditions research program below. P2 ICH is
    # deliberately absent: registered `not available` upstream (no lawful
    # dataset found after two exhaustive checks) - there is nothing to
    # standardize for it, and adding a placeholder entry would misrepresent
    # that as a working pipeline.
    "stroke": ("p1_stroke_clinical.json", "stroke-clinical-risk-tabular.json"),
    # No dedicated "general brain tumor detection" profile exists - the one
    # audited pipeline in this area is glioma MGMT-methylation status from
    # mpMRI radiomics, a much narrower task than "detect a tumor from a
    # CT/MRI + confirming biopsy". Exposed under the "brain-tumor" id with
    # that gap stated in `notes`, rather than silently overclaiming scope.
    "brain-tumor": ("p3_glioma_mgmt.json", "glioma-mgmt-radiomics-tabular.json"),
    "seizure": ("p4_seizure_eeg.json", "seizure-window-risk-tabular.json"),
    "alzheimers": ("p5_alzheimers_clinical.json", "alzheimers-clinical-risk-tabular.json"),
    "parkinsons": ("p6_parkinsons_clinical.json", "parkinsons-voice-risk-tabular.json"),
}

_NOTES_FOR_EMPTY_REQUIRED_FIELDS = (
    "This disease's model contract declares no fixed raw-column whitelist: "
    "its raw input is already the output of a feature-extraction script "
    "(see backend/data/<profile>/*.py), not something this module extracts "
    "from a raw signal/image itself. Upload the extracted-feature CSV, not "
    "the source EEG/imaging file - raw-signal ingestion is a stub, matching "
    "the frontend's DICOM/VCF adapters (fails loudly, not silently)."
)

_NOTES_FOR_BRAIN_TUMOR = (
    "Scope, stated plainly: the only audited pipeline here is glioma MGMT-"
    "promoter-methylation status from mpMRI radiomics - a molecular-"
    "characterization proxy, not general tumor detection or grading. It "
    "does not ingest CT/MRI images directly (radiomics feature-extraction "
    "is a separate, unimplemented step here) or biopsy pathology reports. "
    + _NOTES_FOR_EMPTY_REQUIRED_FIELDS
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StandardizationError(f"registry file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StandardizationError(f"registry file is not valid JSON: {path}") from exc


def _build_registry() -> dict[str, DiseaseSchema]:
    registry: dict[str, DiseaseSchema] = {}
    for disease_id, (profile_name, model_name) in _DISEASE_SOURCES.items():
        profile = _read_json(PROFILES_DIR / profile_name)
        model = _read_json(MODELS_DIR / model_name) if model_name else None

        if model is not None:
            required_fields = tuple(model.get("input_contract", {}).get("required_fields", ()))
            display_name = model.get("display_name", disease_id)
            model_id = model["model_id"]
            status = "trained_model_registered"
        else:
            required_fields = tuple(profile.get("required_fields", ()))
            display_name = profile.get("display_name", disease_id)
            model_id = None
            status = profile.get("status", "no_trained_model_yet")

        if disease_id == "brain-tumor":
            notes = _NOTES_FOR_BRAIN_TUMOR
        else:
            notes = "" if required_fields else _NOTES_FOR_EMPTY_REQUIRED_FIELDS

        registry[disease_id] = DiseaseSchema(
            disease_id=disease_id,
            display_name=display_name,
            target_column=profile["target_column"],
            positive_label=str(profile.get("positive_label") or "1"),
            id_column=profile.get("id_column"),
            group_column=profile.get("group_column"),
            required_fields=required_fields,
            modality=profile.get("modality", "tabular"),
            reduction=profile.get("reduction", "anova"),
            profile_path=str((PROFILES_DIR / profile_name).relative_to(REPO_ROOT)),
            model_id=model_id,
            status=status,
            clinical_context=dict(profile.get("clinical_context", {})),
            notes=notes,
        )
    return registry


_REGISTRY: dict[str, DiseaseSchema] = _build_registry()


def list_supported_diseases() -> list[dict[str, Any]]:
    """For a disease selector/router: one entry per pipeline this module can standardize."""

    return [
        {
            "disease_id": schema.disease_id,
            "display_name": schema.display_name,
            "modality": schema.modality,
            "required_fields": list(schema.required_fields),
            "target_column": schema.target_column,
            "status": schema.status,
            "clinical_context": schema.clinical_context,
            "notes": schema.notes,
        }
        for schema in _REGISTRY.values()
    ]


def get_disease_schema(disease_id: str) -> DiseaseSchema:
    try:
        return _REGISTRY[disease_id]
    except KeyError:
        supported = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise UnknownDiseaseError(
            f"unknown disease_id {disease_id!r}; supported: {supported}",
            disease_id=disease_id,
        ) from None


# --------------------------------------------------------------------------
# Raw file -> rows
# --------------------------------------------------------------------------

RawFile = Union[str, Path, bytes, TextIO, BinaryIO]


def _parse_csv_rows(text: str, *, disease_id: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise UnsupportedFormatError(
            "CSV has no header row", disease_id=disease_id
        )
    rows = list(reader)
    if not rows:
        raise EmptyDatasetError(
            "CSV has a header but no data rows", disease_id=disease_id
        )
    return list(fieldnames), rows


# --------------------------------------------------------------------------
# Column validation
# --------------------------------------------------------------------------


def _resolve_feature_columns(
    fieldnames: list[str], schema: DiseaseSchema, *, has_target: bool
) -> list[str]:
    """Return exactly the raw columns to treat as features, validating that
    everything the schema requires is present. Extra columns in the file
    (patient name, MRN, notes, whatever else a real export carries) are
    ignored, not rejected - matches "extract what's needed, ignore the rest".
    """

    if schema.required_fields:
        missing = [name for name in schema.required_fields if name not in fieldnames]
        if missing:
            raise SchemaMismatchError(
                f"{schema.display_name} requires columns {missing} which are "
                f"not present in the uploaded file (found: {fieldnames})",
                disease_id=schema.disease_id,
                missing_fields=missing,
            )
        return list(schema.required_fields)

    excluded = {schema.target_column, schema.id_column, schema.group_column}
    feature_columns = [name for name in fieldnames if name not in excluded]
    if not feature_columns:
        raise SchemaMismatchError(
            f"{schema.display_name}: no feature columns left in the uploaded "
            f"file after excluding target/id/group columns (found: {fieldnames})",
            disease_id=schema.disease_id,
            missing_fields=[],
        )
    return feature_columns


# --------------------------------------------------------------------------
# Persisted schema cache - the thing that makes predict-time standardization
# reproduce train-time standardization exactly, without needing a full model
# artifact on hand. See STANDARDIZER_CONTRACT.md.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedSchema:
    disease_id: str
    raw_feature_columns: list[str]
    feature_names: list[str]
    """Post one-hot-expansion column names, in fixed order - what `X`'s
    columns actually are. A categorical source column `c` expands to
    `f"{c}={value}"` entries, one per value seen when this was fitted."""
    categorical_columns: dict[str, list[str]]
    fitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "raw_feature_columns": self.raw_feature_columns,
            "feature_names": self.feature_names,
            "categorical_columns": self.categorical_columns,
            "fitted_at": self.fitted_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FittedSchema":
        return cls(
            disease_id=data["disease_id"],
            raw_feature_columns=list(data["raw_feature_columns"]),
            feature_names=list(data["feature_names"]),
            categorical_columns={k: list(v) for k, v in data["categorical_columns"].items()},
            fitted_at=float(data.get("fitted_at", 0.0)),
        )


DEFAULT_SCHEMA_DIR = REPO_ROOT / "backend" / "runtime" / "standardizer_schemas"
RUNTIME_MODELS_DIR = REPO_ROOT / "runtime" / "models"
"""Where a registry-backed model's real trained artifact lands once someone
trains and persists one (`execution.py`'s default `runtime_dir`,
`REPO_ROOT / "runtime"`). Nothing here writes to it - `standardize()` only
reads from it, and only if a file is actually there."""


def _artifact_path_for(schema: DiseaseSchema) -> Path | None:
    if schema.model_id is None:
        return None
    return RUNTIME_MODELS_DIR / f"{schema.model_id}.joblib"


def _load_artifact_schema(schema: DiseaseSchema) -> FittedSchema | None:
    """The moment a teammate's training run drops a real `SavedModelArtifact`
    at the path its own registry entry declares, predict-time standardization
    switches to reading *that* artifact's actual fitted column layout instead
    of this module's own standalone cache - no code change required on
    either side, since both already agree on the artifact's path (the
    registry contract) and its shape (`experiment.SavedModelArtifact`).
    Returns None, not an error, when no artifact exists yet - that's the
    normal state before training has happened, and `standardize()` falls
    back to its own `SchemaStore` cache in that case.
    """

    path = _artifact_path_for(schema)
    if path is None or not path.exists():
        return None

    try:
        artifact = load_model_artifact(path)
        feature_names = list(artifact.preprocessor.feature_names)
        categorical_columns = {
            str(k): [str(v) for v in vs]
            for k, vs in artifact.dataset.get("provenance", {}).get("categorical_columns", {}).items()
        }
    except Exception as exc:
        # A file existing at the declared path but failing to load is not
        # "no model yet" - it's a real problem (corrupt pickle, incompatible
        # schema_version, wrong artifact entirely) that silently falling back
        # to a possibly-stale cache would hide, not fix.
        raise StandardizationError(
            f"{schema.display_name}: found a trained artifact at {path} but "
            f"could not load it ({exc}) - this needs attention, not a silent "
            f"fallback to a different column layout.",
            disease_id=schema.disease_id,
        ) from exc

    raw_feature_columns: list[str] = []
    seen: set[str] = set()
    for name in feature_names:
        source_column = name.split("=", 1)[0] if "=" in name and name.split("=", 1)[0] in categorical_columns else name
        if source_column not in seen:
            seen.add(source_column)
            raw_feature_columns.append(source_column)

    return FittedSchema(
        disease_id=schema.disease_id,
        raw_feature_columns=raw_feature_columns,
        feature_names=feature_names,
        categorical_columns=categorical_columns,
    )


class SchemaStore:
    """A tiny JSON-file cache of the last-fitted `FittedSchema` per disease.

    Lives under backend/runtime/ (gitignored, like every other trained
    artifact) so it's disposable: delete the directory and the next labeled
    upload re-establishes it from scratch.
    """

    def __init__(self, directory: str | Path = DEFAULT_SCHEMA_DIR):
        self.directory = Path(directory)

    def _path(self, disease_id: str) -> Path:
        return self.directory / f"{disease_id}.schema.json"

    def save(self, fitted: FittedSchema) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(fitted.disease_id).write_text(
            json.dumps(fitted.to_dict(), indent=2), encoding="utf-8"
        )

    def load(self, disease_id: str) -> FittedSchema | None:
        path = self._path(disease_id)
        if not path.exists():
            return None
        return FittedSchema.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# The public entry point
# --------------------------------------------------------------------------


def standardize(
    raw_file: RawFile,
    disease_id: str,
    *,
    schema_store: SchemaStore | None = None,
    filename: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Turn an uploaded file - CSV, FHIR R4 Bundle, HL7 v2 feed, PDF report
    (native text or scanned/OCR'd), or DICOM (header metadata) - into (X, y)
    for one disease pipeline. Format is detected automatically from the
    filename and content; nothing about it needs to be declared by the
    caller. See `qhealth_qml.ingest` for the per-format adapters.

    Multi-row sources (CSV/FHIR/HL7) - a genuine cohort:

    Training/benchmark upload (file contains the disease's label column):
    every raw feature column is coerced to numeric, common missing-value
    sentinels (na, n/a, none, null, -, ?, ...) become NaN, and any
    non-numeric column is one-hot encoded - identical logic to the training
    engine's own `load_csv_dataset`, because this function delegates to it.
    Returns (X, y) with y as a 0/1 int array (1 = schema.positive_label).

    Predict upload (no label column present): reuses the column layout the
    most recent training/benchmark call for this disease established (or a
    real trained artifact's, if one exists - see `_load_artifact_schema`),
    so the same source value always lands in the same column of X. Raises
    `SchemaNotFittedError` if no such layout has been fitted yet.

    Single-case sources (PDF/DICOM) - one patient/case, not a cohort, so
    always predict-mode, never a label: matched against the already-fitted
    column layout (same source as above) by canonical field name, filling
    whatever a report/header didn't contain with NaN rather than rejecting
    the upload - a partial match is the realistic, still-useful case for
    these formats. Raises `SchemaMismatchError` if literally nothing in the
    file matched anything the disease's model uses.

    Raises `UnknownDiseaseError`, `UnsupportedFormatError`,
    `EmptyDatasetError`, `SchemaMismatchError`, or `SchemaNotFittedError` -
    all subclasses of `StandardizationError` - never a bare exception.
    """

    schema = get_disease_schema(disease_id)
    store = schema_store or SchemaStore()

    parsed = _ingest_raw_file(raw_file, disease_id=disease_id, filename=filename)

    if isinstance(parsed, SingleCaseFields):
        if parsed.format == "image":
            # Always zero extractable fields for a plain PNG/JPEG/WEBP/BMP,
            # regardless of disease or whether a schema has been fitted yet -
            # raise that plainly rather than an unrelated "not fitted" error.
            raise UnsupportedFormatError(
                f"{schema.display_name}: {'; '.join(parsed.notes)}",
                disease_id=disease_id,
            )
        fitted = _load_artifact_schema(schema) or store.load(disease_id)
        if fitted is None:
            raise SchemaNotFittedError(
                f"{schema.display_name}: a {parsed.format.upper()} upload is a "
                f"single case, matched against an already-fitted column layout - "
                f"but none exists yet. Run standardize() once on a labeled CSV/"
                f"FHIR/HL7 cohort for this disease (or train a real model) first.",
                disease_id=disease_id,
            )
        X = _standardize_single_case(parsed, fitted, schema)
        return X, None

    text = parsed.csv_text
    fieldnames, rows = _parse_csv_rows(text, disease_id=disease_id)
    has_target = schema.target_column in fieldnames

    if has_target:
        # Training/benchmark data: `required_fields` genuinely all being
        # present is a meaningful thing to enforce here, since this becomes
        # the layout every future predict-time upload gets matched against.
        feature_columns = _resolve_feature_columns(fieldnames, schema, has_target=has_target)
        X, y, feature_names, categorical_columns = _standardize_labeled(
            text, feature_columns, schema
        )
        store.save(
            FittedSchema(
                disease_id=disease_id,
                raw_feature_columns=feature_columns,
                feature_names=feature_names,
                categorical_columns=categorical_columns,
            )
        )
        return X, y

    # A real trained artifact, once one exists at the path this disease's own
    # registry entry declares, is the authoritative column layout - it wins
    # over this module's own standalone cache automatically.
    fitted = _load_artifact_schema(schema) or store.load(disease_id)
    if fitted is None:
        raise SchemaNotFittedError(
            f"{schema.display_name}: no fitted column layout on file yet. "
            f"Run standardize() once on a file that contains the "
            f"'{schema.target_column}' column (a training or benchmark "
            f"upload) before predicting.",
            disease_id=disease_id,
        )
    missing = [name for name in fitted.raw_feature_columns if name not in fieldnames]
    if missing:
        if parsed.format == "csv":
            # A hand-prepared or manually-exported CSV is expected to have
            # complete columns - a gap here is a real data-quality problem
            # worth rejecting loudly, not silently NaN-filling around.
            raise SchemaMismatchError(
                f"{schema.display_name}: predict upload is missing columns the "
                f"fitted layout expects: {missing}",
                disease_id=disease_id,
                missing_fields=missing,
            )
        # FHIR/HL7: partial coverage is the realistic, expected case - a
        # generic-labs bundle or feed was never going to carry every raw
        # column a specific disease's model uses. `encode_raw_row` already
        # NaN-fills any column it doesn't find (`.get(name, "")`), so only
        # reject outright if literally nothing in this disease's schema was
        # present at all - same "zero matched" bar the PDF/DICOM path uses.
        if len(missing) == len(fitted.raw_feature_columns):
            raise SchemaMismatchError(
                f"{schema.display_name}: this {parsed.format.upper()} upload "
                f"matched none of the {len(fitted.raw_feature_columns)} raw "
                f"column(s) this disease's model uses (found: {fieldnames}) - "
                f"nothing usable to predict from.",
                disease_id=disease_id,
                missing_fields=missing,
            )
    X = _standardize_unlabeled(rows, fitted)
    return X, None


def _ingest_raw_file(
    raw_file: RawFile, *, disease_id: str, filename: str | None
) -> CanonicalTable | SingleCaseFields:
    """Wraps `qhealth_qml.ingest.ingest()`, translating its exceptions into
    this module's typed `StandardizationError` subclasses so callers only
    ever need to catch one exception family regardless of input format."""

    try:
        return _ingest_pkg.ingest(raw_file, filename)
    except UnrecognizedFormatError as exc:
        raise UnsupportedFormatError(str(exc), disease_id=disease_id) from exc
    except OcrEngineMissingError as exc:
        raise UnsupportedFormatError(str(exc), disease_id=disease_id) from exc
    except PdfTextExtractionError as exc:
        raise UnsupportedFormatError(str(exc), disease_id=disease_id) from exc
    except ValueError as exc:
        # FHIR/HL7/DICOM parse failures (malformed Bundle, no PID segments,
        # unreadable DICOM) - genuinely "this file, as given, can't be read",
        # same bucket as an unparseable CSV.
        raise UnsupportedFormatError(str(exc), disease_id=disease_id) from exc


def _standardize_single_case(
    parsed: SingleCaseFields, fitted: FittedSchema, schema: DiseaseSchema
) -> np.ndarray:
    """Builds the one-row matrix for a PDF/DICOM upload: fills whatever of
    the fitted layout's columns the extraction found (case-insensitive name
    match - the same widening `inference.ts`'s `scoreBatch` applies on the
    frontend, so a predict-time upload naming a measurement differently-cased
    than training still lands in the right column), NaN everywhere else."""

    lookup = {name.lower(): value for name, value in parsed.fields.items()}
    row = np.full(len(fitted.feature_names), np.nan, dtype=float)
    matched = 0
    for i, name in enumerate(fitted.feature_names):
        if name.lower() in lookup:
            row[i] = lookup[name.lower()]
            matched += 1

    if matched == 0:
        raise SchemaMismatchError(
            f"{schema.display_name}: this {parsed.format.upper()} upload matched "
            f"none of the {len(fitted.feature_names)} column(s) this disease's "
            f"model uses ({', '.join(parsed.fields) or 'nothing was extracted at all'} "
            f"found in the file) - nothing usable to predict from.",
            disease_id=schema.disease_id,
            missing_fields=list(fitted.feature_names),
        )

    return row.reshape(1, -1)


def _standardize_labeled(
    csv_text: str, feature_columns: list[str], schema: DiseaseSchema
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[str]]]:
    import tempfile

    # `load_csv_dataset` reads from a path and treats every non-excluded
    # header column as a feature, so the file handed to it is filtered down
    # to exactly [id?, group?, target, *feature_columns] first - that's how
    # "extract only what this disease needs" is enforced even when the
    # user's file has fifty extra columns it doesn't need.
    ordered_columns = [
        c for c in (schema.id_column, schema.group_column, schema.target_column) if c
    ] + feature_columns
    reader = csv.DictReader(io.StringIO(csv_text))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "standardized.csv"
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered_columns)
            writer.writeheader()
            for row in reader:
                writer.writerow({name: row.get(name, "") for name in ordered_columns})

        try:
            dataset = load_csv_dataset(
                tmp_path,
                target=schema.target_column,
                positive_label=schema.positive_label,
                id_column=schema.id_column,
                group_column=schema.group_column,
            )
        except ValueError as exc:
            raise SchemaMismatchError(
                f"{schema.display_name}: {exc}",
                disease_id=schema.disease_id,
                missing_fields=[],
            ) from exc

    categorical_columns = dataset.provenance.get("categorical_columns", {})
    return dataset.X, dataset.y, dataset.feature_names, categorical_columns


def _standardize_unlabeled(rows: list[dict[str, str]], fitted: FittedSchema) -> np.ndarray:
    try:
        encoded_rows = [
            encode_raw_row(row, fitted.feature_names, fitted.categorical_columns)
            for row in rows
        ]
    except ValueError as exc:
        raise SchemaMismatchError(
            f"{fitted.disease_id}: {exc}",
            disease_id=fitted.disease_id,
            missing_fields=[],
        ) from exc
    return np.vstack(encoded_rows)
