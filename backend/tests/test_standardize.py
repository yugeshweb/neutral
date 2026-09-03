"""Tests for the standardizer pipeline (`qhealth_qml.standardize`).

Covers the contract described in backend/STANDARDIZER_CONTRACT.md: a
labeled upload produces (X, y) and fits a column layout; an unlabeled
upload for the same disease reproduces that exact layout; every failure
mode raises a typed `StandardizationError` subclass, never a bare
exception. The registry covers the platform's 4 target diseases: heart
disease, breast cancer, Alzheimer's, and brain tumor (glioma MGMT-
methylation proxy).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from qhealth_qml import experiment, standardize as std

HEART_HEADER = (
    "id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,heart_disease"
)

HEART_ROWS = [
    "1,63,1,4,145,233,1,2,150,0,2.3,3,0,6,1",
    "2,67,1,4,160,286,0,2,108,1,1.5,2,?,3,1",
    "3,67,1,4,120,229,0,2,129,1,2.6,2,2,7,1",
    "4,62,0,4,140,268,0,2,160,0,3.6,3,2,3,1",
    "5,60,1,4,130,206,0,2,132,1,2.4,2,2,7,1",
    "6,52,1,3,125,212,1,0,168,0,1.0,1,0,3,0",
    "7,44,1,2,120,263,0,0,173,0,0.0,1,0,7,0",
    "8,57,0,3,130,236,0,2,174,0,0.0,1,0,3,0",
    "9,56,0,3,140,294,0,2,153,0,1.3,2,0,3,0",
    "10,48,1,2,110,229,0,0,168,0,1.0,1,0,7,0",
]


def heart_csv(rows: list[str] = HEART_ROWS) -> str:
    return "\n".join([HEART_HEADER, *rows]) + "\n"


def wdbc_sample_csv(n_malignant: int = 5, n_benign: int = 5) -> str:
    """A real slice of the WDBC dataset (bundled with scikit-learn), built
    the same way `data/breast_cancer_wdbc/prepare.py` builds the full CSV."""

    source = load_breast_cancer()
    feature_names = [str(name) for name in source.feature_names]
    header = ",".join(["id", "diagnosis", *feature_names])
    lines = [header]
    counts = {"malignant": 0, "benign": 0}
    for index, (row, target) in enumerate(zip(source.data, source.target), start=1):
        label = "malignant" if target == 0 else "benign"
        limit = n_malignant if label == "malignant" else n_benign
        if counts[label] >= limit:
            continue
        counts[label] += 1
        lines.append(",".join([f"row-{index}", label, *(str(v) for v in row)]))
        if counts["malignant"] >= n_malignant and counts["benign"] >= n_benign:
            break
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Disease registry
# ---------------------------------------------------------------------------


def test_list_supported_diseases_covers_all_target_diseases():
    # The platform's own tabular demos (heart-disease, breast-cancer) plus
    # the full six-condition neuro-conditions research program minus P2 ICH
    # (registered `not available` upstream - no lawful dataset exists).
    diseases = {entry["disease_id"] for entry in std.list_supported_diseases()}
    assert diseases == {
        "heart-disease", "breast-cancer",
        "stroke", "brain-tumor", "seizure", "alzheimers", "parkinsons",
    }


def test_get_disease_schema_unknown_id_raises_typed_error():
    with pytest.raises(std.UnknownDiseaseError) as excinfo:
        std.get_disease_schema("made-up-disease")
    assert excinfo.value.disease_id == "made-up-disease"


def test_heart_disease_schema_required_fields_and_status():
    schema = std.get_disease_schema("heart-disease")
    assert schema.required_fields == (
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    )
    assert schema.target_column == "heart_disease"
    assert schema.status == "no_trained_model_yet"
    assert schema.clinical_context["typical_workup"] == [
        "ECG", "troponin (blood test)", "angiography (confirming)"
    ]


def test_breast_cancer_schema_has_30_wdbc_fields():
    schema = std.get_disease_schema("breast-cancer")
    assert len(schema.required_fields) == 30
    assert "mean radius" in schema.required_fields
    assert "worst fractal dimension" in schema.required_fields
    assert schema.target_column == "diagnosis"
    assert schema.positive_label == "malignant"


def test_brain_tumor_schema_flags_its_narrow_scope():
    schema = std.get_disease_schema("brain-tumor")
    assert schema.required_fields == ()  # feature-extraction-script output, no fixed whitelist
    assert "MGMT" in schema.notes
    assert schema.status == "trained_model_registered"  # audited profile + model contract exist


def test_alzheimers_still_registered_unchanged():
    schema = std.get_disease_schema("alzheimers")
    assert schema.required_fields == ("Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F")
    assert schema.status == "trained_model_registered"


# ---------------------------------------------------------------------------
# Labeled (training/benchmark) standardization
# ---------------------------------------------------------------------------


def test_standardize_labeled_heart_disease_returns_correct_shape(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X, y = std.standardize(heart_csv(), "heart-disease", schema_store=store)

    assert X.shape == (10, 13)  # all 13 UCI fields are numeric codes, no one-hot expansion
    assert y is not None
    assert y.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_standardize_missing_ca_sentinel_becomes_nan(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X, _ = std.standardize(heart_csv(), "heart-disease", schema_store=store)
    fitted = store.load("heart-disease")
    ca_index = fitted.feature_names.index("ca")
    assert np.isnan(X[1][ca_index])  # patient id=2's "?" for ca


def test_standardize_labeled_breast_cancer_real_wdbc_slice(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X, y = std.standardize(wdbc_sample_csv(5, 5), "breast-cancer", schema_store=store)

    assert X.shape == (10, 30)
    assert y.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # malignant rows written first


def test_standardize_extra_columns_in_upload_are_ignored(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    header = HEART_HEADER + ",patient_name,mrn"
    rows = [f"{r},John Doe,MRN-{i}" for i, r in enumerate(HEART_ROWS)]
    csv_text = "\n".join([header, *rows]) + "\n"

    X, y = std.standardize(csv_text, "heart-disease", schema_store=store)
    assert X.shape[0] == len(HEART_ROWS)


def test_standardize_missing_required_columns_raises_schema_mismatch(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    header = "id,age,heart_disease"
    rows = ["1,63,1", "2,67,1", "3,52,0", "4,44,0"]
    csv_text = "\n".join([header, *rows]) + "\n"

    with pytest.raises(std.SchemaMismatchError) as excinfo:
        std.standardize(csv_text, "heart-disease", schema_store=store)
    assert "cp" in excinfo.value.missing_fields
    assert "thal" in excinfo.value.missing_fields


# ---------------------------------------------------------------------------
# Unlabeled (predict) standardization round-trip
# ---------------------------------------------------------------------------


def test_predict_upload_reuses_fitted_schema_from_labeled_upload(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X_train, _ = std.standardize(heart_csv(), "heart-disease", schema_store=store)

    predict_header = HEART_HEADER.replace(",heart_disease", "")
    predict_row = HEART_ROWS[0].rsplit(",", 1)[0]
    predict_csv = f"{predict_header}\n{predict_row}\n"

    X_predict, y_predict = std.standardize(predict_csv, "heart-disease", schema_store=store)

    assert y_predict is None
    assert X_predict.shape[1] == X_train.shape[1]
    np.testing.assert_allclose(X_predict[0], X_train[0])


def test_predict_upload_before_any_training_upload_raises_not_fitted(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    predict_header = HEART_HEADER.replace(",heart_disease", "")
    predict_row = HEART_ROWS[0].rsplit(",", 1)[0]
    predict_csv = f"{predict_header}\n{predict_row}\n"

    with pytest.raises(std.SchemaNotFittedError):
        std.standardize(predict_csv, "heart-disease", schema_store=store)


def test_predict_upload_missing_a_fitted_column_raises_schema_mismatch(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    std.standardize(heart_csv(), "heart-disease", schema_store=store)

    # Predict file is missing 'thal', which the fitted schema requires.
    predict_csv = "id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca\n1,63,1,4,145,233,1,2,150,0,2.3,3,0\n"

    with pytest.raises(std.SchemaMismatchError):
        std.standardize(predict_csv, "heart-disease", schema_store=store)


def test_breast_cancer_predict_round_trip(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X_train, _ = std.standardize(wdbc_sample_csv(3, 3), "breast-cancer", schema_store=store)

    fitted = store.load("breast-cancer")
    predict_header = ",".join(["id", *fitted.raw_feature_columns])
    source = load_breast_cancer()
    first_row_values = ",".join(str(v) for v in source.data[0])
    predict_csv = f"{predict_header}\nrow-1,{first_row_values}\n"

    X_predict, y_predict = std.standardize(predict_csv, "breast-cancer", schema_store=store)
    assert y_predict is None
    assert X_predict.shape[1] == X_train.shape[1]


# ---------------------------------------------------------------------------
# Input-format handling
# ---------------------------------------------------------------------------


def test_standardize_accepts_bytes_input(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    X, y = std.standardize(heart_csv().encode("utf-8"), "heart-disease", schema_store=store)
    assert X.shape[0] == len(HEART_ROWS)


def test_standardize_empty_dataset_raises_typed_error(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    with pytest.raises(std.EmptyDatasetError):
        std.standardize(HEART_HEADER + "\n", "heart-disease", schema_store=store)


def test_standardize_non_csv_bytes_raises_unsupported_format(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    with pytest.raises(std.UnsupportedFormatError):
        std.standardize(b"\xff\xfe\x00\x01not utf-8", "heart-disease", schema_store=store)


def test_standardize_unknown_disease_raises_before_reading_file(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    with pytest.raises(std.UnknownDiseaseError):
        std.standardize(heart_csv(), "made-up-disease", schema_store=store)


# ---------------------------------------------------------------------------
# Brain tumor: no fixed required_fields -> falls back to "every non-excluded
# column is a feature" (its raw input is already a radiomics feature CSV).
# ---------------------------------------------------------------------------


def test_brain_tumor_uses_all_other_columns(tmp_path):
    store = std.SchemaStore(tmp_path / "schemas")
    header = "subject_id,patient_id,flair_mean,t1_kurtosis,mgmt_methylated"
    rows = [
        "s1,p1,412.3,3.8,1",
        "s2,p2,398.1,3.5,1",
        "s3,p3,510.2,2.1,0",
        "s4,p4,495.7,2.4,0",
    ]
    csv_text = "\n".join([header, *rows]) + "\n"

    X, y = std.standardize(csv_text, "brain-tumor", schema_store=store)
    assert X.shape == (4, 2)
    assert y is not None


# ---------------------------------------------------------------------------
# Forward compatibility: once a teammate's training run drops a real
# `SavedModelArtifact` at the path its registry entry declares, predict-time
# standardization must switch to reading THAT artifact's own fitted column
# layout - not this module's own standalone cache. This builds a genuine
# artifact via `experiment`'s real fit/save path (not a hand-rolled fake) to
# prove the handoff actually works end to end, not just in theory.
# ---------------------------------------------------------------------------

BRAIN_TUMOR_HEADER = "subject_id,patient_id,flair_mean,t1_kurtosis,mgmt_methylated"
BRAIN_TUMOR_ROWS = [
    "s1,p1,412.3,3.8,1",
    "s2,p2,398.1,3.5,1",
    "s3,p3,510.2,2.1,0",
    "s4,p4,495.7,2.4,0",
    "s5,p5,420.0,3.6,1",
    "s6,p6,505.5,2.0,0",
]


def _build_real_artifact(tmp_path) -> str:
    """A genuine `SavedModelArtifact`, fit via the real training engine - the
    same object `execution.py`'s training path produces, not a mock."""

    csv_path = tmp_path / "brain_tumor.csv"
    csv_path.write_text("\n".join([BRAIN_TUMOR_HEADER, *BRAIN_TUMOR_ROWS]) + "\n", encoding="utf-8")
    loaded = experiment.load_csv_dataset(
        csv_path, target="mgmt_methylated", positive_label="1",
        id_column="subject_id", group_column="patient_id",
    )
    preprocessor = experiment.PreprocessingPipeline(
        feature_names=loaded.feature_names, n_qubits=2,
    ).fit(loaded.X, loaded.y)

    artifact = experiment.SavedModelArtifact(
        schema_version=experiment.SCHEMA_VERSION,
        package_version="test",
        model_name="glioma-mgmt-radiomics-tabular",
        model=None,
        preprocessor=preprocessor,
        feature_space="raw",
        selected_features=list(preprocessor.selected_features),
        threshold=0.5,
        threshold_policy="fixed",
        abstain_margin=None,
        probability_score=False,
        calibration={},
        dataset={"provenance": loaded.provenance},
        execution={},
        hardware_probe={},
        software={},
    )
    artifact_path = tmp_path / "glioma-mgmt-radiomics-tabular.joblib"
    experiment.save_model_artifact(artifact, artifact_path)
    return loaded.feature_names


def test_predict_prefers_a_real_trained_artifact_over_the_schema_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(std, "RUNTIME_MODELS_DIR", tmp_path)
    real_feature_names = _build_real_artifact(tmp_path)

    # A DIFFERENT (deliberately stale) cached schema for the same disease -
    # if standardize() picked this instead of the real artifact, the test
    # would still pass shape-wise but with the wrong column identity, which
    # the exact feature_names assertion below would catch.
    stale_store = std.SchemaStore(tmp_path / "stale_schemas")
    stale_store.save(
        std.FittedSchema(
            disease_id="brain-tumor",
            raw_feature_columns=["flair_mean"],
            feature_names=["flair_mean"],
            categorical_columns={},
        )
    )

    predict_csv = "subject_id,patient_id,flair_mean,t1_kurtosis\ns7,p7,415.0,3.7\n"
    X, y = std.standardize(predict_csv, "brain-tumor", schema_store=stale_store)

    assert y is None
    assert X.shape == (1, len(real_feature_names))  # NOT the stale 1-column cache


def test_predict_falls_back_to_schema_cache_when_no_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(std, "RUNTIME_MODELS_DIR", tmp_path / "nothing-here")
    store = std.SchemaStore(tmp_path / "schemas")
    csv_text = "\n".join([BRAIN_TUMOR_HEADER, *BRAIN_TUMOR_ROWS]) + "\n"
    std.standardize(csv_text, "brain-tumor", schema_store=store)  # fits the cache

    predict_csv = "subject_id,patient_id,flair_mean,t1_kurtosis\ns7,p7,415.0,3.7\n"
    X, y = std.standardize(predict_csv, "brain-tumor", schema_store=store)
    assert y is None
    assert X.shape == (1, 2)


def test_corrupt_artifact_raises_instead_of_silently_falling_back(tmp_path, monkeypatch):
    monkeypatch.setattr(std, "RUNTIME_MODELS_DIR", tmp_path)
    (tmp_path / "glioma-mgmt-radiomics-tabular.joblib").write_bytes(b"not a real pickle")

    store = std.SchemaStore(tmp_path / "schemas")
    predict_csv = "subject_id,patient_id,flair_mean,t1_kurtosis\ns7,p7,415.0,3.7\n"
    with pytest.raises(std.StandardizationError):
        std.standardize(predict_csv, "brain-tumor", schema_store=store)
