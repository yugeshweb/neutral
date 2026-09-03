"""Tests for registry wiring, joblib bundle serialization, seizure safety gating, and score semantics."""

import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

import qhealth_qml.registry as reg
from qhealth_qml.serving import InferenceBundle, save_bundle, load_bundle, SERVING_SCHEMA_VERSION
from qhealth_qml.pipeline.dispatcher import run_training_pipeline
from qhealth_qml.pipeline.model_registry import list_registered_disease_trainers
from qhealth_qml.cohort_audit import check_cohort_spacing
from backend.serve_api import build_app, PredictRequest


SAMPLE_CSV = "f1,f2,f3,f4\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n13.0,14.0,15.0,16.0\n"
SEVEN_DISEASES = [
    "heart-disease",
    "breast-cancer",
    "glioma",
    "stroke",
    "parkinsons",
    "alzheimers",
    "seizure",
]


class TestRegistryAndServingFixes:
    def test_all_seven_diseases_registered(self):
        """Verify all 7 disease models are present in the model registry."""
        reg.register_all_disease_models()
        registered = list_registered_disease_trainers()
        for disease_id in SEVEN_DISEASES:
            assert disease_id in registered

    @pytest.mark.parametrize("disease_id", SEVEN_DISEASES)
    def test_run_training_pipeline_with_registered_models(self, disease_id):
        """Verify all 7 models train cleanly through the 8-step orchestration pipeline."""
        from qhealth_qml.pipeline.fakes import fake_standardize
        result = run_training_pipeline(SAMPLE_CSV, disease_id, standardize_fn=fake_standardize)
        assert result["status"] == "success"
        assert "models" in result
        assert "classical" in result["models"]
        assert "quantum" in result["models"]

    def test_joblib_bundle_roundtrip_with_score_semantics(self, tmp_path):
        """Verify InferenceBundle saves and loads with joblib format and includes score_semantics."""
        bundle = InferenceBundle(
            schema_version=SERVING_SCHEMA_VERSION,
            model_id="stroke-core-volume-mri",
            condition="stroke",
            temporal_framing="characterisation",
            positive_label="large_infarct_core",
            negative_label="small_infarct_core",
            channel_names=["dwi", "adc", "flair"],
            input_grid=[64, 64, 32],
            encoder_state={},
            encoder_kind="none",
            encoder_config={},
            head_kind="rbf_svc",
            head=None,
            standardizer=None,
            reducer=None,
            angle_scaler=None,
            threshold=0.305,
            threshold_policy="youden_j",
            quantum_config=None,
            training_provenance={"dataset": "ISLES-2015"},
            input_stats={},
        )

        target = tmp_path / "test_bundle.joblib"
        saved = save_bundle(bundle, target)
        assert saved.exists()
        assert target.with_suffix(".joblib.manifest.json").exists()

        manifest = bundle.to_manifest()
        assert manifest["score_semantics"] == "normalised_margin_from_threshold"

        loaded = load_bundle(saved)
        assert loaded.model_id == "stroke-core-volume-mri"
        assert loaded.temporal_framing == "characterisation"

    def test_seizure_safety_gated_at_api(self, tmp_path):
        """Verify /predict/seizure-preictal-eeg is gated and returns status: disabled."""
        app = build_app(tmp_path)
        client = TestClient(app)

        res = client.post("/predict/seizure-preictal-eeg", json={"sources": {}, "study_id": "TEST-1"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "disabled"
        assert "LOPO BA 0.505" in data["reason"]
        assert data["score_semantics"] == "normalised_margin_from_threshold"

    def test_cohort_audit_check_spacing_runs(self):
        """Verify cohort spacing audit runs without blocking or crashing."""
        verdict = check_cohort_spacing([])
        assert "d5_exposure" in verdict
