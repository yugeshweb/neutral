"""End-to-end smoke tests and model registry swap verification."""

from pathlib import Path
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from qhealth_qml.pipeline.dispatcher import run_training_pipeline
from qhealth_qml.pipeline.fakes import MockQuantumModel, fake_standardize
from qhealth_qml.pipeline.interfaces import ModelOutput
from qhealth_qml.pipeline.model_registry import (
    clear_model_registry,
    register_disease_models,
)
from qhealth_qml.standardize import standardize

SAMPLE_CSV = "f1,f2,f3,f4\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n13.0,14.0,15.0,16.0\n"
SEVEN_DISEASES = [
    "heart-disease",
    "breast-cancer",
    "alzheimers",
    "glioma",
    "stroke",
    "seizure",
    "parkinsons",
]


class TestEndToEndPipeline:
    """Complete end-to-end orchestration tests across all 7 supported diseases."""

    @pytest.mark.parametrize("disease_id", SEVEN_DISEASES)
    def test_e2e_run_all_diseases(self, disease_id):
        result = run_training_pipeline(
            raw_file=SAMPLE_CSV,
            disease_id=disease_id,
            standardize_fn=fake_standardize,
        )

        assert result["status"] == "success"
        assert result["disease_id"] == disease_id
        assert "run_id" in result
        assert "models" in result
        assert "classical" in result["models"]
        assert "quantum" in result["models"]
        assert "benchmark_comparison" in result
        assert result["benchmark_comparison"]["status"] in {"success", "not_available"}
        assert "artifacts" in result
        assert "manifest" in result["artifacts"]

    @pytest.mark.parametrize("disease_id", SEVEN_DISEASES)
    def test_custom_model_registry_swap_zero_pipeline_changes(self, disease_id):
        """Verify that a real/custom model trainer can be plugged in purely by registration for any disease."""
        clear_model_registry()

        def custom_rf_trainer(X_train, y_train, X_test, y_test):
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            prob = rf.predict_proba(X_test)[:, 1]

            q_model = MockQuantumModel()
            q_prob = q_model.predict_proba(X_test)[:, 1]
            q_pred = q_model.predict(X_test)

            return {
                "classical": ModelOutput(
                    model=rf,
                    y_pred=pred,
                    y_prob=prob,
                    train_time="0.05s",
                    infer_time="1.2ms",
                    metadata={"custom": "true", "trees": 10},
                ),
                "quantum": ModelOutput(
                    model=q_model,
                    y_pred=q_pred,
                    y_prob=q_prob,
                    train_time="0.12s",
                    infer_time="3.5ms",
                    metadata={"custom": "true", "type": "vqc"},
                ),
            }

        register_disease_models(disease_id, custom_rf_trainer)

        result = run_training_pipeline(
            raw_file=SAMPLE_CSV,
            disease_id=disease_id,
            standardize_fn=fake_standardize,
        )

        assert result["status"] == "success"
        assert result["models"]["classical"]["metadata"].get("custom") == "true"
        assert result["models"]["quantum"]["metadata"].get("custom") == "true"

    def test_e2e_run_real_standardizer_heart_disease(self):
        """Verify running training pipeline with real standardizer on real heart disease CSV."""
        sample_path = Path(__file__).resolve().parents[1] / "tests" / "manual_check" / "heart_disease_sample.csv"
        result = run_training_pipeline(
            raw_file=sample_path,
            disease_id="heart-disease",
            standardize_fn=standardize,
        )
        assert result["status"] == "success"
        assert "classical" in result["models"]
        assert "quantum" in result["models"]

    def test_e2e_run_real_standardizer_breast_cancer(self):
        """Verify running training pipeline with real standardizer on real breast cancer WDBC CSV."""
        sample_path = Path(__file__).resolve().parents[2] / "public" / "samples" / "breast_cancer_wdbc.csv"
        if not sample_path.exists():
            sample_path = Path(__file__).resolve().parents[1] / "tests" / "manual_check" / "breast_cancer_sample.csv"
        result = run_training_pipeline(
            raw_file=sample_path,
            disease_id="breast-cancer",
            standardize_fn=standardize,
        )
        assert result["status"] == "success"
        assert "classical" in result["models"]
        assert "quantum" in result["models"]
