"""Tests for artifact persistence, manifest metadata, and model reload round-trips."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from qhealth_qml.pipeline.evaluator import ModelMetrics
from qhealth_qml.pipeline.fakes import MockQuantumModel
from qhealth_qml.pipeline.persistence import (
    compute_dataset_hash,
    load_training_artifacts,
    persist_training_artifacts,
)
from qhealth_qml.pipeline.preprocessor import LeakageSafePreprocessor


class TestPersistenceRoundTrip:
    """Test saving and bit-identical reloading of models and preprocessors."""

    def test_artifact_persistence_and_reload(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            disease_id = "breast-cancer"
            run_id = "test_run_123"

            # Create synthetic training data
            X_train = np.random.randn(40, 10)
            y_train = np.array([0, 1] * 20)
            X_test = np.random.randn(10, 10)

            # Fit preprocessor
            preprocessor = LeakageSafePreprocessor(n_components=4, random_state=42)
            X_train_pre = preprocessor.fit_transform(X_train)
            X_test_pre = preprocessor.transform(X_test)

            # Train classical and quantum models
            classical_model = LogisticRegression(random_state=42)
            classical_model.fit(X_train_pre, y_train)
            original_classical_preds = classical_model.predict_proba(X_test_pre)

            quantum_model = MockQuantumModel()
            original_quantum_preds = quantum_model.predict_proba(X_test_pre)

            # Dummy metrics
            dummy_metrics = {
                "classical": ModelMetrics(
                    accuracy=0.95,
                    precision=0.94,
                    sensitivity=0.96,
                    specificity=0.94,
                    f1=0.95,
                    roc_auc=0.98,
                    confusion_matrix={"tp": 10, "fn": 1, "tn": 10, "fp": 1},
                    roc_points=[{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
                    threshold=0.5,
                )
            }

            data_hash = compute_dataset_hash(X_train, y_train)

            # Persist artifacts
            saved_paths = persist_training_artifacts(
                disease_id=disease_id,
                run_id=run_id,
                preprocessor=preprocessor,
                trained_models={"classical": classical_model, "quantum": quantum_model},
                evaluated_metrics=dummy_metrics,
                benchmark_comparison={"status": "success"},
                dataset_shape=(50, 10),
                train_shape=(40, 10),
                test_shape=(10, 10),
                data_hash=data_hash,
                runtime_dir=tmp_path,
            )

            assert Path(saved_paths["preprocessor"]).exists()
            assert Path(saved_paths["classical_model"]).exists()
            assert Path(saved_paths["quantum_model"]).exists()
            assert Path(saved_paths["manifest"]).exists()

            # Verify manifest content
            with open(saved_paths["manifest"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["run_id"] == run_id
            assert manifest["disease_id"] == disease_id
            assert manifest["data_hash"] == data_hash
            assert manifest["dataset_shape"]["rows"] == 50

            # Reload artifacts
            loaded = load_training_artifacts(
                disease_id=disease_id,
                run_id=run_id,
                runtime_dir=tmp_path,
            )

            reloaded_preprocessor = loaded["preprocessor"]
            reloaded_classical = loaded["models"]["classical"]
            reloaded_quantum = loaded["models"]["quantum"]

            # Verify identical transform on raw test data
            reloaded_X_test_pre = reloaded_preprocessor.transform(X_test)
            np.testing.assert_allclose(X_test_pre, reloaded_X_test_pre, rtol=1e-6)

            # Verify identical predictions
            reloaded_classical_preds = reloaded_classical.predict_proba(reloaded_X_test_pre)
            np.testing.assert_allclose(original_classical_preds, reloaded_classical_preds, rtol=1e-6)

            reloaded_quantum_preds = reloaded_quantum.predict_proba(reloaded_X_test_pre)
            np.testing.assert_allclose(original_quantum_preds, reloaded_quantum_preds, rtol=1e-6)
