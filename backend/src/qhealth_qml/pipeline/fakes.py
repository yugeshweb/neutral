"""Synthetic fakes for standardizer and model-training layers for testing and decoupling."""

import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from qhealth_qml.pipeline.exceptions import (
    EmptyDatasetError,
    SchemaMismatchError,
    SchemaNotFittedError,
    UnknownDiseaseError,
    UnsupportedFormatError,
)
from qhealth_qml.pipeline.interfaces import ModelOutput

SUPPORTED_FAKE_DISEASES = [
    "heart-disease",
    "breast-cancer",
    "alzheimers",
    "glioma",
    "stroke",
    "seizure",
    "parkinsons",
]


def fake_list_supported_diseases() -> List[str]:
    """Return list of diseases supported by the standardizer fake."""
    return list(SUPPORTED_FAKE_DISEASES)


def fake_standardize(
    raw_file: Union[str, bytes, io.BytesIO, io.StringIO, Any],
    disease_id: str,
    simulate_missing_columns: Optional[List[str]] = None,
    simulate_unfitted_schema: bool = False,
    n_samples: int = 100,
    random_seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fake standardizer implementing the exact contract from STANDARDIZER_CONTRACT.md.

    Signature: X, y = standardize(raw_file, disease_id)
    Raises typed StandardizationError subclasses.
    """
    # 1. Disease validation
    if disease_id not in SUPPORTED_FAKE_DISEASES:
        raise UnknownDiseaseError(f"Unknown disease identifier: '{disease_id}'", disease_id=disease_id)

    # 2. Schema not fitted simulation (for predict-before-train scenario)
    if simulate_unfitted_schema:
        raise SchemaNotFittedError(f"Schema not fitted for disease '{disease_id}'", disease_id=disease_id)

    # 3. Schema mismatch simulation
    if simulate_missing_columns:
        raise SchemaMismatchError(
            f"Dataset schema mismatch: missing required columns {simulate_missing_columns}",
            disease_id=disease_id,
            missing_fields=simulate_missing_columns,
        )

    # 4. File content reading and inspection
    content_bytes = b""
    if isinstance(raw_file, bytes):
        content_bytes = raw_file
    elif isinstance(raw_file, str):
        content_bytes = raw_file.encode("utf-8")
    elif hasattr(raw_file, "read"):
        content = raw_file.read()
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content
    else:
        raise UnsupportedFormatError(
            message="Invalid file object provided to standardizer",
            disease_id=disease_id,
        )

    # 5. Format validation
    if b"\x00" in content_bytes:
        raise UnsupportedFormatError(
            message="File contains binary null bytes, not valid UTF-8 CSV",
            disease_id=disease_id,
        )

    text = content_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise EmptyDatasetError(
            message="Raw dataset file is completely empty",
            disease_id=disease_id,
        )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        # Header only or zero data rows
        raise EmptyDatasetError(
            message="Uploaded CSV has header only with no data rows",
            disease_id=disease_id,
        )

    # Check for trigger keywords in test payload
    if "TRIGGER_UNSUPPORTED_FORMAT" in text:
        raise UnsupportedFormatError(disease_id=disease_id)
    if "TRIGGER_EMPTY_DATASET" in text:
        raise EmptyDatasetError(disease_id=disease_id)
    if "TRIGGER_SCHEMA_MISMATCH" in text:
        raise SchemaMismatchError(
            missing_fields=["critical_marker_a", "clinical_grade"],
            disease_id=disease_id,
        )
    if "TRIGGER_SCHEMA_NOT_FITTED" in text:
        raise SchemaNotFittedError(disease_id=disease_id)

    # 6. Generate deterministic synthetic data for the disease
    rng = np.random.RandomState(random_seed)
    
    # Disease-specific feature dimension
    n_features_map = {
        "heart-disease": 13,
        "breast-cancer": 30,
        "alzheimers": 8,
        "glioma": 10,
        "stroke": 10,
        "seizure": 178,
        "parkinsons": 16,
    }
    n_feat = n_features_map.get(disease_id, 10)

    # Actual number of data rows from lines if provided, or fallback
    actual_rows = max(len(lines) - 1, 20)
    
    # Generate continuous features with realistic covariance
    X = rng.randn(actual_rows, n_feat)
    
    # Introduce small realistic missing values (NaNs) to verify preprocessor handles NaNs
    if actual_rows > 10:
        nan_mask = rng.rand(*X.shape) < 0.05
        X[nan_mask] = np.nan

    # Generate balanced binary labels (0 or 1)
    # Signal correlation with first 2 features
    logits = 1.2 * np.nan_to_num(X[:, 0]) - 0.8 * np.nan_to_num(X[:, 1])
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    # Guarantee at least 2 samples per class
    if np.sum(y == 1) == 0:
        y[0] = 1
        y[1] = 1
    elif np.sum(y == 0) == 0:
        y[0] = 0
        y[1] = 0

    return X, y


class MockQuantumModel:
    """Mock quantum model simulating VQC / QSVC decision boundaries."""

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights if weights is not None else np.array([0.5, -0.3, 0.8, -0.2, 0.4, 0.1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        # Pad or slice weights to match features
        w = np.resize(self.weights, X_arr.shape[1])
        scores = np.dot(X_arr, w)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def fake_train_disease_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Union[ModelOutput, Dict[str, Any]]]:
    """Fake model-training function satisfying requirement 1.

    Trains a classical model (Logistic Regression) and a mock quantum model,
    returning structured ModelOutput instances.
    """
    # Classical training
    t0 = time.perf_counter()
    classical_model = LogisticRegression(random_state=42, max_iter=200)
    classical_model.fit(X_train, y_train)
    c_train_time = round(time.perf_counter() - t0, 4)

    t1 = time.perf_counter()
    c_prob = classical_model.predict_proba(X_test)[:, 1]
    c_pred = classical_model.predict(X_test)
    c_infer_time = round((time.perf_counter() - t1) * 1000, 2)

    # Quantum training simulation
    t2 = time.perf_counter()
    quantum_model = MockQuantumModel()
    q_train_time = round(time.perf_counter() - t2 + 0.05, 4)

    t3 = time.perf_counter()
    q_prob = quantum_model.predict_proba(X_test)[:, 1]
    q_pred = quantum_model.predict(X_test)
    q_infer_time = round((time.perf_counter() - t3) * 1000 + 5.0, 2)

    return {
        "classical": ModelOutput(
            model=classical_model,
            y_pred=c_pred,
            y_prob=c_prob,
            train_time=f"{c_train_time}s",
            infer_time=f"{c_infer_time}ms",
            threshold=0.5,
            metadata={"type": "LogisticRegression", "parameters": "L2, C=1.0"},
        ),
        "quantum": ModelOutput(
            model=quantum_model,
            y_pred=q_pred,
            y_prob=q_prob,
            train_time=f"{q_train_time}s",
            infer_time=f"{q_infer_time}ms",
            threshold=0.5,
            metadata={"type": "MockVQC", "qubits": X_train.shape[1], "layers": 3},
        ),
    }
