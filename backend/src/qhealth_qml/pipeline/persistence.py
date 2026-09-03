"""Artifact persistence and manifest management for trained models and preprocessors."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np

from qhealth_qml.pipeline.evaluator import ModelMetrics
from qhealth_qml.pipeline.exceptions import PipelineError
from qhealth_qml.pipeline.preprocessor import LeakageSafePreprocessor

_DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runtime"


def compute_dataset_hash(X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
    """Compute a SHA-256 fingerprint of the input dataset."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(X).tobytes())
    if y is not None:
        hasher.update(np.ascontiguousarray(y).tobytes())
    return hasher.hexdigest()


def persist_training_artifacts(
    disease_id: str,
    run_id: str,
    preprocessor: LeakageSafePreprocessor,
    trained_models: Dict[str, Any],
    evaluated_metrics: Dict[str, ModelMetrics],
    benchmark_comparison: Dict[str, Any],
    dataset_shape: Tuple[int, int],
    train_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    data_hash: str,
    runtime_dir: Optional[Path] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist preprocessor, model weights, and manifest to backend/runtime/models/<disease_id>/<run_id>/."""
    base_runtime = runtime_dir or _DEFAULT_RUNTIME_DIR
    target_dir = base_runtime / "models" / disease_id / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {
        "artifact_dir": str(target_dir),
    }

    # 1. Save Preprocessor
    preprocessor_path = target_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    saved_paths["preprocessor"] = str(preprocessor_path)

    # 2. Save Trained Models
    for model_type, model_obj in trained_models.items():
        model_filename = f"{model_type}_model.joblib"
        model_path = target_dir / model_filename
        joblib.dump(model_obj, model_path)
        saved_paths[f"{model_type}_model"] = str(model_path)

    # 3. Build and Save manifest.json
    metrics_serialized = {
        m_type: metric.to_dict() for m_type, metric in evaluated_metrics.items()
    }
    
    manifest_data = {
        "run_id": run_id,
        "disease_id": disease_id,
        "score_semantics": "normalised_margin_from_threshold",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "created_at_epoch": time.time(),
        "data_hash": data_hash,
        "dataset_shape": {"rows": dataset_shape[0], "features": dataset_shape[1]},
        "train_shape": {"rows": train_shape[0], "features": train_shape[1]},
        "test_shape": {"rows": test_shape[0], "features": test_shape[1]},
        "metrics": metrics_serialized,
        "benchmark_comparison": benchmark_comparison,
        "saved_artifacts": {k: Path(v).name for k, v in saved_paths.items() if k != "artifact_dir"},
        "config": extra_config or {},
    }

    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    saved_paths["manifest"] = str(manifest_path)

    return saved_paths


def load_training_artifacts(
    disease_id: str,
    run_id: str,
    runtime_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load preprocessor, manifest, and model objects from a previous run directory."""
    base_runtime = runtime_dir or _DEFAULT_RUNTIME_DIR
    target_dir = base_runtime / "models" / disease_id / run_id

    if not target_dir.exists():
        raise PipelineError(
            f"Run artifact directory not found: {target_dir}",
            disease_id=disease_id,
        )

    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        raise PipelineError(
            f"Manifest file missing in {target_dir}",
            disease_id=disease_id,
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    preprocessor_path = target_dir / "preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

    models = {}
    for model_type in ["classical", "quantum"]:
        m_path = target_dir / f"{model_type}_model.joblib"
        if m_path.exists():
            models[model_type] = joblib.load(m_path)

    return {
        "manifest": manifest,
        "preprocessor": preprocessor,
        "models": models,
        "artifact_dir": str(target_dir),
    }
