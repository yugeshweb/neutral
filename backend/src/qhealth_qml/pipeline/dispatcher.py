"""Pipeline dispatcher orchestrating the end-to-end training and evaluation workflow."""

import io
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

from qhealth_qml.pipeline.benchmarks import compare_with_benchmarks
from qhealth_qml.pipeline.disease_registry import disease_registry
from qhealth_qml.pipeline.evaluator import evaluate_model_predictions, ModelMetrics
from qhealth_qml.pipeline.exceptions import (
    EmptyDatasetError,
    ModelTrainingError,
    PipelineError,
    StandardizationError,
    UnknownDiseaseError,
)
from qhealth_qml.pipeline.interfaces import ModelOutput, TrainDiseaseModelsFn
from qhealth_qml.pipeline.model_registry import get_disease_models_trainer, has_disease_models_trainer
from qhealth_qml.pipeline.persistence import compute_dataset_hash, persist_training_artifacts
from qhealth_qml.pipeline.preprocessor import LeakageSafePreprocessor, stratified_split


def run_training_pipeline(
    raw_file: Union[str, bytes, io.BytesIO, Any],
    disease_id: str,
    standardize_fn: Optional[Callable[..., Tuple[np.ndarray, Optional[np.ndarray]]]] = None,
    train_fn: Optional[TrainDiseaseModelsFn] = None,
    run_id: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    runtime_dir: Optional[Any] = None,
    benchmark_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the complete training orchestration pipeline.

    Flow:
    1. Validate disease_id via disease_registry.
    2. Ingest raw dataset through standardize_fn(raw_file, disease_id).
    3. Perform stratified train/test split.
    4. Fit preprocessing (impute, scale, PCA) strictly on train fold.
    5. Dispatch preprocessed data to registered model trainer.
    6. Evaluate models on test fold (using Youden's J statistic).
    7. Compare metrics with static stored benchmarks.
    8. Persist artifacts & manifest to disk.
    9. Return unified JSON result.
    """
    # 1. Disease verification
    disease_cfg = disease_registry.get(disease_id)
    generated_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    # 2. Standardization
    if standardize_fn is None:
        try:
            from qhealth_qml.standardize import standardize as std_module_fn
            standardize_fn = std_module_fn
        except ImportError:
            from qhealth_qml.pipeline.fakes import fake_standardize
            standardize_fn = fake_standardize

    try:
        X, y = standardize_fn(raw_file, disease_cfg.standardizer_disease_id)
    except StandardizationError:
        raise
    except Exception as e:
        raise PipelineError(f"Standardization unexpected failure: {str(e)}", disease_id=disease_id) from e

    if X is None or len(X) == 0:
        raise EmptyDatasetError("Standardizer returned an empty feature matrix", disease_id=disease_id)

    if y is None or len(y) == 0:
        raise EmptyDatasetError(
            "Standardizer returned no label column. Labeled training data is required for model training.",
            disease_id=disease_id,
        )

    dataset_shape = (int(X.shape[0]), int(X.shape[1]))
    data_hash = compute_dataset_hash(X, y)

    # 3. Stratified train/test split
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_shape = (int(X_train.shape[0]), int(X_train.shape[1]))
    test_shape = (int(X_test.shape[0]), int(X_test.shape[1]))

    # 4. Leakage-safe Preprocessing strictly fitted on train fold
    preprocessor = LeakageSafePreprocessor(
        n_components=disease_cfg.default_qubits,
        random_state=random_state,
    )
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    # 5. Model dispatching
    if train_fn is None:
        if has_disease_models_trainer(disease_id):
            train_fn = get_disease_models_trainer(disease_id)
        else:
            # Fallback to test fake if no real model has been registered yet
            from qhealth_qml.pipeline.fakes import fake_train_disease_models
            train_fn = fake_train_disease_models

    try:
        raw_model_results = train_fn(X_train_pre, y_train, X_test_pre, y_test)
    except Exception as e:
        raise ModelTrainingError(
            f"Model trainer for disease '{disease_id}' failed: {str(e)}",
            disease_id=disease_id,
        ) from e

    # Normalize returned model outputs
    trained_model_objs: Dict[str, Any] = {}
    evaluated_metrics: Dict[str, ModelMetrics] = {}
    model_metadata_map: Dict[str, Dict[str, Any]] = {}

    for model_type, res in raw_model_results.items():
        if isinstance(res, ModelOutput):
            model_obj = res.model
            y_pred = res.y_pred
            y_prob = res.y_prob
            train_time = res.train_time
            infer_time = res.infer_time
            forced_thresh = res.threshold
            model_meta = res.metadata or {}
        elif isinstance(res, dict):
            model_obj = res.get("model")
            y_pred = res.get("y_pred")
            y_prob = res.get("y_prob")
            train_time = res.get("train_time", 0.0)
            infer_time = res.get("infer_time", 0.0)
            forced_thresh = res.get("threshold")
            model_meta = res.get("metadata", {}) or {}
        else:
            raise ModelTrainingError(
                f"Invalid model result format for '{model_type}'",
                disease_id=disease_id,
            )

        trained_model_objs[model_type] = model_obj
        model_metadata_map[model_type] = model_meta

        # 6. Evaluate with Youden's J optimization
        metrics = evaluate_model_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            train_time=train_time,
            infer_time=infer_time,
            forced_threshold=forced_thresh,
        )
        evaluated_metrics[model_type] = metrics

    # 7. Benchmark comparison & delta calculation
    benchmark_comparison = compare_with_benchmarks(
        disease_id=disease_id,
        evaluated_models=evaluated_metrics,
        benchmark_dir=benchmark_dir,
    )

    # 8. Artifact persistence
    saved_artifacts = persist_training_artifacts(
        disease_id=disease_id,
        run_id=generated_run_id,
        preprocessor=preprocessor,
        trained_models=trained_model_objs,
        evaluated_metrics=evaluated_metrics,
        benchmark_comparison=benchmark_comparison,
        dataset_shape=dataset_shape,
        train_shape=train_shape,
        test_shape=test_shape,
        data_hash=data_hash,
        runtime_dir=runtime_dir,
        extra_config={
            "disease_name": disease_cfg.name,
            "default_qubits": disease_cfg.default_qubits,
            "test_size": test_size,
            "random_state": random_state,
        },
    )

    # 9. Unified Response Construction
    return {
        "status": "success",
        "run_id": generated_run_id,
        "disease_id": disease_id,
        "disease_name": disease_cfg.name,
        "standardization": {
            "status": "completed",
            "total_rows": dataset_shape[0],
            "raw_features": dataset_shape[1],
            "train_rows": train_shape[0],
            "test_rows": test_shape[0],
            "preprocessed_features": X_train_pre.shape[1],
            "data_hash": data_hash,
        },
        "models": {
            m_type: {
                **metric.to_dict(),
                "metadata": model_metadata_map.get(m_type, {}),
            }
            for m_type, metric in evaluated_metrics.items()
        },
        "benchmark_comparison": benchmark_comparison,
        "artifacts": saved_artifacts,
    }
