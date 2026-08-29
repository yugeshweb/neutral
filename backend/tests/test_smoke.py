"""One dependency-light runnable check for the complete local QML path."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from qhealth_qml.experiment import (
    LoadedDataset,
    load_csv_dataset,
    load_high_dimensional_csv_dataset,
    load_breast_cancer_dataset,
    load_model_artifact,
    load_prediction_csv,
    load_profile_dataset,
    predict_with_model_artifact,
    run_experiment,
    run_repeated_experiment,
    prepare_dataset,
    _split_indices,
)
from qhealth_qml.study import run_nested_evaluation, run_resource_sweep


def main() -> None:
    result = run_experiment(
        load_breast_cancer_dataset(),
        models=("classical", "qsvc", "pegasos_qsvc"),
        backend="statevector",
        n_qubits=4,
        max_train=16,
        max_test=8,
        explain=False,
    )
    assert set(result["models"]) == {
        "logistic_regression",
        "rbf_svc",
        "hist_gradient_boosting",
        "qsvc",
        "pegasos_qsvc",
    }
    assert result["hardware_probe"]["backend"] == "statevector"
    assert result["preprocessing"]["qubits"] == 4
    for model in result["models"].values():
        assert 0.0 <= model["metrics"]["accuracy"] <= 1.0
    repeated = run_repeated_experiment(
        load_breast_cancer_dataset(),
        repeats=2,
        models=("classical",),
        max_train=12,
        max_test=8,
        explain=False,
    )
    assert repeated["repeated_evaluation"]["repeats"] == 2
    assert len(repeated["repeated_evaluation"]["seeds"]) == 2
    assert repeated["repeated_evaluation"]["metric_summary"][
        "logistic_regression"
    ]["coverage"]["mean"] == 1.0
    calibrated = run_experiment(
        load_breast_cancer_dataset(),
        models=("rbf_svc", "qsvc"),
        backend="statevector",
        max_train=16,
        max_test=8,
        calibrate=True,
        explain=False,
    )
    assert calibrated["execution"]["calibration"]["enabled"] is True
    assert calibrated["models"]["qsvc"]["metrics"]["brier_score"] is not None
    thresholded = run_experiment(
        load_breast_cancer_dataset(),
        models=("rbf_svc",),
        threshold_policy="target_sensitivity",
        target_sensitivity=0.8,
        abstain_margin=0.05,
        bootstrap_samples=20,
        max_train=20,
        max_test=8,
        explain=False,
    )
    thresholded_model = thresholded["models"]["rbf_svc"]
    assert thresholded["split"]["train_pool_rows"] == 20
    assert thresholded["split"]["train_rows"] == 16
    assert thresholded["split"]["test_rows"] == 8
    assert thresholded["split"]["validation_rows"] > 0
    assert thresholded["execution"]["calibration"]["enabled"] is True
    assert thresholded_model["threshold"]["policy"] == "target_sensitivity"
    assert thresholded_model["confidence_intervals"] is not None
    assert thresholded_model["abstention"]["enabled"] is True

    metadata_X = np.arange(24, dtype=float).reshape(12, 2)
    metadata_y = np.array([0, 1] * 6, dtype=int)
    grouped = LoadedDataset(
        name="grouped-check",
        X=metadata_X,
        y=metadata_y,
        feature_names=["f1", "f2"],
        positive_label="positive",
        negative_label="negative",
        groups=np.repeat([f"p{i}" for i in range(6)], 2),
    )
    grouped_prepared = prepare_dataset(
        grouped,
        n_qubits=1,
        test_size=0.25,
        validation_size=0.25,
        seed=7,
        max_train=0,
        max_test=0,
    )
    assert grouped_prepared.y_validation is not None
    assert grouped.groups is not None
    group_train, group_test = _split_indices(
        metadata_y,
        0.25,
        7,
        groups=grouped.groups,
    )
    assert set(grouped.groups[group_train]).isdisjoint(set(grouped.groups[group_test]))
    chronological = LoadedDataset(
        name="chronological-check",
        X=metadata_X,
        y=metadata_y,
        feature_names=["f1", "f2"],
        positive_label="positive",
        negative_label="negative",
        times=np.array([f"2025-01-{i + 1:02d}" for i in range(12)]),
    )
    chronological_prepared = prepare_dataset(
        chronological,
        n_qubits=1,
        test_size=0.25,
        validation_size=0.25,
        seed=7,
        max_train=0,
        max_test=0,
    )
    assert chronological_prepared.y_validation is not None
    assert chronological.times is not None
    time_train, time_test = _split_indices(
        metadata_y,
        0.25,
        7,
        times=chronological.times,
    )
    assert max(chronological.times[time_train]) < min(chronological.times[time_test])

    tiny = load_csv_dataset(
        Path(__file__).parent / "fixtures" / "tiny.csv",
        target="diagnosis",
        positive_label="malignant",
    )
    matrix = load_high_dimensional_csv_dataset(
        Path(__file__).parent / "fixtures" / "tiny.csv",
        target="diagnosis",
        positive_label="malignant",
    )
    assert matrix.task_profile is not None
    assert matrix.task_profile["modality"] == "gene_expression"
    with TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.pkl"
        trained = run_experiment(
            tiny,
            models=("logistic_regression",),
            n_qubits=2,
            max_train=0,
            max_test=0,
            model_artifact_path=model_path,
            explain=False,
        )
        assert trained["model_artifact"]["model_name"] == "logistic_regression"
        manifest_path = Path(f"{model_path}.manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == trained["schema_version"]
        assert manifest["dataset"]["fingerprint"] == trained["dataset"]["fingerprint"]
        artifact = load_model_artifact(model_path)
        X, feature_names = load_prediction_csv(
            Path(__file__).parent / "fixtures" / "tiny.csv",
            artifact.preprocessor.feature_names,
        )
        predictions = predict_with_model_artifact(artifact, X, feature_names)
        assert len(predictions["predictions"]) == len(tiny.y)

    profiled = load_profile_dataset(Path(__file__).parent / "fixtures" / "early_detection_profile.json")
    profiled_result = run_experiment(
        profiled,
        models=("logistic_regression",),
        n_qubits=2,
        reduction="pca",
        max_train=0,
        max_test=0,
        explain=True,
    )
    assert profiled_result["split"]["strategy"] == "group_and_chronological"
    assert profiled_result["dataset"]["task_profile"]["task_type"] == "early_detection"
    profiled_model = profiled_result["models"]["logistic_regression"]
    assert profiled_model["metrics"]["pr_auc"] is not None
    assert profiled_model["clinical_evaluation"]["decision_curve"]
    assert profiled_model["explanation"]["status"] == "ok"

    site_holdout = run_experiment(
        profiled,
        models=("logistic_regression",),
        n_qubits=2,
        reduction="pca",
        holdout_site="site-b",
        max_train=0,
        max_test=0,
        explain=False,
    )
    assert site_holdout["split"]["strategy"] == "site_holdout"
    assert "sex" in site_holdout["models"]["logistic_regression"]["clinical_evaluation"]["subgroups"]

    nested = run_nested_evaluation(
        profiled,
        models=("logistic_regression",),
        outer_repeats=1,
        inner_repeats=1,
        n_qubits=2,
        reduction="pca",
        tune=False,
    )
    assert nested["study"]["type"] == "nested_repeated_holdout"
    assert nested["metric_summary"]["logistic_regression"]["balanced_accuracy"]["n"] == 1

    sweep = run_resource_sweep(
        profiled,
        qubits=(2,),
        backends=("statevector",),
        models=("logistic_regression",),
        reduction="pca",
        max_train=8,
        max_test=4,
    )
    assert len(sweep["rows"]) == 1
    assert sweep["rows"][0]["resource"]["feature_count"] == 2
    print("smoke test passed")


if __name__ == "__main__":
    main()
