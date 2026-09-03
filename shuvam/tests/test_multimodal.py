from __future__ import annotations

import numpy as np
import pytest

from qhealth_qml.multimodal import (
    ModalityEvidence,
    fuse_latents,
    fuse_modalities,
    skill_weight,
)


def test_skill_weight_is_zero_at_chance_and_full_at_perfect():
    assert skill_weight(0.5) == 0.0
    assert skill_weight(0.4) == 0.0  # worse than chance earns no trust, never negative
    assert skill_weight(1.0) == 1.0
    assert skill_weight(0.75) == pytest.approx(0.5)


def test_aligned_score_puts_every_models_threshold_at_one_half():
    low = ModalityEvidence("signal", "ecg", score=0.30, threshold=0.30)
    high = ModalityEvidence("imaging", "angio", score=0.60, threshold=0.60)

    # Both sit exactly on their own decision boundary, so both align to 0.5
    # despite their raw probabilities differing by 0.30.
    assert low.aligned_score() == pytest.approx(0.5)
    assert high.aligned_score() == pytest.approx(0.5)


def test_aligned_score_is_monotone_and_stays_in_range():
    scores = np.linspace(0.0, 1.0, 21)
    aligned = [
        ModalityEvidence("signal", "ecg", score=float(s), threshold=0.3).aligned_score()
        for s in scores
    ]
    assert all(0.0 <= value <= 1.0 for value in aligned)
    assert all(b >= a for a, b in zip(aligned, aligned[1:]))


def test_missing_modality_is_dropped_not_imputed():
    present = ModalityEvidence("signal", "ecg", score=0.9, threshold=0.5, weight=0.7)
    absent = ModalityEvidence("imaging", "angio", score=None, threshold=0.5, weight=0.9)

    fused = fuse_modalities([present, absent])

    # The ECG alone decides the case; the absent angiogram contributes nothing.
    assert fused.contributing == ["signal"]
    assert fused.missing == ["imaging"]
    assert fused.probability == pytest.approx(present.aligned_score())
    assert fused.prediction == 1


def test_weights_are_renormalised_over_available_modalities():
    both = fuse_modalities(
        [
            ModalityEvidence("signal", "ecg", score=1.0, threshold=0.5, weight=1.0),
            ModalityEvidence("imaging", "angio", score=0.0, threshold=0.5, weight=1.0),
        ]
    )
    # Equal weights, opposite extremes -> exactly the midpoint.
    assert both.probability == pytest.approx(0.5)

    # Dropping one modality must not shrink the other's influence toward zero.
    one = fuse_modalities(
        [
            ModalityEvidence("signal", "ecg", score=1.0, threshold=0.5, weight=1.0),
            ModalityEvidence("imaging", "angio", score=None, threshold=0.5, weight=1.0),
        ]
    )
    assert one.probability == pytest.approx(1.0)


def test_a_more_skilled_modality_pulls_the_decision_further():
    trusted = fuse_modalities(
        [
            ModalityEvidence("signal", "ecg", score=1.0, threshold=0.5, weight=0.9),
            ModalityEvidence("imaging", "angio", score=0.0, threshold=0.5, weight=0.1),
        ]
    )
    assert trusted.probability == pytest.approx(0.9)


def test_case_with_no_usable_evidence_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="no modality provided usable evidence"):
        fuse_modalities(
            [
                ModalityEvidence("signal", "ecg", score=None),
                ModalityEvidence("imaging", "angio", score=float("nan")),
            ]
        )

    with pytest.raises(ValueError, match="at least one modality"):
        fuse_modalities([])


def test_fused_detection_reports_every_modality_including_absent_ones():
    fused = fuse_modalities(
        [
            ModalityEvidence("signal", "ecg", score=0.8, threshold=0.5, weight=0.7),
            ModalityEvidence("imaging", "angio", score=None, weight=0.9),
        ]
    )
    payload = fused.as_dict()

    assert [item["modality"] for item in payload["per_modality"]] == ["signal", "imaging"]
    assert payload["per_modality"][1]["available"] is False
    assert payload["per_modality"][1]["aligned_score"] is None
    assert payload["missing_modalities"] == ["imaging"]


def test_fuse_latents_allocates_a_fixed_qubit_budget_per_modality():
    ecg = np.arange(12, dtype=float).reshape(3, 4)
    angio = np.arange(18, dtype=float).reshape(3, 6)

    fused, names = fuse_latents(
        {"signal": ecg, "imaging": angio},
        qubit_budget={"signal": 3, "imaging": 2},
        modality_order=["signal", "imaging"],
    )

    assert fused.shape == (3, 5)  # head width stays the budget, not the latent widths
    assert names == ["signal_1", "signal_2", "signal_3", "imaging_1", "imaging_2"]
    assert np.allclose(fused[:, :3], ecg[:, :3])
    assert np.allclose(fused[:, 3:], angio[:, :2])


def test_fuse_latents_zeroes_an_absent_modalitys_slice():
    ecg = np.ones((2, 4), dtype=float)

    fused, _ = fuse_latents(
        {"signal": ecg, "imaging": None},
        qubit_budget={"signal": 3, "imaging": 2},
        modality_order=["signal", "imaging"],
    )

    assert fused.shape == (2, 5)
    assert np.allclose(fused[:, 3:], 0.0)


def test_fuse_latents_rejects_unpaired_rows_and_undersized_latents():
    with pytest.raises(ValueError, match="needs the same patients"):
        fuse_latents(
            {"signal": np.ones((3, 4)), "imaging": np.ones((2, 4))},
            qubit_budget={"signal": 2, "imaging": 2},
            modality_order=["signal", "imaging"],
        )

    with pytest.raises(ValueError, match="fewer than its"):
        fuse_latents(
            {"signal": np.ones((2, 2))},
            qubit_budget={"signal": 4},
            modality_order=["signal"],
        )

    with pytest.raises(ValueError, match="no qubit budget"):
        fuse_latents({"signal": np.ones((2, 2))}, qubit_budget={}, modality_order=["signal"])


def _train_modality_artifact(tmp_path, columns, name):
    """Train a real model artifact on a slice of the breast-cancer benchmark."""
    from qhealth_qml.experiment import (
        LoadedDataset,
        load_breast_cancer_dataset,
        load_model_artifact,
        run_experiment,
    )

    source = load_breast_cancer_dataset()
    dataset = LoadedDataset(
        name=name,
        X=source.X[:, columns],
        y=source.y,
        feature_names=[source.feature_names[i] for i in columns],
        positive_label=source.positive_label,
        negative_label=source.negative_label,
    )
    path = tmp_path / f"{name}.pkl"
    run_experiment(
        dataset,
        models=["logistic_regression"],
        max_train=120,
        max_test=60,
        seed=7,
        bootstrap_samples=0,
        calibrate=True,
        model_artifact_path=path,
    )
    return load_model_artifact(path), dataset


def test_detect_multimodal_runs_real_artifacts_and_degrades_when_one_is_absent(tmp_path):
    from qhealth_qml.multimodal import ModalityModel, detect_multimodal

    # Two detectors over disjoint feature sets stand in for two modalities.
    artifact_a, data_a = _train_modality_artifact(tmp_path, [0, 1, 2, 3], "modality_a")
    artifact_b, data_b = _train_modality_artifact(tmp_path, [4, 5, 6, 7], "modality_b")

    models = [
        ModalityModel("signal", "model-a", artifact_a, validated_balanced_accuracy=0.90),
        ModalityModel("imaging", "model-b", artifact_b, validated_balanced_accuracy=0.70),
    ]
    rows = 5
    both = {
        "signal": (data_a.X[:rows], data_a.feature_names),
        "imaging": (data_b.X[:rows], data_b.feature_names),
    }

    fused = detect_multimodal(models, both)
    assert len(fused) == rows
    for item in fused:
        assert sorted(item.contributing) == ["imaging", "signal"]
        assert item.missing == []
        assert 0.0 <= item.probability <= 1.0
        assert item.prediction in (0, 1)

    # The same patients, now without the imaging modality at all.
    ecg_only = detect_multimodal(models, {"signal": both["signal"], "imaging": None})
    assert len(ecg_only) == rows
    for item in ecg_only:
        assert item.contributing == ["signal"]
        assert item.missing == ["imaging"]
        assert 0.0 <= item.probability <= 1.0


def test_detect_multimodal_rejects_mismatched_patient_counts(tmp_path):
    from qhealth_qml.multimodal import ModalityModel, detect_multimodal

    artifact_a, data_a = _train_modality_artifact(tmp_path, [0, 1, 2, 3], "mm_a")
    artifact_b, data_b = _train_modality_artifact(tmp_path, [4, 5, 6, 7], "mm_b")
    models = [
        ModalityModel("signal", "a", artifact_a, validated_balanced_accuracy=0.9),
        ModalityModel("imaging", "b", artifact_b, validated_balanced_accuracy=0.9),
    ]

    with pytest.raises(ValueError, match="same patients"):
        detect_multimodal(
            models,
            {
                "signal": (data_a.X[:5], data_a.feature_names),
                "imaging": (data_b.X[:3], data_b.feature_names),
            },
        )


def test_train_modality_model_derives_its_fusion_weight_from_validation(tmp_path):
    from qhealth_qml.experiment import LoadedDataset, load_breast_cancer_dataset
    from qhealth_qml.multimodal import skill_weight, train_modality_model

    source = load_breast_cancer_dataset()
    dataset = LoadedDataset(
        name="ehr_like_cohort",
        X=source.X[:, :6],
        y=source.y,
        feature_names=list(source.feature_names[:6]),
        positive_label=source.positive_label,
        negative_label=source.negative_label,
    )

    model, result = train_modality_model(
        "structured_clinical",
        "ehr-detector",
        dataset,
        artifact_path=tmp_path / "ehr.pkl",
        max_train=150,
        max_test=80,
        seed=7,
        bootstrap_samples=0,
        validation_size=0.2,
        threshold_policy="target_sensitivity",
        target_sensitivity=0.8,
    )

    validated = result["models"]["logistic_regression"]["threshold"]["validation_metrics"][
        "balanced_accuracy"
    ]
    # The weight is the measured validation score, not a hand-supplied number.
    assert model.validated_balanced_accuracy == pytest.approx(validated)
    assert model.weight == pytest.approx(skill_weight(validated))
    assert model.modality == "structured_clinical"
    assert (tmp_path / "ehr.pkl").exists()
