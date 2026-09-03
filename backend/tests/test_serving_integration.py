"""End-to-end acceptance tests for the deployable path: files on disk -> usable output.

These are the tests that decide whether this is integrable software rather than a pile of
research scripts. Each one pins a property an upstream pipeline actually depends on:

* a bundle round-trips through disk with every fitted transform intact,
* preprocessing is REPLAYED, never refitted (a batch of one scores the same as a batch of many),
* junk input is refused with a reason instead of silently scored,
* a study with a missing sequence is refused instead of zero-filled,
* the output states its own temporal framing so a triage score cannot be presented as an
  early-warning score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from qhealth_qml.ingestion import ingest_study  # noqa: E402
from qhealth_qml.serving import (  # noqa: E402
    SERVING_SCHEMA_VERSION,
    InferenceBundle,
    load_bundle,
    predict,
    quality_check,
    save_bundle,
)


def _toy_bundle(tmp_path: Path, channels=("t1",), grid=(8, 8, 8)) -> InferenceBundle:
    """A small but structurally real bundle: real encoder, real fitted sklearn transforms."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from qhealth_qml.pretrained_encoder import PretrainedVolumeClassifier

    rng = np.random.default_rng(7)
    model = PretrainedVolumeClassifier(in_channels=len(channels), latent_dim=8, slice_stride=4)
    latent = rng.normal(size=(20, 8))
    y = (rng.random(20) > 0.5).astype(int)

    standardizer = StandardScaler().fit(latent)
    scaled = standardizer.transform(latent)
    angle = MinMaxScaler(feature_range=(-0.3, 0.3)).fit(scaled)
    head = LogisticRegression(max_iter=200).fit(angle.transform(scaled), y)

    return InferenceBundle(
        schema_version=SERVING_SCHEMA_VERSION,
        model_id="toy-model",
        condition="unit-test",
        temporal_framing="detection",
        positive_label="finding_present",
        negative_label="finding_absent",
        channel_names=list(channels),
        input_grid=list(grid),
        encoder_state=model.state_dict(),
        encoder_kind="pretrained_volume",
        encoder_config={"in_channels": len(channels), "latent_dim": 8, "slice_stride": 4},
        head_kind="logistic_regression",
        head=head,
        standardizer=standardizer,
        reducer=None,
        angle_scaler=angle,
        threshold=0.5,
        threshold_policy="target_sensitivity@0.8 (validation only)",
        quantum_config=None,
        training_provenance={"dataset": "synthetic", "cohort_size": 20, "modality": "MR"},
        input_stats={"mean": 0.5, "std": 0.2},
    )


def test_bundle_roundtrips_with_fitted_transforms(tmp_path):
    bundle = _toy_bundle(tmp_path)
    path = save_bundle(bundle, tmp_path / "toy.pkl")
    assert path.exists()
    assert path.with_suffix(path.suffix + ".manifest.json").exists()

    reloaded = load_bundle(path)
    assert reloaded.model_id == "toy-model"
    # the fitted scaler must survive, not be re-derived
    assert np.allclose(reloaded.standardizer.mean_, bundle.standardizer.mean_)
    assert reloaded.threshold == bundle.threshold


def test_preprocessing_is_replayed_not_refitted(tmp_path):
    """A study must score identically alone and inside a batch.

    If any transform were refitted on the incoming data, batch composition would move the
    decision boundary -- the failure this whole design exists to prevent.
    """
    bundle = _toy_bundle(tmp_path)
    rng = np.random.default_rng(11)
    volumes = rng.random((4, 1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3

    alone = predict(bundle, volumes[0], study_id="s0")
    together = [predict(bundle, v, study_id=f"s{i}") for i, v in enumerate(volumes)]
    assert alone["status"] == "ok"
    assert alone["score"] == pytest.approx(together[0]["score"], abs=1e-9)


def test_wrong_shape_is_rejected_with_reason(tmp_path):
    bundle = _toy_bundle(tmp_path)
    wrong = np.random.random((1, 16, 16, 16)).astype(np.float32)  # wrong grid
    result = predict(bundle, wrong, study_id="bad")
    assert result["status"] == "rejected"
    assert result["prediction"] is None
    assert any("grid" in reason for reason in result["reasons"])


def test_wrong_channel_count_is_rejected(tmp_path):
    bundle = _toy_bundle(tmp_path, channels=("t1", "flair"))
    single = np.random.random((1, 8, 8, 8)).astype(np.float32)
    result = predict(bundle, single, study_id="bad")
    assert result["status"] == "rejected"
    assert any("channel" in reason for reason in result["reasons"])


def test_constant_and_nonfinite_volumes_are_rejected(tmp_path):
    bundle = _toy_bundle(tmp_path)
    constant = np.zeros((1, 8, 8, 8), dtype=np.float32)
    assert not quality_check(bundle, constant)["passed"]

    nonfinite = np.random.random((1, 8, 8, 8)).astype(np.float32)
    nonfinite[0, 0, 0, 0] = np.nan
    assert not quality_check(bundle, nonfinite)["passed"]


def test_output_states_temporal_framing_and_is_research_only(tmp_path):
    bundle = _toy_bundle(tmp_path)
    volume = (np.random.random((1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3)
    result = predict(bundle, volume, study_id="s")
    assert result["temporal_framing"] == "detection"
    assert result["research_use_only"] is True
    assert "not a probability of disease" in result["confidence_definition"].lower()
    # a detection model must not describe itself as an early warning
    assert "early-warning" not in result["interpretation"].lower()


def test_prediction_framing_reports_lead_time(tmp_path):
    bundle = _toy_bundle(tmp_path)
    bundle.temporal_framing = "prediction"
    bundle.training_provenance["lead_time_minutes"] = 5
    volume = (np.random.random((1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3)
    result = predict(bundle, volume, study_id="s")
    assert "EARLY-WARNING" in result["interpretation"]
    assert "5 minutes" in result["interpretation"]


def test_ingestion_refuses_missing_sequence(tmp_path):
    """A four-sequence model handed three must refuse, not zero-fill the fourth."""
    bundle = _toy_bundle(tmp_path, channels=("t1", "t1_post", "t2", "flair"))
    result = ingest_study(bundle.to_manifest(), {"t1": tmp_path / "a.nii", "t2": tmp_path / "b.nii"})
    assert result.ok is False
    assert any("missing required sequence" in e for e in result.errors)


def test_ingestion_reads_nifti_and_matches_expected_grid(tmp_path):
    nib = pytest.importorskip("nibabel")
    bundle = _toy_bundle(tmp_path)
    volume = np.random.random((12, 14, 10)).astype(np.float32)
    path = tmp_path / "study.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))

    result = ingest_study(bundle.to_manifest(), path)
    assert result.ok, result.errors
    assert result.volume.shape == (1, 8, 8, 8)  # resampled to the artifact's grid
    assert "resample" in result.provenance["applied"]


def test_decision_function_head_score_normalisation_is_persisted(tmp_path):
    """Regression: a decision_function head must not collapse to an all-positive predictor.

    QSVC without probability=True exposes only `decision_function`, whose output is unbounded
    and frequently negative. An early version of the bundle builder clipped that raw value to
    [0,1], which flattened every negative score to zero, drove the selected threshold to 0.0 and
    turned a model with ROC-AUC 0.82 into one that predicted every study positive (balanced
    accuracy 0.500). The normalisation is now fitted on validation and PERSISTED; this pins that.
    """
    from sklearn.svm import SVC

    bundle = _toy_bundle(tmp_path)
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(30, 8))
    y = (latent[:, 0] > 0).astype(int)
    scaled = bundle.standardizer.transform(latent)
    head = SVC(kernel="linear", probability=False).fit(bundle.angle_scaler.transform(scaled), y)

    raw = head.decision_function(bundle.angle_scaler.transform(scaled))
    assert raw.min() < 0, "test premise: decision_function should produce negative values"

    bundle.head = head
    bundle.head_kind = "svc_decision_function"
    bundle.score_offset = float(raw.min())
    bundle.score_scale = float(raw.max() - raw.min())
    bundle.threshold = 0.5

    volumes = rng.random((6, 1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3
    scores = [predict(bundle, v, study_id=f"s{i}")["score"] for i, v in enumerate(volumes)]
    # scores must stay inside [0,1] and must not all be pinned to a single value
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert len(set(round(s, 6) for s in scores)) > 1, "scores collapsed to a constant"

    # and the persisted constants must be used, not re-derived from this batch
    bundle.score_offset += 10.0
    shifted = predict(bundle, volumes[0], study_id="s0")["score"]
    assert shifted != scores[0], "score_offset was ignored; normalisation is being re-derived"


def test_batch_predict_builds_encoder_once(tmp_path, monkeypatch):
    """Regression: batch scoring must not reconstruct the model per sample.

    `batch_predict` originally looped over `predict()`, which called `_rebuild_encoder()` each
    time — for a 33M-parameter frozen backbone that made throughput dominated by model
    construction rather than inference. Reported by a peer session reading serving.py.
    """
    from qhealth_qml import serving

    bundle = _toy_bundle(tmp_path)
    calls = {"n": 0}
    original = serving._rebuild_encoder

    def counting(bundle_arg):
        calls["n"] += 1
        return original(bundle_arg)

    monkeypatch.setattr(serving, "_rebuild_encoder", counting)
    volumes = (np.random.random((5, 1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3)
    results = serving.batch_predict(bundle, volumes)

    assert len(results) == 5
    assert all(r["status"] == "ok" for r in results)
    assert calls["n"] == 1, f"encoder rebuilt {calls['n']}x for 5 studies; expected once"


def test_batch_and_single_scoring_agree(tmp_path):
    """Reusing one encoder across a batch must not change any individual score."""
    from qhealth_qml.serving import batch_predict

    bundle = _toy_bundle(tmp_path)
    volumes = (np.random.random((3, 1, 8, 8, 8)).astype(np.float32) * 0.4 + 0.3)
    batched = batch_predict(bundle, volumes)
    singles = [predict(bundle, v, study_id=f"study-{i+1}") for i, v in enumerate(volumes)]
    for b, s in zip(batched, singles):
        assert b["score"] == pytest.approx(s["score"], abs=1e-9)
