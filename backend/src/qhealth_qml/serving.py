"""Deployable inference for the imaging/signal hybrids: artifact in, clinical-shaped output out.

The research runners in this repo fit an encoder, score a held-out split and print a number.
That is not something another system can call. This module closes that gap: it persists a
*complete* inference bundle (encoder weights + every fitted preprocessing step + the head +
the validation-selected operating threshold + provenance) and reloads it to score new studies.

Three properties matter for this to be integrable rather than merely runnable:

1. **Nothing is refitted at inference.** The standardiser, the dimensionality reducer, the angle
   scaler and the threshold are all fitted on training data, saved, and replayed. A pipeline that
   re-derives normalisation from the incoming batch silently changes its own decision boundary
   with batch composition -- a real and easy-to-miss production failure.
2. **Out-of-distribution input is refused, not scored.** Shape, channel count and intensity
   statistics are checked against what the artifact was trained on; a mismatch returns an
   explicit `status: "rejected"` with a reason. Handing a neck CT to a brain-haemorrhage model
   and getting back a confident number is worse than getting back an error.
3. **The output states its own temporal framing.** Every result carries whether it is a
   `prediction` (early warning, with lead time), a `detection` (event present now) or a
   `characterisation` (property of an existing finding), so a downstream UI cannot present a
   triage score as an early-warning score.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SERVING_SCHEMA_VERSION = 2

TEMPORAL_FRAMINGS = {"prediction", "detection", "characterisation", "screening"}


@dataclass
class InferenceBundle:
    """Everything needed to score a new study, and nothing that must be refitted."""

    schema_version: int
    model_id: str
    condition: str
    temporal_framing: str
    positive_label: str
    negative_label: str
    channel_names: list[str]
    input_grid: list[int]
    encoder_state: dict[str, Any]
    encoder_kind: str          # "pretrained_volume" | "volume_cnn" | "gait_cnn"
    encoder_config: dict[str, Any]
    head_kind: str             # "qsvc" | "rbf_svc" | "logistic_regression" | ...
    head: Any
    standardizer: Any
    reducer: Any | None
    angle_scaler: Any | None
    threshold: float
    threshold_policy: str
    quantum_config: dict[str, Any] | None
    training_provenance: dict[str, Any]
    input_stats: dict[str, float]
    # Raw decision-function outputs are mapped to [0,1] with THESE fitted constants.
    # Re-deriving them per batch would move the decision boundary with batch composition.
    score_offset: float = 0.0
    score_scale: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_manifest(self) -> dict[str, Any]:
        """Human/machine-readable description, safe to publish alongside the binary."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "condition": self.condition,
            "temporal_framing": self.temporal_framing,
            "labels": {"positive": self.positive_label, "negative": self.negative_label},
            "expects": {
                "channels": self.channel_names,
                "grid": self.input_grid,
                "channel_count": len(self.channel_names),
            },
            "encoder": {"kind": self.encoder_kind, **self.encoder_config},
            "head": self.head_kind,
            "quantum": self.quantum_config,
            "operating_point": {
                "threshold": self.threshold,
                "policy": self.threshold_policy,
            },
            "training_provenance": self.training_provenance,
            "created_at": self.created_at,
        }


def save_bundle(bundle: InferenceBundle, path: str | Path) -> Path:
    """Persist the bundle plus a sidecar JSON manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = target.with_suffix(target.suffix + ".manifest.json")
    manifest.write_text(json.dumps(bundle.to_manifest(), indent=2, default=str) + "\n", encoding="utf-8")
    return target


def load_bundle(path: str | Path) -> InferenceBundle:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, InferenceBundle):
        raise ValueError("file is not a qhealth inference bundle")
    if bundle.schema_version != SERVING_SCHEMA_VERSION:
        raise ValueError(
            f"bundle schema {bundle.schema_version} != expected {SERVING_SCHEMA_VERSION}"
        )
    return bundle


def _rebuild_encoder(bundle: InferenceBundle) -> Any:
    """Reconstruct the encoder, or None for feature-vector models that have no encoder.

    Not every condition has a learned front end: the pre-ictal EEG model consumes engineered
    band-power features directly, so `encoder_kind == "none"` is a first-class case rather than
    an error. The rest of the serving path is identical, which is the point of the contract.
    """

    import torch

    if bundle.encoder_kind == "none":
        return None
    if bundle.encoder_kind == "pretrained_volume":
        from .pretrained_encoder import PretrainedVolumeClassifier

        model = PretrainedVolumeClassifier(
            in_channels=bundle.encoder_config["in_channels"],
            latent_dim=bundle.encoder_config["latent_dim"],
            slice_stride=bundle.encoder_config.get("slice_stride", 2),
        )
    elif bundle.encoder_kind == "volume_cnn":
        from .imaging_hybrid import VolumeClassifier

        model = VolumeClassifier(
            in_channels=bundle.encoder_config["in_channels"],
            latent_dim=bundle.encoder_config["latent_dim"],
        )
    elif bundle.encoder_kind == "gait_cnn":
        from .gait_hybrid import RawGaitClassifier

        model = RawGaitClassifier(latent_dim=bundle.encoder_config["latent_dim"])
    else:
        raise ValueError(f"unknown encoder kind {bundle.encoder_kind!r}")
    model.load_state_dict(bundle.encoder_state)
    model.eval()
    return model


def quality_check(bundle: InferenceBundle, volume: np.ndarray) -> dict[str, Any]:
    """Reject inputs that do not look like what the model was trained on."""

    problems: list[str] = []
    expected_channels = len(bundle.channel_names)
    if bundle.encoder_kind == "none":
        # feature vector: one flat row whose length must match the trained feature count
        values = np.asarray(volume, dtype=float).reshape(-1)
        expected_features = int(bundle.input_grid[0]) if bundle.input_grid else len(values)
        if values.size != expected_features:
            problems.append(f"expected {expected_features} features, got {values.size}")
        if not np.all(np.isfinite(values)):
            problems.append("input contains non-finite values")
        if values.size and float(values.std()) == 0.0:
            problems.append("input is constant")
        return {"passed": not problems, "problems": problems}
    expected_ndim = 1 + len(bundle.input_grid)  # [C] + grid dims: 4-D volume, 3-D signal
    if volume.ndim != expected_ndim:
        problems.append(
            f"expected a {expected_ndim}-D array (channels + {len(bundle.input_grid)} grid dims), "
            f"got shape {volume.shape}"
        )
        return {"passed": False, "problems": problems}
    if volume.shape[0] != expected_channels:
        problems.append(
            f"expected {expected_channels} channel(s) {bundle.channel_names}, got {volume.shape[0]}"
        )
    if list(volume.shape[1:]) != list(bundle.input_grid):
        problems.append(f"expected grid {bundle.input_grid}, got {list(volume.shape[1:])}")
    finite = np.isfinite(volume)
    if not finite.all():
        problems.append("input contains non-finite values")
    if finite.any():
        observed_mean = float(volume[finite].mean())
        observed_std = float(volume[finite].std())
        train_mean = bundle.input_stats.get("mean", observed_mean)
        train_std = bundle.input_stats.get("std", observed_std) or 1e-6
        # a very coarse distribution guard: catches wrong modality / unnormalised input,
        # not subtle domain shift, and is documented as such.
        if abs(observed_mean - train_mean) > 6 * train_std:
            problems.append(
                f"intensity mean {observed_mean:.3f} is far from training mean "
                f"{train_mean:.3f} (+/-{train_std:.3f}); wrong modality or missing normalisation?"
            )
        if observed_std < 1e-6:
            problems.append("input is constant (empty or corrupt volume)")
    return {"passed": not problems, "problems": problems}


def predict(
    bundle: InferenceBundle,
    volume: np.ndarray,
    study_id: str = "study",
    abstain_margin: float | None = 0.05,
    _encoder: Any | None = None,
) -> dict[str, Any]:
    """Score one already-preprocessed study and return a clinically-shaped result."""

    import torch

    checked = quality_check(bundle, volume)
    if not checked["passed"]:
        return {
            "schema_version": SERVING_SCHEMA_VERSION,
            "model_id": bundle.model_id,
            "study_id": study_id,
            "status": "rejected",
            "reasons": checked["problems"],
            "score": None,
            "prediction": None,
        }

    if bundle.encoder_kind == "none":
        # feature-vector model: the input IS the representation
        latent = np.asarray(volume, dtype=np.float32).reshape(1, -1)
    else:
        model = _encoder if _encoder is not None else _rebuild_encoder(bundle)
        with torch.no_grad():
            latent = model.encoder(torch.from_numpy(volume[None, ...].astype(np.float32))).numpy()

    features = bundle.standardizer.transform(latent)
    if bundle.reducer is not None:
        features = bundle.reducer.transform(features)
    if bundle.angle_scaler is not None:
        features = bundle.angle_scaler.transform(features)

    head = bundle.head
    if hasattr(head, "predict_proba") and getattr(head, "probability", True):
        score = float(head.predict_proba(features)[0, 1])
    else:
        raw = float(head.decision_function(features)[0])
        # replay the normalisation fitted on validation; never re-derive it here
        score = float(np.clip((raw - bundle.score_offset) / bundle.score_scale, 0.0, 1.0))

    prediction = int(score >= bundle.threshold)
    margin = abs(score - bundle.threshold)
    abstained = abstain_margin is not None and margin < abstain_margin

    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "model_id": bundle.model_id,
        "condition": bundle.condition,
        "study_id": study_id,
        "status": "ok",
        "temporal_framing": bundle.temporal_framing,
        "label": bundle.positive_label if prediction else bundle.negative_label,
        "prediction": prediction,
        "score": score,
        "threshold": bundle.threshold,
        "threshold_policy": bundle.threshold_policy,
        "decision_margin": margin,
        "review_recommended": bool(abstained),
        "confidence_definition": (
            "normalised distance of the calibrated score from the validation-selected operating "
            "threshold; NOT a probability of disease and NOT a confidence interval"
        ),
        "interpretation": _interpretation(bundle, prediction),
        "limitations": bundle.training_provenance.get("limitations", []),
        "model_provenance": {
            "encoder": bundle.encoder_kind,
            "head": bundle.head_kind,
            "quantum": bundle.quantum_config,
            "trained_on": bundle.training_provenance.get("dataset"),
            "cohort_size": bundle.training_provenance.get("cohort_size"),
            "held_out_performance": bundle.training_provenance.get("held_out_performance"),
        },
        "research_use_only": True,
    }


def _interpretation(bundle: InferenceBundle, prediction: int) -> str:
    framing = bundle.temporal_framing
    label = bundle.positive_label if prediction else bundle.negative_label
    if framing == "prediction":
        lead = bundle.training_provenance.get("lead_time_minutes")
        window = f" within roughly the next {lead:.0f} minutes" if lead else ""
        return (
            f"Model indicates {label}{window}. This is an EARLY-WARNING estimate, not a statement "
            "that an event is occurring now."
        )
    if framing == "detection":
        return f"Model indicates {label} is present in the supplied study (current-state detection)."
    if framing == "characterisation":
        return (
            f"Model characterises an already-identified finding as {label}. It does not detect "
            "whether the finding is present."
        )
    return f"Model screening result: {label}."


def batch_predict(
    bundle: InferenceBundle, volumes: np.ndarray, study_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Score many studies while building the encoder exactly once.

    The naive loop over `predict()` reconstructs the torch model per sample, which for a
    33M-parameter frozen backbone dominates runtime and made batch throughput O(n) in model
    construction rather than in inference. The encoder is built once here and injected.
    """

    ids = study_ids or [f"study-{i+1}" for i in range(len(volumes))]
    encoder = _rebuild_encoder(bundle)
    return [
        predict(bundle, volume, study_id=sid, _encoder=encoder)
        for volume, sid in zip(volumes, ids)
    ]
