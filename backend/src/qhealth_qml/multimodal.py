"""Multimodal detection over modality-specific hybrid models.

The platform's modality datasets are *disjoint cohorts*: the PTB-XL patients who
supply 12-lead ECGs are not the CADICA patients who supply coronary angiograms,
and neither are the patients in any EHR export. Joint (intermediate) fusion —
one encoder per modality trained end-to-end into a shared head — needs per-patient
paired modalities to train, so it is not trainable on the data this platform can
lawfully obtain today.

Late fusion is therefore not a compromise here, it is the only honest option: each
modality keeps its own independently-trained, independently-validated detector, and
this module combines their calibrated opinions per case. It needs no paired training
data, it degrades gracefully when a modality is absent, and every modality's
contribution stays separately measurable.

`fuse_latents` covers the other case: when genuinely paired per-patient modalities
*are* available, it packs per-modality latent vectors into one quantum-head-sized
vector under an explicit per-modality qubit budget.

Two consequences of the disjoint-cohort constraint that are easy to get wrong:

1. The combiner here is a *fixed rule*, not a learned fusion model, and that is
   forced rather than chosen. Fitting a stacking meta-learner needs rows where
   every modality's probability is observed for the same patient; with no cohort
   overlap that design matrix does not exist. Learned stacking unlocks only if a
   genuinely paired cohort is ever obtained.
2. **A multimodal gain cannot be measured across disjoint cohorts.** No patient
   has been evaluated under two modalities, so there is no held-out set on which
   "fused beats single-modality" is a meaningful claim. Report each modality's own
   validated performance; do not report a fused accuracy number until a paired
   evaluation cohort exists. Literature also cautions that below roughly 6k
   samples the multimodal gain tends to overlap unimodal performance anyway.

Design evidence: at comparable scale (N~2,100) late-fusion stacking reached
AUC 0.7213 where deep joint attention fusion reached 0.6612 while hitting >0.95
training AUC — the joint model had too few positives to resolve stable cross-modal
weights and memorised noise instead (arXiv 2512.14712). Latent imputation of an
absent modality is likewise reported to inject noise specifically at small sample
sizes (arXiv 2309.15529), which is why absence here is omission, never imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ModalityEvidence:
    """One modality's calibrated opinion about one case.

    `score` is that model's own calibrated detection probability and `threshold`
    its own validation-selected operating point. The two travel together because
    a probability means nothing across models without the threshold it is judged
    against: 0.42 is a positive call for a model whose threshold is 0.30 and a
    negative call for one whose threshold is 0.60.
    """

    modality: str
    model_id: str
    score: float | None
    threshold: float = 0.5
    weight: float = 1.0

    @property
    def available(self) -> bool:
        return self.score is not None and np.isfinite(self.score)

    def aligned_score(self) -> float:
        """Rescale this model's score so its own threshold sits at exactly 0.5.

        Averaging raw probabilities from models with different operating points
        silently favours whichever model has the lowest threshold. The piecewise
        linear map below is monotone, keeps the [0, 1] range, and makes 0.5 mean
        "on this model's decision boundary" for every modality, so a weighted mean
        across modalities compares like with like.
        """

        if not self.available:
            raise ValueError(f"{self.modality} evidence has no score to align")
        score = float(np.clip(self.score, 0.0, 1.0))
        threshold = float(np.clip(self.threshold, 0.0, 1.0))
        if threshold <= 0.0:
            return 0.5 + 0.5 * score
        if threshold >= 1.0:
            return 0.5 * score
        if score <= threshold:
            return 0.5 * score / threshold
        return 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)


def skill_weight(balanced_accuracy: float) -> float:
    """Trust weight for a modality, from its own validated balanced accuracy.

    Balanced accuracy 0.5 is chance and earns zero weight; 1.0 earns full weight.
    Deriving weights from each model's *own* validation run is what keeps this
    fusion trainable on disjoint cohorts — no paired patients are needed to learn
    how much to trust each modality.
    """

    return float(max(0.0, min(1.0, 2.0 * (float(balanced_accuracy) - 0.5))))


@dataclass(frozen=True)
class FusedDetection:
    """The combined decision plus a full account of how it was reached."""

    probability: float
    prediction: int
    threshold: float
    contributing: list[str]
    missing: list[str]
    per_modality: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "prediction": self.prediction,
            "threshold": self.threshold,
            "contributing_modalities": list(self.contributing),
            "missing_modalities": list(self.missing),
            "per_modality": [dict(item) for item in self.per_modality],
            "fusion": "skill-weighted mean of threshold-aligned calibrated probabilities",
            "interpretation": (
                "Combined detection evidence across available modalities. A modality that is "
                "absent for this case is omitted and the remaining weights are renormalised; "
                "it is never imputed."
            ),
        }


def fuse_modalities(
    evidence: Iterable[ModalityEvidence],
    threshold: float = 0.5,
) -> FusedDetection:
    """Combine per-modality calibrated opinions into one detection decision.

    Absent modalities are dropped and the surviving weights renormalised, so a
    case with only an ECG is scored on the ECG alone rather than against an
    invented angiography probability. Raises when nothing is available: a case
    with no evidence has no defensible answer, and returning 0.5 would disguise
    that as an uncertain one.
    """

    items = list(evidence)
    if not items:
        raise ValueError("fusion needs at least one modality's evidence")

    contributing = [item for item in items if item.available and item.weight > 0.0]
    missing = [item.modality for item in items if not item.available]
    if not contributing:
        raise ValueError(
            "no modality provided usable evidence for this case "
            f"(missing: {sorted(set(missing)) or 'none'}; "
            "any present modality scored zero weight)"
        )

    weights = np.asarray([item.weight for item in contributing], dtype=float)
    aligned = np.asarray([item.aligned_score() for item in contributing], dtype=float)
    probability = float(np.sum(weights * aligned) / np.sum(weights))

    return FusedDetection(
        probability=probability,
        prediction=int(probability >= threshold),
        threshold=float(threshold),
        contributing=[item.modality for item in contributing],
        missing=sorted(set(missing)),
        per_modality=[
            {
                "modality": item.modality,
                "model_id": item.model_id,
                "score": item.score,
                "threshold": item.threshold,
                "aligned_score": item.aligned_score() if item.available else None,
                "weight": item.weight,
                "available": item.available,
            }
            for item in items
        ],
    )


@dataclass(frozen=True)
class ModalityModel:
    """One modality's trained detector plus how far it earned our trust.

    `validated_balanced_accuracy` deliberately has no default and is not read off
    the artifact: an artifact records the model, not how well it scored. The
    number must come from that model's registered evaluation, so a modality can
    never silently claim influence it did not demonstrate.
    """

    modality: str
    model_id: str
    artifact: Any
    validated_balanced_accuracy: float

    @property
    def weight(self) -> float:
        return skill_weight(self.validated_balanced_accuracy)


def train_modality_model(
    modality: str,
    model_id: str,
    dataset: Any,
    artifact_path: Any,
    models: Sequence[str] = ("logistic_regression",),
    **run_kwargs: Any,
) -> tuple["ModalityModel", dict[str, Any]]:
    """Train one modality's detector on its own cohort and return it ready to fuse.

    The fusion weight is taken from the model's *validation* balanced accuracy,
    never its test score: the test fold is the locked estimate of how well this
    modality does, and letting it set the modality's influence in the combiner
    would leak that fold into the fused decision. Returning the weight from here
    rather than accepting it by hand is deliberate — a modality cannot be given
    influence that no measurement supports.
    """

    from .experiment import load_model_artifact, run_experiment

    if len(models) != 1:
        raise ValueError("a modality model trains exactly one estimator")
    model_name = models[0]

    result = run_experiment(
        dataset,
        models=[model_name],
        calibrate=True,
        model_artifact_path=artifact_path,
        **run_kwargs,
    )
    threshold_block = result["models"][model_name].get("threshold", {})
    validation_metrics = threshold_block.get("validation_metrics") or {}
    validated = validation_metrics.get("balanced_accuracy")
    if validated is None or not np.isfinite(float(validated)):
        raise ValueError(
            f"{modality} model produced no validation balanced accuracy, so its "
            "fusion weight cannot be established; run with a validation split"
        )

    return (
        ModalityModel(
            modality=modality,
            model_id=model_id,
            artifact=load_model_artifact(artifact_path),
            validated_balanced_accuracy=float(validated),
        ),
        result,
    )


def detect_multimodal(
    models: Sequence[ModalityModel],
    features: Mapping[str, tuple[np.ndarray, Sequence[str]] | None],
    threshold: float = 0.5,
    row_ids: Sequence[str] | None = None,
) -> list[FusedDetection]:
    """Run every available modality's detector over the same patients and fuse.

    `features` maps a modality to its `(X, feature_names)` pair, or to `None` when
    the patients simply do not have that modality. Every present modality must
    describe the *same* patients in the same row order — this is inference over
    one cohort, not the disjoint training cohorts the module docstring describes.
    """

    from .experiment import predict_with_model_artifact

    if not models:
        raise ValueError("multimodal detection needs at least one modality model")

    rows: int | None = None
    for model in models:
        supplied = features.get(model.modality)
        if supplied is None:
            continue
        X = np.asarray(supplied[0], dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{model.modality} features must be 2-D [rows, features]")
        if rows is not None and X.shape[0] != rows:
            raise ValueError(
                f"{model.modality} has {X.shape[0]} rows, expected {rows}; "
                "every present modality must describe the same patients"
            )
        rows = X.shape[0]
    if rows is None:
        raise ValueError("no modality supplied features for these patients")

    scores: dict[str, list[float | None]] = {}
    for model in models:
        supplied = features.get(model.modality)
        if supplied is None:
            scores[model.modality] = [None] * rows
            continue
        X, feature_names = supplied
        result = predict_with_model_artifact(
            model.artifact,
            np.asarray(X, dtype=float),
            feature_names,
            dataset_name=f"multimodal:{model.modality}",
            row_ids=row_ids,
        )
        scores[model.modality] = [row["score"] for row in result["prediction_rows"]]

    return [
        fuse_modalities(
            [
                ModalityEvidence(
                    modality=model.modality,
                    model_id=model.model_id,
                    score=scores[model.modality][index],
                    threshold=(
                        float(model.artifact.threshold)
                        if model.artifact.threshold is not None
                        else 0.5
                    ),
                    weight=model.weight,
                )
                for model in models
            ],
            threshold=threshold,
        )
        for index in range(rows)
    ]


def fuse_latents(
    latents: Mapping[str, np.ndarray | None],
    qubit_budget: Mapping[str, int],
    modality_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Pack per-modality latent vectors into one quantum-head-sized vector.

    Only usable when modalities are genuinely paired per patient. Each modality
    gets a fixed slice of the qubit budget, so the head's width stays known and
    bounded no matter how many modalities are configured — quantum kernels
    concentrate exponentially as qubits grow, so the budget is a hard constraint
    rather than something to widen per modality.

    A modality that is absent contributes zeros in its own slice. That is a real
    imputation, unlike the late-fusion path, so pair it with modality dropout in
    training if you use it: a head that has never seen a zeroed slice will not
    handle one at inference.
    """

    order = list(modality_order) if modality_order is not None else sorted(qubit_budget)
    unknown = set(order) - set(qubit_budget)
    if unknown:
        raise ValueError(f"no qubit budget for modalities: {sorted(unknown)}")
    if any(int(qubit_budget[name]) < 1 for name in order):
        raise ValueError("every modality's qubit budget must be at least 1")

    rows: int | None = None
    for name in order:
        value = latents.get(name)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} latent must be 2-D [rows, features]")
        if rows is not None and array.shape[0] != rows:
            raise ValueError(
                f"{name} latent has {array.shape[0]} rows, expected {rows}; "
                "joint fusion needs the same patients in every modality"
            )
        rows = array.shape[0]
    if rows is None:
        raise ValueError("joint fusion needs at least one present modality")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for name in order:
        width = int(qubit_budget[name])
        value = latents.get(name)
        if value is None:
            blocks.append(np.zeros((rows, width), dtype=float))
        else:
            array = np.asarray(value, dtype=float)
            if array.shape[1] < width:
                raise ValueError(
                    f"{name} latent has {array.shape[1]} features, "
                    f"fewer than its {width}-qubit budget"
                )
            blocks.append(array[:, :width])
        names.extend(f"{name}_{index + 1}" for index in range(width))

    return np.column_stack(blocks), names
