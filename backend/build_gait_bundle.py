"""Package the P6 gait model as a deployable inference bundle.

The imaging builder assumes a volumetric encoder; gait is an 18-channel 1-D force-plate signal,
so it needs its own build path even though it produces the *same* `InferenceBundle` contract and
is served by the same `serving.predict`.

Two things differ from the imaging path and both matter:

* **Subject-grouped splitting.** A subject contributes several walking trials, and one person's
  trials are highly self-similar. A stratified row split would put the same subject either side
  of the boundary and inflate the result; `split_indices_grouped` splits by subject.
* **Temporal framing is `detection`, not prediction.** The model discriminates diagnosed
  Parkinson's from controls on a gait recording; it does not forecast onset. Recording that in
  the artifact stops a downstream UI presenting it as an early-warning score.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from qhealth_qml.experiment import build_quantum_context, classification_metrics, select_threshold
from qhealth_qml.gait_hybrid import (
    RawGaitClassifier,
    load_raw_gaitpdb,
    seed_everything,
    split_indices_grouped,
)
from qhealth_qml.serving import SERVING_SCHEMA_VERSION, InferenceBundle, save_bundle


def _probabilities(model, signals, batch_size=8):
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(signals)), batch_size=batch_size):
            _, logits = model(batch)
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def _encode(model, signals, batch_size=8):
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(signals)), batch_size=batch_size):
            out.append(model.encoder(batch).numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="runtime/bundles/parkinsons-gait-signal.pkl")
    args = parser.parse_args()

    dataset = load_raw_gaitpdb(args.data_root, target_samples=12000)
    y = dataset.y
    print(
        f"gaitpdb: {len(y)} recordings, {len(set(dataset.groups.tolist()))} subjects, "
        f"{int(y.sum())} PD / {int((1-y).sum())} control",
        flush=True,
    )

    seed_everything(args.seed)
    train, validation, test = split_indices_grouped(dataset, args.seed)
    model = RawGaitClassifier(args.latent_dim)
    positive = max(float(y[train].sum()), 1.0)
    negative = max(float(len(train) - y[train].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.signals[train]),
            torch.from_numpy(y[train].astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state, best_auc = None, -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch)
            loss_fn(logits, labels).backward()
            optimizer.step()
        probability = _probabilities(model, dataset.signals[validation], args.batch_size)
        metrics = classification_metrics(
            y[validation], (probability >= 0.5).astype(int), probability, probability_score=True
        )
        auc = float(metrics["roc_auc"] or -np.inf)
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0:
            print(f"  epoch {epoch} val_auroc={auc:.4f}", flush=True)
    model.load_state_dict(best_state)

    latent = _encode(model, dataset.signals, args.batch_size)
    standardizer = StandardScaler().fit(latent[train])
    Xtr, Xva, Xte = (standardizer.transform(latent[i]) for i in (train, validation, test))
    angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Xtr)
    Atr, Ava, Ate = (angle.transform(v) for v in (Xtr, Xva, Xte))

    context = build_quantum_context(
        mode="statevector", n_qubits=args.num_qubits, shots=512, seed=args.seed,
        feature_map_reps=1, feature_map_entanglement="linear",
    )
    from qiskit_machine_learning.algorithms import QSVC

    head = QSVC(quantum_kernel=context.kernel, C=5.0)
    head.fit(Atr, y[train])

    # same persisted-normalisation discipline as the imaging builder: decision_function is
    # unbounded, so the [0,1] map is fitted on validation and saved rather than re-derived.
    raw_validation = head.decision_function(Ava)
    score_offset = float(raw_validation.min())
    score_scale = float(max(raw_validation.max() - raw_validation.min(), 1e-9))
    validation_score = np.clip((raw_validation - score_offset) / score_scale, 0.0, 1.0)
    picked = select_threshold(
        y[validation], validation_score, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_score = np.clip((head.decision_function(Ate) - score_offset) / score_scale, 0.0, 1.0)
    test_metrics = classification_metrics(
        y[test], (test_score >= picked["threshold"]).astype(int), test_score, probability_score=True
    )
    print(f"held-out (subject-grouped) balanced_accuracy={test_metrics['balanced_accuracy']:.4f}", flush=True)

    bundle = InferenceBundle(
        schema_version=SERVING_SCHEMA_VERSION,
        model_id="parkinsons-gait-signal",
        condition="Parkinson's disease (gait)",
        temporal_framing="detection",
        positive_label="parkinsons_gait_pattern",
        negative_label="control_gait_pattern",
        channel_names=[f"vgrf_{i+1}" for i in range(dataset.signals.shape[1])],
        input_grid=[int(dataset.signals.shape[2])],
        encoder_state=model.state_dict(),
        encoder_kind="gait_cnn",
        encoder_config={"latent_dim": args.latent_dim, "in_channels": int(dataset.signals.shape[1])},
        head_kind="qsvc",
        head=head,
        standardizer=standardizer,
        reducer=None,
        angle_scaler=angle,
        threshold=float(picked["threshold"]),
        threshold_policy="target_sensitivity@0.8 (validation only)",
        quantum_config={
            "n_qubits": args.num_qubits, "feature_map": "ZZFeatureMap", "feature_map_reps": 1,
            "entanglement": "linear", "C": 5.0, "angle_scale": 0.2, "backend": "statevector",
        },
        training_provenance={
            "dataset": "PhysioNet gaitpdb (Gait in Parkinson's Disease), ODC-By",
            "modality": "signal",
            "cohort_size": int(len(y)),
            "subjects": len(set(dataset.groups.tolist())),
            "positives": int(y.sum()),
            "split": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
            "split_strategy": "subject-grouped (a subject's trials never straddle the split)",
            "held_out_performance": {
                k: test_metrics.get(k)
                for k in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc")
            },
            "sampling_rate_hz": 100.0,
            "limitations": [
                "Research use only, not a medical device",
                "Discriminates DIAGNOSED Parkinson's from controls; NOT prodromal or at-risk detection",
                "Single public cohort (166 subjects), no external-site validation",
                "Input must be 18-channel vertical ground reaction force at 100 Hz, ~2 minute walk",
                "Calibration (reliability/ECE) not assessed",
            ],
        },
        input_stats={
            "mean": float(dataset.signals[train].mean()),
            "std": float(dataset.signals[train].std()),
        },
        score_offset=score_offset,
        score_scale=score_scale,
    )
    path = save_bundle(bundle, args.out)
    print(f"wrote {path}\nwrote {path}.manifest.json", flush=True)


if __name__ == "__main__":
    main()
