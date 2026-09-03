"""Train once, then emit a deployable inference bundle instead of just printing a number.

This is what turns a research run into something another system can call: it refits the chosen
architecture on train+validation, selects the operating threshold on validation only, scores the
held-out test split for the record, and writes an `InferenceBundle` (+ JSON manifest) containing
every fitted transform so nothing is re-derived at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from qhealth_qml.experiment import build_quantum_context, classification_metrics, select_threshold
from qhealth_qml.imaging_hybrid import seed_everything, stratified_split
from qhealth_qml.pretrained_encoder import PretrainedVolumeClassifier, trainable_parameter_report
from qhealth_qml.serving import SERVING_SCHEMA_VERSION, InferenceBundle, save_bundle


def _probabilities(model, volumes, indices, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(volumes[indices])), batch_size=batch_size):
            _, logits = model(batch)
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def _encode(model, volumes, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(volumes)), batch_size=batch_size):
            out.append(model.encoder(batch).numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--temporal-framing", required=True,
                        choices=["prediction", "detection", "characterisation", "screening"])
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--negative-label", required=True)
    parser.add_argument("--channels", required=True, help="comma-separated channel names")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--modality", default="MR")
    parser.add_argument("--ct-windows", default="", help="JSON list of {level,width} for CT")
    parser.add_argument("--limitations", default="", help="pipe-separated limitation strings")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    volumes = np.load(args.volumes).astype(np.float32)
    y = np.load(args.labels).astype(int)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    print(f"{args.dataset_name}: n={len(y)} channels={volumes.shape[1]} grid={tuple(volumes.shape[2:])}", flush=True)

    seed_everything(args.seed)
    train, validation, test = stratified_split(y, args.seed)

    model = PretrainedVolumeClassifier(
        in_channels=volumes.shape[1], latent_dim=args.latent_dim, slice_stride=2
    )
    positive = max(float(y[train].sum()), 1.0)
    negative = max(float(len(train) - y[train].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(volumes[train]), torch.from_numpy(y[train].astype(np.float32))),
        batch_size=args.batch_size, shuffle=True,
    )
    best_state, best_auc = None, -np.inf
    import copy
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch)
            loss_fn(logits, labels).backward()
            optimizer.step()
        probability = _probabilities(model, volumes, validation, args.batch_size)
        metrics = classification_metrics(
            y[validation], (probability >= 0.5).astype(int), probability, probability_score=True
        )
        auc = float(metrics["roc_auc"] or -np.inf)
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0:
            print(f"  epoch {epoch} val_auroc={auc:.4f}", flush=True)
    model.load_state_dict(best_state)

    latent = _encode(model, volumes, args.batch_size)
    standardizer = StandardScaler().fit(latent[train])
    Xtr, Xva, Xte = (standardizer.transform(latent[i]) for i in (train, validation, test))
    reducer = PCA(n_components=min(args.num_qubits, Xtr.shape[1])).fit(Xtr)
    Qtr, Qva, Qte = (reducer.transform(v) for v in (Xtr, Xva, Xte))
    angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Qtr)
    Atr, Ava, Ate = (angle.transform(v) for v in (Qtr, Qva, Qte))

    context = build_quantum_context(
        mode="statevector", n_qubits=args.num_qubits, shots=512, seed=args.seed,
        feature_map_reps=1, feature_map_entanglement="linear",
    )
    from qiskit_machine_learning.algorithms import QSVC

    head = QSVC(quantum_kernel=context.kernel, C=5.0)
    head.fit(Atr, y[train])

    # QSVC without probability=True exposes only decision_function, whose range is unbounded
    # and often negative. Clipping that to [0,1] collapses every negative score to zero and
    # destroys the ranking -- which silently turns a working model into an all-positive
    # predictor. Fit a min-max map on VALIDATION and persist it so inference replays it.
    raw_validation = head.decision_function(Ava)
    score_offset = float(raw_validation.min())
    score_scale = float(max(raw_validation.max() - raw_validation.min(), 1e-9))
    validation_score = np.clip((raw_validation - score_offset) / score_scale, 0.0, 1.0)
    picked = select_threshold(
        y[validation], validation_score, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_score = np.clip((head.decision_function(Ate) - score_offset) / score_scale, 0.0, 1.0)
    test_metrics = classification_metrics(
        y[test], (test_score >= picked["threshold"]).astype(int),
        test_score, probability_score=True,
    )
    print(f"held-out test balanced_accuracy={test_metrics['balanced_accuracy']:.4f}", flush=True)

    bundle = InferenceBundle(
        schema_version=SERVING_SCHEMA_VERSION,
        model_id=args.model_id,
        condition=args.condition,
        temporal_framing=args.temporal_framing,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        channel_names=channels,
        input_grid=[int(v) for v in volumes.shape[2:]],
        encoder_state=model.state_dict(),
        encoder_kind="pretrained_volume",
        encoder_config={
            "in_channels": int(volumes.shape[1]),
            "latent_dim": args.latent_dim,
            "slice_stride": 2,
            "backbone": "frozen ImageNet ResNet18, slice-wise",
        },
        head_kind="qsvc",
        head=head,
        standardizer=standardizer,
        reducer=reducer,
        angle_scaler=angle,
        threshold=float(picked["threshold"]),
        threshold_policy="target_sensitivity@0.8 (validation only)",
        score_offset=score_offset,
        score_scale=score_scale,
        quantum_config={
            "n_qubits": args.num_qubits, "feature_map": "ZZFeatureMap",
            "feature_map_reps": 1, "entanglement": "linear", "C": 5.0,
            "angle_scale": 0.2, "backend": "statevector",
        },
        training_provenance={
            "dataset": args.dataset_name,
            "cohort_size": int(len(y)),
            "positives": int(y.sum()),
            "split": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
            "held_out_performance": {k: test_metrics.get(k) for k in
                                     ("balanced_accuracy", "sensitivity", "specificity", "roc_auc")},
            "parameters": trainable_parameter_report(model),
            "modality": args.modality,
            "ct_windows": json.loads(args.ct_windows) if args.ct_windows else None,
            "limitations": [s for s in args.limitations.split("|") if s],
        },
        input_stats={"mean": float(volumes[train].mean()), "std": float(volumes[train].std())},
    )
    path = save_bundle(bundle, args.out)
    print(f"wrote {path}\nwrote {path}.manifest.json", flush=True)


if __name__ == "__main__":
    main()
