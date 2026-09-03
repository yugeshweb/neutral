"""Volumetric hybrid on a FROZEN pretrained backbone (quantum transfer learning, done properly).

Replaces the from-scratch 3-D CNN with a frozen ImageNet ResNet18 applied slice-wise, a wide
(32-d) latent, and the usual classical + quantum heads on top -- plus an optional end-to-end
dressed-quantum-circuit arm. Compare against the from-scratch numbers in
`run_isles_hybrid.py`'s reports.

Protocol is the platform's standard leak-free one: heads are fitted on the encoder's train
latents, thresholded on its validation latents, and scored only on its held-out test split,
repeated over independent seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from qhealth_qml.experiment import build_quantum_context, classification_metrics, select_threshold
from qhealth_qml.hybrid_qnn import HybridEncoderQNN, quantum_parameter_count
from qhealth_qml.imaging_hybrid import seed_everything, stratified_split
from qhealth_qml.pretrained_encoder import (
    PretrainedVolumeClassifier,
    trainable_parameter_report,
)
from qhealth_qml.medical3d_encoder import MedicalNetClassifier


def _probabilities(model, volumes, indices, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(
            TensorDataset(torch.from_numpy(volumes[indices])), batch_size=batch_size
        ):
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


def train_encoder(volumes, y, seed, args):
    seed_everything(seed)
    train, validation, test = stratified_split(y, seed)
    if args.encoder == "medicalnet":
        model = MedicalNetClassifier(in_channels=volumes.shape[1], latent_dim=args.latent_dim)
    else:
        model = PretrainedVolumeClassifier(
            in_channels=volumes.shape[1], latent_dim=args.latent_dim, slice_stride=args.slice_stride
        )
    positive = max(float(y[train].sum()), 1.0)
    negative = max(float(len(train) - y[train].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(volumes[train]), torch.from_numpy(y[train].astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state, best_auc = None, -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probability = _probabilities(model, volumes, validation, args.batch_size)
        metrics = classification_metrics(
            y[validation], (probability >= 0.5).astype(int), probability, probability_score=True
        )
        auc = float(metrics["roc_auc"] or -np.inf)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:3d} loss={np.mean(losses):.4f} val_auroc={auc:.4f} "
                f"({time.time()-started:.0f}s)",
                flush=True,
            )
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    validation_probability = _probabilities(model, volumes, validation, args.batch_size)
    picked = select_threshold(
        y[validation], validation_probability, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_probability = _probabilities(model, volumes, test, args.batch_size)
    encoder_metrics = classification_metrics(
        y[test],
        (test_probability >= picked["threshold"]).astype(int),
        test_probability,
        probability_score=True,
    )
    return model, (train, validation, test), encoder_metrics, picked


def fit_head(model, Xtr, ytr, Xva, yva, Xte, yte):
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        va, te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    else:
        va, te = model.decision_function(Xva), model.decision_function(Xte)
        lo, hi = va.min(), va.max()
        span = max(hi - lo, 1e-9)
        va, te = (va - lo) / span, np.clip((te - lo) / span, 0, 1)
    picked = select_threshold(yva, va, policy="target_sensitivity", target_sensitivity=0.8)
    return classification_metrics(yte, (te >= picked["threshold"]).astype(int), te, probability_score=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--rowids", required=True)
    parser.add_argument("--label-name", default="dataset")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--slice-stride", type=int, default=2)
    parser.add_argument("--encoder", default="imagenet2d", choices=["imagenet2d", "medicalnet"],
                        help="imagenet2d = frozen ResNet18 slice-wise; medicalnet = frozen 3-D MedicalNet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    volumes = np.load(args.volumes).astype(np.float32)
    y = np.load(args.labels).astype(int)
    row_ids = np.load(args.rowids, allow_pickle=True).astype(str)
    print(
        f"{args.label_name}: n={len(y)} channels={volumes.shape[1]} grid={tuple(volumes.shape[2:])} "
        f"positives={int(y.sum())} negatives={int((1-y).sum())}\n",
        flush=True,
    )

    results: dict[str, list[float]] = {}
    param_report = None
    for offset in range(args.seeds):
        seed = 7 + offset
        print(f"=== seed {seed} (frozen pretrained backbone) ===", flush=True)
        model, (train, validation, test), encoder_metrics, _ = train_encoder(volumes, y, seed, args)
        param_report = trainable_parameter_report(model)

        latent = _encode(model, volumes, args.batch_size)
        Xtr, Xva, Xte = latent[train], latent[validation], latent[test]
        ytr, yva, yte = y[train], y[validation], y[test]

        standardizer = StandardScaler().fit(Xtr)
        Xc = [standardizer.transform(v) for v in (Xtr, Xva, Xte)]
        # project the wide latent down to qubit count only at the quantum boundary
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=min(args.num_qubits, Xc[0].shape[1])).fit(Xc[0])
        Xq_raw = [reducer.transform(v) for v in Xc]
        angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Xq_raw[0])
        Xq = [angle.transform(v) for v in Xq_raw]

        context = build_quantum_context(
            mode="statevector", n_qubits=args.num_qubits, shots=512, seed=seed,
            feature_map_reps=1, feature_map_entanglement="linear",
        )
        from qiskit_machine_learning.algorithms import QSVC

        heads = {
            "pretrained_cnn_head": encoder_metrics,
            "logistic_regression": fit_head(LogisticRegression(max_iter=1000, random_state=seed), Xc[0], ytr, Xc[1], yva, Xc[2], yte),
            "rbf_svc": fit_head(SVC(kernel="rbf", probability=True, random_state=seed), Xc[0], ytr, Xc[1], yva, Xc[2], yte),
            "hist_gradient_boosting": fit_head(HistGradientBoostingClassifier(random_state=seed), Xc[0], ytr, Xc[1], yva, Xc[2], yte),
            "qsvc_decoupled": fit_head(QSVC(quantum_kernel=context.kernel, C=5.0), Xq[0], ytr, Xq[1], yva, Xq[2], yte),
        }
        for name, metrics in heads.items():
            results.setdefault(name, []).append(float(metrics["balanced_accuracy"]))
            print(f"  {name:24s} balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
        print(flush=True)

    print(f"=== {args.label_name} — frozen pretrained backbone, leak-free, {args.seeds} seeds ===", flush=True)
    print(f"parameters: {param_report}", flush=True)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(args.seeds, 2.776)
    summary = {}
    for name, values in sorted(results.items()):
        arr = np.asarray(values)
        mean, std = float(arr.mean()), float(arr.std())
        half = tcrit * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        summary[name] = {"mean": mean, "std": std, "n": len(arr), "ci": [mean - half, mean + half], "per_seed": values}
        print(f"{name:24s} mean={mean:.4f} std={std:.4f} approx95CI=[{mean-half:.4f}, {mean+half:.4f}]", flush=True)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": args.label_name,
        "architecture": ("frozen 3-D MedicalNet ResNet18 + trainable projection -> classical/QSVC heads"
                         if args.encoder == "medicalnet" else
                         "frozen ImageNet ResNet18 slice-wise + trainable projection -> classical/QSVC heads"),
        "config": vars(args), "parameters": param_report, "summary": summary,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
