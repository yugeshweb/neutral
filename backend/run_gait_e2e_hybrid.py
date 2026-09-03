"""End-to-end hybrid on P6 gait: CNN encoder + trainable quantum layer, jointly backpropped.

Direct comparison against the decoupled result already recorded for this condition
(encoder trained separately, then a QSVC fitted on frozen latents: 0.7501 +/- 0.0396).
Here the quantum circuit sits inside the network and the encoder is shaped by gradients
that pass through it -- the architecture Mari et al. use for quantum + medical imaging.

Protocol is the same leak-free one used for the decoupled comparison: subject-grouped
split, checkpoint chosen on validation AUROC, operating threshold chosen on validation
only, scored once on the held-out test subjects, repeated over independent seeds.
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
from torch.utils.data import DataLoader, TensorDataset

from qhealth_qml.experiment import classification_metrics, select_threshold
from qhealth_qml.gait_hybrid import (
    RawGaitEncoder,
    load_raw_gaitpdb,
    seed_everything,
    split_indices_grouped,
)
from qhealth_qml.hybrid_qnn import HybridEncoderQNN, quantum_parameter_count


def probabilities(model, signals, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(signals)), batch_size=batch_size)
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            _, logits = model(batch)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def run_seed(dataset, seed, args):
    seed_everything(seed)
    train, validation, test = split_indices_grouped(dataset, seed)
    model = HybridEncoderQNN(
        RawGaitEncoder(args.latent_dim),
        latent_dim=args.latent_dim,
        num_qubits=args.num_qubits,
        ansatz_reps=args.ansatz_reps,
        angle_scale=args.angle_scale,
    )
    split = quantum_parameter_count(model)

    positive = max(float(dataset.y[train].sum()), 1.0)
    negative = max(float(len(train) - dataset.y[train].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.signals[train]),
            torch.from_numpy(dataset.y[train].astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state, best_auc = None, -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        losses = []
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_probability = probabilities(model, dataset.signals[validation], args.batch_size)
        metrics = classification_metrics(
            dataset.y[validation],
            (validation_probability >= 0.5).astype(int),
            validation_probability,
            probability_score=True,
        )
        auc = float(metrics["roc_auc"] or -np.inf)
        print(
            f"  epoch {epoch:2d} loss={np.mean(losses):.4f} val_auroc={auc:.4f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    validation_probability = probabilities(model, dataset.signals[validation], args.batch_size)
    threshold = select_threshold(
        dataset.y[validation], validation_probability, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_probability = probabilities(model, dataset.signals[test], args.batch_size)
    test_metrics = classification_metrics(
        dataset.y[test],
        (test_probability >= threshold["threshold"]).astype(int),
        test_probability,
        probability_score=True,
    )
    return test_metrics, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--ansatz-reps", type=int, default=2)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--report", default="runtime/p6_gait_e2e_hybrid.json")
    args = parser.parse_args()

    dataset = load_raw_gaitpdb(args.data_root, target_samples=12000)
    print(
        f"recordings={dataset.signals.shape[0]} subjects={len(set(dataset.groups.tolist()))} "
        f"positives={int(dataset.y.sum())}\n",
        flush=True,
    )

    scores, per_seed, split_info = [], {}, None
    for offset in range(args.seeds):
        seed = 7 + offset
        print(f"=== seed {seed} (end-to-end hybrid) ===", flush=True)
        metrics, split_info = run_seed(dataset, seed, args)
        scores.append(float(metrics["balanced_accuracy"]))
        per_seed[str(seed)] = metrics
        print(f"  -> test balanced_accuracy={metrics['balanced_accuracy']:.4f}\n", flush=True)

    array = np.asarray(scores)
    mean, std = float(array.mean()), float(array.std())
    half = (2.776 if len(array) == 5 else 4.303) * std / np.sqrt(len(array)) if len(array) > 1 else 0.0
    print("=== end-to-end hybrid summary ===", flush=True)
    print(f"parameters: {split_info}", flush=True)
    print(
        f"balanced_accuracy mean={mean:.4f} std={std:.4f} n={len(array)} "
        f"approx95CI=[{mean - half:.4f}, {mean + half:.4f}]",
        flush=True,
    )
    print(f"per-seed: {[round(s, 4) for s in scores]}", flush=True)
    print("\nreference (decoupled, 5 seeds): qsvc 0.7501 +/- 0.0396, rbf_svc 0.7693 +/- 0.0369", flush=True)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "architecture": "end-to-end hybrid (CNN encoder + TorchConnector dressed quantum circuit)",
                "parameters": split_info,
                "config": vars(args),
                "summary": {"mean": mean, "std": std, "n": len(array), "per_seed_balanced_accuracy": scores},
                "per_seed_metrics": per_seed,
                "decoupled_reference": {"qsvc": [0.7501, 0.0396], "rbf_svc": [0.7693, 0.0369]},
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
