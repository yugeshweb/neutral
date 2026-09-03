"""P1 stroke imaging: ISLES 2022 MRI -> 3-D CNN -> classical + quantum heads.

Trains on the sequences a stroke radiologist actually reads (DWI, ADC, FLAIR) instead of
the tabular risk factors the platform's existing P1 arm uses. Label is the DAWN-trial
infarct-core-volume cutoff, derived from the expert lesion mask; the model never sees
the mask.

Uses the cached 64^3 volume array produced during ingest so the raw NIfTI tree can be
deleted -- run with --volumes/--cores pointing at those .npy files.

Two head protocols are reported, both leak-free (heads only ever see the encoder's
held-out test split):
  * decoupled  -- encoder frozen, classical/QSVC heads fitted on its latents
  * end-to-end -- quantum circuit inside the network, trained jointly by backprop
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
from qhealth_qml.imaging_hybrid import (
    VolumeDataset,
    VolumeEncoder,
    encode_volumes,
    seed_everything,
    stratified_split,
    train_volume_encoder,
)
from qhealth_qml.isles_stroke import DAWN_CORE_ML


def build_dataset(
    volumes_path: str,
    cores_path: str,
    rowids_path: str,
    threshold: float,
    binary_labels: bool = False,
) -> VolumeDataset:
    """Load a cached volume array plus labels.

    `cores_path` holds continuous infarct-core volumes for ISLES (thresholded here), or
    already-binary labels when `binary_labels` is set (e.g. BHSD subtype presence).
    """

    volumes = np.load(volumes_path)
    cores = np.load(cores_path)
    row_ids = np.load(rowids_path, allow_pickle=True).astype(str)
    y = cores.astype(int) if binary_labels else (cores >= threshold).astype(int)
    return VolumeDataset(
        volumes=volumes.astype(np.float32),
        y=y,
        groups=row_ids,
        row_ids=row_ids,
        channel_names=["dwi", "adc", "flair"],
        name="isles2022-stroke-core",
        positive_label="large_infarct_core",
        negative_label="small_infarct_core",
        positive_definition=f"expert lesion volume >= {threshold:g} mL (DAWN cutoff)",
        provenance={
            "source": "ISLES 2022 (Zenodo 7153326), CC BY 4.0",
            "raw_imaging": True,
            "core_threshold_ml": threshold,
        },
    )


def fit_head(model, X_train, y_train, X_validation, y_validation, X_test, y_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        validation_score = model.predict_proba(X_validation)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
    else:
        validation_score = model.decision_function(X_validation)
        test_score = model.decision_function(X_test)
        lo, hi = validation_score.min(), validation_score.max()
        span = max(hi - lo, 1e-9)
        validation_score = (validation_score - lo) / span
        test_score = np.clip((test_score - lo) / span, 0.0, 1.0)
    picked = select_threshold(
        y_validation, validation_score, policy="target_sensitivity", target_sensitivity=0.8
    )
    predictions = (test_score >= picked["threshold"]).astype(int)
    return classification_metrics(y_test, predictions, test_score, probability_score=True)


def run_end_to_end(dataset, seed, args):
    seed_everything(seed)
    train, validation, test = stratified_split(dataset.y, seed)
    model = HybridEncoderQNN(
        VolumeEncoder(in_channels=dataset.volumes.shape[1], latent_dim=args.latent_dim),
        latent_dim=args.latent_dim,
        num_qubits=args.latent_dim,
        ansatz_reps=2,
    )
    positive = max(float(dataset.y[train].sum()), 1.0)
    negative = max(float(len(train) - dataset.y[train].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.volumes[train]),
            torch.from_numpy(dataset.y[train].astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    def probabilities(indices):
        model.eval()
        out = []
        with torch.no_grad():
            for (batch,) in DataLoader(
                TensorDataset(torch.from_numpy(dataset.volumes[indices])), batch_size=args.batch_size
            ):
                _, logits = model(batch)
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out)

    best_state, best_auc = None, -np.inf
    for epoch in range(1, args.e2e_epochs + 1):
        model.train()
        started, losses = time.time(), []
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probability = probabilities(validation)
        metrics = classification_metrics(
            dataset.y[validation], (probability >= 0.5).astype(int), probability, probability_score=True
        )
        auc = float(metrics["roc_auc"] or -np.inf)
        print(f"    e2e epoch {epoch:2d} loss={np.mean(losses):.4f} val_auroc={auc:.4f} ({time.time()-started:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    validation_probability = probabilities(validation)
    picked = select_threshold(
        dataset.y[validation], validation_probability, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_probability = probabilities(test)
    return classification_metrics(
        dataset.y[test],
        (test_probability >= picked["threshold"]).astype(int),
        test_probability,
        probability_score=True,
    ), quantum_parameter_count(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--rowids", required=True)
    parser.add_argument("--threshold", type=float, default=DAWN_CORE_ML)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--e2e-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--binary-labels", action="store_true",
                        help="treat --cores as already-binary labels (e.g. BHSD subtype presence)")
    parser.add_argument("--label-name", default="ISLES 2022 (CC BY 4.0)")
    parser.add_argument("--report", default="runtime/p1_isles_hybrid.json")
    args = parser.parse_args()

    dataset = build_dataset(args.volumes, args.cores, args.rowids, args.threshold, args.binary_labels)
    print(
        f"{args.label_name}: cases={len(dataset.y)} channels={dataset.volumes.shape[1]} "
        f"grid={tuple(dataset.volumes.shape[2:])} large_core={int(dataset.y.sum())} "
        f"small_core={int((1-dataset.y).sum())} threshold={args.threshold:g}mL\n",
        flush=True,
    )

    results: dict[str, list[float]] = {}
    quantum_split = None
    for offset in range(args.seeds):
        seed = 7 + offset
        print(f"=== seed {seed} ===", flush=True)
        trained = train_volume_encoder(
            dataset, latent_dim=args.latent_dim, epochs=args.epochs,
            batch_size=args.batch_size, device="auto", seed=seed,
        )
        train_idx, validation_idx, test_idx = (
            trained["train_indices"], trained["validation_indices"], trained["test_indices"]
        )
        latent = encode_volumes(trained["model"], dataset.volumes, trained["device"])
        X_train, X_validation, X_test = latent[train_idx], latent[validation_idx], latent[test_idx]
        y_train, y_validation, y_test = dataset.y[train_idx], dataset.y[validation_idx], dataset.y[test_idx]

        standardizer = StandardScaler().fit(X_train)
        Xc = [standardizer.transform(v) for v in (X_train, X_validation, X_test)]
        angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Xc[0])
        Xq = [angle.transform(v) for v in Xc]

        context = build_quantum_context(
            mode="statevector", n_qubits=args.latent_dim, shots=512, seed=seed,
            feature_map_reps=1, feature_map_entanglement="linear",
        )
        from qiskit_machine_learning.algorithms import QSVC

        heads = {
            "raw_cnn_3d": trained["test_metrics"],
            "logistic_regression": fit_head(LogisticRegression(max_iter=1000, random_state=seed), Xc[0], y_train, Xc[1], y_validation, Xc[2], y_test),
            "rbf_svc": fit_head(SVC(kernel="rbf", probability=True, random_state=seed), Xc[0], y_train, Xc[1], y_validation, Xc[2], y_test),
            "hist_gradient_boosting": fit_head(HistGradientBoostingClassifier(random_state=seed), Xc[0], y_train, Xc[1], y_validation, Xc[2], y_test),
            "qsvc_decoupled": fit_head(QSVC(quantum_kernel=context.kernel, C=5.0), Xq[0], y_train, Xq[1], y_validation, Xq[2], y_test),
        }
        if not args.skip_e2e:
            e2e_metrics, quantum_split = run_end_to_end(dataset, seed, args)
            heads["hybrid_end_to_end"] = e2e_metrics

        for name, metrics in heads.items():
            results.setdefault(name, []).append(float(metrics["balanced_accuracy"]))
            print(f"  {name:24s} balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
        print(flush=True)

    print("=== ISLES 2022 summary (leak-free, independent seeds) ===", flush=True)
    summary = {}
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(args.seeds, 2.776)
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
        "task": ("binary label as provided" if args.binary_labels
                 else f"infarct core >= {args.threshold:g} mL (DAWN)"),
        "cases": int(len(dataset.y)), "positives": int(dataset.y.sum()),
        "quantum_parameters": quantum_split, "summary": summary,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
