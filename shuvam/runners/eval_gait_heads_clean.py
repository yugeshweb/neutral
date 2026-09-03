"""Leak-free head evaluation for the P6 gait hybrid.

The first run trained the CNN encoder on a subject-grouped train split and then
evaluated the classical/quantum heads over latents for ALL recordings -- including
the 196 the encoder was trained on. That inflates the head scores, because those
latents were shaped by labels the encoder had already seen.

This script respects the encoder's split boundary: heads are fit on the encoder's
TRAIN latents, their operating threshold is chosen on the encoder's VALIDATION
latents, and they are scored only on the encoder's held-out TEST latents -- the
same 59 recordings the raw-CNN metric was reported on, none of which the encoder
ever trained on.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qhealth_qml.experiment import (
    build_quantum_context,
    classification_metrics,
    bootstrap_confidence_intervals,
    select_threshold,
)
from qhealth_qml.gait_hybrid import encode_raw_gait, load_raw_gaitpdb, train_raw_gait_encoder


def fit_score(model, X_train, y_train, X_validation, y_validation, X_test, y_test, target_sensitivity=0.8):
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
        y_validation, validation_score, policy="target_sensitivity", target_sensitivity=target_sensitivity
    )
    predictions = (test_score >= picked["threshold"]).astype(int)
    return classification_metrics(y_test, predictions, test_score, probability_score=True), test_score, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5, help="independent encoder+head runs")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--report", default="runtime/p6_gait_clean_report.json")
    args = parser.parse_args()

    dataset = load_raw_gaitpdb(args.data_root, target_samples=12000)
    print(
        f"recordings={dataset.signals.shape[0]} subjects={len(set(dataset.groups.tolist()))} "
        f"positives={int(dataset.y.sum())} negatives={int((1 - dataset.y).sum())}\n",
        flush=True,
    )

    per_seed: dict[str, list[float]] = {}
    last_run: dict[str, object] = {}
    for offset in range(args.seeds):
        seed = 7 + offset
        print(f"=== seed {seed}: training encoder ===", flush=True)
        trained = train_raw_gait_encoder(
            dataset,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device="auto",
            seed=seed,
        )
        train_idx = trained["train_indices"]
        validation_idx = trained["validation_indices"]
        test_idx = trained["test_indices"]

        latent = encode_raw_gait(trained["model"], dataset.signals, trained["device"])
        X_train, y_train = latent[train_idx], dataset.y[train_idx]
        X_validation, y_validation = latent[validation_idx], dataset.y[validation_idx]
        X_test, y_test = latent[test_idx], dataset.y[test_idx]

        standardizer = StandardScaler().fit(X_train)
        X_train_c, X_validation_c, X_test_c = (
            standardizer.transform(X_train),
            standardizer.transform(X_validation),
            standardizer.transform(X_test),
        )
        angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(X_train_c)
        X_train_q, X_validation_q, X_test_q = (
            angle.transform(X_train_c),
            angle.transform(X_validation_c),
            angle.transform(X_test_c),
        )

        context = build_quantum_context(
            mode="statevector", n_qubits=args.latent_dim, shots=512, seed=seed,
            feature_map_reps=1, feature_map_entanglement="linear",
        )
        from qiskit_machine_learning.algorithms import QSVC

        candidates = {
            "raw_cnn": None,
            "logistic_regression": (LogisticRegression(max_iter=1000, random_state=seed), False),
            "rbf_svc": (SVC(kernel="rbf", probability=True, random_state=seed), False),
            "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=seed), False),
            "qsvc": (QSVC(quantum_kernel=context.kernel, C=5.0), True),
        }

        for name, spec in candidates.items():
            if name == "raw_cnn":
                metrics = trained["test_metrics"]
            else:
                model, quantum = spec  # type: ignore[misc]
                if quantum:
                    metrics, _, _ = fit_score(
                        model, X_train_q, y_train, X_validation_q, y_validation, X_test_q, y_test
                    )
                else:
                    metrics, _, _ = fit_score(
                        model, X_train_c, y_train, X_validation_c, y_validation, X_test_c, y_test
                    )
            per_seed.setdefault(name, []).append(float(metrics["balanced_accuracy"]))
            print(f"  {name:24s} balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
        last_run = {
            "seed": seed,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "test_rows": int(len(test_idx)),
        }
        print(flush=True)

    print("=== leak-free summary across independent encoder+head runs ===", flush=True)
    summary = {}
    for name, values in sorted(per_seed.items()):
        array = np.asarray(values, dtype=float)
        mean, std = float(array.mean()), float(array.std())
        half = 2.776 * std / np.sqrt(len(array)) if len(array) > 1 else 0.0  # t(4), 95%
        summary[name] = {
            "mean": mean, "std": std, "n": int(len(array)),
            "approx_ci_lower": mean - half, "approx_ci_upper": mean + half,
            "per_seed": values,
        }
        print(
            f"{name:24s} mean={mean:.4f} std={std:.4f} n={len(array)} "
            f"approx95CI=[{mean - half:.4f}, {mean + half:.4f}]",
            flush=True,
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"protocol": "encoder-split-respecting (leak-free)", "last_run": last_run, "summary": summary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
