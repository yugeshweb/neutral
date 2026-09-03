"""P4 seizure PREDICTION (not detection): leave-one-patient-out pre-ictal classification.

Evaluation is **leave-one-patient-out (LOPO)**, not a random window split, and that choice is the
whole point. A subject's own pre-ictal physiology is highly self-similar, so a random split places
near-duplicate windows from the same patient on both sides of the boundary and reports a number
that collapses on any new patient. Published CHB-MIT results above ~95% are almost always this;
patient-independent state of the art sits around **AUC 0.81**, which is the honest bar here.

Each fold trains on all other patients and tests on the held-out one, so every reported number is
performance on a patient the model has never seen.
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

from qhealth_qml.experiment import build_quantum_context, classification_metrics, select_threshold


def fit_and_score(model, Xtr, ytr, Xva, yva, Xte, yte):
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        va, te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    else:
        va, te = model.decision_function(Xva), model.decision_function(Xte)
        lo, span = va.min(), max(va.max() - va.min(), 1e-9)
        va, te = (va - lo) / span, np.clip((te - lo) / span, 0, 1)
    picked = select_threshold(yva, va, policy="target_sensitivity", target_sensitivity=0.8)
    return classification_metrics(
        yte, (te >= picked["threshold"]).astype(int), te, probability_score=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/unichronic/neutral_data/p4_eeg")
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--report", default="runtime/p4_preictal_lopo.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    groups = np.load(root / "groups.npy", allow_pickle=True).astype(str)
    patients = sorted(set(groups.tolist()))
    print(
        f"pre-ictal dataset: {len(y)} windows, {int(y.sum())} pre-ictal / {int((1-y).sum())} "
        f"interictal, {len(patients)} patients {patients}\n",
        flush=True,
    )

    results: dict[str, list[float]] = {}
    aucs: dict[str, list[float]] = {}
    for held_out in patients:
        test_mask = groups == held_out
        pool = ~test_mask
        pool_idx = np.flatnonzero(pool)
        # carve a validation patient out of the training pool so the threshold is never
        # selected on the held-out patient
        other = [p for p in patients if p != held_out]
        validation_patient = other[0]
        va_mask = groups == validation_patient
        tr_mask = pool & ~va_mask
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue

        standardizer = StandardScaler().fit(X[tr_mask])
        Xtr, Xva, Xte = (standardizer.transform(X[m]) for m in (tr_mask, va_mask, test_mask))
        ytr, yva, yte = y[tr_mask], y[va_mask], y[test_mask]

        angle = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Xtr)
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=min(args.num_qubits, Xtr.shape[1])).fit(Xtr)
        Qtr, Qva, Qte = (reducer.transform(v) for v in (Xtr, Xva, Xte))
        angle_q = MinMaxScaler(feature_range=(-np.pi / 2 * 0.2, np.pi / 2 * 0.2)).fit(Qtr)
        Atr, Ava, Ate = (angle_q.transform(v) for v in (Qtr, Qva, Qte))

        context = build_quantum_context(
            mode="statevector", n_qubits=args.num_qubits, shots=512, seed=7,
            feature_map_reps=1, feature_map_entanglement="linear",
        )
        from qiskit_machine_learning.algorithms import QSVC

        heads = {
            "logistic_regression": (LogisticRegression(max_iter=1000), Xtr, Xva, Xte),
            "rbf_svc": (SVC(kernel="rbf", probability=True, random_state=7), Xtr, Xva, Xte),
            "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=7), Xtr, Xva, Xte),
            "qsvc": (QSVC(quantum_kernel=context.kernel, C=5.0), Atr, Ava, Ate),
        }
        print(f"=== held-out patient {held_out} (test n={int(test_mask.sum())}) ===", flush=True)
        for name, (model, a, b, c) in heads.items():
            metrics = fit_and_score(model, a, ytr, b, yva, c, yte)
            ba = float(metrics["balanced_accuracy"])
            auc = float(metrics["roc_auc"] or np.nan)
            results.setdefault(name, []).append(ba)
            aucs.setdefault(name, []).append(auc)
            print(f"  {name:24s} BA={ba:.4f}  AUC={auc:.4f}", flush=True)
        print(flush=True)

    print("=== LOPO summary (every number is a patient the model never saw) ===", flush=True)
    summary = {}
    for name in sorted(results):
        ba = np.asarray(results[name])
        auc = np.asarray([v for v in aucs[name] if np.isfinite(v)])
        summary[name] = {
            "balanced_accuracy_mean": float(ba.mean()),
            "balanced_accuracy_std": float(ba.std()),
            "auc_mean": float(auc.mean()) if len(auc) else None,
            "per_patient_ba": results[name],
        }
        print(
            f"{name:24s} BA={ba.mean():.4f}±{ba.std():.4f}   "
            f"AUC={auc.mean():.4f}" if len(auc) else f"{name:24s} BA={ba.mean():.4f}±{ba.std():.4f}",
            flush=True,
        )
    print("\nliterature reference: patient-independent CHB-MIT SOTA ≈ AUC 0.81;", flush=True)
    print("anything >0.95 in the literature is near-certainly window-wise split leakage.", flush=True)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": "pre-ictal prediction (early detection)",
                "protocol": "leave-one-patient-out",
                "preictal_window": "35-5 min before onset (5 min seizure prediction horizon)",
                "windows": int(len(y)),
                "patients": patients,
                "summary": summary,
                "literature_reference_auc": 0.81,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
