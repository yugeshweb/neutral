"""End-to-end P6 Parkinson's gait hybrid run: raw force-plate signal -> CNN encoder ->
compact latent -> classical and quantum heads, evaluated with the platform's usual rigor.

This replaces the platform's prior P6 representation (22 hand-computed voice features)
with the modality actually closest to the clinical motor exam -- 18-channel vertical
ground-reaction-force recordings from PhysioNet gaitpdb -- following the session's
"richer input, not more hand-engineering" finding.

Usage (from backend/):
    ./.venv/bin/python run_gait_hybrid.py --data-root ~/neutral_data/p6_gait/gait-in-parkinsons-disease-1.0.0
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

from qhealth_qml import experiment
from qhealth_qml.gait_hybrid import (
    encode_raw_gait,
    load_raw_gaitpdb,
    train_raw_gait_encoder,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--report", default="runtime/p6_gait_hybrid_report.json")
    args = parser.parse_args()

    print("=== loading raw gait recordings ===", flush=True)
    dataset = load_raw_gaitpdb(args.data_root, target_samples=args.target_samples)
    print(
        f"recordings={dataset.signals.shape[0]} channels={dataset.signals.shape[1]} "
        f"samples={dataset.signals.shape[2]} subjects={len(set(dataset.groups.tolist()))} "
        f"positives={int(dataset.y.sum())} negatives={int((1 - dataset.y).sum())}",
        flush=True,
    )

    print("\n=== training raw CNN encoder (subject-grouped split) ===", flush=True)
    trained = train_raw_gait_encoder(
        dataset,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(
        f"train={trained['train_rows']} validation={trained['validation_rows']} test={trained['test_rows']}",
        flush=True,
    )
    print(f"raw-CNN test metrics: {json.dumps(trained['test_metrics'], indent=2)}", flush=True)

    print("\n=== encoding all recordings to latent vectors ===", flush=True)
    latent = encode_raw_gait(trained["model"], dataset.signals, trained["device"])
    print(f"latent matrix {latent.shape}", flush=True)

    loaded = dataset.as_loaded_dataset(latent)

    print(
        "\n=== classical + quantum heads on the learned latent "
        f"(repeated evaluation, repeats={args.repeats}, subject-grouped) ===",
        flush=True,
    )
    result = experiment.run_repeated_experiment(
        loaded,
        repeats=args.repeats,
        models=["classical", "qsvc"],
        n_qubits=args.latent_dim,
        reduction="anova",
        seed=args.seed,
        max_train=0,
        max_test=0,
        validation_size=0.2,
        threshold_policy="target_sensitivity",
        target_sensitivity=0.8,
        bootstrap_samples=0,
        explain=False,
    )
    summary = result["repeated_evaluation"]["metric_summary"]
    for name in sorted(summary):
        balanced = summary[name]["balanced_accuracy"]
        print(
            f"{name:24s} balanced_accuracy mean={balanced['mean']:.4f} "
            f"std={balanced['std']:.4f} n={balanced['n']}",
            flush=True,
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": dataset.name,
                    "recordings": int(dataset.signals.shape[0]),
                    "channels": int(dataset.signals.shape[1]),
                    "samples_per_recording": int(dataset.signals.shape[2]),
                    "subjects": len(set(dataset.groups.tolist())),
                    "positives": int(dataset.y.sum()),
                    "negatives": int((1 - dataset.y).sum()),
                },
                "encoder": {
                    "latent_dim": args.latent_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "device": trained["device"],
                    "train_rows": trained["train_rows"],
                    "validation_rows": trained["validation_rows"],
                    "test_rows": trained["test_rows"],
                    "history": trained["history"],
                    "threshold": trained["threshold"],
                    "validation_metrics": trained["validation_metrics"],
                    "test_metrics": trained["test_metrics"],
                },
                "heads_repeated_evaluation": summary,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
