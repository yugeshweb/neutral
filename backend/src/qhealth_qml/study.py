"""Nested study and hardware-resource runners built on the core experiment path."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .experiment import (
    LoadedDataset,
    _normalise_models,
    _has_temporal_variation,
    _split_indices,
    dataset_fingerprint,
    package_version,
    run_experiment,
    run_repeated_experiment,
    runtime_manifest,
)


DEFAULT_PARAMETER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "rbf_svc": [
        {"C": 0.1, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
    ],
    "hist_gradient_boosting": [
        {"max_leaf_nodes": 15, "learning_rate": 0.1},
        {"max_leaf_nodes": 31, "learning_rate": 0.1},
        {"max_leaf_nodes": 31, "learning_rate": 0.05},
    ],
    "qsvc": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "pegasos_qsvc": [
        {"C": 0.1, "num_steps": 50},
        {"C": 1.0, "num_steps": 50},
        {"C": 1.0, "num_steps": 200},
    ],
    "vqc": [{"maxiter": 25}, {"maxiter": 50}],
}


def _subset_dataset(dataset: LoadedDataset, indices: np.ndarray, name: str) -> LoadedDataset:
    """Preserve every metadata column while creating an inner-study view."""

    return LoadedDataset(
        name=name,
        X=dataset.X[indices],
        y=dataset.y[indices],
        feature_names=list(dataset.feature_names),
        positive_label=dataset.positive_label,
        negative_label=dataset.negative_label,
        provenance=dict(dataset.provenance),
        groups=dataset.groups[indices] if dataset.groups is not None else None,
        times=dataset.times[indices] if dataset.times is not None else None,
        row_ids=dataset.row_ids[indices] if dataset.row_ids is not None else None,
        sites=dataset.sites[indices] if dataset.sites is not None else None,
        outcome_times=(
            dataset.outcome_times[indices] if dataset.outcome_times is not None else None
        ),
        subgroups={name: values[indices] for name, values in dataset.subgroups.items()},
        task_profile=dict(dataset.task_profile) if dataset.task_profile is not None else None,
    )


def _parameter_grid(
    model_name: str,
    pegasos_steps: int,
    vqc_maxiter: int,
) -> list[dict[str, Any]]:
    if model_name == "pegasos_qsvc":
        return [
            {"C": 0.1, "num_steps": max(1, pegasos_steps)},
            {"C": 1.0, "num_steps": max(1, pegasos_steps)},
            {"C": 1.0, "num_steps": max(1, pegasos_steps * 4)},
        ]
    if model_name == "vqc":
        return [{"maxiter": max(2, vqc_maxiter)}, {"maxiter": max(2, vqc_maxiter * 2)}]
    return [dict(candidate) for candidate in DEFAULT_PARAMETER_GRIDS[model_name]]


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None or not np.isfinite(float(value)):
        value = metrics.get("balanced_accuracy")
    if value is None or not np.isfinite(float(value)):
        value = metrics.get("accuracy")
    return float(value) if value is not None and np.isfinite(float(value)) else -np.inf


def _summarise_metric(values: list[float]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "std": float(np.std(finite)) if finite else None,
        "n": len(finite),
    }


def _paired_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return {
        "mean": float(np.mean(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "lower": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "upper": float(np.percentile(finite, 97.5)) if len(finite) else None,
        "n": int(len(finite)),
    }


def run_nested_evaluation(
    dataset: LoadedDataset,
    models: Iterable[str] = ("classical", "qsvc"),
    outer_repeats: int = 3,
    inner_repeats: int = 2,
    test_size: float = 0.2,
    inner_test_size: float = 0.25,
    seed: int = 7,
    backend: str = "statevector",
    n_qubits: int = 4,
    shots: int = 512,
    aer_noise: str = "none",
    max_train: int = 0,
    max_test: int = 0,
    pegasos_steps: int = 50,
    vqc_maxiter: int = 25,
    reduction: str = "anova",
    validation_size: float | None = None,
    threshold_policy: str = "default",
    target_sensitivity: float | None = None,
    abstain_margin: float | None = None,
    bootstrap_samples: int = 0,
    holdout_site: str | None = None,
    selection_metric: str = "balanced_accuracy",
    tune: bool = True,
    feature_map_reps: int = 1,
    feature_map_entanglement: str = "linear",
    angle_scale: float = 1.0,
) -> dict[str, Any]:
    """Tune on inner repeated holdouts, then score each choice on outer holdouts."""

    if outer_repeats < 1 or inner_repeats < 1:
        raise ValueError("outer_repeats and inner_repeats must be positive")
    if holdout_site is not None:
        if dataset.sites is None:
            raise ValueError("holdout_site requires site metadata")
        outer_repeats = 1
        outer_train = np.flatnonzero(dataset.sites != str(holdout_site))
        outer_test = np.flatnonzero(dataset.sites == str(holdout_site))
        if not len(outer_train) or not len(outer_test):
            raise ValueError("holdout_site must leave rows in both partitions")
        split_pairs = [(outer_train, outer_test)]
    else:
        split_pairs = [
            _split_indices(
                dataset.y,
                test_size,
                seed + fold,
                groups=dataset.groups,
                times=dataset.times,
                split_name=f"outer fold {fold + 1}",
            )
            for fold in range(outer_repeats)
        ]

    model_names = _normalise_models(models)
    folds: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(split_pairs):
        fold_seed = seed + fold
        inner_dataset = _subset_dataset(
            dataset,
            np.asarray(train_index, dtype=int),
            name=f"{dataset.name}:outer-train-{fold + 1}",
        )
        selected_parameters: dict[str, dict[str, Any]] = {}
        search_records: dict[str, list[dict[str, Any]]] = {}
        for model_name in model_names:
            candidates = _parameter_grid(model_name, pegasos_steps, vqc_maxiter) if tune else [{}]
            records: list[dict[str, Any]] = []
            for candidate in candidates:
                inner_result = run_repeated_experiment(
                    inner_dataset,
                    repeats=inner_repeats,
                    models=(model_name,),
                    backend=backend,
                    n_qubits=n_qubits,
                    shots=shots,
                    test_size=inner_test_size,
                    # Every candidate sees the same inner splits; only its
                    # fitted parameters vary.
                    seed=fold_seed * 1000,
                    max_train=max_train,
                    max_test=max_test,
                    aer_noise=aer_noise,
                    pegasos_steps=pegasos_steps,
                    vqc_maxiter=vqc_maxiter,
                    reduction=reduction,
                    explain=False,
                    model_params={model_name: candidate},
                    feature_map_reps=feature_map_reps,
                    feature_map_entanglement=feature_map_entanglement,
                    angle_scale=angle_scale,
                )
                if inner_repeats > 1:
                    summary = inner_result["repeated_evaluation"]["metric_summary"][model_name]
                    score = _metric_value(
                        {name: value["mean"] for name, value in summary.items()},
                        selection_metric,
                    )
                    score_summary = {
                        "mean": score,
                        "std": summary.get(selection_metric, {}).get("std"),
                        "n": summary.get(selection_metric, {}).get("n"),
                    }
                else:
                    score = _metric_value(
                        inner_result["models"][model_name]["metrics"],
                        selection_metric,
                    )
                    score_summary = {"mean": score, "std": None, "n": 1}
                records.append(
                    {
                        "parameters": dict(candidate),
                        "selection_metric": selection_metric,
                        "score": score_summary,
                    }
                )
            chosen = max(
                enumerate(records),
                key=lambda item: (
                    item[1]["score"]["mean"],
                    -item[0],
                ),
            )[1]
            selected_parameters[model_name] = dict(chosen["parameters"])
            search_records[model_name] = records

        outer_result = run_experiment(
            dataset,
            models=model_names,
            backend=backend,
            n_qubits=n_qubits,
            shots=shots,
            test_size=test_size,
            seed=fold_seed,
            max_train=max_train,
            max_test=max_test,
            aer_noise=aer_noise,
            pegasos_steps=pegasos_steps,
            vqc_maxiter=vqc_maxiter,
            reduction=reduction,
            validation_size=validation_size,
            threshold_policy=threshold_policy,
            target_sensitivity=target_sensitivity,
            abstain_margin=abstain_margin,
            bootstrap_samples=bootstrap_samples,
            explain=False,
            holdout_site=holdout_site,
            split_indices=(np.asarray(train_index), np.asarray(test_index)),
            model_params=selected_parameters,
            feature_map_reps=feature_map_reps,
            feature_map_entanglement=feature_map_entanglement,
            angle_scale=angle_scale,
        )
        folds.append(
            {
                "fold": fold + 1,
                "seed": fold_seed,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "selected_parameters": selected_parameters,
                "inner_search": search_records,
                "models": {
                    name: {
                        "parameters": result["parameters"],
                        "metrics": result["metrics"],
                        "clinical_evaluation": result["clinical_evaluation"],
                        "resource": result["resource"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                    for name, result in outer_result["models"].items()
                },
            }
        )

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
    )
    metric_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in model_names:
        metric_summary[model_name] = {}
        for metric_name in metric_names:
            metric_summary[model_name][metric_name] = _summarise_metric(
                [fold["models"][model_name]["metrics"].get(metric_name) for fold in folds]
            )

    reference_model = next(
        (name for name in ("rbf_svc", "logistic_regression", "hist_gradient_boosting") if name in model_names),
        model_names[0],
    )
    paired_deltas: dict[str, Any] = {}
    for model_name in model_names:
        deltas = [
            float(fold["models"][model_name]["metrics"]["balanced_accuracy"])
            - float(fold["models"][reference_model]["metrics"]["balanced_accuracy"])
            for fold in folds
            if fold["models"][model_name]["metrics"].get("balanced_accuracy") is not None
            and fold["models"][reference_model]["metrics"].get("balanced_accuracy") is not None
        ]
        paired_deltas[model_name] = {
            "reference_model": reference_model,
            "metric": "balanced_accuracy",
            "delta": _paired_summary(deltas),
        }

    return {
        "schema_version": 1,
        "package_version": package_version(),
        "software": runtime_manifest(),
        "study": {
            "type": "nested_repeated_holdout",
            "dataset": dataset.name,
            "outer_repeats": len(folds),
            "inner_repeats": inner_repeats,
            "test_size": test_size,
            "inner_test_size": inner_test_size,
            "selection_metric": selection_metric,
            "tuning_enabled": tune,
            "split_strategy": "site_holdout"
            if holdout_site is not None
            else "group_and_chronological"
            if dataset.groups is not None and _has_temporal_variation(dataset.times)
            else "group"
            if dataset.groups is not None
            else "chronological"
            if dataset.times is not None
            else "stratified_random",
            "holdout_site": holdout_site,
            "models": model_names,
            "backend": backend,
            "qubits": n_qubits,
            "shots": shots,
            "max_train": max_train,
            "max_test": max_test,
            "reduction": reduction,
        },
        "dataset": {
            "name": dataset.name,
            "rows": int(len(dataset.y)),
            "features": int(dataset.X.shape[1]),
            "fingerprint": dataset_fingerprint(dataset),
            "positive_label": dataset.positive_label,
            "negative_label": dataset.negative_label,
            "provenance": dataset.provenance,
            "task_profile": dataset.task_profile,
        },
        "folds": folds,
        "metric_summary": metric_summary,
        "paired_deltas": paired_deltas,
    }


def run_resource_sweep(
    dataset: LoadedDataset,
    qubits: Iterable[int] = (2, 4, 6),
    backends: Iterable[str] = ("statevector", "aer"),
    models: Iterable[str] = ("rbf_svc", "qsvc"),
    seed: int = 7,
    shots: int = 512,
    test_size: float = 0.2,
    max_train: int = 80,
    max_test: int = 40,
    aer_noise: str = "none",
    reduction: str = "anova",
    repeats: int = 1,
) -> dict[str, Any]:
    """Compare quality and execution cost as the quantum budget changes."""

    qubit_values = list(dict.fromkeys(int(value) for value in qubits))
    backend_values = list(dict.fromkeys(str(value) for value in backends))
    model_names = _normalise_models(models)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for qubit_count in qubit_values:
        if qubit_count < 1 or qubit_count > dataset.X.shape[1]:
            skipped.append(
                {
                    "qubits": qubit_count,
                    "reason": "qubit count must be between one and the input feature count",
                }
            )
            continue
        for backend in backend_values:
            result = run_repeated_experiment(
                dataset,
                repeats=repeats,
                models=model_names,
                backend=backend,
                n_qubits=qubit_count,
                shots=shots,
                test_size=test_size,
                seed=seed,
                max_train=max_train,
                max_test=max_test,
                aer_noise=aer_noise,
                reduction=reduction,
                explain=False,
            )
            summary = result.get("repeated_evaluation", {}).get("metric_summary", {})
            for model_name in result["models"]:
                metrics = summary.get(model_name, {})
                rows.append(
                    {
                        "backend": backend,
                        "resolved_backend": result["execution"].get("resolved_backend", backend),
                        "qubits": qubit_count,
                        "model": model_name,
                        "metrics": {
                            name: metrics.get(name, {}).get("mean")
                            if metrics
                            else result["models"][model_name]["metrics"].get(name)
                            for name in (
                                "accuracy",
                                "balanced_accuracy",
                                "sensitivity",
                                "specificity",
                                "roc_auc",
                                "pr_auc",
                                "brier_score",
                            )
                        },
                        "resource": result["models"][model_name]["resource"],
                        "elapsed_seconds": (
                            float(np.mean(
                                [
                                    run["models"][model_name]["elapsed_seconds"]
                                    for run in result.get("repeated_evaluation", {}).get("runs", [])
                                ]
                            ))
                            if result.get("repeated_evaluation")
                            else result["models"][model_name]["elapsed_seconds"]
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "package_version": package_version(),
        "software": runtime_manifest(),
        "study": {
            "type": "quantum_resource_sweep",
            "dataset": dataset.name,
            "models": model_names,
            "qubits": qubit_values,
            "backends": backend_values,
            "seed": seed,
            "repeats": repeats,
            "shots": shots,
            "reduction": reduction,
        },
        "dataset": {
            "name": dataset.name,
            "rows": int(len(dataset.y)),
            "features": int(dataset.X.shape[1]),
            "fingerprint": dataset_fingerprint(dataset),
            "positive_label": dataset.positive_label,
            "negative_label": dataset.negative_label,
            "provenance": dataset.provenance,
            "task_profile": dataset.task_profile,
        },
        "rows": rows,
        "skipped": skipped,
    }


def write_study_report(results: dict[str, Any], path: str | Path) -> None:
    """Write a reviewer-friendly HTML summary without a plotting dependency."""

    study = results.get("study", {})
    title = html.escape(str(study.get("type", "quantum health study")).replace("_", " ").title())
    if "metric_summary" in results:
        headers = ["Model", "Balanced accuracy", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC"]
        body = []
        for model, metrics in results["metric_summary"].items():
            cells = [html.escape(model.replace("_", " "))]
            for metric in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc"):
                value = metrics.get(metric, {}).get("mean")
                cells.append("n/a" if value is None else f"{float(value):.3f}")
            body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        table = (
            "<table><thead><tr>"
            + "".join(f"<th>{header}</th>" for header in headers)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table>"
        )
    else:
        table = "<pre>" + html.escape(json.dumps(results, indent=2)) + "</pre>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#14242a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dbe3e5;text-align:left}}th{{font-size:12px;text-transform:uppercase;letter-spacing:.08em}}pre{{white-space:pre-wrap;background:#f2f6f5;padding:20px}}</style>
</head><body><p>Quantum Health research record</p><h1>{title}</h1>{table}</body></html>"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
