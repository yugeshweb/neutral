"""Benchmark loader and signed delta computation against reference metrics."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from qhealth_qml.pipeline.evaluator import ModelMetrics
from qhealth_qml.pipeline.exceptions import BenchmarkNotFoundError

_DEFAULT_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks"


def load_benchmark_reference(
    disease_id: str,
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load pre-computed static benchmark reference JSON for a disease."""
    base_dir = benchmark_dir or _DEFAULT_BENCHMARK_DIR
    bench_file = base_dir / f"{disease_id}.json"

    if not bench_file.exists():
        raise BenchmarkNotFoundError(
            message=f"Benchmark reference file not found at {bench_file}",
            disease_id=disease_id,
        )

    with open(bench_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metric_deltas(
    run_metrics: ModelMetrics,
    benchmark_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute signed deltas (run - benchmark) for all common scalar metrics."""
    deltas = {}
    metric_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]

    for key in metric_keys:
        run_val = getattr(run_metrics, key, None)
        # Handle variations like "roc_auc" vs "rocAuc"
        bench_val = benchmark_metrics.get(key)
        if bench_val is None and key == "roc_auc":
            bench_val = benchmark_metrics.get("rocAuc")

        if run_val is not None and bench_val is not None:
            delta = float(run_val) - float(bench_val)
            deltas[f"delta_{key}"] = round(delta, 4)

    return deltas


def compare_with_benchmarks(
    disease_id: str,
    evaluated_models: Dict[str, ModelMetrics],
    benchmark_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare evaluation metrics against stored benchmarks for all model types."""
    try:
        benchmark_data = load_benchmark_reference(disease_id, benchmark_dir=benchmark_dir)
    except BenchmarkNotFoundError:
        return {
            "status": "not_available",
            "message": f"No benchmark reference found for disease '{disease_id}'",
            "models": {},
        }

    bench_models = benchmark_data.get("models", {})
    comparison_results: Dict[str, Any] = {}

    for model_type, run_metric in evaluated_models.items():
        bench_model_data = bench_models.get(model_type, {})
        bench_metric_data = bench_model_data.get("metrics", {})
        
        deltas = compute_metric_deltas(run_metric, bench_metric_data)
        comparison_results[model_type] = {
            "current_metrics": run_metric.to_dict(),
            "benchmark_metrics": bench_metric_data,
            "deltas": deltas,
            "benchmark_model_id": bench_model_data.get("id", "unknown"),
            "benchmark_model_name": bench_model_data.get("name", "unknown"),
        }

    return {
        "status": "success",
        "disease_id": disease_id,
        "models": comparison_results,
    }
