"""Tests for benchmark loader and signed delta computation."""

import pytest
from qhealth_qml.pipeline.benchmarks import (
    compare_with_benchmarks,
    compute_metric_deltas,
    load_benchmark_reference,
)
from qhealth_qml.pipeline.evaluator import ModelMetrics
from qhealth_qml.pipeline.exceptions import BenchmarkNotFoundError


class TestBenchmarks:
    """Test suite for benchmark loading and signed delta calculations."""

    @pytest.mark.parametrize("disease_id", ["heart-disease", "breast-cancer", "alzheimers", "glioma", "stroke", "seizure", "parkinsons"])
    def test_benchmark_handling_across_all_seven_diseases(self, disease_id):
        dummy_run = ModelMetrics(
            accuracy=0.90,
            precision=0.88,
            sensitivity=0.85,
            specificity=0.92,
            f1=0.86,
            roc_auc=0.94,
            confusion_matrix={"tp": 10, "fn": 2, "tn": 20, "fp": 2},
            roc_points=[],
            threshold=0.5,
        )

        comparison = compare_with_benchmarks(
            disease_id=disease_id,
            evaluated_models={"classical": dummy_run},
        )
        # Returns success if static benchmark exists, or not_available if pending separate addition
        assert comparison["status"] in {"success", "not_available"}

    def test_signed_delta_calculation(self):
        # Create dummy ModelMetrics with accuracy = 0.90, sensitivity = 0.85
        dummy_run = ModelMetrics(
            accuracy=0.90,
            precision=0.88,
            sensitivity=0.85,
            specificity=0.92,
            f1=0.86,
            roc_auc=0.94,
            confusion_matrix={"tp": 10, "fn": 2, "tn": 20, "fp": 2},
            roc_points=[],
            threshold=0.5,
        )

        benchmark_metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "sensitivity": 0.90,  # lower in run -> negative delta
            "specificity": 0.90,
            "f1": 0.84,
            "roc_auc": 0.91,
        }

        deltas = compute_metric_deltas(dummy_run, benchmark_metrics)

        assert pytest.approx(deltas["delta_accuracy"], rel=1e-3) == +0.05
        assert pytest.approx(deltas["delta_precision"], rel=1e-3) == +0.08
        assert pytest.approx(deltas["delta_sensitivity"], rel=1e-3) == -0.05
        assert pytest.approx(deltas["delta_specificity"], rel=1e-3) == +0.02
        assert pytest.approx(deltas["delta_f1"], rel=1e-3) == +0.02
        assert pytest.approx(deltas["delta_roc_auc"], rel=1e-3) == +0.03

    def test_compare_with_benchmarks_end_to_end(self):
        dummy_run_c = ModelMetrics(
            accuracy=0.97,
            precision=0.98,
            sensitivity=0.95,
            specificity=0.98,
            f1=0.96,
            roc_auc=0.99,
            confusion_matrix={"tp": 30, "fn": 1, "tn": 50, "fp": 1},
            roc_points=[],
            threshold=0.5,
        )

        comparison = compare_with_benchmarks(
            disease_id="breast-cancer",
            evaluated_models={"classical": dummy_run_c},
        )

        assert comparison["status"] == "success"
        assert "classical" in comparison["models"]
        c_res = comparison["models"]["classical"]
        assert "deltas" in c_res
        assert "delta_accuracy" in c_res["deltas"]
