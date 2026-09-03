"""Unit tests for metric evaluation and Youden's J threshold optimization."""

import numpy as np
import pytest

from qhealth_qml.pipeline.evaluator import (
    evaluate_model_predictions,
    find_optimal_threshold_youden_j,
)


class TestEvaluatorCorrectness:
    """Test suite for metrics evaluation and threshold optimization."""

    def test_hand_computed_metrics_and_confusion_matrix(self):
        # 10 samples: 4 positives, 6 negatives
        # Predictions: 3 TP, 1 FN, 4 TN, 2 FP
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

        metrics = evaluate_model_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=None,
            train_time="1.5s",
            infer_time="2.0ms",
        )

        cm = metrics.confusion_matrix
        assert cm["tp"] == 3
        assert cm["fn"] == 1
        assert cm["tn"] == 4
        assert cm["fp"] == 2

        assert pytest.approx(metrics.accuracy, rel=1e-3) == 0.70
        assert pytest.approx(metrics.precision, rel=1e-3) == 0.60
        assert pytest.approx(metrics.sensitivity, rel=1e-3) == 0.75
        assert pytest.approx(metrics.specificity, rel=1e-3) == 4 / 6  # 0.6667
        assert pytest.approx(metrics.f1, rel=1e-3) == 2 * (0.6 * 0.75) / (0.6 + 0.75)  # 0.6667
        assert metrics.train_time == "1.5s"
        assert metrics.infer_time == "2.0ms"

    def test_youden_j_threshold_optimization(self):
        # Known probability distribution where optimal cutoff is ~0.4 instead of 0.5
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # Suppose probabilities for positives are [0.45, 0.48, 0.7, 0.9]
        # and negatives are [0.1, 0.15, 0.2, 0.35]
        # A threshold of 0.5 would miss the 2 positives at 0.45 and 0.48.
        # But Youden's J will pick threshold ~0.40 giving 100% TPR and 0% FPR (J = 1.0).
        y_prob = np.array([0.45, 0.48, 0.70, 0.90, 0.10, 0.15, 0.20, 0.35])
        y_pred_flat = (y_prob >= 0.5).astype(int)

        optimal_thresh, roc_points = find_optimal_threshold_youden_j(y_true, y_prob)

        # Optimal threshold should be between 0.35 and 0.45
        assert 0.35 <= optimal_thresh <= 0.46

        metrics = evaluate_model_predictions(
            y_true=y_true,
            y_pred=y_pred_flat,
            y_prob=y_prob,
        )

        # With Youden's J optimized threshold, sensitivity should be 1.0 (4 TP, 0 FN) and specificity 1.0 (4 TN, 0 FP)
        assert metrics.sensitivity == 1.0
        assert metrics.specificity == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0
        assert len(metrics.roc_points) > 0
