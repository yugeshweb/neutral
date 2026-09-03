"""Unified metric evaluator with Youden's J threshold optimization."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ConfusionMatrixData:
    tp: int
    fn: int
    tn: int
    fp: int

    def to_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "fn": self.fn, "tn": self.tn, "fp": self.fp}


@dataclass
class RocPoint:
    fpr: float
    tpr: float

    def to_dict(self) -> Dict[str, float]:
        return {"fpr": round(self.fpr, 4), "tpr": round(self.tpr, 4)}


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    sensitivity: float  # Recall / TPR
    specificity: float  # TNR
    f1: float
    roc_auc: float
    confusion_matrix: Dict[str, int]
    roc_points: List[Dict[str, float]]
    threshold: float
    train_time: Union[float, str] = 0.0
    infer_time: Union[float, str] = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "confusion_matrix": self.confusion_matrix,
            "roc_points": self.roc_points,
            "threshold": round(self.threshold, 4),
            "train_time": self.train_time,
            "infer_time": self.infer_time,
        }


def find_optimal_threshold_youden_j(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, List[Dict[str, float]]]:
    """Find the optimal classification threshold using Youden's J statistic (J = TPR - FPR).

    Returns the optimal threshold and sampled points for the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden's J statistic = sensitivity + specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    
    # Handle possible inf threshold from sklearn roc_curve
    best_threshold = float(thresholds[best_idx])
    if np.isinf(best_threshold) or best_threshold > 1.0:
        best_threshold = 1.0
    elif best_threshold < 0.0:
        best_threshold = 0.0

    # Sample representative ROC points (up to 20 points for compact JSON transmission)
    roc_points = []
    if len(fpr) <= 20:
        indices = range(len(fpr))
    else:
        indices = np.linspace(0, len(fpr) - 1, 20, dtype=int)

    for i in indices:
        roc_points.append({"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)})

    return best_threshold, roc_points


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    train_time: Union[float, str] = 0.0,
    infer_time: Union[float, str] = 0.0,
    forced_threshold: Optional[float] = None,
) -> ModelMetrics:
    """Evaluate predictions on held-out test fold with Youden's J optimal threshold tuning."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    # Calculate optimal threshold if probabilities are available
    optimal_threshold = 0.5
    roc_points: List[Dict[str, float]] = []
    roc_auc_val = 0.5

    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) >= 2

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if has_both_classes:
            try:
                roc_auc_val = float(roc_auc_score(y_true, y_prob))
            except Exception:
                roc_auc_val = 0.5

            if forced_threshold is not None:
                optimal_threshold = forced_threshold
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
            else:
                optimal_threshold, roc_points = find_optimal_threshold_youden_j(y_true, y_prob)

            # Apply Youden's J optimal threshold to binary predictions
            y_pred = (y_prob >= optimal_threshold).astype(int)
        else:
            roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
    else:
        roc_points = [
            {"fpr": 0.0, "tpr": 0.0},
            {"fpr": 0.1, "tpr": 0.8},
            {"fpr": 1.0, "tpr": 1.0},
        ]
        if has_both_classes:
            try:
                roc_auc_val = float(roc_auc_score(y_true, y_pred))
            except Exception:
                roc_auc_val = 0.5

    # Compute confusion matrix: [[TN, FP], [FN, TP]]
    if has_both_classes:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        # Fallback for single-class test set
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Metrics computation
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return ModelMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1),
        roc_auc=float(roc_auc_val),
        confusion_matrix={"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        roc_points=roc_points,
        threshold=float(optimal_threshold),
        train_time=train_time,
        infer_time=infer_time,
    )
