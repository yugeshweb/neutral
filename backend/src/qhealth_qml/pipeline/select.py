"""Feature selection and dimensionality reduction to a qubit budget
(design.md §9.5, §11.3, §11.5). Implements FR-096 through FR-101, FR-093,
FR-098, FR-099.

Two selection paths:
- **Plain** (`select_features`): one global pass, same as today's
  `experiment.PreprocessingPipeline` - correct when `n_features` is
  comfortably below `n_samples`.
- **Stability-checked** (`select_features_stable`): fits the selector
  within each grouped fold's training part only and keeps features
  selected in at least `min_selected_in` folds (FR-098) - the lever that
  matters when a cohort has more features than patients (design.md §11.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GroupKFold

from .types import Issue, IssueCode


class SelectionError(Exception):
    """A source has fewer features than the qubit budget (FR-097) - refused,
    never silently padded."""


@dataclass
class SelectionResult:
    selector: object | None  # SelectKBest, fitted; None for PCA reduction
    pca: PCA | None
    selected_indices: np.ndarray
    selected_features: list[str]
    selection_frequency: dict[str, float] | None  # stability mode only
    issues: list[Issue]


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_qubits: int,
    reduction: str,
) -> SelectionResult:
    """One global pass. FR-096, FR-097, FR-101."""

    n_samples, n_features = X.shape
    if n_features < n_qubits:
        raise SelectionError(
            f"source has {n_features} features but n_qubits={n_qubits} were requested; "
            f"refusing rather than padding with fabricated components."
        )

    issues: list[Issue] = []
    if n_features > n_samples:
        issues.append(
            Issue(
                "feature_count_exceeds_sample_count",
                "warn",
                f"{n_features} features exceeds {n_samples} samples; a single global "
                f"selection pass is prone to fitting noise - consider selection.stability.enabled.",
            )
        )

    k = min(n_qubits, n_features)
    if reduction == "pca":
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(X)
        selected_indices = np.arange(k)  # components, not original feature indices
        selected_features = [f"PC{i + 1}" for i in range(k)]
        return SelectionResult(None, pca, selected_indices, selected_features, None, issues)

    if reduction == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=0)
        selected_indices = np.argsort(scores)[::-1][:k]
        selected_indices.sort()
        selected_features = [feature_names[i] for i in selected_indices]
        return SelectionResult(None, None, selected_indices, selected_features, None, issues)

    # default: anova
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    return SelectionResult(selector, None, selected_indices, selected_features, None, issues)


def select_features_stable(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    groups: np.ndarray,
    *,
    n_qubits: int,
    reduction: str,
    folds: int,
    min_selected_in: int,
) -> SelectionResult:
    """FR-098, FR-099: selection run inside each grouped fold's training
    part only; a feature's `selection_frequency` is how many of `folds` it
    was picked in. The final selected set is every feature picked in at
    least `min_selected_in` folds, trimmed to `n_qubits` by frequency then
    by the full-data score as a tiebreak."""

    n_samples, n_features = X.shape
    if n_features < n_qubits:
        raise SelectionError(
            f"source has {n_features} features but n_qubits={n_qubits} were requested."
        )

    n_unique_groups = len(np.unique(groups))
    effective_folds = min(folds, n_unique_groups)
    if effective_folds < 2:
        # Too few subjects to run stability folds meaningfully - fall back
        # to the plain pass rather than raising, with the reason recorded.
        result = select_features(X, y, feature_names, n_qubits=n_qubits, reduction=reduction)
        result.issues.append(
            Issue(
                "stability_selection_too_few_groups",
                "warn",
                f"only {n_unique_groups} subject group(s) available; stability-checked "
                f"selection needs at least 2 and fell back to a single global pass.",
            )
        )
        return result

    gkf = GroupKFold(n_splits=effective_folds)
    counts = np.zeros(n_features, dtype=int)
    for train_idx, _ in gkf.split(X, y, groups):
        fold_result = select_features(
            X[train_idx], y[train_idx], feature_names, n_qubits=n_qubits, reduction="anova"
        )
        counts[fold_result.selected_indices] += 1

    selection_frequency = {feature_names[i]: counts[i] / effective_folds for i in range(n_features)}

    stable_mask = counts >= min_selected_in
    stable_indices = np.where(stable_mask)[0]

    if len(stable_indices) == 0:
        raise SelectionError(
            f"no feature was selected in at least {min_selected_in}/{effective_folds} folds; "
            f"the cohort's signal did not survive stability checking with these settings."
        )

    # Trim to n_qubits by frequency (desc), then by full-data ANOVA score.
    if len(stable_indices) > n_qubits:
        full_scores = f_classif(X[:, stable_indices], y)[0]
        order = np.lexsort((-np.nan_to_num(full_scores), -counts[stable_indices]))
        stable_indices = stable_indices[order][:n_qubits]
        stable_indices.sort()

    selected_features = [feature_names[i] for i in stable_indices]
    return SelectionResult(None, None, stable_indices, selected_features, selection_frequency, [])


def leakage_suspicion_check(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], *, threshold: float = 0.98
) -> list[Issue]:
    """FR-093: report, never auto-drop, any single feature whose univariate
    train AUROC exceeds `threshold`. A flag for a human - a genuinely
    near-perfect biomarker exists, and this pipeline does not get to
    decide that question by itself (design.md §11.3)."""

    from sklearn.metrics import roc_auc_score

    issues: list[Issue] = []
    if len(np.unique(y)) < 2:
        return issues
    for i, name in enumerate(feature_names):
        col = X[:, i]
        finite = np.isfinite(col)
        if finite.sum() < 2 or len(np.unique(y[finite])) < 2:
            continue
        try:
            auroc = roc_auc_score(y[finite], col[finite])
        except ValueError:
            continue
        auroc = max(auroc, 1 - auroc)  # direction-agnostic
        if auroc > threshold:
            issues.append(
                Issue(
                    IssueCode.LEAKAGE_SUSPECTED,
                    "warn",
                    f"Feature '{name}' alone reaches train AUROC {auroc:.3f} (> {threshold}), "
                    f"which is unusual for a genuine predictor - review before trusting it.",
                    field=name,
                    detail={"auroc": auroc},
                )
            )
    return issues
