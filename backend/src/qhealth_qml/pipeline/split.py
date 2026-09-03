"""Splitting strategies and the assertions the splitter always makes
(design.md §11.1-§11.2, FR-086 through FR-092).

Three of the five declared strategies are implemented in this pass -
`grouped`, `chronological`, `grouped_chronological` - covering every
profile currently in `backend/profiles/`. `site_holdout` and
`predeclared_folds` are declared in `spec.py`'s `SplitSpec` but raise
`NotImplementedError` here rather than silently falling back to a
different strategy; a spec that names one gets a clear, typed refusal
until it's built, not a wrong split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Issue, IssueCode, Sample


class SplitError(Exception):
    """A split could not be produced honestly - e.g. one class is entirely
    absent from a partition after every retry. Never silently returned."""


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray | None
    test: np.ndarray
    issues: list[Issue]


def split_batch(
    samples: list[Sample],
    y: np.ndarray,
    *,
    strategy: str,
    test_size: float,
    validation_size: float,
    seed: int,
    max_attempts: int = 50,
) -> SplitIndices:
    """Dispatches to the declared strategy. `y` excludes nothing - callers
    pass the tri-state label array; `LABEL_EXCLUDE` rows are still split
    (so they can be counted and dropped later) but never count toward the
    both-classes-present assertion."""

    if strategy == "grouped":
        return _split_grouped(samples, y, test_size=test_size, validation_size=validation_size, seed=seed, max_attempts=max_attempts)
    if strategy == "chronological":
        return _split_chronological(samples, y, test_size=test_size, validation_size=validation_size)
    if strategy == "grouped_chronological":
        return _split_grouped_chronological(samples, y, test_size=test_size, validation_size=validation_size, seed=seed, max_attempts=max_attempts)
    if strategy in ("site_holdout", "predeclared_folds"):
        raise NotImplementedError(
            f"split strategy {strategy!r} is declared in spec.py's SplitSpec but not yet "
            f"implemented in this pipeline build (Phase 1 tabular slice implements grouped, "
            f"chronological and grouped_chronological only)."
        )
    raise SplitError(f"unknown split strategy: {strategy!r}")


def _both_classes_present(y: np.ndarray, indices: np.ndarray, *, from_label: int = 0) -> bool:
    scored = y[indices]
    scored = scored[scored != -1]  # LABEL_EXCLUDE never counts
    return len(np.unique(scored)) >= 2


def _split_grouped(
    samples: list[Sample], y: np.ndarray, *, test_size: float, validation_size: float, seed: int, max_attempts: int,
) -> SplitIndices:
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([s.subject_id or s.sample_id for s in samples])
    n = len(samples)
    indices = np.arange(n)

    for attempt in range(max_attempts):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_val_idx, test_idx = next(gss.split(indices, y, groups))
        if not _both_classes_present(y, test_idx) or not _both_classes_present(y, train_val_idx):
            continue

        if validation_size > 0:
            inner_groups = groups[train_val_idx]
            inner_y = y[train_val_idx]
            rel_val_size = validation_size / (1 - test_size)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed + attempt)
            train_idx_rel, val_idx_rel = next(gss_val.split(train_val_idx, inner_y, inner_groups))
            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]
            if not _both_classes_present(y, train_idx) or not _both_classes_present(y, val_idx):
                continue
        else:
            train_idx = train_val_idx
            val_idx = None

        _assert_subject_disjoint(samples, train_idx, val_idx, test_idx)
        return SplitIndices(train=train_idx, validation=val_idx, test=test_idx, issues=[])

    raise SplitError(
        f"grouped split could not produce both classes in every partition after "
        f"{max_attempts} attempts - the cohort may be too small or too imbalanced "
        f"for the declared test_size/validation_size."
    )


def _split_chronological(
    samples: list[Sample], y: np.ndarray, *, test_size: float, validation_size: float,
) -> SplitIndices:
    times = [s.index_time for s in samples]
    if len(set(times)) <= 1:
        raise SplitError(
            "chronological split requires index_time to vary across records; every "
            "record shares one timestamp (or none) - declare 'grouped' instead."
        )
    order = np.argsort([t or "" for t in times])
    n = len(samples)
    n_test = max(1, round(n * test_size))
    n_val = max(1, round(n * validation_size)) if validation_size > 0 else 0

    test_idx = order[n - n_test:]
    val_idx = order[n - n_test - n_val: n - n_test] if n_val else None
    train_idx = order[: n - n_test - n_val]

    issues: list[Issue] = []
    if not _both_classes_present(y, test_idx) or not _both_classes_present(y, train_idx):
        issues.append(
            Issue(
                "chronological_split_class_imbalance",
                "warn",
                "A chronological cut left one partition without both classes; "
                "consider 'grouped_chronological' or a different cut fraction.",
            )
        )
    return SplitIndices(train=train_idx, validation=val_idx, test=test_idx, issues=issues)


def _split_grouped_chronological(
    samples: list[Sample], y: np.ndarray, *, test_size: float, validation_size: float, seed: int, max_attempts: int,
) -> SplitIndices:
    """Chronological cut, scanning cutoffs until group sets are disjoint
    AND both classes survive on both sides. Degrades to `grouped` with an
    info issue when the timestamp is constant (no ordering to cut on) -
    design.md §11.1, kept verbatim from existing behaviour."""

    times = [s.index_time for s in samples]
    if len(set(times)) <= 1:
        result = _split_grouped(samples, y, test_size=test_size, validation_size=validation_size, seed=seed, max_attempts=max_attempts)
        return SplitIndices(
            train=result.train,
            validation=result.validation,
            test=result.test,
            issues=[
                Issue(
                    "constant_timestamp_degraded_to_grouped",
                    "info",
                    "index_time is constant across records; grouped_chronological degraded "
                    "to a plain grouped split rather than an uninformative chronological cut.",
                )
            ],
        )

    groups = np.array([s.subject_id or s.sample_id for s in samples])
    order = np.argsort([t or "" for t in times])
    n = len(samples)

    for shift in range(max_attempts):
        n_test = max(1, round(n * test_size)) + shift
        if n_test >= n:
            break
        test_idx = order[n - n_test:]
        train_val_idx = order[: n - n_test]

        test_groups = set(groups[test_idx])
        train_val_groups = set(groups[train_val_idx])
        if test_groups & train_val_groups:
            continue  # a subject straddles the cut; widen and retry
        if not _both_classes_present(y, test_idx) or not _both_classes_present(y, train_val_idx):
            continue

        if validation_size > 0:
            n_val = max(1, round(n * validation_size))
            val_idx = train_val_idx[-n_val:]
            train_idx = train_val_idx[:-n_val]
            if set(groups[val_idx]) & set(groups[train_idx]):
                continue
            if not _both_classes_present(y, train_idx) or not _both_classes_present(y, val_idx):
                continue
        else:
            train_idx = train_val_idx
            val_idx = None

        _assert_subject_disjoint(samples, train_idx, val_idx, test_idx)
        return SplitIndices(train=train_idx, validation=val_idx, test=test_idx, issues=[])

    raise SplitError(
        "grouped_chronological split could not find a cutoff with disjoint subject "
        "groups and both classes present on every side."
    )


def _assert_subject_disjoint(
    samples: list[Sample], train_idx: np.ndarray, val_idx: np.ndarray | None, test_idx: np.ndarray
) -> None:
    """FR-088: no subject appears in more than one partition. Raises, not
    just warns - a split that fails this is not usable, ever."""

    def subjects(idx: np.ndarray) -> set[str]:
        return {samples[i].subject_id or samples[i].sample_id for i in idx}

    train_s, test_s = subjects(train_idx), subjects(test_idx)
    if train_s & test_s:
        raise SplitError(f"subject(s) {sorted(train_s & test_s)} appear in both train and test")
    if val_idx is not None:
        val_s = subjects(val_idx)
        if train_s & val_s:
            raise SplitError(f"subject(s) {sorted(train_s & val_s)} appear in both train and validation")
        if test_s & val_s:
            raise SplitError(f"subject(s) {sorted(test_s & val_s)} appear in both test and validation")


def assert_window_cohesion(samples: list[Sample], train_idx: np.ndarray, val_idx: np.ndarray | None, test_idx: np.ndarray) -> None:
    """FR-089: no two windows of one recording fall in different
    partitions. A window's parent recording id is expected under
    `sample.provenance["parent_recording_id"]`; samples without one
    (the common tabular case) are exempt."""

    def parents(idx: np.ndarray) -> set[str]:
        return {
            samples[i].provenance.get("parent_recording_id")
            for i in idx
            if samples[i].provenance.get("parent_recording_id")
        }

    train_p, val_p, test_p = parents(train_idx), parents(val_idx if val_idx is not None else np.array([], dtype=int)), parents(test_idx)
    overlap = (train_p & test_p) | (train_p & val_p) | (val_p & test_p)
    if overlap:
        raise SplitError(f"recording(s) {sorted(overlap)} have windows split across partitions")
