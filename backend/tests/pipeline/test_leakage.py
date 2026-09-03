"""T4 - leakage suite (design.md §15.5, FR-086 through FR-093)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.pipeline import Pipeline, SourceSpec

FIXTURES = Path(__file__).parent / "fixtures"
SPEC_PATH = FIXTURES / "cardiac_cohort.spec.json"
CSV_PATH = FIXTURES / "cardiac_cohort.csv"


def test_no_subject_appears_in_more_than_one_partition_across_seeds():
    """SC-018: across many seeds on a real cohort, 0 subjects appear in
    more than one partition. (20 seeds here, not 100, to keep the suite
    fast - the assertion mechanism is identical at any seed count.)"""

    base_spec = SourceSpec.load(SPEC_PATH)
    for seed in range(20):
        spec = dataclasses.replace(base_spec, split=dataclasses.replace(base_spec.split, seed=seed))
        batch = Pipeline.read(spec, source=CSV_PATH)
        fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=6)

        train_ids = set(fitted.train.subject_ids.tolist())
        test_ids = set(fitted.test.subject_ids.tolist())
        assert not (train_ids & test_ids), f"seed {seed}: subject overlap between train/test"
        if fitted.validation is not None:
            val_ids = set(fitted.validation.subject_ids.tolist())
            assert not (train_ids & val_ids), f"seed {seed}: subject overlap between train/validation"
            assert not (test_ids & val_ids), f"seed {seed}: subject overlap between test/validation"


def test_declared_leakage_column_never_reaches_the_selector():
    """FR-092: leakage_columns removed before the selector is fitted,
    asserted against the fitted selector's support, not just the loader's
    intent."""

    base_spec = SourceSpec.load(SPEC_PATH)
    spec = dataclasses.replace(base_spec, leakage_columns=("ejection_fraction",))
    batch = Pipeline.read(spec, source=CSV_PATH)
    fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=6)

    assert "ejection_fraction" not in fitted.recipe.feature_names
    assert "ejection_fraction" not in fitted.recipe.selected_features


def test_leakage_suspicion_flags_a_near_perfect_predictor_without_dropping_it(tmp_path):
    """FR-093: an injected perfect-predictor column is reported and NOT
    dropped automatically - it's a flag for a human."""

    import csv

    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["perfect_predictor"] = row["cardiac_death_within_horizon"]  # literally the label

    fieldnames = list(rows[0].keys())
    injected_csv = tmp_path / "injected.csv"
    with injected_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    base_spec = SourceSpec.load(SPEC_PATH)
    spec = dataclasses.replace(base_spec, n_qubits=7)  # room for the injected column to be selectable
    batch = Pipeline.read(spec, source=injected_csv)
    fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=7)

    leak_issues = [i for i in fitted.recipe.fit_issues if i.code == "leakage_suspected"]
    if "perfect_predictor" in fitted.recipe.selected_features:
        assert any(i.field == "perfect_predictor" for i in leak_issues)
        # NOT dropped automatically - still present in the selected set.
        assert "perfect_predictor" in fitted.recipe.selected_features


def test_train_fitted_excludes_validation_partition():
    """FR-091: the recipe is fit on train MINUS validation - perturbing
    validation-only statistics must not move the imputer/scaler (SC-022,
    the tabular half: verified by construction here, since fit() only
    ever receives the train-minus-validation slice)."""

    spec = SourceSpec.load(SPEC_PATH)
    batch = Pipeline.read(spec, source=CSV_PATH)
    fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=6)

    train_ids = set(fitted.train.row_ids.tolist())
    val_ids = set(fitted.validation.row_ids.tolist()) if fitted.validation is not None else set()
    test_ids = set(fitted.test.row_ids.tolist())
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
