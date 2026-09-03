"""T3 - metamorphic tests, the definitive tier (design.md §15.4). These are
the tests that fail against pre-Phase-1 code and must pass after it -
T3.4 (train ≡ predict) especially is the reason this phase exists.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.pipeline import Pipeline, Recipe, SourceSpec
from qhealth_qml.pipeline.types import Batch

FIXTURES = Path(__file__).parent / "fixtures"
SPEC_PATH = FIXTURES / "cardiac_cohort.spec.json"
CSV_PATH = FIXTURES / "cardiac_cohort.csv"


@pytest.fixture(scope="module")
def fitted():
    spec = SourceSpec.load(SPEC_PATH)
    batch = Pipeline.read(spec, source=CSV_PATH)
    return Pipeline.from_spec(spec).fit(batch, n_qubits=6)


def _csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _predict_csv_text(rows: list[dict[str, str]], *, drop_target: bool = True, header_order: list[str] | None = None) -> str:
    fieldnames = header_order or list(rows[0].keys())
    if drop_target:
        fieldnames = [f for f in fieldnames if f != "cardiac_death_within_horizon"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in fieldnames})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# T3.4 - train ≡ predict. The single test that proves the central claim.
# ---------------------------------------------------------------------------


def test_train_equals_predict_bit_identical(fitted):
    recipe = fitted.recipe
    rows = _csv_rows()

    # Pick a row that actually made it into the TRAIN partition (not
    # validation/test) - identifiable by row_id in fitted.train.
    train_ids = set(fitted.train.row_ids.tolist())
    target_row = next(r for r in rows if r["patient_id"] in train_ids)
    train_row_index = list(fitted.train.row_ids).index(target_row["patient_id"])
    expected_classical = fitted.train.X_classical[train_row_index]
    expected_quantum = fitted.train.X_quantum[train_row_index]

    predict_csv = _predict_csv_text([target_row])
    predict_batch = Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8"))
    run = Pipeline.from_recipe(recipe).run(predict_batch)

    assert run.arrays.X_classical.shape[0] == 1
    np.testing.assert_array_equal(run.arrays.X_classical[0], expected_classical)
    np.testing.assert_array_equal(run.arrays.X_quantum[0], expected_quantum)


def test_train_equals_predict_for_every_train_row(fitted):
    """Not just one row - every row the training path actually produced a
    vector for, predict-time re-encoding must reproduce exactly (SC-001)."""

    recipe = fitted.recipe
    rows = {r["patient_id"]: r for r in _csv_rows()}
    train_ids = list(fitted.train.row_ids)

    sample_rows = [rows[rid] for rid in train_ids]
    predict_csv = _predict_csv_text(sample_rows)
    predict_batch = Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8"))
    run = Pipeline.from_recipe(recipe).run(predict_batch)

    # Re-associate by row_id, not position - the contract explicitly does
    # not promise the run preserves input order beyond what verdicts give.
    predict_index = {rid: i for i, rid in enumerate(run.arrays.row_ids)}
    for i, rid in enumerate(train_ids):
        j = predict_index[rid]
        np.testing.assert_array_equal(run.arrays.X_classical[j], fitted.train.X_classical[i])


# ---------------------------------------------------------------------------
# T3.1 - single ≡ batch
# ---------------------------------------------------------------------------


def test_single_record_equals_same_record_in_a_batch(fitted):
    recipe = fitted.recipe
    rows = _csv_rows()[:5]

    solo_csv = _predict_csv_text([rows[2]])
    solo_batch = Pipeline.read(recipe.spec, source=solo_csv.encode("utf-8"))
    solo_run = Pipeline.from_recipe(recipe).run(solo_batch)

    group_csv = _predict_csv_text(rows)
    group_batch = Pipeline.read(recipe.spec, source=group_csv.encode("utf-8"))
    group_run = Pipeline.from_recipe(recipe).run(group_batch)

    group_index = list(group_run.arrays.row_ids).index(rows[2]["patient_id"])
    np.testing.assert_array_equal(solo_run.arrays.X_classical[0], group_run.arrays.X_classical[group_index])


def test_unrelated_records_added_to_batch_change_nothing(fitted):
    """SC-003: adding unrelated records to a batch changes 0 existing
    records' output."""

    recipe = fitted.recipe
    rows = _csv_rows()
    small = rows[:3]
    bigger = rows[:3] + rows[10:20]

    run_small = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=_predict_csv_text(small).encode("utf-8")))
    run_bigger = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=_predict_csv_text(bigger).encode("utf-8")))

    small_by_id = {rid: run_small.arrays.X_classical[i] for i, rid in enumerate(run_small.arrays.row_ids)}
    bigger_by_id = {rid: run_bigger.arrays.X_classical[i] for i, rid in enumerate(run_bigger.arrays.row_ids)}
    for rid, vec in small_by_id.items():
        np.testing.assert_array_equal(vec, bigger_by_id[rid])


# ---------------------------------------------------------------------------
# T3.3 - column-order permutation invariance
# ---------------------------------------------------------------------------


def test_reordered_columns_produce_identical_output(fitted):
    """SC-009: a source whose columns are reordered produces output
    bit-identical to the same source in training order (FR-021)."""

    recipe = fitted.recipe
    rows = _csv_rows()[:5]
    original_order = list(rows[0].keys())
    original_order = [c for c in original_order if c != "cardiac_death_within_horizon"]
    shuffled_order = list(reversed(original_order))

    run_a = Pipeline.from_recipe(recipe).run(
        Pipeline.read(recipe.spec, source=_predict_csv_text(rows, header_order=original_order).encode("utf-8"))
    )
    run_b = Pipeline.from_recipe(recipe).run(
        Pipeline.read(recipe.spec, source=_predict_csv_text(rows, header_order=shuffled_order).encode("utf-8"))
    )

    a_by_id = {rid: run_a.arrays.X_classical[i] for i, rid in enumerate(run_a.arrays.row_ids)}
    b_by_id = {rid: run_b.arrays.X_classical[i] for i, rid in enumerate(run_b.arrays.row_ids)}
    for rid in a_by_id:
        np.testing.assert_array_equal(a_by_id[rid], b_by_id[rid])


# ---------------------------------------------------------------------------
# T3.6 - chunk-size invariance
# ---------------------------------------------------------------------------


def test_chunk_size_does_not_change_output(fitted):
    """SC-004: scoring a source in chunks of 1, 7 and all-at-once produces
    identical matrices (re-associated by row_id)."""

    recipe = fitted.recipe
    rows = _csv_rows()[:14]

    def run_in_chunks(chunk_size: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=_predict_csv_text(chunk).encode("utf-8")))
            for i, rid in enumerate(run.arrays.row_ids):
                result[rid] = run.arrays.X_classical[i]
        return result

    all_at_once = run_in_chunks(len(rows))
    chunks_of_7 = run_in_chunks(7)
    chunks_of_1 = run_in_chunks(1)

    for rid in all_at_once:
        np.testing.assert_array_equal(all_at_once[rid], chunks_of_7[rid])
        np.testing.assert_array_equal(all_at_once[rid], chunks_of_1[rid])


# ---------------------------------------------------------------------------
# T3.5 - save / load round-trip
# ---------------------------------------------------------------------------


def test_recipe_save_load_round_trip_bit_identical(fitted, tmp_path):
    recipe = fitted.recipe
    saved_path = recipe.save(tmp_path / "test.recipe.pkl")
    reloaded = Recipe.load(saved_path)

    rows = _csv_rows()[:5]
    predict_csv = _predict_csv_text(rows)

    run_original = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8")))
    run_reloaded = Pipeline.from_recipe(reloaded).run(Pipeline.read(reloaded.spec, source=predict_csv.encode("utf-8")))

    np.testing.assert_array_equal(run_original.arrays.X_classical, run_reloaded.arrays.X_classical)
    np.testing.assert_array_equal(run_original.arrays.X_quantum, run_reloaded.arrays.X_quantum)
    assert reloaded.fingerprint() == recipe.fingerprint()


def test_two_fits_same_seed_produce_identical_fingerprint():
    """SC-025: identical input, spec and seed, in separate fit() calls,
    produce identical arrays and an identical pipeline fingerprint."""

    spec = SourceSpec.load(SPEC_PATH)
    batch1 = Pipeline.read(spec, source=CSV_PATH)
    batch2 = Pipeline.read(spec, source=CSV_PATH)

    fit1 = Pipeline.from_spec(spec).fit(batch1, n_qubits=6)
    fit2 = Pipeline.from_spec(spec).fit(batch2, n_qubits=6)

    assert fit1.recipe.fingerprint() == fit2.recipe.fingerprint()
    np.testing.assert_array_equal(fit1.train.X_classical, fit2.train.X_classical)


# ---------------------------------------------------------------------------
# fit_transform == fit then transform (FR-009)
# ---------------------------------------------------------------------------


def test_fit_transform_equals_fit_then_transform():
    """FR-009: fit_transform(batch, y) == fit(batch, y) followed by a
    separate transform(batch) call. Both recipes here are fit on the exact
    same batch - a Recipe fit on a different row subset (e.g. the
    train-minus-validation partition Pipeline.fit() uses internally) can
    legitimately make different quarantine/constant-column decisions, so
    that comparison does not belong in this test; it would conflate
    FR-009 (fit_transform determinism) with split-dependent fitting."""

    spec = SourceSpec.load(SPEC_PATH)
    batch = Pipeline.read(spec, source=CSV_PATH)

    from qhealth_qml.pipeline.recipe import Recipe as RecipeClass, RECIPE_SCHEMA_VERSION
    from qhealth_qml.pipeline.pipeline import _code_version

    y = np.array([1 if s.label == 1 else 0 for s in batch.samples])

    recipe_b = RecipeClass(schema_version=RECIPE_SCHEMA_VERSION, spec=spec, code_version=_code_version(), fitted_at="", n_qubits=6)
    arrays_b, _ = recipe_b.fit_transform(batch, y)

    recipe_c = RecipeClass(schema_version=RECIPE_SCHEMA_VERSION, spec=spec, code_version=_code_version(), fitted_at="", n_qubits=6)
    recipe_c.fit(batch, y)
    arrays_c, _ = recipe_c.transform(batch)

    np.testing.assert_array_equal(arrays_b.X_raw.shape, arrays_c.X_raw.shape)
    np.testing.assert_array_equal(arrays_b.X_raw, arrays_c.X_raw)
    np.testing.assert_array_equal(arrays_b.X_classical, arrays_c.X_classical)
    np.testing.assert_array_equal(arrays_b.X_quantum, arrays_c.X_quantum)
    assert recipe_b.feature_names == recipe_c.feature_names
    assert recipe_b.fingerprint() == recipe_c.fingerprint()
