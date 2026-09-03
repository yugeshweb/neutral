"""Phase 2 (signal: ecg, eeg, gait) + Phase 3 (imaging: mr, ct) adapter
coverage - tasks.md T036-T062, scoped down (see PIPELINE_STATUS.md for what
was and wasn't built). One parametrized suite per modality: read a real
cohort, fit a real Recipe, verify train==predict bit-identical (T3.4) on a
record re-read fresh from disk, and confirm a real corruption case
(missing channel/sequence, or a header modality mismatch) is refused by
name rather than silently coerced or crashed on."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.pipeline import Pipeline, SourceSpec

FIXTURES = Path(__file__).parent / "fixtures"

MODALITIES = [
    # (spec_file, cohort_dir, corruption_dir, expected_corrupt_records: {sample_id: (status, [(code, field), ...])})
    (
        "ecg_cohort.spec.json", "ecg_cohort", "ecg_corruption",
        {
            "MISSING_LEAD": ("reject", [("channel_missing", "V1")]),
            "FLATLINE": ("accept_with_flags", None),  # exact issues not asserted - OOD flag, not a fixed code
        },
    ),
    (
        "eeg_cohort.spec.json", "eeg_cohort", "eeg_corruption",
        {
            "MISSING_CHANNEL": ("reject", [("channel_missing", "Fp2"), ("channel_missing", "C3")]),
            "FLATLINE": ("accept_with_flags", None),
        },
    ),
    (
        "gait_cohort.spec.json", "gait_cohort", "gait_corruption",
        {"MISSING_CHANNEL": ("reject", [("channel_missing", "accel_z")])},
    ),
    (
        "mri_cohort.spec.json", "mri_cohort", "mri_corruption",
        {"MISSING_SEQUENCE": ("reject", [("sequence_missing", "t2")])},
    ),
    (
        "ct_cohort.spec.json", "ct_cohort", "ct_corruption",
        {"WRONG_MODALITY": ("reject", [("modality_mismatch", None)])},
    ),
    (
        "image2d_cohort.spec.json", "image2d_cohort", "image2d_corruption",
        {"BLANK": ("reject", [("low_foreground", None)])},
    ),
]


@pytest.fixture(scope="module", params=MODALITIES, ids=[m[1] for m in MODALITIES])
def modality_fixture(request):
    spec_file, cohort_dir, corruption_dir, expected = request.param
    spec = SourceSpec.load(FIXTURES / spec_file)
    batch = Pipeline.read(spec, source=FIXTURES / cohort_dir)
    fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=4)
    return {
        "spec": spec, "batch": batch, "fitted": fitted, "recipe": fitted.recipe,
        "cohort_dir": FIXTURES / cohort_dir, "corruption_dir": FIXTURES / corruption_dir,
        "expected_corrupt": expected,
    }


def test_cohort_reads_and_fits_with_real_numeric_output(modality_fixture):
    fitted = modality_fixture["fitted"]
    assert fitted.train.X_classical.shape[0] > 0
    assert fitted.train.X_quantum.shape == fitted.train.X_classical.shape
    assert np.isfinite(fitted.train.X_classical).all()
    assert np.isfinite(fitted.train.X_quantum).all()
    # angle-scaled quantum features stay within the declared range
    assert np.all(fitted.train.X_quantum >= -np.pi / 2 - 1e-9)
    assert np.all(fitted.train.X_quantum <= np.pi / 2 + 1e-9)


def test_train_equals_predict_bit_identical_on_a_fresh_read(modality_fixture):
    """T3.4 for the representation-based path: the SAME record, re-read
    from disk in complete isolation from the rest of the cohort, must
    reproduce the exact training-time feature vector."""

    recipe = modality_fixture["recipe"]
    fitted = modality_fixture["fitted"]
    cohort_dir = modality_fixture["cohort_dir"]

    train_ids = list(fitted.train.row_ids)
    target_id = train_ids[0]
    train_idx = train_ids.index(target_id)
    expected = fitted.train.X_classical[train_idx]

    tmp = Path(tempfile.mkdtemp())
    try:
        # `target_id` is the label-stripped sample_id: adapters parse off
        # either a `__labelN` filename suffix (wfdb/edf/gait/nifti/dicom
        # fixtures), or a class-subdirectory prefix (`<folder>__<stem>`,
        # image2d's `yes/`/`no/` layout) - both are reconstructed here the
        # same way the adapter builds them, then matched against target_id.
        candidates = {p: (f"{p.parent.name}__{p.stem}" if p.parent != cohort_dir else p.stem) for p in cohort_dir.iterdir()}
        matches = [
            p for p, candidate_id in candidates.items()
            if candidate_id == target_id or candidate_id.split("__label")[0] == target_id
        ]
        if not matches:
            # class-subdirectory layout: files live one level deeper.
            candidates = {p: f"{p.parent.name}__{p.stem}" for p in cohort_dir.rglob("*") if p.is_file()}
            matches = [p for p, candidate_id in candidates.items() if candidate_id == target_id]
        assert matches, f"no on-disk match for sample_id {target_id!r} under {cohort_dir}"
        for item in matches:
            if item.is_dir():
                shutil.copytree(item, tmp / target_id)
            else:
                shutil.copy(item, tmp / item.name)

        predict_batch = Pipeline.read(recipe.spec, source=tmp)
        run = Pipeline.from_recipe(recipe).run(predict_batch)
        assert run.arrays.X_classical.shape[0] == 1
        np.testing.assert_array_equal(run.arrays.X_classical[0], expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corruption_is_refused_by_name_not_silently_coerced(modality_fixture):
    recipe = modality_fixture["recipe"]
    corruption_dir = modality_fixture["corruption_dir"]
    expected_corrupt = modality_fixture["expected_corrupt"]

    corrupt_batch = Pipeline.read(recipe.spec, source=corruption_dir)
    run = Pipeline.from_recipe(recipe).run(corrupt_batch)
    by_id = {sid: v for sid, v in zip((s.sample_id for s in corrupt_batch.samples), run.verdicts)}

    for sample_id, (expected_status, expected_issues) in expected_corrupt.items():
        assert sample_id in by_id, f"expected corruption record {sample_id!r} not found in {corruption_dir}"
        verdict = by_id[sample_id]
        assert verdict.status == expected_status, f"{sample_id}: expected status {expected_status!r}, got {verdict.status!r}"
        if expected_issues is not None:
            actual_pairs = {(i.code, i.field) for i in verdict.issues}
            for pair in expected_issues:
                assert pair in actual_pairs, f"{sample_id}: expected issue {pair} not in {actual_pairs}"

    # A rejected record never reaches the scored matrix (FR-082) - checked
    # against whichever corrupt records were declared "reject" above.
    rejected_ids = {sid for sid, (status, _) in expected_corrupt.items() if status == "reject"}
    scored_ids = set(run.arrays.row_ids.tolist())
    assert not (rejected_ids & scored_ids)


def test_recipe_save_load_round_trip_bit_identical(modality_fixture, tmp_path):
    recipe = modality_fixture["recipe"]
    cohort_dir = modality_fixture["cohort_dir"]

    saved_path = recipe.save(tmp_path / "test.recipe.pkl")
    from qhealth_qml.pipeline import Recipe
    reloaded = Recipe.load(saved_path)

    batch = Pipeline.read(recipe.spec, source=cohort_dir)
    run_original = Pipeline.from_recipe(recipe).run(batch)
    run_reloaded = Pipeline.from_recipe(reloaded).run(batch)

    np.testing.assert_array_equal(run_original.arrays.X_classical, run_reloaded.arrays.X_classical)
    assert reloaded.fingerprint() == recipe.fingerprint()
