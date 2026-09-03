"""Integration tests against REAL, teammate-built feature extractors -
`qhealth_qml.ecg.extract_ecg_features` and
`qhealth_qml.angiography.extract_angiography_features`, both pulled in
from `origin/main`'s `shuvam/src/qhealth_qml/` (not yet merged to
`backend/`; copied here read-only via `git show`). `shuvam/sync.sh`'s own
MODULES list confirms both are meant to live in `backend/src/qhealth_qml/`
- this isn't an improvised integration, it's the intended one, done early.
ecg.py is the exact function spec.md's own worked example names
(`representation.extractor: "ecg.extract_ecg_features"`).

`pipeline/extractors/real_ecg.py` and `real_angiography.py` are the ENTIRE
integration surface in each case: thin shims from this package's
`(Sample, SourceSpec) -> (vector, names)` representation contract to each
real function's own contract. No changes were needed on either side -
proving the `representation.extractor` dispatch mechanism (design.md
§9.2.7) actually works for real third-party model code, not just this
build's own placeholders.

Two of `sync.sh`'s eight sanctioned modules (`hybrid_qnn.py`,
`pretrained_encoder.py`) fail to import - both depend on a `raw_hybrid.py`
that was never pushed to `origin/main` at all (confirmed via
`git ls-tree`), not a gap on this side. `cardiovascular.py`,
`cardiovascular_cli.py`, `prs.py` and `multimodal.py` import cleanly but
are higher-level orchestration, not single-record feature extractors -
not wired into `representation.extractor` here."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.pipeline import Pipeline, SourceSpec

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fitted_real():
    spec = SourceSpec.load(FIXTURES / "ecg_cohort_real_extractor.spec.json")
    batch = Pipeline.read(spec, source=FIXTURES / "ecg_cohort")
    return Pipeline.from_spec(spec).fit(batch, n_qubits=6)


def test_real_extractor_produces_the_documented_290_features(fitted_real):
    """12 leads x 18 per-lead features + C(12,2)=66 lead-pair correlations
    + 4 lead_std stats + 4 rhythm stats = 290 - the extractor's own
    documented, deterministic feature count (data-model.md's worked
    example cites ~294 for a different cohort's rhythm-detection yield;
    290 is what THIS cohort's synthetic signal produces, and it must be
    exactly reproducible)."""

    assert len(fitted_real.recipe.feature_names) == 290
    assert fitted_real.recipe.feature_names[0] == "I_raw_mean"
    assert fitted_real.recipe.feature_names[-1] == "detected_rr_intervals"
    assert np.isfinite(fitted_real.train.X_classical).all()


def test_train_equals_predict_bit_identical_through_real_extractor(fitted_real):
    """T3.4, through genuinely third-party model code rather than this
    build's own placeholder - the whole point of this integration test."""

    recipe = fitted_real.recipe
    train_ids = list(fitted_real.train.row_ids)
    target_id = train_ids[0]
    train_idx = train_ids.index(target_id)
    expected = fitted_real.train.X_classical[train_idx]

    tmp = Path(tempfile.mkdtemp())
    try:
        for ext in (".hea", ".dat"):
            shutil.copy(FIXTURES / "ecg_cohort" / f"{target_id}{ext}", tmp / f"{target_id}{ext}")
        run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=tmp))
        assert run.arrays.X_classical.shape[0] == 1
        np.testing.assert_array_equal(run.arrays.X_classical[0], expected)
        assert run.verdicts[0].status == "accept"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_real_extractor_refuses_a_lead_count_mismatch_with_a_clear_error():
    """FR-relevant honesty check: the real extractor hard-requires all 12
    standard leads. A spec declaring fewer must fail loudly and
    specifically, not silently truncate or zero-pad - this is exactly the
    kind of standardizer<->model contract mismatch the project's stated
    goal ('no bugs between the standardizing pipeline and the training
    backend') exists to catch before it reaches production."""

    import dataclasses

    spec = SourceSpec.load(FIXTURES / "ecg_cohort_real_extractor.spec.json")
    raw = dict(spec._raw)
    raw["signal"] = {**raw["signal"], "channels": ["I", "II", "V1"]}
    bad_spec = dataclasses.replace(spec, _raw=raw)

    batch = Pipeline.read(bad_spec, source=FIXTURES / "ecg_cohort")
    with pytest.raises(ValueError, match="expected 12 ECG leads"):
        Pipeline.from_spec(bad_spec).fit(batch, n_qubits=6)


# ---------------------------------------------------------------------------
# angiography (qhealth_qml.angiography.extract_angiography_features)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_angio():
    spec = SourceSpec.load(FIXTURES / "angio_cohort_real_extractor.spec.json")
    batch = Pipeline.read(spec, source=FIXTURES / "angio_cohort")
    return Pipeline.from_spec(spec).fit(batch, n_qubits=4)


def test_real_angiography_extractor_produces_the_documented_34_features(fitted_angio):
    """4 scales x 5 per-scale vesselness stats + 12 global intensity/
    gradient stats + 2 calibre-profile stats = 34 - this extractor's own
    documented, deterministic feature count."""

    assert len(fitted_angio.recipe.feature_names) == 34
    assert fitted_angio.recipe.feature_names[0] == "scale1_vesselness_mean"
    assert fitted_angio.recipe.feature_names[-1] == "coarse_to_fine_vesselness_ratio"
    assert np.isfinite(fitted_angio.train.X_classical).all()


def test_train_equals_predict_bit_identical_through_real_angiography_extractor(fitted_angio):
    """T3.4 through the real vesselness extractor, mirroring the ECG
    check above."""

    recipe = fitted_angio.recipe
    cohort_dir = FIXTURES / "angio_cohort"
    train_ids = list(fitted_angio.train.row_ids)
    target_id = train_ids[0]
    train_idx = train_ids.index(target_id)
    expected = fitted_angio.train.X_classical[train_idx]

    match = next(p for p in cohort_dir.rglob("*") if p.is_file() and f"{p.parent.name}__{p.stem}" == target_id)
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(match, tmp / match.name)
        run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=tmp))
        assert run.arrays.X_classical.shape[0] == 1
        np.testing.assert_array_equal(run.arrays.X_classical[0], expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
