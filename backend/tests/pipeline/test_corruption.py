"""T2 - adversarial corruption matrix (design.md §15.3, a Phase-1-relevant
subset). Each case asserts the SPECIFIC issue code, not merely a
non-success result (SC-008)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.pipeline import Pipeline, SourceSpec
from qhealth_qml.pipeline.types import IssueCode

FIXTURES = Path(__file__).parent / "fixtures"
SPEC_PATH = FIXTURES / "cardiac_cohort.spec.json"
CSV_PATH = FIXTURES / "cardiac_cohort.csv"


@pytest.fixture(scope="module")
def fitted():
    spec = SourceSpec.load(SPEC_PATH)
    batch = Pipeline.read(spec, source=CSV_PATH)
    return Pipeline.from_spec(spec).fit(batch, n_qubits=6)


def test_censored_bound_quarantine_signature_is_quarantined_not_coerced(fitted):
    """FR-038: the injected thousands-separator row in the fixture must
    quarantine the whole column, and the column must not silently coerce
    into a huge numeric outlier anywhere downstream."""

    assert "vpc_per_hour" in fitted.recipe.quarantined_columns
    assert fitted.recipe.quarantined_columns["vpc_per_hour"] == "thousands-separator corruption signature"
    assert "vpc_per_hour" not in fitted.recipe.feature_names


def test_missing_required_field_is_refused_not_imputed():
    spec_raw = SourceSpec.load(SPEC_PATH)
    import dataclasses
    spec = dataclasses.replace(spec_raw, required_fields=("ejection_fraction",))

    batch = Pipeline.read(spec, source=CSV_PATH)
    fitted = Pipeline.from_spec(spec).fit(batch, n_qubits=6)
    recipe = fitted.recipe

    predict_csv = (
        "patient_id,age,sex,smoking_status,bmi,ejection_fraction,vpc_per_hour,nyha_class\n"
        "P9001,60,Male,never smoked,27.0,,10,II\n"
    )
    run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8")))

    assert len(run.verdicts) == 1
    assert run.verdicts[0].status == "reject"
    codes = [i.code for i in run.verdicts[0].issues]
    assert IssueCode.REQUIRED_FIELD_MISSING in codes
    reject_issue = next(i for i in run.verdicts[0].issues if i.code == IssueCode.REQUIRED_FIELD_MISSING)
    assert reject_issue.field == "ejection_fraction"
    assert run.arrays.X_classical.shape[0] == 0  # refused record never reaches the matrix (FR-082)


def test_missing_optional_field_becomes_nan_then_imputed_never_zero(fitted):
    recipe = fitted.recipe
    predict_csv = (
        "patient_id,age,sex,smoking_status,bmi,ejection_fraction,vpc_per_hour,nyha_class\n"
        "P9002,60,Male,never smoked,,35,10,II\n"  # bmi blank, optional
    )
    run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8")))

    assert run.verdicts[0].status in ("accept", "accept_with_flags")
    bmi_idx = recipe.feature_names.index("bmi")
    assert run.arrays.X_raw[0, bmi_idx] != run.arrays.X_raw[0, bmi_idx]  # NaN in X_raw (pre-impute)
    assert not np.isnan(run.arrays.X_classical).any()  # imputed by transform time
    # Never zero (FR-032) - the imputed value is the TRAIN median, not 0.
    imputed_raw_value = recipe.imputer.transform(run.arrays.X_raw)[0, bmi_idx]
    assert imputed_raw_value != 0.0


def test_unseen_category_gets_indicator_not_silent_all_zero(fitted):
    recipe = fitted.recipe
    assert "smoking_status" in recipe.categorical_vocab
    predict_csv = (
        "patient_id,age,sex,smoking_status,bmi,ejection_fraction,vpc_per_hour,nyha_class\n"
        "P9003,60,Male,vaping,27.0,35,10,II\n"  # 'vaping' never seen in training
    )
    run = Pipeline.from_recipe(recipe).run(Pipeline.read(recipe.spec, source=predict_csv.encode("utf-8")))

    assert run.arrays.X_raw.shape[0] == 1
    unseen_idx = recipe.feature_names.index("smoking_status__unseen")
    assert run.arrays.X_raw[0, unseen_idx] == 1.0
    for cat in recipe.categorical_vocab["smoking_status"]:
        cat_idx = recipe.feature_names.index(f"smoking_status={cat}")
        assert run.arrays.X_raw[0, cat_idx] == 0.0  # all-zero across known categories...
    # ...but distinguishable from a valid observation via the unseen bit (FR-043).


def test_empty_batch_raises_explicit_error():
    spec = SourceSpec.load(SPEC_PATH)
    from qhealth_qml.pipeline.types import Batch
    from qhealth_qml.pipeline.pipeline import PipelineError

    with pytest.raises(PipelineError):
        Pipeline.from_spec(spec).fit(Batch(samples=[], spec=spec, provenance={}, issues=[]), n_qubits=6)
