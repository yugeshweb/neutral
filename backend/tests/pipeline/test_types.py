"""T0: contract tests for the type definitions themselves (T002, T005)."""

from __future__ import annotations

import pytest

from qhealth_qml.pipeline.types import (
    Batch, IssueCode, ISSUE_CATALOGUE, LABEL_EXCLUDE, LABEL_NEGATIVE, LABEL_POSITIVE, Ledger, Sample,
)


def _sample(label=None, sample_id="s1") -> Sample:
    return Sample(
        sample_id=sample_id, subject_id="subj1", index_time=None, outcome_time=None,
        site=None, label=label, fields={}, arrays={}, subgroups={}, provenance={}, issues=[],
    )


def test_batch_of_one_is_not_a_special_case():
    batch = Batch(samples=[_sample()])
    assert len(batch) == 1
    assert list(batch) == batch.samples


def test_sample_scored_property():
    assert _sample(label=LABEL_POSITIVE).scored is True
    assert _sample(label=LABEL_NEGATIVE).scored is True
    assert _sample(label=LABEL_EXCLUDE).scored is False
    assert _sample(label=None).scored is True  # None (predict time) is not EXCLUDE


def test_ledger_invariant_holds():
    ledger = Ledger(n_in=10, n_scored=7, n_excluded=2, n_rejected=1)
    assert ledger.n_in == ledger.n_scored + ledger.n_excluded + ledger.n_rejected


def test_ledger_invariant_violation_raises():
    with pytest.raises(ValueError, match="Ledger invariant violated"):
        Ledger(n_in=10, n_scored=7, n_excluded=2, n_rejected=2)  # 11 != 10


def test_issue_catalogue_has_no_duplicate_codes():
    codes = list(ISSUE_CATALOGUE.keys())
    assert len(codes) == len(set(codes))


def test_every_issue_code_constant_is_in_the_catalogue():
    constants = [
        v for k, v in vars(IssueCode).items()
        if not k.startswith("_") and isinstance(v, str)
    ]
    assert set(constants) == set(ISSUE_CATALOGUE.keys())


def test_dataclasses_are_frozen():
    s = _sample()
    with pytest.raises(Exception):
        s.label = LABEL_POSITIVE  # type: ignore[misc]
