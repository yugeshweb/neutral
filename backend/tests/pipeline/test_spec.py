"""T3, T004: SourceSpec loading and the nine validation rules (design.md §6.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qhealth_qml.pipeline.spec import SourceSpec, SpecValidationError

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "cardiac_cohort.spec.json"


@pytest.mark.parametrize("profile_path", sorted(PROFILES_DIR.glob("*.json")), ids=lambda p: p.name)
def test_all_existing_profiles_load_unmodified(profile_path):
    """T003's done-condition: every profile already committed under
    backend/profiles/ loads without modification."""

    spec = SourceSpec.load(profile_path)
    assert spec.name
    assert spec.target_column


def test_fixture_spec_loads():
    spec = SourceSpec.load(FIXTURE_SPEC)
    assert spec.spec_version == 2
    assert spec.target_column == "cardiac_death_within_horizon"


def test_rule_leakage_columns_disjoint_from_required_fields():
    raw = {
        "name": "x", "target_column": "y", "modality": "tabular",
        "required_fields": ["age", "chol"], "leakage_columns": ["chol"],
    }
    with pytest.raises(SpecValidationError, match="leakage_columns"):
        SourceSpec.from_dict(raw).validate()


def test_rule_unknown_modality_rejected():
    raw = {"name": "x", "target_column": "y", "modality": "not-a-real-modality"}
    with pytest.raises(SpecValidationError, match="modality"):
        SourceSpec.from_dict(raw).validate()


def test_rule_prediction_framing_requires_horizon_fields():
    raw = {"name": "x", "target_column": "y", "modality": "tabular", "temporal_framing": "prediction"}
    with pytest.raises(SpecValidationError) as excinfo:
        SourceSpec.from_dict(raw).validate()
    assert excinfo.value.field_name in ("horizon_days", "index_time_column", "outcome_time_column", "group_column")


def test_rule_target_column_cannot_double_as_group_column():
    raw = {"name": "x", "target_column": "y", "modality": "tabular", "group_column": "y"}
    with pytest.raises(SpecValidationError, match="group_column"):
        SourceSpec.from_dict(raw).validate()


def test_rule_id_and_group_column_may_coincide():
    """The spec's own worked example (design.md §6.1) declares
    `id_column` == `group_column` == "patient_id" deliberately - the
    common one-row-per-patient case. Must NOT raise."""

    raw = {"name": "x", "target_column": "y", "modality": "tabular", "id_column": "patient_id", "group_column": "patient_id"}
    SourceSpec.from_dict(raw).validate()  # does not raise


def test_rule_no_absolute_host_path():
    raw = {
        "name": "x", "target_column": "y", "modality": "tabular",
        "source": {"root": "C:\\Users\\someone\\data", "pattern": "cohort.csv"},
    }
    with pytest.raises(SpecValidationError, match="source.root"):
        SourceSpec.from_dict(raw).validate()


def test_rule_env_expanded_root_is_allowed():
    raw = {
        "name": "x", "target_column": "y", "modality": "tabular",
        "source": {"root": "${QHEALTH_DATA_ROOT}/cohort", "pattern": "cohort.csv"},
    }
    SourceSpec.from_dict(raw).validate()  # does not raise


def test_rule_unknown_spec_version_rejected():
    raw = {"name": "x", "target_column": "y", "modality": "tabular", "spec_version": 99}
    with pytest.raises(SpecValidationError, match="spec_version"):
        SourceSpec.from_dict(raw).validate()
