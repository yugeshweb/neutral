"""P0 acceptance tests for the platform layer: registry, routing, and the safety invariant.

These are the tests that back SC-002 ("100% of compatible models routed, 100% of
incompatible models reported with a reason") and SC-003 ("0 missing/invalid/unsupported
modality cases reported as a negative finding") — the two hardest safety guarantees in the
spec. They construct schema objects directly rather than going through adapters.py, so they
do not depend on that module and can run independently of it.
"""

from __future__ import annotations

import pytest

from qhealth_qml.platform.registry import load_registry
from qhealth_qml.platform.routing import route
from qhealth_qml.platform.safety import SafetyError, finalize_finding, not_evaluated, redact
from qhealth_qml.platform.schema import (
    ArtifactRef,
    CalibrationSpec,
    ConditionDefinition,
    CoverageReport,
    DataBundle,
    ExplainabilitySpec,
    InputContract,
    ModalityAsset,
    ModelCard,
    ModelDefinition,
    OptionalModalitySpec,
    PreprocessingSpec,
    RoutingDecision,
    SafetySpec,
)


def make_condition(**overrides) -> ConditionDefinition:
    defaults = dict(
        condition_id="test.condition",
        name="Test condition",
        domain="neurological",
        priority="P1",
        task_type="binary_classification",
        readiness_tier="high_reference",
        reference_label_tier="high",
        target_definition="test",
        population="test",
        required_modalities=["structured_clinical"],
        optional_modalities=[],
        expected_output="test",
        reference_datasets=[],
        evidence_links=[],
        limitations=[],
    )
    defaults.update(overrides)
    return ConditionDefinition(**defaults)


def make_model_card() -> ModelCard:
    return ModelCard(
        intended_use="test",
        excluded_use="test",
        training_population="test",
        evaluation_population="test",
        label_policy="test",
        data_sources=[],
        data_licenses=[],
        limitations=[],
        calibration_status="not_assessed",
        explanation_method="none",
        maintainer="test@example.com",
        reuse_manifest=[],
        research_record="specs/001-neurological-conditions/spec.md",
    )


def make_model(**overrides) -> ModelDefinition:
    defaults = dict(
        model_id="test-model",
        condition_id="test.condition",
        version="0.1.0",
        display_name="Test model",
        task_type="binary_classification",
        availability="available",
        lifecycle="experimental",
        executor="tabular_qml",
        input_contract=InputContract(
            required_modalities=["structured_clinical"],
            optional_modalities=[],
            required_fields=[],
            min_rows=1,
            population="test",
            population_filters={},
            quality_constraints={},
        ),
        dataset_profile_path=None,
        temporal_validation="n/a",
        preprocessing=PreprocessingSpec(
            imputation="median", scaling="standard", reduction="anova",
            n_components=4, angle_scaling="minmax", fitted_on="train_partition_only",
        ),
        quantum=None,
        classical_baseline_model_id=None,
        artifact=ArtifactRef(
            kind="saved_model_artifact", path="runtime/models/test.joblib",
            manifest_path=None, sha256=None, schema_version=4,
        ),
        calibration=CalibrationSpec(method="none", status="not_assessed", note=None),
        explainability=ExplainabilitySpec(method="none", scope="none", surrogate=False),
        output_score_type="calibrated_probability",
        evaluation_record_ids=["eval-1"],
        safety=SafetySpec(
            allows_negative_finding=True, abstain_margin=0.05,
            requires_full_coverage=True, disclaimer="test disclaimer", limitations=["test limitation"],
        ),
        model_card=make_model_card(),
    )
    defaults.update(overrides)
    return ModelDefinition(**defaults)


def make_asset(**overrides) -> ModalityAsset:
    defaults = dict(
        asset_id="asset-1",
        bundle_id="bundle-1",
        modality="structured_clinical",
        format="csv",
        role="clinical_table",
        visit_id=None,
        uri="memory:asset-1",
        content_hash="deadbeef",
        byte_size=100,
        rows=10,
        dimensions=None,
        units={},
        acquired_at=None,
        validation_status="accepted",
        validation_issues=[],
        field_mappings=[],
        derived_from=[],
    )
    defaults.update(overrides)
    return ModalityAsset(**defaults)


def make_bundle(assets) -> DataBundle:
    return DataBundle(
        bundle_id="bundle-1",
        case_id="local-test",
        case_id_source="assigned_local",
        created_at="2026-08-29T00:00:00Z",
        source="fixture",
        synthetic=False,
        visits=[],
        assets=list(assets),
        validation=[],
        content_hashes={a.asset_id: a.content_hash for a in assets},
        provenance={},
    )


class FakeRegistry:
    """Just enough of Registry's interface for route() to work against fixture models."""

    def __init__(self, conditions, models):
        self._conditions = conditions
        self._models = models

    def conditions(self):
        return self._conditions

    def models(self, condition_id=None):
        if condition_id is None:
            return self._models
        return [m for m in self._models if m.condition_id == condition_id]


# --- Routing (FR-009, FR-010, SC-002) -----------------------------------------------


def test_no_artifact_routes_not_available():
    model = make_model(availability="not available", artifact=ArtifactRef(
        kind="none", path=None, manifest_path=None, sha256=None, schema_version=None,
    ))
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset()])
    decisions = route(bundle, registry)
    assert decisions[0].status == "not available"
    assert decisions[0].reason


def test_missing_required_modality_routes_incompatible():
    model = make_model()
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([])  # no assets at all -> structured_clinical missing entirely
    decisions = route(bundle, registry)
    assert decisions[0].status == "incompatible"
    assert "structured_clinical" in decisions[0].reason


def test_quality_failed_required_modality_routes_insufficient_data():
    model = make_model()
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset(validation_status="quality_failed")])
    decisions = route(bundle, registry)
    assert decisions[0].status == "insufficient data"


def test_rejected_required_modality_routes_insufficient_data_not_incompatible():
    """A modality that was SUPPLIED but rejected is a different case from never supplied."""
    model = make_model()
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset(validation_status="rejected")])
    decisions = route(bundle, registry)
    assert decisions[0].status == "insufficient data"


def test_untrained_optional_gap_routes_insufficient_data():
    model = make_model(
        input_contract=InputContract(
            required_modalities=["structured_clinical"],
            optional_modalities=[OptionalModalitySpec(modality="imaging", trained_for_absence=False)],
            required_fields=[], min_rows=1, population="test",
            population_filters={}, quality_constraints={},
        )
    )
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset()])  # only structured_clinical, no imaging
    decisions = route(bundle, registry)
    assert decisions[0].status == "insufficient data"
    assert "imaging" in decisions[0].reason


def test_trained_for_absence_optional_gap_does_not_block():
    model = make_model(
        input_contract=InputContract(
            required_modalities=["structured_clinical"],
            optional_modalities=[OptionalModalitySpec(modality="imaging", trained_for_absence=True)],
            required_fields=[], min_rows=1, population="test",
            population_filters={}, quality_constraints={},
        )
    )
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset()])
    decisions = route(bundle, registry)
    assert decisions[0].status == "ready"


def test_full_contract_satisfied_routes_ready():
    model = make_model()
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset()])
    decisions = route(bundle, registry)
    assert decisions[0].status == "ready"
    assert decisions[0].reason == "input contract satisfied"


def test_preview_downgrades_ready_to_compatible():
    model = make_model()
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset()])
    decisions = route(bundle, registry, preview=True)
    assert decisions[0].status == "compatible"


def test_domain_filter_excludes_other_domains():
    model = make_model()
    registry = FakeRegistry([make_condition(domain="other")], [model])
    bundle = make_bundle([make_asset()])
    decisions = route(bundle, registry, domain="neurological")
    assert decisions == []


def test_missing_required_field_routes_insufficient_data():
    model = make_model(
        input_contract=InputContract(
            required_modalities=["structured_clinical"], optional_modalities=[],
            required_fields=["age"], min_rows=1, population="test",
            population_filters={}, quality_constraints={},
        )
    )
    registry = FakeRegistry([make_condition()], [model])
    bundle = make_bundle([make_asset(field_mappings=[])])  # no "age" field mapped
    decisions = route(bundle, registry)
    assert decisions[0].status == "insufficient data"
    assert "age" in decisions[0].reason


# --- Safety invariant (FR-011, FR-018, SC-003) --------------------------------------


def test_finalize_finding_rejects_non_terminal_decision():
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="ready", reason="input contract satisfied",
        satisfied_modalities=["structured_clinical"], missing_required=[], missing_optional=[],
        unmet_constraints=[],
    )
    coverage = CoverageReport(
        required_present=1, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=1.0, missing=[], quality_failed=[],
    )
    with pytest.raises(SafetyError):
        finalize_finding(
            model=model, condition=condition, decision=decision, coverage=coverage,
            raw={"score": 0.9, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=False,
        )


def test_finalize_finding_rejects_incomplete_coverage():
    """This is the core SC-003 guarantee: a scored finding cannot be produced from
    incomplete or quality-failed coverage, even if a score somehow made it this far."""
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="completed", reason="", satisfied_modalities=["structured_clinical"],
        missing_required=[], missing_optional=[], unmet_constraints=[],
    )
    incomplete_coverage = CoverageReport(
        required_present=0, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=0.0, missing=["structured_clinical"], quality_failed=[],
    )
    with pytest.raises(SafetyError):
        finalize_finding(
            model=model, condition=condition, decision=decision, coverage=incomplete_coverage,
            raw={"score": 0.9, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=False,
        )


def test_finalize_finding_rejects_negative_when_disallowed():
    model = make_model(safety=SafetySpec(
        allows_negative_finding=False, abstain_margin=None, requires_full_coverage=True,
        disclaimer="test", limitations=[],
    ))
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="completed", reason="", satisfied_modalities=["structured_clinical"],
        missing_required=[], missing_optional=[], unmet_constraints=[],
    )
    full_coverage = CoverageReport(
        required_present=1, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=1.0, missing=[], quality_failed=[],
    )
    with pytest.raises(SafetyError):
        finalize_finding(
            model=model, condition=condition, decision=decision, coverage=full_coverage,
            raw={"score": 0.1, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=False,
        )


def test_finalize_finding_positive_and_negative_happy_path():
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="completed", reason="", satisfied_modalities=["structured_clinical"],
        missing_required=[], missing_optional=[], unmet_constraints=[],
    )
    full_coverage = CoverageReport(
        required_present=1, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=1.0, missing=[], quality_failed=[],
    )
    positive = finalize_finding(
        model=model, condition=condition, decision=decision, coverage=full_coverage,
        raw={"score": 0.9, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=False,
    )
    assert positive.status == "positive"
    assert positive.disclaimer == model.safety.disclaimer

    negative = finalize_finding(
        model=model, condition=condition, decision=decision, coverage=full_coverage,
        raw={"score": 0.1, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=False,
    )
    assert negative.status == "negative"


def test_finalize_finding_abstained_carries_no_label():
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="abstained", reason="", satisfied_modalities=["structured_clinical"],
        missing_required=[], missing_optional=[], unmet_constraints=[],
    )
    full_coverage = CoverageReport(
        required_present=1, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=1.0, missing=[], quality_failed=[],
    )
    finding = finalize_finding(
        model=model, condition=condition, decision=decision, coverage=full_coverage,
        raw={"score": 0.5, "abstained": True}, run_id="run-1", synthetic=False,
    )
    assert finding.status == "abstained"
    assert finding.abstained is True
    assert finding.reason


def test_synthetic_finding_carries_marker_and_flag():
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status="completed", reason="", satisfied_modalities=["structured_clinical"],
        missing_required=[], missing_optional=[], unmet_constraints=[],
    )
    full_coverage = CoverageReport(
        required_present=1, required_total=1, optional_present=0, optional_total=0,
        coverage_ratio=1.0, missing=[], quality_failed=[],
    )
    finding = finalize_finding(
        model=model, condition=condition, decision=decision, coverage=full_coverage,
        raw={"score": 0.9, "abstained": False, "threshold": 0.5}, run_id="run-1", synthetic=True,
    )
    assert finding.synthetic is True
    assert "SYNTHETIC" in finding.disclaimer


@pytest.mark.parametrize(
    "decision_status,expected_finding_status",
    [
        ("not available", "not available"),
        ("incompatible", "not evaluated"),
        ("insufficient data", "insufficient data"),
        ("failed", "not evaluated"),
    ],
)
def test_not_evaluated_status_table(decision_status, expected_finding_status):
    model = make_model()
    condition = make_condition()
    decision = RoutingDecision(
        model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
        status=decision_status, reason="test reason", satisfied_modalities=[],
        missing_required=["structured_clinical"], missing_optional=[], unmet_constraints=[],
    )
    finding = not_evaluated(model, condition, decision, run_id="run-1", reason="test reason", synthetic=False)
    assert finding.status == expected_finding_status
    assert finding.score is None
    assert finding.abstained is False


def test_not_evaluated_never_produces_positive_or_negative():
    """SC-003: 0 missing/invalid/unsupported modality cases reported as a negative finding."""
    model = make_model()
    condition = make_condition()
    for decision_status in ("not available", "incompatible", "insufficient data", "failed"):
        decision = RoutingDecision(
            model_id=model.model_id, model_version=model.version, condition_id=model.condition_id,
            status=decision_status, reason="x", satisfied_modalities=[], missing_required=[],
            missing_optional=[], unmet_constraints=[],
        )
        finding = not_evaluated(model, condition, decision, run_id="run-1", reason="x", synthetic=False)
        assert finding.status not in ("positive", "negative")


def test_redact_strips_quoted_values_not_in_allowlist():
    message = redact("feature 'bmi' at CSV row 41 is not numeric", allowed_fields={"bmi"})
    assert "'bmi'" in message  # allowed field name preserved
    assert "41" not in message  # positional row number redacted


def test_redact_strips_disallowed_quoted_value():
    message = redact("patient name 'John Smith' rejected", allowed_fields=set())
    assert "John Smith" not in message
    assert "<redacted>" in message


# --- Registry integration (loads the real, shipped registry) -----------------------


def test_real_registry_loads_and_routes_without_fabricating_findings():
    registry = load_registry()
    empty_bundle = make_bundle([])
    decisions = route(empty_bundle, registry)
    assert decisions  # at least one registered model in the domain
    assert all(d.status in ("not available", "incompatible", "insufficient data") for d in decisions)

    condition = registry.model("stroke-clinical-risk-tabular")
    stroke_condition = next(c for c in registry.conditions() if c.condition_id == condition.condition_id)
    for decision in decisions:
        if decision.status in ("not available", "incompatible", "insufficient data", "failed"):
            finding = not_evaluated(
                registry.model(decision.model_id), stroke_condition if decision.model_id == condition.model_id
                else next(c for c in registry.conditions() if c.condition_id == decision.condition_id),
                decision, run_id="run-1", reason=decision.reason, synthetic=False,
            )
            assert finding.status not in ("positive", "negative")
