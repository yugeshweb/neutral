"""Tests for the pluggable cardiovascular frame.

These cover the properties an integrator depends on: that readiness can be inspected
without running anything, that an unimplemented or starved modality is reported rather
than raised at an unpredictable moment, that influence cannot be granted without
measurement, and that pooling never over-claims how far into the future its answer
reaches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qhealth_qml.cardiovascular import (
    CARDIOVASCULAR_MODALITIES,
    CardiovascularFrame,
    ModalitySpec,
    modality_registry,
)
from qhealth_qml.multimodal import ModalityModel


def test_every_declared_modality_has_a_valid_framing() -> None:
    for spec in CARDIOVASCULAR_MODALITIES:
        assert spec.temporal_framing in {
            "prediction",
            "detection",
            "characterisation",
            "screening",
        }
        assert spec.clinical_role.strip()
        assert spec.data_requirement.strip()


def test_invalid_framing_is_rejected_at_declaration() -> None:
    with pytest.raises(ValueError, match="temporal_framing"):
        ModalitySpec(
            name="bogus",
            clinical_role="x",
            temporal_framing="early_detection",  # not a member of the enum
            data_requirement="y",
        )


def test_readiness_reports_without_touching_models(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    report = frame.readiness_report()
    states = {entry["modality"]: entry["state"] for entry in report["modalities"]}
    # Nothing exists under tmp_path, so no built-in modality can be ready.
    assert report["runnable_now"] == []
    assert states["echocardiography"] == "stub"
    assert states["ecg_12lead"] == "needs_data"


def test_ready_when_the_source_exists(tmp_path: Path) -> None:
    (tmp_path / "data" / "ptb-xl" / "1.0.3").mkdir(parents=True)
    frame = CardiovascularFrame(root=tmp_path)
    assert "ecg_12lead" in frame.readiness_report()["runnable_now"]


def test_source_override_is_honoured(tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere"
    custom.mkdir()
    frame = CardiovascularFrame(root=tmp_path, sources={"ecg_12lead": "elsewhere"})
    entry = next(
        item
        for item in frame.readiness_report()["modalities"]
        if item["modality"] == "ecg_12lead"
    )
    assert entry["state"] == "ready"
    assert entry["source"] == str(custom)


def test_loading_a_stub_modality_fails_with_a_specific_reason(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    with pytest.raises(ValueError, match="no loader implemented"):
        frame.load("echocardiography")


def test_loading_an_unknown_modality_names_the_known_ones(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    with pytest.raises(KeyError, match="unknown modality"):
        frame.load("mri_perfusion")


def test_registering_a_custom_modality_is_the_extension_point(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    spec = ModalitySpec(
        name="ct_angiography",
        clinical_role="Coronary CT angiography",
        temporal_framing="detection",
        data_requirement="CCTA volumes",
        loader=lambda source, **_: None,
        default_source="ccta",
    )
    frame.register(spec)
    assert "ct_angiography" in frame.readiness_report()["summary"]["needs_data"]
    (tmp_path / "ccta").mkdir()
    assert "ct_angiography" in frame.readiness_report()["runnable_now"]


def test_registry_rejects_non_specs() -> None:
    with pytest.raises(TypeError):
        modality_registry(extra=[{"name": "nope"}])  # type: ignore[list-item]


def test_untrained_modality_pools_at_zero_weight(tmp_path: Path) -> None:
    """A channel with no measured skill must not acquire influence by assertion."""

    frame = CardiovascularFrame(root=tmp_path)
    with pytest.raises(ValueError, match="no modality provided usable evidence"):
        frame.pool({"ecg_12lead": 0.9})


def test_pooling_weights_by_demonstrated_skill(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    frame.trained["ecg_12lead"] = ModalityModel(
        modality="ecg_12lead",
        model_id="ecg_12lead:logistic_regression",
        artifact=None,
        validated_balanced_accuracy=0.86,
    )
    frame.trained["ehr_tabular"] = ModalityModel(
        modality="ehr_tabular",
        model_id="ehr_tabular:logistic_regression",
        artifact=None,
        validated_balanced_accuracy=0.52,
    )
    payload = frame.pool({"ecg_12lead": 0.9, "ehr_tabular": 0.9})
    weights = {item["modality"]: item["weight"] for item in payload["per_modality"]}
    # 2*(0.86-0.5) = 0.72 versus 2*(0.52-0.5) = 0.04.
    assert weights["ecg_12lead"] == pytest.approx(0.72)
    assert weights["ehr_tabular"] == pytest.approx(0.04)
    assert weights["ecg_12lead"] > weights["ehr_tabular"] * 10


def test_absent_modality_is_omitted_not_imputed(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    frame.trained["ecg_12lead"] = ModalityModel(
        modality="ecg_12lead",
        model_id="ecg_12lead:logistic_regression",
        artifact=None,
        validated_balanced_accuracy=0.86,
    )
    payload = frame.pool({"ecg_12lead": 0.8, "angiography": None})
    assert payload["contributing_modalities"] == ["ecg_12lead"]
    assert payload["missing_modalities"] == ["angiography"]


def test_pool_rejects_unknown_modalities(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    with pytest.raises(KeyError, match="unknown modalities"):
        frame.pool({"tea_leaves": 0.9})


def test_pooled_framing_takes_the_weakest_contributor(tmp_path: Path) -> None:
    """A 4-year prediction pooled with a present-state detection is not a prediction."""

    frame = CardiovascularFrame(root=tmp_path)
    for name, accuracy in (("ehr_tabular", 0.70), ("ecg_12lead", 0.86)):
        frame.trained[name] = ModalityModel(
            modality=name,
            model_id=f"{name}:logistic_regression",
            artifact=None,
            validated_balanced_accuracy=accuracy,
        )

    prognostic_only = frame.pool({"ehr_tabular": 0.8})
    assert prognostic_only["temporal_framing"] == "prediction"

    mixed = frame.pool({"ehr_tabular": 0.8, "ecg_12lead": 0.8})
    assert mixed["temporal_framing"] == "detection"


def test_report_states_the_disjoint_cohort_limitation(tmp_path: Path) -> None:
    frame = CardiovascularFrame(root=tmp_path)
    report = frame.report()
    assert "disjoint" in report["fusion"]["limitation"]
    assert report["fusion"]["weight_source"].startswith("each modality's own validation")


def test_report_round_trips_as_json(tmp_path: Path) -> None:
    import json

    frame = CardiovascularFrame(root=tmp_path)
    path = frame.save_report(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["condition"] == "cardiovascular"
    assert payload["schema_version"] >= 1


def test_fit_report_round_trips_into_pooling_weights(tmp_path: Path) -> None:
    """The documented integration path: fit once, then pool elsewhere with those weights."""

    import json

    from qhealth_qml.cardiovascular_cli import _load_weights

    source = CardiovascularFrame(root=tmp_path)
    source.trained["ecg_12lead"] = ModalityModel(
        modality="ecg_12lead",
        model_id="ecg_12lead:logistic_regression",
        artifact=None,
        validated_balanced_accuracy=0.8486,
    )
    report_path = source.save_report(tmp_path / "fit-report.json")

    restored = CardiovascularFrame(root=tmp_path)
    assert _load_weights(restored, report_path) is True
    assert restored.trained["ecg_12lead"].weight == pytest.approx(0.6972, abs=1e-4)

    payload = restored.pool({"ecg_12lead": 0.81})
    assert payload["contributing_modalities"] == ["ecg_12lead"]

    # A report carrying no trained modalities must leave the frame untrusted rather
    # than silently defaulting every channel to equal influence.
    empty = CardiovascularFrame(root=tmp_path)
    empty_path = empty.save_report(tmp_path / "empty.json")
    assert json.loads(empty_path.read_text(encoding="utf-8"))["weights"] == {}
    assert _load_weights(CardiovascularFrame(root=tmp_path), empty_path) is False


def test_prs_modality_is_not_marked_quantum_capable() -> None:
    """One feature per patient gives a quantum feature map nothing to entangle."""

    spec = next(item for item in CARDIOVASCULAR_MODALITIES if item.name == "cad_prs")
    assert spec.quantum_capable is False
