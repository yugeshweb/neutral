from __future__ import annotations

import numpy as np

from qhealth_qml.ecg import LEADS, align_raw_ecg_features, extract_ecg_features
from qhealth_qml.experiment import LoadedDataset, detection_confidence, prepare_dataset
from qhealth_qml.raw_hybrid import normalize_raw_ecg


def test_extract_ecg_features_accepts_twelve_lead_array():
    t = np.arange(5000, dtype=float) / 500.0
    signal = np.column_stack(
        [np.sin(2 * np.pi * (1.0 + lead / 20.0) * t) for lead in range(len(LEADS))]
    )
    features, names = extract_ecg_features(signal, sampling_rate=500.0)

    assert signal.shape == (5000, 12)
    assert features.shape == (290,)
    assert len(names) == len(features)
    assert np.isfinite(features).all()


def test_detection_confidence_is_distance_from_operating_threshold():
    confidence = detection_confidence(np.asarray([0.1, 0.5, 0.9]), 0.5)
    assert confidence is not None
    assert np.allclose(confidence, [0.8, 0.0, 0.8])


def test_raw_features_can_fill_optional_training_metadata():
    aligned, missing = align_raw_ecg_features(
        np.asarray([1.0, 2.0]),
        ["wave_a", "wave_b"],
        ["wave_b", "age", "wave_a"],
    )
    assert np.allclose(aligned[[0, 2]], [2.0, 1.0])
    assert np.isnan(aligned[1])
    assert missing == ["age"]


def test_raw_waveform_normalization_returns_leads_first_tensor():
    signal = np.arange(500 * len(LEADS), dtype=np.float32).reshape(500, len(LEADS))
    normalized = normalize_raw_ecg(signal, target_samples=250)

    assert normalized.shape == (len(LEADS), 250)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert np.allclose(np.median(normalized, axis=1), 0.0, atol=0.1)


def test_explicit_validation_indices_are_not_replaced_by_random_split():
    X = np.arange(20 * 4, dtype=float).reshape(20, 4)
    y = np.asarray([0, 1] * 10)
    dataset = LoadedDataset(
        name="fold-check",
        X=X,
        y=y,
        feature_names=["a", "b", "c", "d"],
        positive_label="positive",
        negative_label="negative",
    )
    train = np.arange(0, 12)
    validation = np.arange(12, 16)
    test = np.arange(16, 20)
    prepared = prepare_dataset(
        dataset,
        n_qubits=4,
        test_size=0.2,
        seed=7,
        max_train=0,
        max_test=0,
        validation_size=0.25,
        split_indices=(np.concatenate((train, validation)), test),
        validation_indices=validation,
    )
    assert prepared.X_train_raw.shape[0] == len(train)
    assert prepared.X_validation_raw is not None
    assert prepared.X_validation_raw.shape[0] == len(validation)
    assert prepared.X_test_raw.shape[0] == len(test)


def _write_ptbxl_fixture(tmp_path):
    """Minimal PTB-XL-shaped metadata/statement pair for row-selection tests."""

    statements = tmp_path / "scp_statements.csv"
    statements.write_text(
        ",diagnostic_class\nNORM,NORM\nIMI,MI\nASMI,MI\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "ptbxl_database.csv"
    metadata.write_text(
        "ecg_id,patient_id,scp_codes,validated_by_human,infarction_stadium1,strat_fold\n"
        "1,101,\"{'NORM': 100.0}\",True,,1\n"
        "2,102,\"{'IMI': 100.0}\",True,Stadium I,1\n"
        "3,103,\"{'ASMI': 100.0}\",True,Stadium II-III,2\n"
        "4,104,\"{'IMI': 100.0}\",True,Stadium III,2\n"
        "5,105,\"{'ASMI': 100.0}\",True,,3\n"
        "6,106,\"{'IMI': 100.0}\",False,Stadium I,3\n",
        encoding="utf-8",
    )
    return metadata, statements


def test_mi_stage_filter_keeps_only_acute_infarcts(tmp_path):
    from qhealth_qml.ecg import ACUTE_MI_STAGES, select_ptbxl_rows

    metadata, statements = _write_ptbxl_fixture(tmp_path)

    unfiltered = select_ptbxl_rows(metadata, statements, target="mi")
    acute = select_ptbxl_rows(metadata, statements, target="mi", mi_stages=ACUTE_MI_STAGES)

    # Unfiltered keeps every human-validated MI record regardless of stage.
    assert sorted(row["ecg_id"] for row in unfiltered) == ["1", "2", "3", "4", "5"]
    # Acute keeps the normal control plus only the Stadium I infarct.
    assert sorted(row["ecg_id"] for row in acute) == ["1", "2"]
    # Healed infarcts are dropped, never relabelled as normal.
    assert all(row["label"] == 0 for row in acute if row["ecg_id"] == "1")
    assert all(row["label"] == 1 for row in acute if row["ecg_id"] == "2")


def test_mi_stage_filter_rejected_for_non_mi_target(tmp_path):
    import pytest

    from qhealth_qml.ecg import ACUTE_MI_STAGES, select_ptbxl_rows

    metadata, statements = _write_ptbxl_fixture(tmp_path)

    with pytest.raises(ValueError, match="mi_stages only applies"):
        select_ptbxl_rows(metadata, statements, target="abnormal", mi_stages=ACUTE_MI_STAGES)


def test_ptbxl_age_clamps_the_deidentified_300_sentinel():
    from qhealth_qml.ecg import PTBXL_AGE_CEILING, ptbxl_age

    # PTB-XL stores every age above 89 as the literal 300.
    assert ptbxl_age("300") == PTBXL_AGE_CEILING
    assert ptbxl_age("62") == 62.0
    assert ptbxl_age("89") == 89.0
    assert np.isnan(ptbxl_age(""))
    assert np.isnan(ptbxl_age("not-a-number"))
