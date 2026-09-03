"""T0/T2 unit tests for `clean.py`'s per-value and per-column functions
(FR-028, FR-029, FR-038, FR-040, FR-044)."""

from __future__ import annotations

from qhealth_qml.pipeline.clean import (
    detect_categorical_columns,
    detect_constant_or_missing_columns,
    high_cardinality_columns,
    looks_like_thousands_separator_corruption,
    map_sentinel,
    normalize_boolean,
    parse_censored_bound,
)

SENTINELS = ("na", "n/a", "nan", "none", "null", "missing", "-", "?", "unknown", ".")


def test_map_sentinel_case_insensitive_and_trimmed():
    assert map_sentinel(" N/A ", SENTINELS)
    assert map_sentinel("Unknown", SENTINELS)
    assert not map_sentinel("31", SENTINELS)


def test_parse_censored_bound_recognises_declared_tokens():
    bound = parse_censored_bound(">89", (">", "<"))
    assert bound is not None
    assert bound.value == 89.0
    assert bound.token == ">"


def test_parse_censored_bound_ignores_plain_numbers():
    assert parse_censored_bound("89", (">", "<")) is None


def test_parse_censored_bound_ignores_undeclared_token():
    assert parse_censored_bound(">89", ("<",)) is None  # '>' not declared for this column


def test_thousands_separator_signature_matches_the_documented_example():
    # design.md's own example: vpc_per_hour plausible_range [0, 5000],
    # corrupted value 4970833333.
    assert looks_like_thousands_separator_corruption("4970833333", (0, 5000))


def test_thousands_separator_signature_does_not_fire_on_ordinary_large_values():
    assert not looks_like_thousands_separator_corruption("4200", (0, 5000))
    assert not looks_like_thousands_separator_corruption("31.5", (0, 5000))  # has a decimal point


def test_normalize_boolean_recognises_common_forms():
    assert normalize_boolean("Yes") == 1.0
    assert normalize_boolean("no") == 0.0
    assert normalize_boolean("Male") == 1.0
    assert normalize_boolean("F") == 0.0
    assert normalize_boolean("maybe") is None


def test_detect_categorical_columns_flags_text_not_censored_bounds():
    columns = {
        "sex": ["Male", "Female", "Male"],
        "age": ["45", ">89", "60"],  # censored bound must NOT make this categorical
    }
    result = detect_categorical_columns(columns, SENTINELS, {})
    assert "sex" in result
    assert result["sex"] == ["Female", "Male"]
    assert "age" not in result  # FR-029: the whole point of censored-bound handling


def test_detect_constant_or_missing_columns():
    columns = {
        "study_arm": ["A", "A", "A"],
        "notes": ["", "N/A", ""],
        "age": ["45", "60", "70"],
    }
    result = detect_constant_or_missing_columns(columns, SENTINELS)
    assert result["study_arm"] == "constant (no variance)"
    assert result["notes"] == "all values missing"
    assert "age" not in result


def test_high_cardinality_guard():
    categorical = {"time_of_day": [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]}  # 48 categories
    offenders = high_cardinality_columns(categorical, n_rows=100, max_categories=50, declared_transforms=set())
    assert "time_of_day" in offenders  # min(50, 0.05*100=5) = 5 < 48


def test_high_cardinality_guard_exempts_declared_transform():
    categorical = {"admission_time": [f"{h:02d}:00" for h in range(24)]}
    offenders = high_cardinality_columns(categorical, n_rows=1000, max_categories=50, declared_transforms={"admission_time"})
    assert "admission_time" not in offenders
