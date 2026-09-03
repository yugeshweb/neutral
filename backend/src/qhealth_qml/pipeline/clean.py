"""Cleaning: sentinels, censored bounds, quarantine, categorical detection,
unit/boolean normalisation, constant/all-missing dropping.

Implements FR-028 through FR-047 (design.md §9.2). Two kinds of function
live here, deliberately kept apart per spec.md's stateless/fitted line:

- **Per-value, stateless** (`map_sentinel`, `parse_censored_bound`,
  `normalize_boolean`): pure functions of one raw string. Called from
  `harmonize()`, which reads exactly one record (FR-004).
- **Per-column, fit-time** (`detect_categorical_columns`,
  `detect_quarantine_columns`, `detect_constant_or_missing_columns`):
  read the whole training column to make one decision, frozen into the
  `Recipe` and never re-decided at predict time (FR-041).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .types import Issue, IssueCode

_CENSOR_PATTERN = re.compile(r"^\s*([><≥≤])\s*(-?[0-9]+\.?[0-9]*)\s*$")
_LONG_DIGIT_RUN = re.compile(r"^-?\d{7,}$")  # no decimal point, 7+ digits


@dataclass(frozen=True)
class CensoredBound:
    value: float
    token: str


def map_sentinel(raw: str, sentinels: tuple[str, ...]) -> bool:
    """True if `raw` (already stripped) is a declared missing-value
    sentinel, matched case-insensitively (FR-028)."""

    return raw.strip().lower() in sentinels


def parse_censored_bound(raw: str, censor_tokens: tuple[str, ...]) -> CensoredBound | None:
    """A value carrying a declared censoring token prefix (`>`, `<`, `≥`,
    `≤`) is a censored bound, not a category (FR-029). Returns the parsed
    numeric bound so the caller can clamp it to the column's declared
    `value_ceiling`; returns None for anything else, including a bare
    numeric string."""

    match = _CENSOR_PATTERN.match(raw)
    if not match:
        return None
    token, number = match.groups()
    if token not in censor_tokens:
        return None
    try:
        return CensoredBound(value=float(number), token=token)
    except ValueError:
        return None


def normalize_boolean(raw: str) -> float | None:
    """Booleans, yes/no, M/F normalised to 0/1 (FR-045). Returns None for
    anything not recognised, so the caller falls through to normal
    categorical/numeric handling rather than guessing."""

    lower = raw.strip().lower()
    truthy = {"true", "yes", "y", "1", "m", "male"}
    falsy = {"false", "no", "n", "0", "f", "female"}
    if lower in truthy:
        return 1.0
    if lower in falsy:
        return 0.0
    return None


def looks_like_thousands_separator_corruption(
    raw: str, plausible_range: tuple[float, float] | None
) -> bool:
    """FR-038: a long, decimal-point-free digit run that blows past a
    column's declared plausible ceiling by two orders of magnitude is the
    signature this repo's own audit found (design.md, `vpc_per_hour`) - an
    exporter that stripped a decimal point and thousands separators from a
    normal-scale value. Quarantines regardless of the normal fraction
    threshold (FR-038 is unconditional, unlike FR-037)."""

    if not _LONG_DIGIT_RUN.match(raw.strip()):
        return False
    if plausible_range is None:
        return False
    try:
        value = float(raw)
    except ValueError:
        return False
    _, ceiling = plausible_range
    return abs(value) > max(ceiling, 1.0) * 100


def detect_categorical_columns(
    columns: dict[str, list[str]],
    sentinels: tuple[str, ...],
    type_overrides: dict[str, str],
) -> dict[str, list[str]]:
    """A column is categorical if any non-sentinel, non-censored value
    fails `float()` (existing, kept behaviour per design.md §9.2.3).
    Returns column -> sorted distinct categories seen. This is a FIT-TIME
    decision: called once on the training split, the result is frozen into
    `Recipe.categorical_vocab` and never recomputed at predict time."""

    result: dict[str, list[str]] = {}
    for name, values in columns.items():
        override = type_overrides.get(name)
        if override == "numeric":
            continue
        seen: set[str] = set()
        is_categorical = override == "categorical"
        for raw in values:
            stripped = raw.strip()
            if not stripped or map_sentinel(stripped, sentinels):
                continue
            if _CENSOR_PATTERN.match(stripped):
                continue
            try:
                float(stripped)
            except ValueError:
                is_categorical = True
            seen.add(stripped)
        if is_categorical:
            result[name] = sorted(seen)
    return result


def detect_quarantine_columns(
    columns: dict[str, list[str]],
    sentinels: tuple[str, ...],
    plausible_ranges: dict[str, tuple[float, float]],
    quarantine_threshold: float,
    categorical_columns: set[str],
) -> dict[str, str]:
    """FR-037, FR-038: quarantine a numeric column whose implausible
    fraction exceeds `quarantine_threshold`, or which shows the
    thousands-separator corruption signature even once. Never runs on a
    column already detected categorical - implausibility is a numeric-
    column concept."""

    quarantined: dict[str, str] = {}
    for name, values in columns.items():
        if name in categorical_columns:
            continue
        plausible = plausible_ranges.get(name)
        total = 0
        implausible = 0
        for raw in values:
            stripped = raw.strip()
            if not stripped or map_sentinel(stripped, sentinels):
                continue
            if looks_like_thousands_separator_corruption(stripped, plausible):
                quarantined[name] = "thousands-separator corruption signature"
                break
            if plausible is None:
                continue
            try:
                value = float(stripped)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            total += 1
            lo, hi = plausible
            if value < lo or value > hi:
                implausible += 1
        if name in quarantined:
            continue
        if plausible is not None and total > 0 and (implausible / total) > quarantine_threshold:
            quarantined[name] = (
                f"{implausible}/{total} values ({implausible / total:.1%}) outside "
                f"plausible_range {plausible}, exceeding threshold {quarantine_threshold:.1%}"
            )
    return quarantined


def detect_constant_or_missing_columns(
    columns: dict[str, list[str]],
    sentinels: tuple[str, ...],
) -> dict[str, str]:
    """FR-040: drop a column with zero variance or zero observed values.
    A single-category "categorical" column (every row the same string) is
    constant too, not a one-column one-hot expansion nobody asked for."""

    dropped: dict[str, str] = {}
    for name, values in columns.items():
        seen: set[str] = set()
        any_present = False
        for raw in values:
            stripped = raw.strip()
            if not stripped or map_sentinel(stripped, sentinels):
                continue
            any_present = True
            seen.add(stripped)
        if not any_present:
            dropped[name] = "all values missing"
        elif len(seen) <= 1:
            dropped[name] = "constant (no variance)"
    return dropped


def high_cardinality_columns(
    categorical_columns: dict[str, list[str]],
    n_rows: int,
    max_categories: int,
    declared_transforms: set[str],
) -> dict[str, int]:
    """FR-044: a categorical column whose cardinality exceeds
    `min(max_categories, 5% of rows)` and has no declared `transform` is a
    specification error, not a silent 500-dummy-column expansion."""

    guard = min(max_categories, max(1, round(0.05 * n_rows)))
    offenders: dict[str, int] = {}
    for name, categories in categorical_columns.items():
        if name in declared_transforms:
            continue
        if len(categories) > guard:
            offenders[name] = len(categories)
    return offenders
