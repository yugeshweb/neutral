"""`QCGate` - the universal quality-control checks (design.md §9.3, T006).

Implements FR-075, FR-076, FR-077, FR-080. Runs against one `Sample` at a
time (stateless: reads no other sample) and returns exactly one
`QCVerdict`.

This intentionally re-implements the same small population-filter and
quality-constraint expression grammar `platform/execution.py` already has
(`_POPULATION_FILTER_RE`/`_QUALITY_RANGE_RE`), rather than importing its
private helpers - Phase 0-3 of this build calls existing modules without
editing them (tasks.md's ownership note); consolidating onto this single
implementation is Phase 4 task T068, done once, deliberately, with its own
gate - not smuggled in early by a cross-module import of underscored
names.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .types import Issue, IssueCode, QCVerdict, Sample

_COMPARISONS = {
    ">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t, ">": lambda v, t: v > t, "<": lambda v, t: v < t,
}
_POPULATION_FILTER_RE = re.compile(r"\s*(>=|<=|==|!=|>|<)\s*(-?\d+\.?\d*)")
_QUALITY_RANGE_RE = re.compile(r"(-?\d+\.?\d*)\s*<=\s*\S+\s*<=\s*(-?\d+\.?\d*)")


@dataclass(frozen=True)
class QCContract:
    """The universal-check inputs a `SourceSpec` supplies. Kept as its own
    small type so `QCGate.check` doesn't need the full `SourceSpec` (and
    so a fixture-only unit test can build one without a whole spec)."""

    required_fields: tuple[str, ...] = ()
    population_filters: dict[str, str] = field(default_factory=dict)
    quality_constraints: dict[str, str] = field(default_factory=dict)
    min_required_coverage: float = 1.0


class QCGate:
    """Runs universal checks. A modality adapter's own `qc()` runs its
    signal/imaging checks on top of this, per design.md §9.3 - this gate
    never needs to know about leads or voxels."""

    def check(self, sample: Sample, contract: QCContract) -> QCVerdict:
        issues: list[Issue] = []

        # required field blank or missing -> reject, field named (FR-030,
        # FR-076). Checked first and independently per field, so a record
        # missing two required fields is refused naming both, not just one.
        missing_required = [
            name
            for name in contract.required_fields
            if _is_blank(sample.fields.get(name))
        ]
        for name in missing_required:
            issues.append(
                Issue(
                    IssueCode.REQUIRED_FIELD_MISSING,
                    "reject",
                    f"Required field '{name}' is blank and will not be imputed.",
                    field=name,
                )
            )

        # coverage threshold (FR-035) - only meaningful when there are
        # required fields to have coverage over.
        if contract.required_fields and contract.min_required_coverage < 1.0:
            present = sum(1 for n in contract.required_fields if not _is_blank(sample.fields.get(n)))
            coverage = present / len(contract.required_fields)
            if coverage < contract.min_required_coverage:
                issues.append(
                    Issue(
                        IssueCode.INSUFFICIENT_COVERAGE,
                        "reject",
                        f"Required-field coverage {coverage:.0%} is below the declared "
                        f"threshold {contract.min_required_coverage:.0%}.",
                        detail={"coverage": coverage, "threshold": contract.min_required_coverage},
                    )
                )

        # population filters (e.g. age >= 18) (FR-076).
        for name, expr in contract.population_filters.items():
            value = sample.fields.get(name)
            if _is_blank(value):
                continue  # absence is reported via required-field checks, not here
            match = _POPULATION_FILTER_RE.match(expr)
            if not match:
                continue
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            op, threshold = match.group(1), float(match.group(2))
            if not _COMPARISONS[op](numeric, threshold):
                issues.append(
                    Issue(
                        IssueCode.POPULATION_FILTER_UNMET,
                        "reject",
                        f"'{name}'={numeric} fails population filter '{expr}'.",
                        field=name,
                    )
                )

        # quality constraints (e.g. 0 <= age <= 120) (FR-076).
        for name, expr in contract.quality_constraints.items():
            value = sample.fields.get(name)
            if _is_blank(value):
                continue
            bounds = _QUALITY_RANGE_RE.search(expr)
            if not bounds:
                continue
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            low, high = float(bounds.group(1)), float(bounds.group(2))
            if not (low <= numeric <= high):
                issues.append(
                    Issue(
                        IssueCode.QUALITY_CONSTRAINT_UNMET,
                        "reject",
                        f"'{name}'={numeric} outside quality constraint '{expr}'.",
                        field=name,
                    )
                )

        # non-finite values among the declared fields (FR-076 spirit;
        # FR-055/imaging equivalents live in the adapter's own qc()).
        for name, value in sample.fields.items():
            if isinstance(value, float) and not math.isnan(value) and not math.isfinite(value):
                issues.append(
                    Issue(
                        IssueCode.NON_FINITE_INPUT,
                        "reject",
                        f"'{name}' is non-finite (inf/-inf), which no imputer can repair.",
                        field=name,
                    )
                )

        if any(issue.severity == "reject" for issue in issues):
            return QCVerdict(status="reject", issues=issues)
        if issues:
            return QCVerdict(status="accept_with_flags", issues=issues)
        return QCVerdict(status="accept", issues=[])


def check_duplicate_sample_ids(sample_ids: list[str]) -> list[Issue]:
    """FR-027, dataset-level (not per-sample): duplicate identifiers are a
    `warn`, reported once per duplicated id, not once per occurrence."""

    seen: dict[str, int] = {}
    for sid in sample_ids:
        seen[sid] = seen.get(sid, 0) + 1
    return [
        Issue(
            IssueCode.DUPLICATE_SAMPLE_ID,
            "warn",
            f"sample_id '{sid}' appears {count} times.",
            field="sample_id",
            detail={"count": count},
        )
        for sid, count in seen.items()
        if count > 1
    ]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False
