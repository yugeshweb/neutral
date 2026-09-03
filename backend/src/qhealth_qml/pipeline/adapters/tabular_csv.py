"""`tabular_csv` adapter: CSV/TSV. Ports the parse logic and conventions
from `experiment.load_csv_dataset` (FR-013, FR-016) into the new
stream-one-record-at-a-time shape (design.md §8.3).

**What `harmonize()` does NOT do, on purpose**: it does not one-hot expand
a categorical column. `Sample.fields["sex"]` stays the raw string
`"Female"` - the frozen vocabulary that turns it into `sex=Female` /
`sex=Male` columns lives in the `Recipe` and is applied by `Recipe.align()`
(design.md §7.1), because that vocabulary can only be known after fitting.
`harmonize()` is stateless (FR-004): it reads exactly one row plus the
spec, so anything requiring the whole training column cannot happen here.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Iterator

from ..clean import looks_like_thousands_separator_corruption, map_sentinel, normalize_boolean, parse_censored_bound
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source


class TabularCsvAdapter:
    name = "tabular_csv"
    modalities = ("tabular",)
    formats = (".csv", ".tsv")

    def sniff(self, source: Source) -> float:
        text = _peek_text(source)
        if text is None:
            return 0.0
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if "," in first_line or "\t" in first_line:
            return 0.7
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        text, digest = _read_text_and_digest(source)
        delimiter = "\t" if source.locator.endswith(".tsv") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV/TSV has no header row")

        _validate_columns_present(fieldnames, spec)

        for row_number, row in enumerate(reader, start=2):
            yield RawRecord(
                source=source,
                payload=dict(row),
                meta={"row": row_number, "sha256": digest, "fieldnames": fieldnames},
            )

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        row: dict[str, Any] = dict(raw.payload)
        fieldnames: list[str] = raw.meta["fieldnames"]
        issues: list[Issue] = []
        applied: list[str] = []

        excluded = {
            spec.target_column, spec.id_column, spec.group_column,
            spec.index_time_column, spec.outcome_time_column, spec.site_column,
            *spec.subgroup_columns, *spec.leakage_columns,
        }
        feature_columns = [c for c in fieldnames if c not in excluded and c is not None]

        fields: dict[str, Any] = {}
        for name in feature_columns:
            raw_value = str(row.get(name, "")).strip()
            column_spec = spec.columns.get(name)
            sentinels = spec.missing_sentinels
            censor_tokens = column_spec.censor_tokens if column_spec else (">", "<", "≥", "≤")

            if not raw_value or map_sentinel(raw_value, sentinels):
                fields[name] = float("nan")
                if raw_value:
                    issues.append(
                        Issue(IssueCode.SENTINEL_TO_NAN, "info", f"'{name}' value {raw_value!r} was treated as missing.", field=name)
                    )
                applied.append("sentinel_to_nan")
                continue

            censored = parse_censored_bound(raw_value, censor_tokens)
            if censored is not None:
                ceiling = column_spec.value_ceiling if column_spec and column_spec.value_ceiling is not None else censored.value
                fields[name] = ceiling
                fields[f"{name}__censored"] = 1.0
                issues.append(
                    Issue(
                        IssueCode.SENTINEL_CLAMPED, "warn",
                        f"'{name}' value '{raw_value}' is a censored bound; clamped to the declared ceiling of {ceiling}.",
                        field=name,
                    )
                )
                applied.append("censored_bound_clamped")
                continue

            boolean = normalize_boolean(raw_value) if column_spec and column_spec.type_override != "categorical" else None
            if boolean is not None and raw_value.lower() not in ("0", "1"):
                fields[name] = boolean
                applied.append("boolean_normalized")
                continue

            try:
                numeric = float(raw_value)
                fields[name] = numeric
                plausible = column_spec.plausible_range if column_spec else None
                if looks_like_thousands_separator_corruption(raw_value, plausible):
                    # Per-record SIGNAL, not a per-record decision: quarantine
                    # is a column-level, fit-time call (FR-038) made by
                    # aggregating this issue across the training batch, never
                    # by one row acting alone.
                    issues.append(
                        Issue(
                            "thousands_separator_signature_candidate", "info",
                            f"'{name}'={raw_value} matches the thousands-separator corruption "
                            f"signature (a long digit run with no decimal point, far past the "
                            f"declared plausible ceiling).",
                            field=name,
                        )
                    )
                elif plausible:
                    lo, hi = plausible
                    if not (lo <= numeric <= hi):
                        issues.append(
                            Issue(IssueCode.IMPLAUSIBLE_VALUE, "info", f"'{name}'={numeric} is outside plausible_range {plausible}.", field=name)
                        )
            except ValueError:
                fields[name] = raw_value  # left raw; categorical-ness decided at fit time

        target_raw = str(row.get(spec.target_column, "")).strip() if spec.target_column in fieldnames else None
        label = None
        if target_raw:
            label = 1 if target_raw == spec.positive_label else 0

        sample_id = str(row.get(spec.id_column, "")).strip() or f"row-{raw.meta['row'] - 1}"
        subject_raw = str(row.get(spec.group_column, "")).strip() if spec.group_column else None
        subgroups = {c: str(row.get(c, "")).strip() for c in spec.subgroup_columns if row.get(c)}

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(subject_raw) if subject_raw else None,
            index_time=str(row.get(spec.index_time_column, "")).strip() or None if spec.index_time_column else None,
            outcome_time=str(row.get(spec.outcome_time_column, "")).strip() or None if spec.outcome_time_column else None,
            site=str(row.get(spec.site_column, "")).strip() or None if spec.site_column else None,
            label=label,
            fields=fields,
            arrays={},
            subgroups=subgroups,
            provenance={
                "adapter": self.name,
                "source": {"path": source_locator(raw.source), "sha256": raw.meta.get("sha256"), "row": raw.meta["row"]},
                "applied": applied,
            },
            issues=issues,
        )

    def qc(self, sample: Sample, spec: SourceSpec) -> QCVerdict:
        # Tabular has no modality-specific checks beyond the universal
        # QCGate - returning a clean accept here; QCGate runs separately.
        return QCVerdict(status="accept", issues=[])


def source_locator(source: Source) -> str:
    return source.locator


def _hash_subject_id(raw_subject_id: str) -> str:
    """FR-129: the subject identifier is hashed before it enters the
    canonical record. A fixed, unsalted hash here (not a secret; this is
    hygiene per FR-131, not de-identification) still guarantees the same
    subject always maps to the same hash within one dataset, which is all
    grouped splitting needs."""

    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


def _peek_text(source: Source) -> str | None:
    try:
        if source.kind == "bytes" and source.payload is not None:
            return source.payload.decode("utf-8-sig", errors="ignore")
        if source.kind == "path":
            return Path(source.locator).read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    except Exception:
        return None
    return None


def _read_text_and_digest(source: Source) -> tuple[str, str]:
    if source.kind == "bytes" and source.payload is not None:
        raw_bytes = source.payload
    elif source.kind == "path":
        raw_bytes = Path(source.locator).read_bytes()
    else:
        raise ValueError(f"tabular_csv adapter cannot read source kind {source.kind!r}")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return raw_bytes.decode("utf-8-sig"), digest


def _validate_columns_present(fieldnames: list[str], spec: SourceSpec) -> None:
    # target_column is deliberately NOT required here: its absence is what
    # distinguishes a predict-time upload from a training upload (mirrors
    # `standardize.py`'s has_target check) - only id/group/time/site
    # columns and declared required_fields must actually be present.
    required: list[str] = []
    for name, col in (
        ("id_column", spec.id_column), ("group_column", spec.group_column),
        ("index_time_column", spec.index_time_column), ("outcome_time_column", spec.outcome_time_column),
        ("site_column", spec.site_column),
    ):
        if col:
            required.append(col)
    if spec.required_fields:
        required.extend(spec.required_fields)
    missing = [c for c in required if c and c not in fieldnames]
    if missing:
        raise ValueError(f"declared column(s) {missing} not present in source header {fieldnames}")
