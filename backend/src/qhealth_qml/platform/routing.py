"""Routes a DataBundle to every compatible registered model (FR-009, FR-010, FR-011).

See design doc section 5 for the normative algorithm this implements. The order of the
checks below is deliberate: the first matching branch wins, and every branch sets a
non-empty reason so a rejected/incompatible/insufficient-data model always explains itself.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from .registry import Registry
from .schema import DataBundle, Modality, ModelDefinition, RoutingDecision


def route(
    bundle: DataBundle,
    registry: Registry,
    domain: str = "neurological",
    *,
    preview: bool = False,
) -> list[RoutingDecision]:
    """Return one RoutingDecision per registered model in `domain`.

    `preview=True` downgrades a `"ready"` decision to `"compatible"` (used by
    `POST /api/bundle` to answer "could this model ever run on data shaped like this"
    without implying "it will run now"); every other status is unchanged.
    """

    condition_domains = {c.condition_id: c.domain for c in registry.conditions()}
    present = {a.modality for a in bundle.assets if a.validation_status == "accepted"}
    quality_failed = {a.modality for a in bundle.assets if a.validation_status == "quality_failed"}
    rejected = {a.modality for a in bundle.assets if a.validation_status == "rejected"}

    decisions: list[RoutingDecision] = []
    for model in registry.models():
        if condition_domains.get(model.condition_id) != domain:
            continue
        decision = _route_one(model, bundle, present, quality_failed, rejected)
        if preview and decision.status == "ready":
            decision = dataclasses.replace(decision, status="compatible")
        decisions.append(decision)
    return decisions


def _route_one(
    model: ModelDefinition,
    bundle: DataBundle,
    present: set[Modality],
    quality_failed: set[Modality],
    rejected: set[Modality],
) -> RoutingDecision:
    contract = model.input_contract
    satisfied = sorted(present & set(contract.required_modalities))

    def decision(
        status: str,
        reason: str,
        *,
        missing_required: Iterable[Modality] = (),
        missing_optional: Iterable[Modality] = (),
        unmet: Iterable[str] = (),
    ) -> RoutingDecision:
        return RoutingDecision(
            model_id=model.model_id,
            model_version=model.version,
            condition_id=model.condition_id,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            satisfied_modalities=satisfied,
            missing_required=list(missing_required),
            missing_optional=list(missing_optional),
            unmet_constraints=list(unmet),
        )

    # 1. No artifact -> the model cannot run, and it is not the case's fault.
    if model.availability == "not available" or model.artifact.path is None:
        return decision(
            "not available",
            f"no registered model artifact for {model.model_id} v{model.version}",
        )

    # 2. Required modality absent entirely.
    missing_required = [m for m in contract.required_modalities if m not in present]
    if missing_required:
        if any(m in quality_failed or m in rejected for m in missing_required):
            return decision(
                "insufficient data",
                f"{', '.join(missing_required)} failed validation and cannot be used",
                missing_required=missing_required,
            )
        return decision(
            "incompatible",
            f"requires {', '.join(missing_required)}; not present in this case",
            missing_required=missing_required,
        )

    # 3. Optional modality absent AND the model was not trained for its absence.
    untrained_gaps = [
        opt.modality
        for opt in contract.optional_modalities
        if opt.modality not in present and not opt.trained_for_absence
    ]
    if untrained_gaps:
        return decision(
            "insufficient data",
            f"{', '.join(untrained_gaps)} is absent and this model was not evaluated for its absence",
            missing_optional=untrained_gaps,
        )

    # 4. Full-coverage models reject any required-modality QC failure even if a
    #    substitute asset of the same modality was accepted.
    coverage_conflict = sorted(set(contract.required_modalities) & quality_failed)
    if model.safety.requires_full_coverage and coverage_conflict:
        return decision(
            "insufficient data",
            f"{', '.join(coverage_conflict)} failed quality control",
            missing_required=coverage_conflict,
        )

    # 5. Field-level and row-count contract. Population filters and quality
    # constraints (e.g. "age >= 18") are per-row predicates that require the
    # asset's actual data, which a ModalityAsset does not carry (it references
    # a file/URI, not embedded rows) — those are enforced at execution time in
    # execution.py, not here. Routing only checks what the bundle's provenance
    # (field_mappings, rows) can answer without reading the source file again.
    unmet = _check_required_fields(bundle, contract.required_fields, contract.min_rows)
    if unmet:
        return decision("insufficient data", "; ".join(unmet), unmet=unmet)

    # 6. Contract satisfied.
    return decision("ready", "input contract satisfied")


def _check_required_fields(bundle: DataBundle, required_fields: list[str], min_rows: int) -> list[str]:
    available_fields: set[str] = set()
    max_rows = 0
    for asset in bundle.assets:
        if asset.validation_status != "accepted":
            continue
        available_fields.update(mapping.canonical_field for mapping in asset.field_mappings)
        if asset.rows is not None:
            max_rows = max(max_rows, asset.rows)

    unmet: list[str] = []
    missing_fields = [name for name in required_fields if name not in available_fields]
    if missing_fields:
        unmet.append(f"missing required fields: {', '.join(missing_fields)}")
    if min_rows and max_rows < min_rows:
        unmet.append(f"requires at least {min_rows} row(s), found {max_rows}")
    return unmet
