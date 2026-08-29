"""The single choke point for producing a clinically meaningful Finding (FR-011, FR-018,
FR-030, SC-003, SC-011). No other module may construct a Finding whose status is
"positive", "negative", or "abstained" — see `finalize_finding`. Every other status comes
from `not_evaluated`, which cannot carry a score.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any
from uuid import uuid4

from .schema import (
    ConditionDefinition,
    CoverageReport,
    Finding,
    ModelDefinition,
    RoutingDecision,
    Uncertainty,
)

DISCLAIMER = (
    "Research prediction, not a medical diagnosis. This platform does not provide "
    "diagnosis, treatment recommendations, or triage decisions."
)
SYNTHETIC_MARKER = "SYNTHETIC / DEMO OUTPUT — not model inference and not clinical evidence."

# RoutingDecision statuses that never carry a score. finalize_finding handles the
# remaining two terminal statuses ("completed", "abstained") that may carry one.
_NOT_EVALUATED_STATUS_MAP: dict[str, str] = {
    "not available": "not available",
    "incompatible": "not evaluated",
    "insufficient data": "insufficient data",
    "failed": "not evaluated",
}


class SafetyError(Exception):
    """Raised when a caller tries to construct a Finding that violates a safety invariant."""


def finalize_finding(
    *,
    model: ModelDefinition,
    condition: ConditionDefinition,
    decision: RoutingDecision,
    coverage: CoverageReport,
    raw: dict[str, Any],
    run_id: str,
    synthetic: bool,
) -> Finding:
    """Construct a `positive`/`negative`/`abstained` Finding. The only place allowed to.

    `raw` is the executor's result dict for one case, expected to carry at least
    `score` (float | None) and `abstained` (bool), and optionally `threshold`,
    `threshold_policy`, `uncertainty` (an `Uncertainty` instance), `evidence`
    (a list of `EvidenceItem`), `explanation_status`, `output`, `modalities_used`,
    and `reason` (used only for the abstained branch).
    """

    if decision.status not in ("completed", "abstained"):
        raise SafetyError(
            f"finalize_finding requires decision.status in ('completed', 'abstained'), "
            f"got {decision.status!r} for model {model.model_id}; use not_evaluated() instead"
        )
    if model.artifact.path is None:
        raise SafetyError(f"model {model.model_id} has no artifact; cannot finalize a scored finding")
    if coverage.required_present != coverage.required_total or coverage.quality_failed:
        raise SafetyError(
            f"model {model.model_id}: input coverage is incomplete or quality-failed "
            f"({coverage.required_present}/{coverage.required_total} required, "
            f"quality_failed={coverage.quality_failed}); a scored finding requires full coverage"
        )

    score = raw.get("score")
    abstained = bool(raw.get("abstained", False))

    if decision.status == "abstained" or abstained:
        status = "abstained"
        reason = raw.get("reason") or "Model abstained: score too close to the decision threshold."
    else:
        if score is None:
            raise SafetyError(
                f"model {model.model_id}: decision.status is 'completed' but raw['score'] is None"
            )
        threshold = raw.get("threshold")
        if threshold is None:
            raise SafetyError(
                f"model {model.model_id}: decision.status is 'completed' but no threshold was supplied"
            )
        status = "positive" if score >= threshold else "negative"
        if status == "negative" and not model.safety.allows_negative_finding:
            raise SafetyError(
                f"model {model.model_id}: computed status 'negative' but "
                "safety.allows_negative_finding is False for this model"
            )
        reason = ""

    return _build_finding(
        model=model,
        condition=condition,
        run_id=run_id,
        status=status,
        reason=reason,
        score=score,
        threshold=raw.get("threshold"),
        threshold_policy=raw.get("threshold_policy"),
        output=raw.get("output") or {},
        uncertainty=raw.get("uncertainty")
        or Uncertainty(
            kind="none", value=None, lower=None, upper=None, calibration_status=model.calibration.status
        ),
        abstained=(status == "abstained"),
        input_coverage=coverage,
        modalities_used=list(raw.get("modalities_used") or model.input_contract.required_modalities),
        evidence=list(raw.get("evidence") or []),
        explanation_status=raw.get("explanation_status")
        or ("available" if model.explainability.method != "none" else "unavailable"),
        synthetic=synthetic,
    )


def not_evaluated(
    model: ModelDefinition,
    condition: ConditionDefinition,
    decision: RoutingDecision,
    run_id: str,
    reason: str,
    synthetic: bool,
) -> Finding:
    """Construct a Finding for a model that did not produce a score.

    Maps RoutingDecision.status through the FR-011/SC-003 safety table: `incompatible`
    and `failed` both become `"not evaluated"`; `not available` and `insufficient data`
    pass through unchanged. Never carries a score.
    """

    if decision.status not in _NOT_EVALUATED_STATUS_MAP:
        raise SafetyError(
            f"not_evaluated() cannot handle decision.status={decision.status!r}; "
            "use finalize_finding() for 'completed'/'abstained'"
        )
    finding_status = _NOT_EVALUATED_STATUS_MAP[decision.status]
    required_total = len(decision.satisfied_modalities) + len(decision.missing_required)
    coverage = CoverageReport(
        required_present=len(decision.satisfied_modalities),
        required_total=required_total,
        optional_present=0,
        optional_total=len(decision.missing_optional),
        coverage_ratio=(
            len(decision.satisfied_modalities) / required_total if required_total else 0.0
        ),
        missing=list(decision.missing_required) + list(decision.missing_optional),
        quality_failed=[],
    )
    return _build_finding(
        model=model,
        condition=condition,
        run_id=run_id,
        status=finding_status,
        reason=reason,
        score=None,
        threshold=None,
        threshold_policy=None,
        output={},
        uncertainty=Uncertainty(
            kind="none", value=None, lower=None, upper=None, calibration_status="not_assessed"
        ),
        abstained=False,
        input_coverage=coverage,
        modalities_used=list(decision.satisfied_modalities),
        evidence=[],
        explanation_status="not_applicable",
        synthetic=synthetic,
    )


def _build_finding(
    *,
    model: ModelDefinition,
    condition: ConditionDefinition,
    run_id: str,
    status: str,
    reason: str,
    score: float | None,
    threshold: float | None,
    threshold_policy: str | None,
    output: dict[str, Any],
    uncertainty: Uncertainty,
    abstained: bool,
    input_coverage: CoverageReport,
    modalities_used: list[str],
    evidence: list[Any],
    explanation_status: str,
    synthetic: bool,
) -> Finding:
    if status not in ("positive", "negative") and not reason:
        raise SafetyError(f"model {model.model_id}: Finding.status={status!r} requires a non-empty reason")

    disclaimer = model.safety.disclaimer or DISCLAIMER
    if synthetic:
        disclaimer = f"{SYNTHETIC_MARKER} {disclaimer}"

    finding_id = str(uuid4())
    evidence = [replace(item, finding_id=finding_id) for item in evidence]

    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        model_id=model.model_id,
        model_version=model.version,
        condition_id=model.condition_id,
        condition_name=condition.name,
        task_type=model.task_type,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        score=score,
        score_type=model.output_score_type,
        output=output,
        threshold=threshold,
        threshold_policy=threshold_policy,
        uncertainty=uncertainty,
        abstained=abstained,
        reference_label_tier=condition.reference_label_tier,
        input_coverage=input_coverage,
        modalities_used=modalities_used,  # type: ignore[arg-type]
        evidence=evidence,
        explanation_status=explanation_status,  # type: ignore[arg-type]
        limitations=list(model.safety.limitations),
        disclaimer=disclaimer,
        synthetic=bool(synthetic),
    )


_QUOTED_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")
_POSITIONAL_NUMBER_RE = re.compile(r"\b(row|line|index|record)\s+(\d+)\b", re.IGNORECASE)


def redact(message: str, *, allowed_fields: set[str]) -> str:
    """Strip likely source values from an error/log message (FR-030, SC-011).

    Permits canonical field names (from `allowed_fields`), asset/model identifiers, and
    generic text; replaces quoted values not in `allowed_fields` and row/line/index
    numbers with `<redacted>`. This is a defensive heuristic, not a guarantee for
    arbitrary free text — callers must still avoid interpolating raw source values into
    messages in the first place.
    """

    def _quote_sub(match: re.Match[str]) -> str:
        content = match.group(2)
        if content in allowed_fields:
            return match.group(0)
        return "<redacted>"

    text = _QUOTED_RE.sub(_quote_sub, message)
    text = _POSITIONAL_NUMBER_RE.sub(lambda m: f"{m.group(1)} <redacted>", text)
    return text


def fingerprints(bundle: Any, model: ModelDefinition, artifact: Any) -> dict[str, str]:
    """Best-effort model/dataset/preprocessing/package/backend/input fingerprints (FR-028).

    `artifact` is whatever the executor loaded to run the model (e.g. a
    `SavedModelArtifact`-shaped object/dict); its exact fields are engine-owned, so this
    reads them defensively rather than importing experiment.py (this module makes no
    engine calls).
    """

    def _dig(obj: Any, *path: str, default: str = "unknown") -> str:
        for key in path:
            if obj is None:
                return default
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key, None)
        return str(obj) if obj is not None else default

    input_fingerprint = "no-input-assets"
    if getattr(bundle, "content_hashes", None):
        input_fingerprint = ",".join(sorted(bundle.content_hashes.values()))

    return {
        "model": f"{model.model_id}@{model.version}",
        "dataset": _dig(artifact, "dataset", "fingerprint"),
        "preprocessing": _dig(artifact, "preprocessor", "reduction", default=model.preprocessing.reduction),
        "package": _dig(artifact, "software", "package_version"),
        "backend": model.quantum.backend_mode if model.quantum else "classical",
        "input": input_fingerprint,
    }
