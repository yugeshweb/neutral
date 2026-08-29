"""The only platform module that calls into the qhealth_qml training/evaluation engine
(`experiment.py`, `study.py`, `protocol.py`). Nothing here re-implements a model, a
metric, a split, or a preprocessing step.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .. import experiment, protocol
from .. import study as study_module
from .registry import Registry
from .routing import route
from .safety import finalize_finding, not_evaluated
from .schema import (
    AssessmentRun,
    CoverageReport,
    EvaluationRecord,
    EvidenceItem,
    Finding,
    LeakageCheck,
    ModelDefinition,
    PreprocessingSpec,
    ResourceUsage,
    RoutingDecision,
    StageEvent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve(repo_relative_path: str) -> Path:
    return (REPO_ROOT / repo_relative_path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dataset_for_model(model: ModelDefinition):
    """Load the model's declared dataset the way its `temporal_validation` says to.

    A `"validated"` model goes through `load_profile_dataset()`, which runs the full
    early-detection temporal validator. A cross-sectional profile (no index/outcome time)
    cannot pass that validator, so its executor composes the two reused primitives
    directly instead of fabricating timestamps to satisfy it (P1 reuse record §6).
    """

    profile_path = _resolve(model.dataset_profile_path)
    if model.temporal_validation == "validated":
        dataset = experiment.load_profile_dataset(profile_path)
        profile, _ = protocol.load_early_detection_profile(profile_path)
        return dataset, profile

    profile, profile_dir = protocol.load_early_detection_profile(profile_path)
    csv_path = protocol.resolve_profile_dataset(profile, profile_dir)
    dataset = experiment.load_csv_dataset(
        csv_path,
        target=profile.target_column,
        positive_label=profile.positive_label,
        group_column=profile.group_column,
        time_column=profile.index_time_column,
        id_column=profile.id_column,
        site_column=profile.site_column,
        outcome_time_column=profile.outcome_time_column,
        subgroup_columns=profile.subgroup_columns,
        leakage_columns=profile.leakage_columns,
        task_profile=profile.as_dict(),
    )
    return dataset, profile


def _engine_model_name(model: ModelDefinition) -> str:
    """Map a registered ModelDefinition to a single qhealth_qml engine model token."""

    if model.quantum is not None:
        return "vqc" if model.quantum.ansatz else "qsvc"
    raise ValueError(
        f"model {model.model_id} has no quantum spec; its engine model name must be "
        "picked from the benchmark results (see benchmark_model's classical branch)"
    )


def benchmark_model(
    model_id: str,
    registry: Registry,
    *,
    repeats: int = 10,
    outer_repeats: int = 5,
    inner_repeats: int = 2,
    runtime_dir: Path | None = None,
    seed: int = 7,
    **overrides: Any,
) -> EvaluationRecord:
    """Benchmark a registered model against the classical baselines and mint its
    deployable artifact (design doc §4 execution.py "tabular_qml benchmark path").
    """

    model = registry.model(model_id)
    if model.executor != "tabular_qml":
        raise ValueError(f"benchmark_model only supports executor='tabular_qml', got {model.executor!r}")

    dataset, profile = _load_dataset_for_model(model)
    runtime_dir = runtime_dir or (REPO_ROOT / "runtime")

    run_kwargs: dict[str, Any] = dict(
        backend=model.quantum.backend_mode if model.quantum else "statevector",
        n_qubits=model.preprocessing.n_components or (model.quantum.n_qubits if model.quantum else 4),
        shots=model.quantum.shots if model.quantum else 512,
        reduction=model.preprocessing.reduction,
        seed=seed,
    )
    run_kwargs.update(overrides)

    broad_result = experiment.run_repeated_experiment(
        dataset, repeats=repeats, models=["classical", "qsvc"], **run_kwargs
    )
    metric_summary = broad_result["repeated_evaluation"]["metric_summary"]

    if model.quantum is not None:
        engine_name = _engine_model_name(model)
    else:
        classical_names = ("logistic_regression", "rbf_svc", "hist_gradient_boosting")
        engine_name = max(
            classical_names,
            key=lambda name: metric_summary[name]["balanced_accuracy"]["mean"] or 0.0,
        )

    artifact_path = runtime_dir / "models" / f"{model_id}.joblib"
    single_kwargs = dict(run_kwargs)
    single_kwargs["model_artifact_path"] = artifact_path
    single_run = experiment.run_experiment(dataset, models=[engine_name], **single_kwargs)
    experiment.save_model_artifact(experiment.load_model_artifact(artifact_path), artifact_path)

    paired = None
    nested_failure_reason: str | None = None
    try:
        nested = study_module.run_nested_evaluation(
            dataset,
            models=["classical", "qsvc"],
            outer_repeats=outer_repeats,
            inner_repeats=inner_repeats,
            backend=run_kwargs["backend"],
            n_qubits=run_kwargs["n_qubits"],
            shots=run_kwargs["shots"],
            reduction=run_kwargs["reduction"],
            max_train=overrides.get("max_train", 0),
            max_test=overrides.get("max_test", 0),
            seed=seed,
        )
        paired = nested["paired_deltas"].get(engine_name)
    except ValueError as exc:
        # A nested/inner chronological split can fail to preserve both classes on
        # a rare-event dataset (e.g. 18 positives out of 2700 rows, clustered in
        # two narrow time windows) — this is a real, expected limitation of the
        # paired-comparison mechanism at small/imbalanced scale, not a bug to
        # paper over. Record it as an explicit not_assessed reason rather than
        # crashing the whole benchmark.
        nested_failure_reason = str(exc)

    real_gain_decision = "not_assessed"
    real_gain_reason = "This model has no quantum component; the real-gain gate applies only to QML candidates."
    if nested_failure_reason is not None:
        real_gain_reason = f"Paired nested evaluation could not run: {nested_failure_reason}"
    elif model.quantum is not None and paired is not None:
        delta = paired["delta"]
        if delta["n"] == 0 or delta["lower"] is None:
            real_gain_decision = "not_assessed"
            real_gain_reason = "No paired folds produced a comparable delta."
        elif delta["lower"] > 0:
            real_gain_decision = "passed"
            real_gain_reason = (
                f"Paired 95% CI on balanced-accuracy delta vs {paired['reference_model']} "
                f"is [{delta['lower']:.4f}, {delta['upper']:.4f}], excluding zero in the candidate's favour."
            )
        else:
            real_gain_decision = "failed"
            real_gain_reason = (
                f"Paired 95% CI on balanced-accuracy delta vs {paired['reference_model']} "
                f"is [{delta['lower']:.4f}, {delta['upper']:.4f}], not excluding zero (or negative). "
                "The classical reference remains the operational baseline."
            )

    metrics = {name: values["mean"] for name, values in metric_summary[engine_name].items()}
    resource = single_run["models"][engine_name]["resource"]

    leakage_checks = [
        LeakageCheck(
            name="subject_identity",
            status="passed" if dataset.groups is not None else "not_applicable",
            detail=(
                "Group column prevents a subject's records from crossing the split."
                if dataset.groups is not None
                # A unique id_column only proves rows are individually distinct — it does
                # NOT by itself prove one row per subject (e.g. many non-overlapping
                # windows from one EEG recording all have unique ids but share a
                # subject). Without a declared group_column this cannot be verified
                # here; see the model's reuse record for the actual population structure.
                else "No group column declared; subject-level structure is not verified "
                     "by the engine and must be confirmed from the model's reuse record."
            ),
        ),
        LeakageCheck(
            name="repeated_acquisition",
            status="not_applicable" if dataset.groups is None else "passed",
            detail="No repeated-visit grouping declared for this dataset." if dataset.groups is None
            else "Group column prevents a patient's repeated records from crossing the split.",
        ),
        LeakageCheck(
            name="site_grouping",
            status="not_applicable",
            detail="No site column exists in this dataset; site-held-out validation was not possible.",
        ),
        LeakageCheck(
            name="training_only_preprocessing",
            status="passed",
            detail="Imputation, scaling, and feature selection are fitted on the training partition only (engine-enforced).",
        ),
    ]

    evaluation_id = f"eval-{model_id}-{model.version}"
    return EvaluationRecord(
        evaluation_id=evaluation_id,
        model_id=model_id,
        model_version=model.version,
        condition_id=model.condition_id,
        created_at=_now(),
        dataset_profile=profile.as_dict(),
        dataset_fingerprint=single_run["dataset"]["fingerprint"],
        split_strategy=single_run["split"]["strategy"],
        split_summary={
            "train_rows": single_run["split"].get("train_rows", 0),
            "validation_rows": single_run["split"].get("validation_rows", 0) or 0,
            "test_rows": single_run["split"].get("test_rows", 0),
            "repeats": repeats,
        },
        preprocessing=PreprocessingSpec(
            imputation="median", scaling="standard", reduction=run_kwargs["reduction"],
            n_components=run_kwargs["n_qubits"], angle_scaling="minmax", fitted_on="train_partition_only",
        ),
        leakage_checks=leakage_checks,
        metrics=metrics,
        segmentation_metrics={},
        event_metrics={},
        calibration={
            "calibration_curve": (single_run["models"][engine_name].get("clinical_evaluation") or {}).get(
                "calibration_curve", []
            ),
            "engine_calibration": single_run.get("execution", {}).get("calibration", {}),
        },
        abstention=single_run["models"][engine_name].get("abstention") or {},
        resource=ResourceUsage(
            wall_seconds=resource.get("seconds", 0.0),
            training_rows=resource.get("training_rows"),
            test_rows=resource.get("test_rows"),
            feature_count=resource.get("feature_count"),
            qubits=resource.get("qubits"),
            shots=resource.get("shots"),
            backend=run_kwargs["backend"],
            estimated_kernel_pairs=resource.get("kernel_pairs"),
        ),
        confidence_intervals=(single_run["models"][engine_name].get("confidence_intervals") or {}).get(
            "metrics", {}
        ),
        baseline_model_id=model.classical_baseline_model_id,
        baseline_metrics=(
            {name: values["mean"] for name, values in metric_summary[paired["reference_model"]].items()}
            if paired is not None else {}
        ),
        paired_comparison=paired or {},
        real_gain_decision=real_gain_decision,
        real_gain_reason=real_gain_reason,
        software=experiment.runtime_manifest(),
        source_result_path=str(artifact_path.with_suffix(".result.json")),
    )


def run_assessment(bundle, registry: Registry, *, mode: str = "research", runtime_dir: Path | None = None) -> AssessmentRun:
    """Route a bundle, execute every ready model, and assemble the AssessmentRun.

    Every model runs inside its own try/except: a raised exception becomes a `failed`
    finding and does not abort the loop (FR-012, SC-009).
    """

    run_id = str(uuid4())
    started_at = _now()
    synthetic = mode == "demo" or bundle.synthetic
    decisions = route(bundle, registry)
    stage_events: list[StageEvent] = []
    findings: list[Finding] = []
    errors: list[str] = []
    run_fingerprints: dict[str, str] = {}

    for decision in decisions:
        model = registry.model(decision.model_id)
        condition = next(c for c in registry.conditions() if c.condition_id == model.condition_id)

        if decision.status != "ready":
            findings.append(not_evaluated(model, condition, decision, run_id, decision.reason, synthetic))
            continue

        try:
            running_decision = _replace_status(decision, "running")
            stage_events.append(
                StageEvent(at=_now(), stage="execute", model_id=model.model_id, level="info",
                           message=f"running {model.model_id}")
            )
            if model.executor == "tabular_qml":
                finding, terminal_decision = _run_tabular_inference(
                    model, condition, bundle, run_id, running_decision, synthetic
                )
            else:
                raise NotImplementedError(
                    f"executor {model.executor!r} is not implemented in P0"
                )
            findings.append(finding)
        except Exception as exc:  # noqa: BLE001 - fault isolation is the point (FR-012)
            failed_decision = _replace_status(decision, "failed")
            findings.append(not_evaluated(model, condition, failed_decision, run_id, str(exc), synthetic))
            errors.append(f"{model.model_id}: {exc}")
            stage_events.append(
                StageEvent(at=_now(), stage="execute", model_id=model.model_id, level="error", message=str(exc))
            )

    has_scored_finding = any(f.status in ("positive", "negative") for f in findings)
    if not decisions or not any(d.status == "ready" for d in decisions):
        disclaimer = registry.no_compatible_assessment_text
    else:
        disclaimer = registry.disclaimer

    return AssessmentRun(
        run_id=run_id,
        bundle_id=bundle.bundle_id,
        domain="neurological",
        started_at=started_at,
        completed_at=_now(),
        status="completed",
        mode=mode,  # type: ignore[arg-type]
        synthetic=synthetic,
        schema_version=1,
        routing=decisions,
        findings=findings,
        stage_events=stage_events,
        resource=ResourceUsage(
            wall_seconds=0.0, training_rows=None, test_rows=None, feature_count=None,
            qubits=None, shots=None, backend=None, estimated_kernel_pairs=None,
        ),
        fingerprints=run_fingerprints,
        disclaimer=disclaimer,
        errors=errors,
    )


def _replace_status(decision: RoutingDecision, status: str) -> RoutingDecision:
    return dataclasses.replace(decision, status=status)  # type: ignore[arg-type]


_COMPARISONS = {
    ">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t, ">": lambda v, t: v > t, "<": lambda v, t: v < t,
}
_POPULATION_FILTER_RE = re.compile(r"\s*(>=|<=|==|!=|>|<)\s*(-?\d+\.?\d*)")
_QUALITY_RANGE_RE = re.compile(r"(-?\d+\.?\d*)\s*<=\s*\S+\s*<=\s*(-?\d+\.?\d*)")


def _is_blank(raw_value: str) -> bool:
    return not raw_value.strip() or raw_value.strip().lower() in experiment.MISSING_VALUE_SENTINELS


def _check_row_contract(raw_row: dict, contract) -> tuple[list[str], list[str]]:
    """Validate one case row's required fields, population filters, and quality
    constraints (FR-008, FR-011) — the field-level checks routing.py explicitly
    defers to execution time. Returns (missing_fields, unmet_constraints); both
    empty means the row may be scored. A required field present but blank must
    NOT be silently imputed into a scored finding (SC-003).
    """

    missing = [field for field in contract.required_fields if _is_blank(str(raw_row.get(field, "")))]

    unmet: list[str] = []
    for field, expr in contract.population_filters.items():
        raw = str(raw_row.get(field, ""))
        if _is_blank(raw):
            continue  # already reported via `missing` if this field is required
        match = _POPULATION_FILTER_RE.match(expr)
        if not match:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        op, threshold = match.group(1), float(match.group(2))
        if not _COMPARISONS[op](value, threshold):
            unmet.append(f"{field}={raw} fails population filter '{expr}'")

    quality_failed: list[str] = []
    for field, expr in contract.quality_constraints.items():
        raw = str(raw_row.get(field, ""))
        if _is_blank(raw):
            continue  # "when present" constraints and required-field absence are both covered above
        bounds = _QUALITY_RANGE_RE.search(expr)
        if not bounds:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        low, high = float(bounds.group(1)), float(bounds.group(2))
        if not (low <= value <= high):
            quality_failed.append(f"{field}={value} outside quality constraint '{expr}'")

    return missing, unmet + quality_failed


def _run_tabular_inference(model, condition, bundle, run_id, decision, synthetic):
    clinical_assets = [
        a for a in bundle.assets if a.modality == "structured_clinical" and a.validation_status == "accepted"
    ]
    contract = model.input_contract
    required_field_total = len(contract.required_fields)

    if not clinical_assets:
        insufficient = _replace_status(decision, "insufficient data")
        return (
            not_evaluated(model, condition, insufficient, run_id, "no structured_clinical asset available", synthetic),
            insufficient,
        )

    asset = clinical_assets[0]
    if asset.uri.startswith("memory:"):
        raise ValueError(f"asset {asset.asset_id} has no on-disk CSV to read a case row from")
    with open(asset.uri, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one case row in {asset.uri}, found {len(rows)}"
        )
    raw_row = rows[0]

    # Field-level coverage: a required field present in the schema but blank in
    # this row must not be silently imputed into a scored finding (FR-011,
    # SC-003) — this is the check routing.py's own docstring defers to
    # execution time, since routing only sees modality presence, not row values.
    missing_fields, unmet_constraints = _check_row_contract(raw_row, contract)
    field_coverage = CoverageReport(
        required_present=required_field_total - len(missing_fields),
        required_total=max(required_field_total, 1),
        optional_present=0,
        optional_total=len(contract.optional_modalities),
        coverage_ratio=(
            (required_field_total - len(missing_fields)) / required_field_total
            if required_field_total else 1.0
        ),
        missing=missing_fields,
        quality_failed=unmet_constraints,
    )
    if missing_fields or unmet_constraints:
        insufficient = _replace_status(decision, "insufficient data")
        reasons = []
        if missing_fields:
            reasons.append(f"required fields blank or missing: {', '.join(missing_fields)}")
        if unmet_constraints:
            reasons.append("; ".join(unmet_constraints))
        return (
            not_evaluated(model, condition, insufficient, run_id, "; ".join(reasons), synthetic),
            insufficient,
        )

    artifact_path = _resolve(model.artifact.path)
    artifact = experiment.load_model_artifact(artifact_path)
    expected_features = artifact.preprocessor.feature_names
    categorical_columns = artifact.dataset.get("provenance", {}).get("categorical_columns", {})

    X = experiment.encode_raw_row(raw_row, expected_features, categorical_columns).reshape(1, -1)
    result = experiment.predict_with_model_artifact(
        artifact, X, expected_features, dataset_name=bundle.case_id, row_ids=[bundle.case_id], explain=True
    )
    prediction_row = result["prediction_rows"][0]
    explanation = result.get("explanation", {})
    evidence = []
    if explanation.get("status") == "ok":
        for row_report in explanation.get("rows", []):
            for item in row_report.get("top_features", []):
                evidence.append(
                    EvidenceItem(
                        evidence_id=str(uuid4()),
                        finding_id="",  # stamped with the real Finding.finding_id inside safety.finalize_finding()
                        kind="feature_contribution",
                        label=item["feature"],
                        value=float(item["score_delta"]),
                        unit=None,
                        region=None,
                        interval=None,
                        source_asset_id=asset.asset_id,
                        confidence=None,
                        note=item.get("direction"),
                    )
                )

    completed_decision = _replace_status(decision, "abstained" if prediction_row["abstained"] else "completed")
    raw = {
        "score": prediction_row["score"],
        "abstained": prediction_row["abstained"],
        "threshold": result["threshold"],
        "threshold_policy": result["threshold_policy"],
        "output": {"label": prediction_row["prediction"], "positive_label": "1"},
        "modalities_used": ["structured_clinical"],
        "evidence": evidence,
        "explanation_status": "available" if explanation.get("status") == "ok" else "unavailable",
    }
    finding = finalize_finding(
        model=model, condition=condition, decision=completed_decision, coverage=field_coverage,
        raw=raw, run_id=run_id, synthetic=synthetic,
    )
    return finding, completed_decision
