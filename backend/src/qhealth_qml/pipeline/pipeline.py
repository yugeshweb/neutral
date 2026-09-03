"""`Pipeline` - the only orchestration (design.md §3.4, FR-001, FR-007,
FR-008). Two calls: `.fit()` for training, `.run()` for prediction. There
is no third and no single-record variant (FR-006) - a lone record is
`Pipeline.run(Batch(samples=[one_sample]))`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .qc import QCContract, QCGate, check_duplicate_sample_ids
from .recipe import Recipe, RECIPE_SCHEMA_VERSION
from .registry import default_registry
from .spec import SourceSpec
from .split import assert_window_cohesion, split_batch
from .types import (
    Batch, FitResult, Issue, IssueCode, LABEL_EXCLUDE, LABEL_NEGATIVE, LABEL_POSITIVE,
    Ledger, PreparedArrays, QCVerdict, RunResult, Sample, Source,
)


class PipelineError(Exception):
    """A pipeline-level failure - an empty batch, an unreadable source, an
    unmet dependency - as opposed to a per-record refusal (FR-080)."""


class Pipeline:
    def __init__(self, spec: SourceSpec):
        self.spec = spec
        self._recipe: Recipe | None = None

    @classmethod
    def from_spec(cls, spec: SourceSpec) -> "Pipeline":
        return cls(spec)

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "Pipeline":
        pipeline = cls(recipe.spec)
        pipeline._recipe = recipe
        return pipeline

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    @staticmethod
    def read(spec: SourceSpec, source: str | Path | bytes | None = None) -> Batch:
        """Resolves an adapter, reads every record, harmonizes each
        (stateless, FR-004), and returns a `Batch`. Adapter selection is
        the pipeline's responsibility, never the caller's (FR-010)."""

        if isinstance(source, bytes):
            src = Source(kind="bytes", locator="<bytes>", payload=source)
        elif source is not None:
            src = Source(kind="path", locator=str(source))
        else:
            src = Source(kind="path", locator=spec.source.resolved_pattern(fallback_dir=spec._loaded_from_dir))

        registry = default_registry()
        adapter = registry.resolve(src, spec)

        samples: list[Sample] = []
        dataset_issues: list[Issue] = []
        try:
            for raw in adapter.read(src, spec):
                sample = adapter.harmonize(raw, spec)
                # Modality-specific checks on top of the universal QCGate
                # (design.md §9.3) - previously declared on the adapter
                # Protocol but never actually invoked anywhere.
                modality_verdict = adapter.qc(sample, spec)
                if modality_verdict.issues:
                    sample = dataclasses.replace(sample, issues=[*sample.issues, *modality_verdict.issues])
                sample = _triage_label(sample, spec, dataset_issues)
                samples.append(sample)
        except FileNotFoundError as exc:
            raise PipelineError(f"source not readable: {exc}") from exc
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc

        if not samples:
            raise PipelineError(f"source {src.locator!r} produced zero records")

        dataset_issues.extend(check_duplicate_sample_ids([s.sample_id for s in samples]))

        provenance = {
            "adapter": adapter.name,
            "source": src.locator,
            "n_records": len(samples),
        }
        return Batch(samples=samples, spec=spec, provenance=provenance, issues=dataset_issues)

    # ------------------------------------------------------------------
    # fit (training entry point)
    # ------------------------------------------------------------------

    def fit(self, batch: Batch, n_qubits: int = 6) -> FitResult:
        t0 = time.perf_counter()
        if len(batch) == 0:
            raise PipelineError("cannot fit on an empty batch")

        gate = QCGate()
        contract = QCContract(
            required_fields=self.spec.required_fields,
            population_filters=self._population_filters(),
            quality_constraints=self._quality_constraints(),
            min_required_coverage=self.spec.min_required_coverage,
        )

        verdicts: list[QCVerdict] = []
        rejected_by_code: dict[str, int] = {}
        excluded_by_reason: dict[str, int] = {}
        scorable_samples: list[Sample] = []
        n_rejected_records = 0

        for sample in batch.samples:
            if sample.label == LABEL_EXCLUDE:
                reason = next(
                    (i.code for i in sample.issues if i.code in (IssueCode.LABEL_EXCLUDED_CENSORED, IssueCode.LABEL_EXCLUDED_COMPETING_RISK)),
                    "label_excluded",
                )
                excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                verdicts.append(QCVerdict(status="reject", issues=list(sample.issues)))
                continue
            verdict = gate.check(sample, contract)
            # Reject-severity issues from harmonize()/qc() (e.g. a signal
            # adapter's channel_missing, an imaging adapter's
            # sequence_missing) must reach the verdict too, deduped against
            # anything the universal gate already reported for the same
            # (code, field).
            existing = {(i.code, i.field) for i in verdict.issues}
            carried = [i for i in sample.issues if i.severity == "reject" and (i.code, i.field) not in existing]
            if carried:
                verdict = QCVerdict(status="reject", issues=list(verdict.issues) + carried)
            verdicts.append(verdict)
            if verdict.status == "reject":
                # n_rejected counts RECORDS; rejected_by_code is a breakdown
                # that may sum to more than n_rejected when one record
                # carries more than one distinct reject-severity code.
                n_rejected_records += 1
                seen_codes_this_record: set[str] = set()
                for issue in verdict.issues:
                    if issue.severity == "reject" and issue.code not in seen_codes_this_record:
                        rejected_by_code[issue.code] = rejected_by_code.get(issue.code, 0) + 1
                        seen_codes_this_record.add(issue.code)
            else:
                scorable_samples.append(sample)

        if not scorable_samples:
            raise PipelineError(
                "every record was refused or excluded before fitting - nothing to train on. "
                f"rejected_by_code={rejected_by_code}, excluded_by_reason={excluded_by_reason}"
            )

        y_all = np.array([s.label for s in scorable_samples], dtype=int)
        scorable_batch = dataclasses.replace(batch, samples=scorable_samples)

        split_result = split_batch(
            scorable_samples, y_all,
            strategy=self.spec.split.strategy,
            test_size=self.spec.split.test_size,
            validation_size=self.spec.split.validation_size,
            seed=self.spec.split.seed,
        )
        assert_window_cohesion(scorable_samples, split_result.train, split_result.validation, split_result.test)

        train_idx = split_result.train
        val_idx = split_result.validation
        test_idx = split_result.test

        # FR-091: fit on train MINUS validation.
        train_batch = dataclasses.replace(scorable_batch, samples=[scorable_samples[i] for i in train_idx])
        train_y = y_all[train_idx]

        recipe = Recipe(
            schema_version=RECIPE_SCHEMA_VERSION, spec=self.spec,
            code_version=_code_version(), fitted_at="", n_qubits=n_qubits,
        )
        recipe.fit(train_batch, train_y)

        train_arrays, train_issues = recipe.transform(train_batch)
        validation_arrays = None
        if val_idx is not None:
            val_batch = dataclasses.replace(scorable_batch, samples=[scorable_samples[i] for i in val_idx])
            validation_arrays, _ = recipe.transform(val_batch)
        test_batch = dataclasses.replace(scorable_batch, samples=[scorable_samples[i] for i in test_idx])
        test_arrays, _ = recipe.transform(test_batch)

        n_rejected = n_rejected_records
        n_excluded = sum(excluded_by_reason.values())
        ledger = Ledger(
            n_in=len(batch), n_scored=len(scorable_samples), n_excluded=n_excluded, n_rejected=n_rejected,
            excluded_by_reason=excluded_by_reason, rejected_by_code=rejected_by_code,
            quarantined_columns=recipe.quarantined_columns, dropped_columns=recipe.dropped_columns,
            missingness={k: v["missing_frac"] for k, v in recipe.train_stats.items()},
            fingerprints={"spec": _spec_fingerprint(self.spec), "recipe": recipe.fingerprint(), "pipeline": _pipeline_fingerprint(self.spec, recipe)},
            timings_ms={"total": (time.perf_counter() - t0) * 1000},
        )

        self._recipe = recipe
        return FitResult(recipe=recipe, train=train_arrays, validation=validation_arrays, test=test_arrays, ledger=ledger)

    # ------------------------------------------------------------------
    # run (prediction entry point)
    # ------------------------------------------------------------------

    def run(self, batch: Batch) -> RunResult:
        t0 = time.perf_counter()
        if self._recipe is None:
            raise PipelineError("Pipeline.run() requires a recipe - use Pipeline.from_recipe(recipe)")
        if len(batch) == 0:
            raise PipelineError("cannot run on an empty batch")

        recipe = self._recipe
        gate = QCGate()
        contract = QCContract(
            required_fields=self.spec.required_fields,
            population_filters=self._population_filters(),
            quality_constraints=self._quality_constraints(),
            min_required_coverage=self.spec.min_required_coverage,
        )

        verdicts: list[QCVerdict] = []
        rejected_by_code: dict[str, int] = {}
        scorable_indices: list[int] = []
        n_rejected_records = 0

        for idx, sample in enumerate(batch.samples):
            verdict = gate.check(sample, contract)
            # Reject-severity issues from harmonize()/qc() (channel_missing,
            # sequence_missing, low_foreground, ...) must reach the verdict,
            # deduped against anything the universal gate already reported.
            existing = {(i.code, i.field) for i in verdict.issues}
            carried = [i for i in sample.issues if i.severity == "reject" and (i.code, i.field) not in existing]
            if carried:
                verdict = QCVerdict(status="reject", issues=list(verdict.issues) + carried)
                existing |= {(i.code, i.field) for i in carried}

            # align()-level refusals (required field missing per the FITTED
            # recipe, which may declare requirements the raw QC contract
            # doesn't know about) are folded in below, per-row. Deduped by
            # (code, field): the QC contract and the recipe's own
            # required_fields commonly overlap, and one record must not be
            # reported twice for the same missing field.
            _, align_issues = recipe.align(sample)
            reject_issues = [i for i in align_issues if i.severity == "reject" and (i.code, i.field) not in existing]
            if reject_issues:
                verdict = QCVerdict(status="reject", issues=list(verdict.issues) + reject_issues)

            verdicts.append(verdict)
            if verdict.status == "reject":
                # n_rejected counts RECORDS; rejected_by_code is a
                # breakdown that may sum to more than n_rejected when one
                # record carries more than one distinct reject-severity code.
                n_rejected_records += 1
                seen_codes_this_record: set[str] = set()
                for issue in verdict.issues:
                    if issue.severity == "reject" and issue.code not in seen_codes_this_record:
                        rejected_by_code[issue.code] = rejected_by_code.get(issue.code, 0) + 1
                        seen_codes_this_record.add(issue.code)
            else:
                scorable_indices.append(idx)

        scorable_samples = [batch.samples[i] for i in scorable_indices]
        if scorable_samples:
            scorable_batch = dataclasses.replace(batch, samples=scorable_samples)
            arrays, _ = recipe.transform(scorable_batch)
            ood_issues = recipe.ood_report(arrays.X_raw)
            if ood_issues:
                for i in scorable_indices:
                    v = verdicts[i]
                    verdicts[i] = QCVerdict(status="accept_with_flags", issues=list(v.issues) + ood_issues)
        else:
            arrays = PreparedArrays(
                X_raw=np.zeros((0, len(recipe.feature_names))),
                X_classical=np.zeros((0, len(recipe.selected_features))),
                X_quantum=np.zeros((0, len(recipe.selected_features))),
                y=None, row_ids=np.array([]), subject_ids=None, sites=None, subgroups={},
                feature_names=list(recipe.feature_names), selected_features=list(recipe.selected_features),
            )

        n_rejected = n_rejected_records
        ledger = Ledger(
            n_in=len(batch), n_scored=len(scorable_samples), n_excluded=0, n_rejected=n_rejected,
            excluded_by_reason={}, rejected_by_code=rejected_by_code,
            quarantined_columns=recipe.quarantined_columns, dropped_columns=recipe.dropped_columns,
            missingness={}, fingerprints={"spec": _spec_fingerprint(self.spec), "recipe": recipe.fingerprint()},
            timings_ms={"total": (time.perf_counter() - t0) * 1000},
        )
        return RunResult(arrays=arrays, verdicts=verdicts, ledger=ledger)

    # ------------------------------------------------------------------
    def _population_filters(self) -> dict[str, str]:
        return {k: v for k, v in self.spec._raw.get("population_filters", {}).items()}

    def _quality_constraints(self) -> dict[str, str]:
        return {k: v for k, v in self.spec._raw.get("quality_constraints", {}).items()}


def _triage_label(sample: Sample, spec: SourceSpec, dataset_issues: list[Issue]) -> Sample:
    """FR-094, FR-095: three label states, not two. Only engages when the
    spec declares a horizon and both times are present - a cross-sectional
    profile (the common case among the seven profiles already committed)
    is unaffected and keeps its binary label exactly as harmonize() set it."""

    if sample.label is None or sample.label == LABEL_POSITIVE:
        return sample
    if spec.temporal_framing != "prediction" or spec.horizon_days is None:
        return sample
    if not sample.index_time or not sample.outcome_time:
        return sample
    try:
        delta_days = (date.fromisoformat(sample.outcome_time) - date.fromisoformat(sample.index_time)).days
    except ValueError:
        return sample
    if delta_days < spec.horizon_days:
        issue = Issue(
            IssueCode.LABEL_EXCLUDED_CENSORED, "info",
            f"Follow-up ended at {delta_days}d, short of the {spec.horizon_days}d horizon; "
            f"excluded rather than counted negative.",
        )
        return dataclasses.replace(sample, label=LABEL_EXCLUDE, issues=[*sample.issues, issue])
    return sample


def _code_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("quantum-health")
    except Exception:
        return "0.0.0-dev"


def _spec_fingerprint(spec: SourceSpec) -> str:
    import json

    payload = json.dumps(spec.as_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _pipeline_fingerprint(spec: SourceSpec, recipe: Recipe) -> str:
    payload = f"{_spec_fingerprint(spec)}:{recipe.fingerprint()}:{_code_version()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
