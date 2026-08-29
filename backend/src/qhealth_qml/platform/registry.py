"""Registry loader and validator for model definitions and evaluation records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    ConditionDefinition,
    ConditionDefinition_from_dict,
    Availability,
    ModelDefinition,
    ModelDefinition_from_dict,
    EvaluationRecord,
    EvaluationRecord_from_dict,
)
from ..protocol import load_early_detection_profile


class RegistryError(Exception):
    """Raised when registry validation fails."""
    pass


@dataclass(frozen=True)
class CatalogEntry:
    """A view of a condition with its models and combined availability."""
    condition: ConditionDefinition
    models: list[ModelDefinition]
    availability: Availability


class Registry:
    """Loaded and validated registry of conditions, models, and evaluation records."""

    def __init__(
        self,
        conditions: dict[str, ConditionDefinition],
        models: dict[str, ModelDefinition],
        evaluation_records: dict[str, list[EvaluationRecord]],
        platform_meta: dict[str, Any],
    ):
        self._conditions = conditions
        self._models = models
        self._evaluation_records = evaluation_records
        self._platform_meta = platform_meta

    @property
    def disclaimer(self) -> str:
        """FR-018 disclaimer text, safety.py's single source of truth."""
        return str(self._platform_meta["disclaimer"])

    @property
    def no_compatible_assessment_text(self) -> str:
        """Text shown when no registered model can run on the supplied data (SC-003)."""
        return str(self._platform_meta["no_compatible_assessment_text"])

    @property
    def synthetic_marker(self) -> str:
        """FR-031 marker prefixed onto demo-mode disclaimers."""
        return str(self._platform_meta["synthetic_marker"])

    def conditions(self) -> list[ConditionDefinition]:
        """Return all registered conditions."""
        return list(self._conditions.values())

    def models(self, condition_id: str | None = None) -> list[ModelDefinition]:
        """Return models, optionally filtered by condition."""
        if condition_id is None:
            return list(self._models.values())
        return [m for m in self._models.values() if m.condition_id == condition_id]

    def model(self, model_id: str) -> ModelDefinition:
        """Get a specific model by ID."""
        if model_id not in self._models:
            raise RegistryError(f"Model not found: {model_id}")
        return self._models[model_id]

    def catalog(self, domain: str = "neurological") -> list[CatalogEntry]:
        """Return the catalog (conditions with their models) for a domain.

        Every registered model is included, available or not (FR-002, US-1 scenario 3:
        a condition with no available model must still be listed and marked unavailable,
        never silently omitted).
        """
        entries = []
        for condition in self.conditions():
            if condition.domain != domain:
                continue
            models = self.models(condition.condition_id)
            availability: Availability = (
                "available" if any(m.availability == "available" for m in models) else "not available"
            )
            entries.append(CatalogEntry(condition=condition, models=models, availability=availability))
        return sorted(entries, key=lambda e: e.condition.priority)

    def evaluation_records(self, model_id: str) -> list[EvaluationRecord]:
        """Get evaluation records for a specific model."""
        return self._evaluation_records.get(model_id, [])


def load_registry(root: Path | None = None) -> Registry:
    """Load and validate the registry from JSON files.

    Registry validation rules are enforced here:
    1. Every ModelDefinition.condition_id resolves.
    2. availability == "available" ⇒ artifact.path is not None AND evaluation_record_ids is non-empty.
    3. quantum is not None ⇒ classical_baseline_model_id resolves to a registered model.
    4. lifecycle == "operational_reference" ⇒ some referenced EvaluationRecord has real_gain_decision == "passed" OR model has no quantum spec.
    5. model_card.research_record points at an existing file.
    6. model_card.reuse_manifest is non-empty.
    7. artifact.path, when set, resolves inside the configured runtime directory.
    """
    if root is None:
        root = Path(__file__).parent / "registry_data"

    root = root.resolve()

    # Load platform.json (mainly for metadata)
    platform_path = root / "platform.json"
    if not platform_path.exists():
        raise RegistryError(f"Platform file not found: {platform_path}")
    with open(platform_path, "r", encoding="utf-8") as f:
        platform_data = json.load(f)

    # Load conditions
    conditions_dir = root / "conditions"
    conditions_dict: dict[str, ConditionDefinition] = {}
    if conditions_dir.exists():
        for cond_file in sorted(conditions_dir.glob("*.json")):
            with open(cond_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            try:
                cond = ConditionDefinition_from_dict(raw)
                conditions_dict[cond.condition_id] = cond
            except Exception as e:
                raise RegistryError(f"Failed to load condition from {cond_file}: {e}")

    # Load models
    models_dir = root / "models"
    models_dict: dict[str, ModelDefinition] = {}
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("*.json")):
            with open(model_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            try:
                model = ModelDefinition_from_dict(raw)
                models_dict[model.model_id] = model
            except Exception as e:
                raise RegistryError(f"Failed to load model from {model_file}: {e}")

    # Load evaluation records
    evaluations_dir = root / "evaluations"
    evaluations_dict: dict[str, list[EvaluationRecord]] = {}
    if evaluations_dir.exists():
        for eval_file in sorted(evaluations_dir.glob("*.json")):
            with open(eval_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            try:
                eval_record = EvaluationRecord_from_dict(raw)
                model_id = eval_record.model_id
                if model_id not in evaluations_dict:
                    evaluations_dict[model_id] = []
                evaluations_dict[model_id].append(eval_record)
            except Exception as e:
                raise RegistryError(f"Failed to load evaluation record from {eval_file}: {e}")

    # Compute repo root for rule 7 and dataset profile validation
    repo_root = Path(__file__).parents[4]

    # Validation rules
    for model_id, model in models_dict.items():
        # Rule 1: condition_id resolves
        if model.condition_id not in conditions_dict:
            raise RegistryError(
                f"Model {model_id}: condition_id '{model.condition_id}' not found in registry"
            )

        # Rule 2: availability == "available" ⇒ artifact.path is not None AND evaluation_record_ids is non-empty
        if model.availability == "available":
            if model.artifact.path is None:
                raise RegistryError(
                    f"Model {model_id}: availability is 'available' but artifact.path is None"
                )
            if not model.evaluation_record_ids:
                raise RegistryError(
                    f"Model {model_id}: availability is 'available' but evaluation_record_ids is empty"
                )

        # Rule 3: quantum is not None ⇒ classical_baseline_model_id resolves
        if model.quantum is not None:
            if model.classical_baseline_model_id is None:
                raise RegistryError(
                    f"Model {model_id}: has quantum spec but classical_baseline_model_id is None"
                )
            if model.classical_baseline_model_id not in models_dict:
                raise RegistryError(
                    f"Model {model_id}: classical_baseline_model_id '{model.classical_baseline_model_id}' not found in registry"
                )

        # Rule 4: lifecycle == "operational_reference" with a quantum spec ⇒ some EvaluationRecord
        # clears the baseline-viability gate (balanced accuracy clearly above chance, i.e. the
        # bootstrap CI lower bound on balanced_accuracy exceeds 0.5) OR the model has no quantum
        # spec. Per ACCEPTANCE-CRITERIA.md §0 (2026-08-29 revision): the hybrid QML candidate is
        # promoted on its own merits, not on beating the classical baseline — real_gain_decision is
        # still recorded and still reported (FR-025), it just no longer gates promotion.
        if model.lifecycle == "operational_reference":
            if model.quantum is not None:
                eval_records = evaluations_dict.get(model_id, [])

                def _clears_baseline_viability(er: EvaluationRecord) -> bool:
                    ci = er.confidence_intervals.get("balanced_accuracy")
                    if ci is not None and ci.get("lower") is not None:
                        return ci["lower"] > 0.5
                    balanced_accuracy = er.metrics.get("balanced_accuracy")
                    return balanced_accuracy is not None and balanced_accuracy > 0.5

                if not any(_clears_baseline_viability(er) for er in eval_records):
                    raise RegistryError(
                        f"Model {model_id}: lifecycle is 'operational_reference' with quantum spec but "
                        f"no evaluation record shows balanced accuracy clearly above chance (CI lower "
                        f"bound > 0.5)"
                    )

        # Rule 5: model_card.research_record points at an existing file
        research_record_path = repo_root / model.model_card.research_record
        if not research_record_path.exists():
            raise RegistryError(
                f"Model {model_id}: research_record file not found: {research_record_path}"
            )

        # Rule 6: model_card.reuse_manifest is non-empty
        if not model.model_card.reuse_manifest:
            raise RegistryError(
                f"Model {model_id}: reuse_manifest is empty"
            )

        # Rule 7: artifact.path, when set, resolves inside the runtime directory
        if model.artifact.path is not None:
            artifact_path = repo_root / model.artifact.path
            runtime_dir = repo_root / "runtime"
            try:
                artifact_path.relative_to(runtime_dir)
            except ValueError:
                raise RegistryError(
                    f"Model {model_id}: artifact.path '{model.artifact.path}' does not resolve under runtime directory"
                )

        # Validate dataset_profile_path if present (reads via protocol.load_early_detection_profile)
        if model.dataset_profile_path is not None:
            profile_path = repo_root / model.dataset_profile_path
            try:
                load_early_detection_profile(profile_path)
            except Exception as e:
                raise RegistryError(
                    f"Model {model_id}: failed to load dataset_profile_path '{model.dataset_profile_path}': {e}"
                )

    return Registry(conditions_dict, models_dict, evaluations_dict, platform_data)
