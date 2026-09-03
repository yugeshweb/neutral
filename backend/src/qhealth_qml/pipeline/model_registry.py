"""Registry for disease model training functions.

Allows swapping model implementations without changing orchestration pipeline code.
"""

from typing import Dict, List, Optional
from qhealth_qml.pipeline.interfaces import TrainDiseaseModelsFn
from qhealth_qml.pipeline.exceptions import UnknownDiseaseError, ModelTrainingError

_MODEL_TRAINER_REGISTRY: Dict[str, TrainDiseaseModelsFn] = {}


def register_disease_models(disease_id: str, train_fn: TrainDiseaseModelsFn) -> None:
    """Register a model training function for a specific disease_id.

    The train_fn must match the signature:
    train_fn(X_train, y_train, X_test, y_test) -> dict {"classical": ..., "quantum": ...}
    """
    if not callable(train_fn):
        raise TypeError(f"train_fn for disease '{disease_id}' must be callable.")
    _MODEL_TRAINER_REGISTRY[disease_id] = train_fn


def unregister_disease_models(disease_id: str) -> Optional[TrainDiseaseModelsFn]:
    """Remove registration for a disease_id."""
    return _MODEL_TRAINER_REGISTRY.pop(disease_id, None)


def get_disease_models_trainer(disease_id: str) -> TrainDiseaseModelsFn:
    """Look up the registered model training function for the given disease_id."""
    if disease_id not in _MODEL_TRAINER_REGISTRY:
        raise UnknownDiseaseError(
            disease_id=disease_id,
            message=f"No model-training function registered for disease '{disease_id}'. "
                    f"Registered diseases: {list(_MODEL_TRAINER_REGISTRY.keys())}",
        )
    return _MODEL_TRAINER_REGISTRY[disease_id]


def has_disease_models_trainer(disease_id: str) -> bool:
    """Check whether a model training function is registered for disease_id."""
    return disease_id in _MODEL_TRAINER_REGISTRY


def list_registered_disease_trainers() -> List[str]:
    """List all disease IDs that have registered training functions."""
    return list(_MODEL_TRAINER_REGISTRY.keys())


def clear_model_registry() -> None:
    """Clear all registered model trainers (useful for test isolation)."""
    _MODEL_TRAINER_REGISTRY.clear()
