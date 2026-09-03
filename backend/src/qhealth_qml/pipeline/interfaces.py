"""Model training interfaces and standard signatures for disease models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Union
import numpy as np


@dataclass
class ModelOutput:
    """Standardized output produced by a classical or quantum model trainer."""

    model: Any
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    train_time: Union[float, str] = 0.0
    infer_time: Union[float, str] = 0.0
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "y_pred": self.y_pred,
            "y_prob": self.y_prob,
            "train_time": self.train_time,
            "infer_time": self.infer_time,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


# Standard signature required for any model trainer plugging into this orchestration pipeline:
# train_fn(X_train, y_train, X_test, y_test) -> {"classical": ModelOutput | dict, "quantum": ModelOutput | dict}
TrainDiseaseModelsFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, Union[ModelOutput, Dict[str, Any]]],
]


class DiseaseModelTrainer(Protocol):
    """Protocol defining the model trainer interface."""

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Union[ModelOutput, Dict[str, Any]]]:
        ...
