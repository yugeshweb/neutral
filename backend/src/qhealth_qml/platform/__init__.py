"""Platform package: registry, schema, routing, execution, and safety."""

from .execution import benchmark_model, run_assessment
from .registry import load_registry
from .routing import route

__all__ = [
    "load_registry",
    "route",
    "run_assessment",
    "benchmark_model",
]
