"""Adapter registry and three-stage dispatch (design.md §8.2, FR-010,
FR-014, FR-015, FR-134). Mirrors what `src/lib/ingest/index.ts` already
does in the browser (extension -> content-sniff, with an explicit
not-implemented for a declared-but-missing format), which the backend
previously lacked."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.base import ModalityAdapter
from .spec import SourceSpec
from .types import Source

SNIFF_FLOOR = 0.5


class AdapterUnavailableError(Exception):
    """The adapter exists but its optional dependency is missing
    (FR-134) - reported as unavailable, not as an import traceback at
    call time."""

    def __init__(self, adapter_name: str, missing_dependency: str):
        super().__init__(
            f"adapter '{adapter_name}' is registered but its dependency "
            f"'{missing_dependency}' is not installed"
        )
        self.adapter_name = adapter_name
        self.missing_dependency = missing_dependency


class DispatchError(Exception):
    """No adapter met the sniff floor, or two tied (FR-015) - an error,
    never a guess."""


@dataclass
class Registration:
    adapter: ModalityAdapter
    available: bool
    unavailable_reason: str | None = None


class AdapterRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, Registration] = {}

    def register(self, adapter: ModalityAdapter, *, available: bool = True, unavailable_reason: str | None = None) -> None:
        self._registrations[adapter.name] = Registration(adapter, available, unavailable_reason)

    def get(self, name: str) -> ModalityAdapter:
        reg = self._registrations.get(name)
        if reg is None:
            raise DispatchError(f"no adapter named {name!r} is registered")
        if not reg.available:
            raise AdapterUnavailableError(name, reg.unavailable_reason or "unknown")
        return reg.adapter

    def is_available(self, name: str) -> bool:
        reg = self._registrations.get(name)
        return reg is not None and reg.available

    def names(self) -> list[str]:
        return list(self._registrations)

    def resolve(self, source: Source, spec: SourceSpec) -> ModalityAdapter:
        """(1) `spec.adapter` if declared; (2) highest `sniff()` confidence
        above `SNIFF_FLOOR`; (3) extension as a last resort. A tie or an
        all-below-floor result is a `DispatchError`, never a guess."""

        if spec.adapter:
            return self.get(spec.adapter)

        scored: list[tuple[float, Registration]] = []
        for reg in self._registrations.values():
            if not reg.available:
                continue
            try:
                confidence = reg.adapter.sniff(source)
            except Exception:
                confidence = 0.0
            if confidence >= SNIFF_FLOOR:
                scored.append((confidence, reg))

        if not scored:
            raise DispatchError(
                f"no adapter reached the sniff confidence floor ({SNIFF_FLOOR}) for "
                f"source {source.locator!r}; declare `adapter` explicitly in the spec."
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score = scored[0][0]
        tied = [reg for score, reg in scored if score == top_score]
        if len(tied) > 1:
            names = [t.adapter.name for t in tied]
            raise DispatchError(
                f"adapters {names} tied at confidence {top_score} for source "
                f"{source.locator!r}; declare `adapter` explicitly in the spec."
            )
        return tied[0].adapter


DEFAULT_REGISTRY = AdapterRegistry()


def default_registry() -> AdapterRegistry:
    """Lazily populates and returns the module-level default registry, so
    importing `pipeline.registry` never eagerly imports every adapter
    (some of which have optional heavy dependencies - FR-133)."""

    if not DEFAULT_REGISTRY.names():
        _register_builtin_adapters(DEFAULT_REGISTRY)
    return DEFAULT_REGISTRY


def _register_builtin_adapters(registry: AdapterRegistry) -> None:
    from .adapters.tabular_csv import TabularCsvAdapter

    registry.register(TabularCsvAdapter())

    _register_optional(registry, "wfdb_ecg", "adapters.wfdb_ecg", "WfdbEcgAdapter", "wfdb")
    _register_optional(registry, "edf_eeg", "adapters.edf_eeg", "EdfEegAdapter", "mne")
    _register_optional(registry, "gait_txt", "adapters.gait_txt", "GaitTxtAdapter", None)
    _register_optional(registry, "nifti_volume", "adapters.nifti_volume", "NiftiVolumeAdapter", "nibabel")
    _register_optional(registry, "dicom_series", "adapters.dicom_series", "DicomSeriesAdapter", "pydicom")
    _register_optional(registry, "image_2d", "adapters.image_2d", "Image2dAdapter", "PIL")


def _register_optional(registry: AdapterRegistry, name: str, module_path: str, class_name: str, dependency: str | None) -> None:
    """Registers an adapter whose class import never fails (it only
    imports its module, not the heavy optional library, at class-definition
    time), but marks it unavailable if its runtime dependency truly isn't
    installed (FR-133/FR-134) - reported as `AdapterUnavailableError` at
    dispatch time, never an import traceback."""

    import importlib

    try:
        module = importlib.import_module(f".{module_path}", package=__package__)
        adapter_cls = getattr(module, class_name)
    except ImportError:
        return  # the adapter module itself is missing - nothing to register

    available = True
    reason = None
    if dependency is not None:
        try:
            importlib.import_module(dependency)
        except ImportError:
            available = False
            reason = dependency

    registry.register(adapter_cls(), available=available, unavailable_reason=reason)
