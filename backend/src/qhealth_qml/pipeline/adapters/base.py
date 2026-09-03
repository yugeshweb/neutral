"""`ModalityAdapter` protocol (design.md §8.1). Every adapter in this
directory implements this; the registry (`../registry.py`) dispatches to
one by declared spec, then content sniff, then extension (FR-014)."""

from __future__ import annotations

from typing import Iterator, Protocol

from ..spec import SourceSpec
from ..types import Modality, QCVerdict, RawRecord, Sample, Source


class ModalityAdapter(Protocol):
    name: str
    modalities: tuple[Modality, ...]
    formats: tuple[str, ...]  # extensions, for the file picker only - never used to dispatch alone

    def sniff(self, source: Source) -> float:
        """0..1 confidence from CONTENT, not extension."""
        ...

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        """Streaming. One RawRecord per record/row/series."""
        ...

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        """Stateless: reads exactly this RawRecord plus the spec."""
        ...

    def qc(self, sample: Sample, spec: SourceSpec) -> QCVerdict:
        """Modality-specific checks, on top of the universal QCGate."""
        ...
