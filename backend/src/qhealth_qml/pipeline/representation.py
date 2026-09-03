"""Representation building - the declared, pluggable step that turns a
harmonized signal/imaging `Sample.arrays` into the feature vector
`Recipe.fit`/`align` work on (design.md §9.2.7). `SourceSpec.representation`
NAMES which function computes features; this module only knows HOW to call
it - it contains zero domain-specific ECG/EEG/imaging feature logic
itself, keeping intact the split the spec draws between "declared" and
"stateless adapter code" (design.md's Kind/Lives-in/Origin table, §6.2).

Only `kind == "deterministic"` is implemented here. `learned` and
`multimodal` (design.md §9.2.7, tasks.md T046 - fit-before-encode
ordering, multimodal availability masks) need an actual trainable encoder
and are explicitly out of this build's scope, the same way `split.py`
declares but does not implement `site_holdout`/`predeclared_folds`."""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from .spec import SourceSpec
from .types import Sample

ExtractorFn = Callable[[Sample, SourceSpec], "tuple[np.ndarray, list[str]]"]


class RepresentationError(Exception):
    """A misconfigured or unimplemented `representation.kind`, or a
    declared extractor whose dotted path could not be imported."""


def extract(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    """Returns `(vector, names)` for one sample, per the extractor
    `spec.representation` declares. Deterministic extractors are pure
    functions of one sample plus the spec (design.md §9.2.7) - calling
    this once per record, at fit time AND at predict time, is what keeps
    train and predict from diverging (the same mechanism `Recipe.align`
    uses for tabular data)."""

    config = spec._raw.get("representation")
    if not config:
        raise RepresentationError(
            f"modality '{spec.modality}' requires a declared `representation` block "
            f"(design.md §9.2.7) - there is no default feature extractor for non-tabular data."
        )
    kind = config.get("kind", "deterministic")
    if kind == "deterministic":
        fn = _load_extractor(config["extractor"])
        vector, names = fn(sample, spec)
        return np.asarray(vector, dtype=float), list(names)
    if kind in ("learned", "multimodal"):
        raise RepresentationError(
            f"representation.kind={kind!r} is declared by the spec (design.md §9.2.7 - "
            f"fit-before-encode ordering, multimodal availability masks) but not implemented in "
            f"this build; only 'deterministic' extractors are wired up."
        )
    raise RepresentationError(f"unknown representation.kind {kind!r}")


_EXTRACTOR_CACHE: dict[str, ExtractorFn] = {}


def _load_extractor(dotted: str) -> ExtractorFn:
    """`dotted` is `module.path:function_name`. Cached so a batch of
    records doesn't re-import per row."""

    if dotted in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[dotted]
    if ":" not in dotted:
        raise RepresentationError(f"representation.extractor {dotted!r} must be 'module.path:function_name'")
    module_name, _, func_name = dotted.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RepresentationError(f"could not import representation.extractor module {module_name!r}: {exc}") from exc
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise RepresentationError(f"module {module_name!r} has no callable {func_name!r}")
    _EXTRACTOR_CACHE[dotted] = fn
    return fn
