"""Reference deterministic feature extractor for imaging modalities (ct,
mr, angio) - design.md §9.2.7. Same placeholder caveat as
`reference_signal.py`: ordinary per-channel intensity statistics over the
harmonized volume, not a trained encoder or the team's real feature set.
Swap `SourceSpec.representation.extractor` for the real one when it
exists; no pipeline code changes required."""

from __future__ import annotations

import numpy as np

from ..spec import SourceSpec
from ..types import Sample

_STATS = ("mean", "std", "p10", "p50", "p90", "foreground_fraction")


def extract_imaging_features(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    sequences: list[str] = list(spec._raw.get("imaging", {}).get("sequences", []))
    names = [f"{seq}__{stat}" for seq in sequences for stat in _STATS]

    array = sample.arrays.get("volume")
    if array is None or array.size == 0:
        return np.full(len(names), np.nan), names

    values: list[float] = []
    for c in range(min(array.shape[0], len(sequences))):
        vol = array[c].astype(float)
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            values.extend([float("nan")] * len(_STATS))
            continue
        foreground = finite[finite > finite.min()]
        foreground_fraction = float(foreground.size) / float(finite.size)
        values.extend(
            [
                float(finite.mean()), float(finite.std()),
                float(np.percentile(finite, 10)), float(np.percentile(finite, 50)), float(np.percentile(finite, 90)),
                foreground_fraction,
            ]
        )
    while len(values) < len(names):
        values.append(float("nan"))
    return np.asarray(values, dtype=float), names
