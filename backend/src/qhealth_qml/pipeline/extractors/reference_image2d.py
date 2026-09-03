"""Reference deterministic feature extractor for the `image2d` modality -
same placeholder caveat as `reference_signal.py`/`reference_imaging.py`:
ordinary per-channel intensity statistics over the harmonized image, not a
trained CNN (the reference notebook's own VGG-16 transfer-learning model
is the kind of thing that would actually replace this). `image2d` itself
is not part of the team-head spec - see `..adapters.image_2d` and
`PIPELINE_STATUS.md`."""

from __future__ import annotations

import numpy as np

from ..spec import SourceSpec
from ..types import Sample

_STATS = ("mean", "std", "p10", "p50", "p90", "foreground_fraction")


def extract_image2d_features(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    color_mode = spec._raw.get("image", {}).get("color_mode", "grayscale")
    n_channels = 1 if color_mode == "grayscale" else 3
    channel_names = ["gray"] if n_channels == 1 else ["r", "g", "b"]
    names = [f"{ch}__{stat}" for ch in channel_names for stat in _STATS]

    array = sample.arrays.get("image")
    if array is None or array.size == 0:
        return np.full(len(names), np.nan), names

    values: list[float] = []
    for c in range(min(array.shape[0], n_channels)):
        chan = array[c].astype(float)
        finite = chan[np.isfinite(chan)]
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
