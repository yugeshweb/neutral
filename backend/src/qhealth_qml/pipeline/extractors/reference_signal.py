"""Reference deterministic feature extractor for signal modalities (ecg,
eeg, gait) - design.md §9.2.7 `representation.kind == "deterministic"`.

**THIS IS A PLACEHOLDER, not the team's real feature-extraction model.**
spec.md's own worked example names a specific extractor
(`ecg.extract_ecg_features`) that is expected to already exist as part of
the disease-detection "model scripts" mentioned elsewhere in this project;
those scripts are not in this repository. This module exists so the
signal ingestion path (decode -> filter -> resample -> normalize ->
FEATURE VECTOR) is real and testable end-to-end today, using ordinary
per-channel summary statistics instead of the team's actual hand-tuned
ECG/EEG/gait features.

Swapping this out for the real thing is a one-line change to
`SourceSpec.representation.extractor` in the spec JSON - no pipeline code
changes required, by design (see `..representation`)."""

from __future__ import annotations

import numpy as np

from ..spec import SourceSpec
from ..types import Sample

_STATS = ("mean", "std", "rms", "zero_crossing_rate", "min", "max")


def extract_signal_features(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    channels: list[str] = list(spec._raw.get("signal", {}).get("channels", []))
    names = [f"{ch}__{stat}" for ch in channels for stat in _STATS]

    array = sample.arrays.get("signal")
    if array is None or array.size == 0:
        return np.full(len(names), np.nan), names

    values: list[float] = []
    for c in range(min(array.shape[0], len(channels))):
        chan = array[c].astype(float)
        finite = chan[np.isfinite(chan)]
        if finite.size == 0:
            values.extend([float("nan")] * len(_STATS))
            continue
        mean = float(finite.mean())
        std = float(finite.std())
        rms = float(np.sqrt(np.mean(finite ** 2)))
        zero_crossings = float(np.sum(np.diff(np.sign(finite)) != 0)) / max(finite.size - 1, 1)
        values.extend([mean, std, rms, zero_crossings, float(finite.min()), float(finite.max())])
    # A channel the array didn't have (shorter than declared) contributes NaN, imputed downstream.
    while len(values) < len(names):
        values.append(float("nan"))
    return np.asarray(values, dtype=float), names
