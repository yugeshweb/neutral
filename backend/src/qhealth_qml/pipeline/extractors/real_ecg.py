"""Adapter shim: wires `qhealth_qml.ecg.extract_ecg_features` (the real,
teammate-built deterministic ECG extractor, pulled in from
`shuvam/src/qhealth_qml/ecg.py` on `origin/main`) to this package's
`representation.extractor` contract - `(Sample, SourceSpec) -> (vector,
names)` - without modifying either side. The real function's own contract
is `(signal: np.ndarray, sampling_rate: float) -> (vector, names)`; this
module is the ENTIRE integration surface between them.

Requires all 12 standard leads (`qhealth_qml.ecg.LEADS`) in that exact
set - `wfdb_ecg`'s `signal.channels` must declare all 12 for this to work;
a spec declaring fewer leads (this build's original 3-lead demo fixture,
for instance) fails here with a clear error, which is the correct,
honest behaviour - not a bug in either side."""

from __future__ import annotations

import numpy as np

from ..spec import SourceSpec
from ..types import Sample


def extract_signal_features(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    from qhealth_qml.ecg import extract_ecg_features

    array = sample.arrays.get("signal")
    signal_cfg = spec._raw.get("signal", {}) or {}
    sampling_rate = float(signal_cfg.get("resample_to_hz") or signal_cfg.get("native_hz") or 500.0)

    if array is None or array.size == 0:
        raise ValueError("real_ecg extractor requires sample.arrays['signal'] to be present")

    return extract_ecg_features(array, sampling_rate)
