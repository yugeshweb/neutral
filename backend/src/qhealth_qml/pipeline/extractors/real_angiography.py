"""Adapter shim: wires `qhealth_qml.angiography.extract_angiography_features`
(the real, teammate-built deterministic angiography extractor, pulled in
from `shuvam/src/qhealth_qml/angiography.py` on `origin/main`) to this
package's `representation.extractor` contract, the same pattern as
`real_ecg.py`. The real function takes a single 2-D grayscale frame
(`np.ndarray`, shape `(H, W)`); `image_2d`'s harmonized `Sample.arrays["image"]`
is `(C, H, W)` - this shim takes channel 0 (grayscale, or luma-equivalent
for RGB)."""

from __future__ import annotations

import numpy as np

from ..spec import SourceSpec
from ..types import Sample


def extract_image2d_features(sample: Sample, spec: SourceSpec) -> tuple[np.ndarray, list[str]]:
    from qhealth_qml.angiography import extract_angiography_features

    array = sample.arrays.get("image")
    if array is None or array.size == 0:
        raise ValueError("real_angiography extractor requires sample.arrays['image'] to be present")

    frame = array[0]  # first channel - grayscale or luma-equivalent
    return extract_angiography_features(frame)
