from __future__ import annotations

import numpy as np
import pytest

from qhealth_qml.angiography import (
    GLOBAL_FEATURES,
    PER_SCALE_FEATURES,
    VESSEL_SCALES,
    extract_angiography_features,
    normalize_frame,
    vesselness,
)


def _synthetic_vessel(size: int = 128, width: float = 3.0, value: float = 0.2) -> np.ndarray:
    """A dark vertical tube on a bright background, like a contrast-filled vessel."""
    frame = np.ones((size, size), dtype=float)
    xs = np.arange(size)
    profile = np.exp(-((xs - size / 2) ** 2) / (2 * width**2))
    frame -= (1.0 - value) * profile[None, :]
    return frame


def _synthetic_blob(size: int = 128, radius: float = 12.0) -> np.ndarray:
    """A dark round blob — the structure vesselness is supposed to reject."""
    frame = np.ones((size, size), dtype=float)
    ys, xs = np.mgrid[0:size, 0:size]
    distance = np.hypot(ys - size / 2, xs - size / 2)
    frame -= 0.8 * np.exp(-(distance**2) / (2 * radius**2))
    return frame


def test_normalize_frame_is_robust_to_a_saturated_outlier():
    # A frame with real dynamic range, plus a burned-in marker / collimator edge.
    rng = np.random.default_rng(7)
    frame = rng.uniform(100.0, 200.0, size=(32, 32))
    frame[0, 0] = 10000.0

    normalized = normalize_frame(frame)

    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    # The outlier must not compress the genuine 100-200 range into a sliver:
    # without percentile scaling everything real would land near 0.
    assert normalized.std() > 0.15


def test_normalize_frame_handles_a_flat_image_without_dividing_by_zero():
    assert np.allclose(normalize_frame(np.full((16, 16), 0.7)), 0.0)


def test_vesselness_prefers_a_tube_over_a_blob():
    tube = vesselness(normalize_frame(_synthetic_vessel()), scale=3.0)
    blob = vesselness(normalize_frame(_synthetic_blob()), scale=3.0)

    # The whole point of the Hessian eigenvalue ratio is rejecting blobs.
    assert tube.max() > blob.max()


def test_vesselness_peaks_near_the_matching_calibre_scale():
    narrow = normalize_frame(_synthetic_vessel(width=2.0))
    wide = normalize_frame(_synthetic_vessel(width=8.0))

    # Peak, not mean: a coarse scale smears its response over a wider area, so
    # the mean rises with scale regardless of the calibre actually present.
    narrow_best = max(VESSEL_SCALES, key=lambda s: vesselness(narrow, s).max())
    wide_best = max(VESSEL_SCALES, key=lambda s: vesselness(wide, s).max())

    # A wider vessel must be best described at a scale no finer than a narrow one.
    assert wide_best >= narrow_best


def test_vesselness_ignores_bright_tubes_when_looking_for_contrast_filled_vessels():
    bright_tube = normalize_frame(2.0 - _synthetic_vessel())  # inverted: bright on dark

    dark_seeking = vesselness(bright_tube, scale=3.0, dark_vessels=True)
    bright_seeking = vesselness(bright_tube, scale=3.0, dark_vessels=False)

    assert bright_seeking.max() > dark_seeking.max()


def test_extract_features_shape_and_names_are_consistent():
    features, names = extract_angiography_features(_synthetic_vessel())

    expected = len(VESSEL_SCALES) * len(PER_SCALE_FEATURES) + len(GLOBAL_FEATURES) + 2
    assert features.shape == (expected,)
    assert len(names) == expected
    assert len(set(names)) == expected  # no duplicate feature names
    assert np.isfinite(features).all()


def _soft_tissue_background(size: int = 128) -> np.ndarray:
    """A vessel-free angiogram region: smooth tissue gradients, no tubes."""
    rng = np.random.default_rng(7)
    ys, xs = np.mgrid[0:size, 0:size]
    gradient = 0.8 + 0.2 * (ys / size) - 0.1 * (xs / size)
    return gradient + rng.normal(0.0, 0.01, (size, size))


def test_extract_features_separates_a_vessel_frame_from_vessel_free_tissue():
    vessel, names = extract_angiography_features(_synthetic_vessel())
    background, _ = extract_angiography_features(_soft_tissue_background())

    # Peak/far-tail statistics, not the mean: a vessel covers a few percent of
    # the frame, so its signal lives in the tail while diffuse background noise
    # dominates the mean.
    for feature in ("scale2_vesselness_p99", "scale2_vesselness_max"):
        index = names.index(feature)
        assert vessel[index] > background[index], feature
    # Area fraction of strong tubular response also separates them.
    area = names.index("scale2_vesselness_area_fraction")
    assert vessel[area] > background[area]


def test_blank_frame_is_not_amplified_into_apparent_vessels():
    """A dropped/blank acquisition must not have its sensor noise stretched into
    structure — relative normalisation would otherwise manufacture vessels."""
    rng = np.random.default_rng(7)
    blank = np.ones((128, 128)) + rng.normal(0.0, 0.01, (128, 128))

    features, names = extract_angiography_features(blank)

    for feature in ("scale2_vesselness_mean", "scale2_vesselness_max"):
        assert features[names.index(feature)] == pytest.approx(0.0, abs=1e-9)


def test_extract_features_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D grayscale"):
        extract_angiography_features(np.zeros((4, 4, 3)))
