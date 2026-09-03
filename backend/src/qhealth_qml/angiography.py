"""Deterministic frame features for X-ray coronary angiography.

This is the imaging counterpart to `ecg.py`: a small, dependency-light, fully
deterministic feature extractor that turns an angiography frame into a numeric
row the existing classical and Qiskit runners can consume. It exists so the
imaging modality does not depend on torch — the learned 2-D CNN encoder is the
better representation when a GPU is available, exactly as `raw_hybrid.py` is for
ECG, but the platform should not be unable to use imaging at all without one.

The features are chosen for the actual clinical target rather than being generic
image statistics. A coronary angiogram is a contrast-filled vessel tree against
soft-tissue background, and a stenosis presents as a *local narrowing of vessel
calibre*. So the core descriptor here is multi-scale Hessian vesselness: at each
smoothing scale the Hessian's eigenvalues separate tube-like ridges (one large
and one small eigenvalue) from blobs and flat background, and sweeping the scale
reports which vessel calibres are present. A vessel tree that is well filled at
coarse scales but drops away at finer ones carries different scale statistics
from one that narrows, which is the signal a stenosis model needs.

Frangi et al., "Multiscale vessel enhancement filtering" (MICCAI 1998) is the
standard formulation; the implementation below is a deliberately simple 2-D
version of it built on `scipy.ndimage` Gaussian derivatives, with no new
dependency and no third-party vesselness code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage

VESSEL_SCALES = (1.0, 2.0, 4.0, 8.0)

# A vessel tree covers only a few percent of a frame, so mid percentiles land in
# empty background and read 0 no matter how strong the vessel is. Peak and far-tail
# statistics are what actually register a sparse tubular structure.
PER_SCALE_FEATURES = (
    "vesselness_mean",
    "vesselness_std",
    "vesselness_max",
    "vesselness_p99",
    "vesselness_area_fraction",
)

GLOBAL_FEATURES = (
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_p90",
    "intensity_skew",
    "gradient_mean",
    "gradient_std",
    "gradient_p90",
    "laplacian_std",
    "dark_area_fraction",
    "local_contrast_mean",
    "entropy",
)


def load_frame(path: str | Path, target_size: int = 256) -> np.ndarray:
    """Load one angiography frame as a normalised square grayscale array."""

    from PIL import Image

    with Image.open(path) as handle:
        frame = handle.convert("L").resize((target_size, target_size), Image.BILINEAR)
    return normalize_frame(np.asarray(frame, dtype=float))


def normalize_frame(frame: np.ndarray, min_dynamic_range: float = 0.05) -> np.ndarray:
    """Scale a frame to [0, 1] robustly, ignoring extreme outliers.

    Percentile scaling rather than min/max: angiography frames routinely carry
    saturated collimator edges and burned-in markers, and a single white pixel
    would otherwise compress the whole vessel range into a sliver.

    `min_dynamic_range` guards the opposite failure. Stretching is relative, so a
    blank or dropped acquisition carrying nothing but sensor noise would have that
    noise amplified to full contrast and then score as genuine structure
    downstream — the extractor would hallucinate vessels in an empty frame. When
    the 1-99 percentile spread is below this fraction of the frame's own level,
    the frame is reported as empty instead of being amplified. The default is a
    heuristic, not a calibrated constant; raise it for noisier detectors.
    """

    values = np.asarray(frame, dtype=float)
    if values.ndim != 2:
        raise ValueError("angiography frame must be a 2-D grayscale array")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(values, [1.0, 99.0])
    spread = float(high - low)
    level = max(abs(float(np.median(values))), 1e-8)
    if spread < 1e-8 or spread < min_dynamic_range * level:
        return np.zeros_like(values)
    return np.clip((values - low) / spread, 0.0, 1.0)


def vesselness(frame: np.ndarray, scale: float, dark_vessels: bool = True) -> np.ndarray:
    """Frangi-style 2-D tubularity response at one smoothing scale.

    Returns a per-pixel score that is high on tube-like ridges of roughly `scale`
    width and low on blobs and flat background. `dark_vessels` matches X-ray
    angiography, where iodine contrast makes vessels *darker* than surroundings.
    """

    values = np.asarray(frame, dtype=float)
    # Gamma-normalised Gaussian derivatives keep responses comparable across
    # scales, which is what makes the scale sweep interpretable as calibre.
    dxx = ndimage.gaussian_filter(values, scale, order=(0, 2)) * scale**2
    dyy = ndimage.gaussian_filter(values, scale, order=(2, 0)) * scale**2
    dxy = ndimage.gaussian_filter(values, scale, order=(1, 1)) * scale**2

    # Closed-form eigenvalues of the symmetric 2x2 Hessian at every pixel.
    half_trace = 0.5 * (dxx + dyy)
    gap = np.sqrt(np.maximum(0.25 * (dxx - dyy) ** 2 + dxy**2, 0.0))
    lambda_1 = half_trace - gap
    lambda_2 = half_trace + gap
    # Order by magnitude: |small| <= |large|.
    swap = np.abs(lambda_1) > np.abs(lambda_2)
    small = np.where(swap, lambda_2, lambda_1)
    large = np.where(swap, lambda_1, lambda_2)

    # Blobness: near 0 for an ideal tube, near 1 for a blob.
    blobness = np.abs(small) / np.maximum(np.abs(large), 1e-10)
    # Structureness: suppresses response in flat noise.
    structureness = np.sqrt(small**2 + large**2)
    # Frangi's `c` is conventionally half the maximum Hessian norm in the image,
    # but that is purely relative: on an almost-empty frame the maximum is itself
    # tiny, every noise ripple then clears it, and pure noise scores *higher*
    # vesselness than a real vessel. Flooring `c` in absolute terms — meaningful
    # because the frame is normalised to [0, 1] and the derivatives are
    # gamma-normalised — keeps a frame with no real structure scoring near zero.
    STRUCTURENESS_FLOOR = 5e-3
    scale_c = max(0.5 * float(np.max(structureness)), STRUCTURENESS_FLOOR)

    response = np.exp(-(blobness**2) / (2 * 0.5**2)) * (
        1.0 - np.exp(-(structureness**2) / (2 * scale_c**2))
    )
    # A dark tube on a bright background has a positive large eigenvalue; the
    # wrong sign is background structure, not a contrast-filled vessel.
    valid = large > 0 if dark_vessels else large < 0
    return np.where(valid, response, 0.0)


def _global_features(frame: np.ndarray) -> list[float]:
    flat = frame.ravel()
    gy, gx = np.gradient(frame)
    gradient = np.hypot(gx, gy)
    laplacian = ndimage.laplace(frame)
    local_mean = ndimage.uniform_filter(frame, size=9)
    centered = flat - float(np.mean(flat))
    deviation = float(np.std(flat))
    skew = float(np.mean(centered**3) / (deviation**3 + 1e-12))

    histogram, _ = np.histogram(flat, bins=32, range=(0.0, 1.0), density=False)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0

    return [
        float(np.mean(flat)),
        deviation,
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 90)),
        skew,
        float(np.mean(gradient)),
        float(np.std(gradient)),
        float(np.percentile(gradient, 90)),
        float(np.std(laplacian)),
        float(np.mean(frame < 0.35)),
        float(np.mean(np.abs(frame - local_mean))),
        entropy,
    ]


def extract_angiography_features(
    frame: np.ndarray,
    scales: Sequence[float] = VESSEL_SCALES,
) -> tuple[np.ndarray, list[str]]:
    """Extract deterministic vesselness and appearance features from one frame."""

    normalized = normalize_frame(frame)
    features: list[float] = []
    names: list[str] = []

    for scale in scales:
        response = vesselness(normalized, float(scale))
        peak = float(np.max(response))
        strong = response > (0.5 * peak) if peak > 0 else np.zeros_like(response, dtype=bool)
        features.extend(
            [
                float(np.mean(response)),
                float(np.std(response)),
                peak,
                float(np.percentile(response, 99)),
                float(np.mean(strong)),
            ]
        )
        names.extend(f"scale{scale:g}_{name}" for name in PER_SCALE_FEATURES)

    features.extend(_global_features(normalized))
    names.extend(GLOBAL_FEATURES)

    # Calibre profile across scales: which vessel widths dominate the frame.
    # A tree that loses coarse-scale response relative to fine carries different
    # ratios from a fully-filled one, which is the stenosis-relevant summary.
    per_scale_mean = np.asarray(
        [features[index * len(PER_SCALE_FEATURES)] for index in range(len(scales))],
        dtype=float,
    )
    total = float(per_scale_mean.sum())
    if total > 0:
        dominant = int(np.argmax(per_scale_mean))
        coarse_to_fine = float(per_scale_mean[-1] / (per_scale_mean[0] + 1e-12))
    else:
        dominant = -1
        coarse_to_fine = 0.0
    features.extend([float(dominant), coarse_to_fine])
    names.extend(["dominant_calibre_scale_index", "coarse_to_fine_vesselness_ratio"])

    return np.asarray(features, dtype=float), names


def extract_frames_dataset(
    frame_paths: Iterable[str | Path],
    labels: Iterable[int],
    target_size: int = 256,
    scales: Sequence[float] = VESSEL_SCALES,
    row_ids: Iterable[str] | None = None,
    name: str = "coronary-angiography",
) -> Any:
    """Build a `LoadedDataset` of deterministic frame features for the runners."""

    from .experiment import LoadedDataset

    paths = [Path(item) for item in frame_paths]
    y = np.asarray(list(labels), dtype=int)
    if len(paths) != len(y):
        raise ValueError("frame_paths and labels must have equal length")
    if not paths:
        raise ValueError("no angiography frames supplied")

    matrix: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for index, path in enumerate(paths, start=1):
        features, names = extract_angiography_features(load_frame(path, target_size), scales)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("angiography frames produced inconsistent feature names")
        matrix.append(features)
        if index % 100 == 0 or index == len(paths):
            print(f"extracted angiography features {index}/{len(paths)}", flush=True)

    assert feature_names is not None
    identifiers = (
        np.asarray([str(item) for item in row_ids], dtype=str)
        if row_ids is not None
        else np.asarray([path.stem for path in paths], dtype=str)
    )
    return LoadedDataset(
        name=name,
        X=np.asarray(matrix, dtype=float),
        y=y,
        feature_names=list(feature_names),
        positive_label="significant_coronary_lesion",
        negative_label="no_significant_lesion",
        provenance={
            "source": "X-ray coronary angiography frames",
            "features": "multi-scale Hessian vesselness plus appearance statistics",
            "reference": "Frangi et al., Multiscale vessel enhancement filtering, MICCAI 1998",
        },
        row_ids=identifiers,
    )
