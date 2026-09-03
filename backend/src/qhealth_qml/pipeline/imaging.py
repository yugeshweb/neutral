"""Imaging harmonisation chain: decode -> rescale -> orient -> respace ->
intensity -> grid -> stack, in that FIXED order (design.md §9.2.6,
tasks.md T050, FR-057 through FR-064). Respacing must precede grid
resizing (FR-059) - two sequences at different native voxel spacing must
describe the same physical field of view once stacked into one tensor;
skip that step and they'd look identical in shape while covering different
tissue, which is the defect this ordering exists to prevent."""

from __future__ import annotations

import numpy as np

from .types import Issue, IssueCode


def rescale_hu(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """DICOM only: HU = pixel * RescaleSlope + RescaleIntercept (FR-057)."""

    return pixel_array.astype(float) * slope + intercept


def orient_canonical(volume: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalise to RAS+ via nibabel's own orientation machinery, so six
    axis permutations of the same physical volume converge on one
    identical array (FR-058, SC-011)."""

    import nibabel as nib

    img = nib.Nifti1Image(volume, affine)
    canonical = nib.as_closest_canonical(img)
    return np.asarray(canonical.dataobj), canonical.affine


def respace(volume: np.ndarray, current_zooms: tuple[float, float, float], target_mm: tuple[float, float, float]) -> np.ndarray:
    """Resample to the declared voxel spacing BEFORE grid resizing
    (FR-059)."""

    from scipy import ndimage

    zoom_factors = tuple(cz / tm for cz, tm in zip(current_zooms, target_mm))
    if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
        return volume
    return ndimage.zoom(volume, zoom_factors, order=1)


def apply_ct_window(volume_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip to [center-width/2, center+width/2] and rescale to [0, 1]
    (FR-061). A constant affine rescale of the whole volume MUST change
    CT output (SC-013) - HU values are diagnostic, unlike MR intensities."""

    lo, hi = center - width / 2, center + width / 2
    clipped = np.clip(volume_hu, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-9)


def apply_mr_normalize(volume: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile clip, then z-score over the foreground only (FR-062). A
    constant rescale must NOT change MR output (SC-013) - MR intensities
    carry no absolute physical scale the way HU does."""

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return volume
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    clipped = np.clip(volume, lo, hi)
    foreground = clipped[clipped > lo]
    if foreground.size == 0:
        return clipped
    mean, std = float(foreground.mean()), float(foreground.std())
    std = std if std > 1e-9 else 1.0
    return (clipped - mean) / std


def resize_grid(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Crop/pad (never interpolate) to the declared grid, AFTER respacing
    has already put every channel in the same physical units (FR-060)."""

    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current > target:
            start = (current - target) // 2
            out = np.take(out, range(start, start + target), axis=axis)
        elif current < target:
            pad = target - current
            pad_before = pad // 2
            pad_after = pad - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant")
    return out


def stack_channels(
    channel_arrays: dict[str, np.ndarray], order: list[str], required: set[str],
) -> tuple[np.ndarray | None, list[Issue]]:
    """Assembled in DECLARED order, never the caller's key order (FR-063).
    A missing required channel refuses the whole record rather than
    zero-filling it (FR-064) - a zero-filled channel reads as uniformly
    dark tissue to a downstream encoder, a real and wrong signal."""

    issues: list[Issue] = []
    missing_required = [name for name in order if name in required and name not in channel_arrays]
    for name in missing_required:
        issues.append(
            Issue(IssueCode.SEQUENCE_MISSING, "reject", f"Required sequence '{name}' is absent; refusing rather than substituting zeros.", field=name)
        )
    if missing_required:
        return None, issues
    present = [name for name in order if name in channel_arrays]
    if not present:
        issues.append(Issue(IssueCode.SEQUENCE_MISSING, "reject", "none of the declared sequences were present in this record.", field=None))
        return None, issues
    return np.stack([channel_arrays[name] for name in present], axis=0), issues


def check_foreground(volume: np.ndarray, *, min_foreground_fraction: float = 0.01) -> list[Issue]:
    """FR-067: an all-air / all-background volume is refused rather than
    scored as if it were a normal, uninformative case."""

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return [Issue(IssueCode.NON_FINITE_INPUT, "reject", "volume has no finite voxels.", field=None)]
    foreground = finite[finite > finite.min()]
    fraction = float(foreground.size) / float(finite.size)
    if fraction < min_foreground_fraction:
        return [Issue(IssueCode.LOW_FOREGROUND, "reject", f"only {fraction:.1%} of voxels are above background - likely an empty/air volume.", field=None)]
    return []
