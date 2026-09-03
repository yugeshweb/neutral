"""Preprocessing built on MONAI transforms rather than hand-rolled equivalents.

Replaces the bespoke loading/orienting/windowing/resampling in `ingestion.py` and
`imaging_hybrid.normalize_volume`. Three reasons, in order of importance:

1. **It fixes the respacing defect (D5).** The hand-rolled path resampled straight to a fixed
   array grid, which normalises voxel *counts* but not physical extent — two scans at different
   mm/voxel end up representing different real-world fields of view in the same tensor slot.
   `Spacingd` resamples to a common physical voxel size *before* the grid resize, which is the
   step that was missing entirely.
2. **It is the field's standard implementation.** `LoadImage` already handles DICOM series
   assembly, RescaleSlope/Intercept for Hounsfield units, and NIfTI affines; `Orientation`
   already canonicalises axis order. Re-deriving these by hand is how subtle bugs enter.
3. The canonical ordering is enforced in one place:

       load -> (HU rescale, by LoadImage) -> orient RAS -> respace -> intensity -> grid -> stack

Note on modality: CT carries absolute Hounsfield units so intensity is *windowed* to a clinical
range; MR has no absolute scale so it is percentile-normalised. Applying the wrong one destroys
contrast, so `modality` is required rather than guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SPACING_MM = (1.0, 1.0, 1.0)


def _monai():
    import monai.transforms as T

    return T


def load_volume(path: str | Path, ensure_channel_first: bool = True) -> tuple[np.ndarray, Any]:
    """Load a NIfTI file or a DICOM series directory via MONAI, returning (array, affine).

    MONAI's reader applies the DICOM rescale slope/intercept, so CT comes back in Hounsfield
    units rather than raw stored values — skipping that is a classic and silent CT bug.
    """

    T = _monai()
    loader = T.LoadImage(image_only=False, ensure_channel_first=ensure_channel_first)
    data, meta = loader(str(path))
    array = np.asarray(data, dtype=np.float32)
    affine = meta.get("affine", meta.get("original_affine"))
    return array, affine


def preprocess_volume(
    array: np.ndarray,
    affine: Any,
    target_grid: tuple[int, int, int],
    modality: str = "MR",
    target_spacing_mm: tuple[float, float, float] | None = DEFAULT_SPACING_MM,
    ct_window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Canonical pipeline: orient -> respace -> intensity -> grid. Returns [D,H,W].

    `target_spacing_mm=None` skips respacing (only appropriate for a cohort verified
    spacing-homogeneous by `cohort_audit`); anything else respaces first, which is the D5 fix.
    """

    T = _monai()
    from monai.data import MetaTensor

    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 3:
        values = values[None, ...]
    tensor = MetaTensor(values, affine=affine) if affine is not None else MetaTensor(values)

    tensor = T.Orientation(axcodes="RAS")(tensor)
    if target_spacing_mm is not None and affine is not None:
        tensor = T.Spacing(pixdim=target_spacing_mm, mode="bilinear")(tensor)

    if modality.upper() == "CT":
        if ct_window is None:
            raise ValueError("CT preprocessing requires an explicit (level, width) window")
        level, width = ct_window
        low, high = level - width / 2.0, level + width / 2.0
        tensor = T.ScaleIntensityRange(
            a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True
        )(tensor)
    else:
        # MR has no absolute scale: clip the extremes so one bright artefact voxel cannot
        # crush the useful dynamic range, then rescale.
        tensor = T.ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        )(tensor)

    tensor = T.Resize(spatial_size=target_grid, mode="trilinear")(tensor)
    return np.asarray(tensor, dtype=np.float32)[0]


def preprocess_study(
    sources: dict[str, str | Path],
    channel_order: list[str],
    target_grid: tuple[int, int, int],
    modality: str = "MR",
    target_spacing_mm: tuple[float, float, float] | None = DEFAULT_SPACING_MM,
    ct_windows: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the model-ready [C,D,H,W] tensor for one study, plus provenance.

    Multi-window CT (one scan, several clinical windows as channels) is handled by passing a
    single source and a `ct_windows` list; multi-sequence MR is handled by passing one source
    per channel.
    """

    provenance: dict[str, Any] = {
        "backend": "monai",
        "ordering": "load -> orient(RAS) -> respace -> intensity -> grid",
        "target_grid": list(target_grid),
        "target_spacing_mm": list(target_spacing_mm) if target_spacing_mm else None,
        "modality": modality,
        "sources": {},
    }

    channels: list[np.ndarray] = []
    if modality.upper() == "CT" and ct_windows and len(sources) == 1:
        only_path = next(iter(sources.values()))
        array, affine = load_volume(only_path)
        provenance["sources"]["ct"] = str(only_path)
        provenance["ct_windows"] = ct_windows
        for window in ct_windows:
            channels.append(
                preprocess_volume(
                    array, affine, target_grid, "CT", target_spacing_mm,
                    (float(window["level"]), float(window["width"])),
                )
            )
    else:
        for name in channel_order:
            if name not in sources:
                raise ValueError(f"missing required channel {name!r}; have {sorted(sources)}")
            array, affine = load_volume(sources[name])
            provenance["sources"][name] = str(sources[name])
            channels.append(
                preprocess_volume(array, affine, target_grid, modality, target_spacing_mm)
            )

    return np.stack(channels, axis=0).astype(np.float32), provenance
