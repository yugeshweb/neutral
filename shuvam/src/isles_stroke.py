"""ISLES 2022 ingestion: acute ischemic stroke MRI -> infarct-core-volume classification.

Task framing (important, and deliberately not "detect stroke"): every ISLES 2022 case is a
stroke patient, so the release contains no negative class and a detection task cannot be
built from it honestly. What the release does support is the measurement that actually
drives acute stroke treatment decisions -- **infarct core volume**. DAWN used core cutoffs of
21/31/51 mL (age- and NIHSS-stratified) and DEFUSE-3 excluded cores >= 70 mL; core estimation
from diffusion imaging is exactly what deployed tools (RAPID, Viz.ai) compute for thrombectomy
triage. The default here is DAWN's 21 mL -- see the constants below for why.

So the label is derived from the expert lesion mask's physical volume, while the model sees
only the DWI/ADC/FLAIR images -- never the mask. This is a real radiology task on the real
sequences a stroke radiologist reads, not a proxy invented to fit the pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .imaging_hybrid import VolumeDataset, load_nifti_volume, normalize_volume

CHANNELS = ("dwi", "adc", "flair")

# Published thrombectomy core-volume cutoffs. DEFUSE-3 excluded cores >= 70 mL; DAWN used
# age/NIHSS-stratified cutoffs of 21, 31 and 51 mL. Measured on this release, ISLES 2022's
# lesions skew small (median 6.7 mL, mean 23.4 mL, max 482 mL), so the DEFUSE-3 cutoff leaves
# only 22/250 positives -- too thin to evaluate honestly. The DAWN 21 mL cutoff is used as the
# default instead: it is an equally real clinical decision threshold and yields 65/250 (26%).
DEFUSE3_CORE_ML = 70.0
DAWN_CORE_ML = 21.0
DEFAULT_CORE_ML = DAWN_CORE_ML


def _find_case_files(case_dir: Path, derivatives_dir: Path | None) -> dict[str, Path]:
    """Locate this case's dwi/adc/flair images and its lesion mask (BIDS layout)."""

    found: dict[str, Path] = {}
    for path in case_dir.rglob("*.nii*"):
        name = path.name.lower()
        if "adc" in name:
            found.setdefault("adc", path)
        elif "dwi" in name:
            found.setdefault("dwi", path)
        elif "flair" in name:
            found.setdefault("flair", path)
    if derivatives_dir is not None and derivatives_dir.exists():
        for path in derivatives_dir.rglob("*.nii*"):
            if "msk" in path.name.lower() or "mask" in path.name.lower() or "lesion" in path.name.lower():
                found.setdefault("mask", path)
                break
    return found


def _voxel_volume_ml(path: Path) -> float:
    """Physical volume of one voxel in millilitres, from the NIfTI header."""

    import nibabel as nib

    header = nib.load(str(path)).header
    zooms = np.asarray(header.get_zooms()[:3], dtype=float)  # mm per axis
    return float(np.prod(zooms)) / 1000.0  # mm^3 -> mL


def load_isles2022(
    root: str | Path,
    target_shape: tuple[int, int, int] = (64, 64, 64),
    core_threshold_ml: float = DEFAULT_CORE_ML,
    limit: int = 0,
) -> VolumeDataset:
    """Load ISLES 2022 into a 3-channel volume dataset labelled by infarct core volume."""

    root = Path(root)
    # BIDS: rawdata/sub-strokecaseXXXX/..., derivatives/sub-strokecaseXXXX/...
    raw_root = next((p for p in (root / "rawdata", root) if p.exists()), root)
    derivatives_root = next(
        (p for p in (root / "derivatives", root.parent / "derivatives") if p.exists()), None
    )

    case_dirs = sorted(
        p for p in raw_root.iterdir() if p.is_dir() and re.match(r"sub-", p.name, re.IGNORECASE)
    )
    if not case_dirs:
        raise ValueError(f"no BIDS sub-* case directories found under {raw_root}")
    if limit:
        case_dirs = case_dirs[:limit]

    volumes: list[np.ndarray] = []
    core_volumes: list[float] = []
    row_ids: list[str] = []
    skipped: list[str] = []

    for index, case_dir in enumerate(case_dirs, start=1):
        derivative_dir = (derivatives_root / case_dir.name) if derivatives_root else None
        files = _find_case_files(case_dir, derivative_dir)
        if not all(key in files for key in CHANNELS) or "mask" not in files:
            skipped.append(f"{case_dir.name}(missing {sorted(set(CHANNELS + ('mask',)) - set(files))})")
            continue

        channels = [normalize_volume(load_nifti_volume(files[name]), target_shape) for name in CHANNELS]
        mask = load_nifti_volume(files["mask"])
        core_ml = float((mask > 0).sum()) * _voxel_volume_ml(files["mask"])

        volumes.append(np.stack(channels, axis=0))
        core_volumes.append(core_ml)
        row_ids.append(case_dir.name)
        if index % 25 == 0 or index == len(case_dirs):
            print(f"loaded ISLES case {index}/{len(case_dirs)}", flush=True)

    if not volumes:
        raise ValueError("ISLES 2022 produced no usable cases")

    core_array = np.asarray(core_volumes, dtype=float)
    y = (core_array >= core_threshold_ml).astype(int)

    return VolumeDataset(
        volumes=np.asarray(volumes, dtype=np.float32),
        y=y,
        groups=np.asarray(row_ids, dtype=str),  # one scan per patient
        row_ids=np.asarray(row_ids, dtype=str),
        channel_names=list(CHANNELS),
        name="isles2022-stroke-core",
        positive_label="large_infarct_core",
        negative_label="small_infarct_core",
        positive_definition=(
            f"expert-annotated diffusion lesion volume >= {core_threshold_ml:.0f} mL "
            f"({'DAWN' if abs(core_threshold_ml - DAWN_CORE_ML) < 1e-6 else 'DEFUSE-3' if abs(core_threshold_ml - DEFUSE3_CORE_ML) < 1e-6 else 'custom'} "
            "thrombectomy core-volume cutoff); measured from the expert mask, "
            "while the model observes only DWI/ADC/FLAIR"
        ),
        provenance={
            "source": "ISLES 2022 (Zenodo 7153326), CC BY 4.0",
            "modalities": list(CHANNELS),
            "raw_imaging": True,
            "target_shape": list(target_shape),
            "core_threshold_ml": core_threshold_ml,
            "core_volume_ml_summary": {
                "min": float(core_array.min()),
                "median": float(np.median(core_array)),
                "max": float(core_array.max()),
                "mean": float(core_array.mean()),
            },
            "skipped_cases": skipped,
        },
    )


def summarize_core_volumes(dataset: VolumeDataset) -> dict[str, Any]:
    summary = dataset.provenance.get("core_volume_ml_summary", {})
    return {
        "cases": int(len(dataset.y)),
        "positives_large_core": int(dataset.y.sum()),
        "negatives_small_core": int((1 - dataset.y).sum()),
        "core_volume_ml": summary,
        "skipped": dataset.provenance.get("skipped_cases", []),
    }


if __name__ == "__main__":  # quick structural check
    import sys

    ds = load_isles2022(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(json.dumps(summarize_core_volumes(ds), indent=2))
