"""BHSD ingestion: non-contrast head CT -> intracranial-haemorrhage subtype classification.

BHSD (Brain Haemorrhage Segmentation Dataset, MIT licence) ships 192 head-CT volumes with
voxel-level annotations for the five subtypes a neuroradiologist reports: epidural,
intraparenchymal, intraventricular, subarachnoid and subdural. It is a re-annotation of
RSNA ICH data, which is how it is distributable without RSNA's access gate.

Task framing: every volume is haemorrhage-positive, so "bleed vs no bleed" has no negative
class here. What the annotations do support -- and what actually changes management -- is
*which compartment* the blood is in. Subdural, epidural and intraventricular bleeds have
different causes, urgency and surgical answers, so subtype presence is the clinically real
binary. Measured over the 192 volumes: intraparenchymal 127 (66.1%), subarachnoid 109
(56.8%), intraventricular 104 (54.2%), subdural 70 (36.5%), epidural 23 (12.0%).
Intraventricular is the default -- best class balance, and IVH independently predicts
hydrocephalus and worse outcome, often driving external-ventricular-drain placement.

Unlike MR, CT carries absolute Hounsfield units, so intensity is windowed rather than
percentile-normalised: the three channels are the three windows a radiologist actually
toggles through at the workstation (brain, subdural, bone), which is also the standard
multi-window input used across the RSNA ICH literature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .imaging_hybrid import VolumeDataset, load_nifti_volume

SUBTYPES = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}
DEFAULT_SUBTYPE = 3  # intraventricular -- best balance at 104/192

# (window level, window width) in Hounsfield units.
CT_WINDOWS = (
    (40.0, 80.0),    # brain: separates acute blood (~50-70 HU) from parenchyma (~20-40 HU)
    (75.0, 215.0),   # subdural: widened to catch thin extra-axial collections next to bone
    (600.0, 2800.0), # bone: skull/fracture context
)


def apply_ct_window(volume: np.ndarray, level: float, width: float) -> np.ndarray:
    """Clip a Hounsfield-unit volume to one radiological window and scale it to [0, 1]."""

    low, high = level - width / 2.0, level + width / 2.0
    windowed = np.clip(np.asarray(volume, dtype=np.float32), low, high)
    return ((windowed - low) / max(high - low, 1e-5)).astype(np.float32, copy=False)


def _resample(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    values = np.asarray(volume, dtype=np.float32)
    if values.shape == target_shape:
        return values
    factors = tuple(t / c for t, c in zip(target_shape, values.shape))
    resampled = zoom(values, factors, order=1).astype(np.float32, copy=False)
    fixed = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(t, c)) for t, c in zip(target_shape, resampled.shape))
    fixed[slices] = resampled[slices]
    return fixed


def load_bhsd(
    root: str | Path,
    subtype: int = DEFAULT_SUBTYPE,
    target_shape: tuple[int, int, int] = (64, 64, 32),
    limit: int = 0,
) -> VolumeDataset:
    """Load BHSD into a 3-channel (multi-window CT) volume dataset labelled by subtype presence."""

    if subtype not in SUBTYPES:
        raise ValueError(f"subtype must be one of {sorted(SUBTYPES)}")
    root = Path(root)
    image_dir, mask_dir = root / "images", root / "ground truths"
    if not image_dir.exists() or not mask_dir.exists():
        raise ValueError(f"expected 'images' and 'ground truths' under {root}")

    image_paths = sorted(image_dir.glob("*.nii.gz"))
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"no NIfTI volumes found under {image_dir}")

    volumes: list[np.ndarray] = []
    labels: list[int] = []
    row_ids: list[str] = []
    subtype_counts: dict[str, int] = {name: 0 for name in SUBTYPES.values()}
    skipped: list[str] = []

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            skipped.append(f"{image_path.name}(no mask)")
            continue
        hounsfield = load_nifti_volume(image_path)
        channels = [
            _resample(apply_ct_window(hounsfield, level, width), target_shape)
            for level, width in CT_WINDOWS
        ]
        mask = load_nifti_volume(mask_path).astype(int)
        present = {int(v) for v in np.unique(mask) if v > 0}
        for code in present:
            if code in SUBTYPES:
                subtype_counts[SUBTYPES[code]] += 1

        volumes.append(np.stack(channels, axis=0))
        labels.append(int(subtype in present))
        row_ids.append(image_path.name.replace(".nii.gz", ""))
        if index % 25 == 0 or index == len(image_paths):
            print(f"loaded BHSD volume {index}/{len(image_paths)}", flush=True)

    if not volumes:
        raise ValueError("BHSD produced no usable volumes")

    y = np.asarray(labels, dtype=int)
    name = SUBTYPES[subtype]
    return VolumeDataset(
        volumes=np.asarray(volumes, dtype=np.float32),
        y=y,
        groups=np.asarray(row_ids, dtype=str),  # one CT per patient
        row_ids=np.asarray(row_ids, dtype=str),
        channel_names=[f"ct_window_L{int(l)}_W{int(w)}" for l, w in CT_WINDOWS],
        name=f"bhsd-ich-{name}",
        positive_label=f"{name}_haemorrhage_present",
        negative_label=f"{name}_haemorrhage_absent",
        positive_definition=(
            f"expert voxel annotation contains {name} haemorrhage; every volume in this "
            "release is haemorrhage-positive overall, so this is a compartment/subtype "
            "question, not bleed-vs-no-bleed"
        ),
        provenance={
            "source": "BHSD (HuggingFace Wendy-Fly/BHSD, label_192), MIT licence",
            "note": "re-annotation of RSNA ICH data",
            "raw_imaging": True,
            "modality": "non-contrast head CT (Hounsfield units)",
            "ct_windows": [{"level": l, "width": w} for l, w in CT_WINDOWS],
            "target_shape": list(target_shape),
            "subtype": name,
            "subtype_volume_counts": subtype_counts,
            "skipped": skipped,
        },
    )


def summarize(dataset: VolumeDataset) -> dict[str, Any]:
    return {
        "volumes": int(len(dataset.y)),
        "positive": int(dataset.y.sum()),
        "negative": int((1 - dataset.y).sum()),
        "subtype": dataset.provenance.get("subtype"),
        "subtype_volume_counts": dataset.provenance.get("subtype_volume_counts"),
        "channels": dataset.channel_names,
    }


if __name__ == "__main__":
    import sys

    ds = load_bhsd(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(json.dumps(summarize(ds), indent=2))
