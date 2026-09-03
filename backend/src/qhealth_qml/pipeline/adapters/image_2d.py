"""`image_2d` adapter: plain 2D photos (PNG/JPEG/BMP/WEBP) with NO scanner
metadata - no DICOM header, no NIfTI affine, no voxel spacing. This
modality (`image2d`) is **not part of the team-head spec** (spec.md/
design.md name only tabular/ecg/eeg/gait/ct/mr/angio/genomic) - it exists
to close a real, concrete gap: datasets like Kaggle's "Brain MRI Images
for Brain Tumor Detection" (plain `.jpg` files in `yes/`/`no/` folders,
the exact shape of the `brain-tumor-detection-v1-0-cnn-vgg-16.ipynb`
notebook's input) have no adapter anywhere in the formal spec, and
`dicom_series`/`nifti_volume` correctly refuse them (they need real
scanner metadata this format doesn't have). See `PIPELINE_STATUS.md` for
the full caveat and how this differs from what the spec actually defines.

**Source contract**: `source` is a root directory. Two layouts are
supported:
  - **class subdirectories** (the common ML-dataset shape - `yes/`,
    `no/`, or any two folder names): each image's label is 1 if its
    parent folder name matches `spec.positive_label` (case-insensitive),
    else 0. This is what a Kaggle-style binary image dataset looks like
    out of the box, no renaming required.
  - **a flat directory of images**: one Sample per file, unlabeled
    (predict-time upload), unless the filename carries the
    `<id>__label<0|1>` suffix convention this build's other new adapters
    use for their own synthetic fixtures.

Harmonisation loosely mirrors what the reference notebook does by hand
(decode -> optional foreground crop -> resize -> intensity-normalise),
generalised and made declarative via the spec's `image` block, reusing
`..imaging.apply_mr_normalize`/`check_foreground` since both are already
dimension-agnostic numpy operations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import imaging as img
from ..spec import SourceSpec
from ..types import Issue, QCVerdict, RawRecord, Sample, Source

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class Image2dAdapter:
    name = "image_2d"
    modalities = ("image2d",)
    formats = _IMAGE_SUFFIXES

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            return 0.6 if _find_images(path) else 0.0
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            try:
                header = path.read_bytes()[:8]
            except OSError:
                return 0.0
            if header.startswith(_PNG_MAGIC) or header.startswith(_JPEG_MAGIC):
                return 0.55
            return 0.3
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        path = _as_path(source)
        if path is None:
            raise ValueError(f"image_2d adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if path.is_file():
            files_with_label = [(path, None, path.stem)]
        else:
            files_with_label = _collect_labeled_files(path, spec.positive_label)
        if not files_with_label:
            raise ValueError(f"no image files ({', '.join(_IMAGE_SUFFIXES)}) found under {path}")

        for file_path, folder_label, record_name in files_with_label:
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            yield RawRecord(
                source=source,
                payload={"path": file_path, "folder_label": folder_label},
                meta={"record_name": record_name, "path": str(file_path), "sha256": digest},
            )

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        from PIL import Image

        image_cfg = spec._raw.get("image", {}) or {}
        color_mode = image_cfg.get("color_mode", "grayscale")
        target_size = tuple(image_cfg.get("target_size", (128, 128)))  # (H, W)
        crop_foreground = image_cfg.get("crop_foreground", True)

        file_path: Path = raw.payload["path"]
        pil_img = Image.open(file_path).convert("L" if color_mode == "grayscale" else "RGB")
        array = np.asarray(pil_img, dtype=float)
        if color_mode == "grayscale":
            array = array[np.newaxis, :, :]  # (1, H, W)
        else:
            array = np.moveaxis(array, -1, 0)  # (3, H, W)

        applied: list[str] = ["decode"]

        if crop_foreground:
            array = _crop_to_foreground(array)
            applied.append("crop_foreground")

        array = np.stack([_resize_2d(array[c], target_size) for c in range(array.shape[0])], axis=0)
        applied.append(f"resize_{target_size}")

        array = img.apply_mr_normalize(array)
        applied.append("percentile_clip_zscore")

        issues: list[Issue] = list(img.check_foreground(array))

        record_name = raw.meta["record_name"]
        sample_id, filename_label = _split_label_suffix(record_name)
        label = raw.payload["folder_label"] if raw.payload["folder_label"] is not None else filename_label

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(sample_id),
            index_time=None,
            outcome_time=None,
            site=None,
            label=label,
            fields={},
            arrays={"image": array},
            subgroups={},
            provenance={
                "adapter": self.name,
                "source": {"path": raw.meta.get("path"), "sha256": raw.meta.get("sha256")},
                "color_mode": color_mode,
                "applied": applied,
            },
            issues=issues,
        )

    def qc(self, sample: Sample, spec: SourceSpec) -> QCVerdict:
        return QCVerdict(status="accept", issues=[])


def _as_path(source: Source) -> Path | None:
    if source.kind == "path":
        return Path(source.locator)
    return None


def _find_images(path: Path) -> list[Path]:
    return [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES]


def _collect_labeled_files(root: Path, positive_label: str | None) -> list[tuple[Path, int | None, str]]:
    """Class-subdirectory layout (`yes/`, `no/`, ...) if the root's
    immediate children are directories; otherwise a flat unlabeled
    directory of files. Returns `(file_path, label, record_name)` -
    `record_name` is namespaced by folder (`<folder>__<stem>`) so two
    classes each numbering their own files from 1 (exactly this Kaggle
    dataset's shape) don't collide into one sample_id, which would
    otherwise silently merge two different images into one "subject" for
    grouped splitting."""

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        out: list[tuple[Path, int | None, str]] = []
        for d in sorted(subdirs):
            label = 1 if positive_label and d.name.strip().lower() == positive_label.strip().lower() else 0
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES:
                    out.append((f, label, f"{d.name}__{f.stem}"))
        return out
    return [
        (f, None, f.stem) for f in sorted(root.iterdir()) if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES
    ]


def _crop_to_foreground(array: np.ndarray, threshold_frac: float = 0.08) -> np.ndarray:
    """A simplified stand-in for the reference notebook's contour-based
    crop (`cv2.findContours` + extreme points): a bounding box around
    pixels above a low intensity threshold, summed across channels. Good
    enough to drop black borders/corners; not a segmentation."""

    combined = array.sum(axis=0)
    threshold = combined.max() * threshold_frac
    mask = combined > threshold
    if not mask.any():
        return array
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return array[:, r0:r1 + 1, c0:c1 + 1]


def _resize_2d(channel: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    finite = np.nan_to_num(channel, nan=0.0)
    pil_img = Image.fromarray(finite.astype(np.float32), mode="F")
    resized = pil_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=float)


def _hash_subject_id(raw_subject_id: str) -> str:
    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


_LABEL_SUFFIX = re.compile(r"^(?P<id>.+)__label(?P<label>[01])$")


def _split_label_suffix(record_name: str) -> tuple[str, int | None]:
    match = _LABEL_SUFFIX.match(record_name)
    if not match:
        return record_name, None
    return match.group("id"), int(match.group("label"))
