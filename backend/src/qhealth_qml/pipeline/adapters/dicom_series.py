"""`dicom_series` adapter: a directory of `.dcm` slice files sharing one
`SeriesInstanceUID`, reconstructed into a 3D volume - tasks.md T056.
Wraps `pydicom` (already a project dependency for header-only parsing in
`ingest/dicom_adapter.py`; this adapter additionally reads PIXEL data,
which that older module deliberately does not).

**Source contract**: `source` is a directory containing one series's slice
files directly, one Sample per directory; a root directory of several such
series-subdirectories also works, one Sample per subdirectory - mirroring
`nifti_volume`'s layout convention for consistency across the two imaging
adapters.

Slices are ordered by `InstanceNumber` (falling back to filename order
when it's absent), so a series with shuffled instance numbers still
reconstructs in correct position order (tasks.md T056's done-condition).
Harmonisation follows the same fixed chain as `nifti_volume`: rescale (HU,
DICOM-only) -> orient -> respace -> intensity (CT window / MR percentile,
branching on the HEADER-confirmed modality, never assumed) -> grid ->
stack.

**A known simplification**: the affine built from `PixelSpacing` +
`SliceThickness` here is diagonal (axis-aligned) - it does not read
`ImageOrientationPatient`/`ImagePositionPatient` direction cosines, so a
series acquired at a genuine gantry tilt or oblique orientation will not
canonicalise correctly. Every synthetic/axial fixture in this build is
axis-aligned, so this does not show up in tests here; it is a real gap for
tilted real-world DICOM."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import imaging as img
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source


class DicomSeriesAdapter:
    name = "dicom_series"
    modalities = ("ct", "mr", "angio")
    formats = (".dcm",)

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            if _dicom_files(path):
                return 0.8
            if any(_dicom_files(sub) for sub in path.iterdir() if sub.is_dir()):
                return 0.7
            return 0.0
        if path.suffix.lower() == ".dcm":
            return 0.6
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        path = _as_path(source)
        if path is None:
            raise ValueError(f"dicom_series adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if not path.is_dir():
            raise ValueError(
                f"dicom_series expects a directory of .dcm slice files (or a root directory of "
                f"such series-subdirectories), not a single file: {path}"
            )

        if _dicom_files(path):
            series_dirs = [path]
        else:
            series_dirs = sorted(sub for sub in path.iterdir() if sub.is_dir() and _dicom_files(sub))
        if not series_dirs:
            raise ValueError(f"no DICOM series directories found under {path}")

        for series_dir in series_dirs:
            yield RawRecord(
                source=source,
                payload={"files": _dicom_files(series_dir)},
                meta={"record_name": series_dir.name, "path": str(series_dir)},
            )

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        import pydicom

        files: list[Path] = raw.payload["files"]
        datasets = [pydicom.dcmread(str(f)) for f in files]

        def sort_key(pair: tuple[Path, "pydicom.Dataset"]) -> int:
            f, ds = pair
            return int(getattr(ds, "InstanceNumber", 0) or 0)

        ordered = sorted(zip(files, datasets), key=sort_key)
        files_sorted = [f for f, _ in ordered]
        datasets = [ds for _, ds in ordered]

        first = datasets[0]
        header_modality = str(getattr(first, "Modality", "")).upper()
        issues: list[Issue] = []
        applied: list[str] = []

        expected = {"ct": "CT", "mr": "MR", "angio": "XA"}.get(spec.modality)
        if expected and header_modality and header_modality != expected:
            issues.append(
                Issue(
                    IssueCode.MODALITY_MISMATCH, "reject",
                    f"DICOM header declares modality '{header_modality}', spec declares '{spec.modality}'.",
                    field=None,
                )
            )

        volume = np.stack([ds.pixel_array.astype(float) for ds in datasets], axis=0)  # (slices, rows, cols)

        slope = float(getattr(first, "RescaleSlope", 1.0))
        intercept = float(getattr(first, "RescaleIntercept", 0.0))
        if header_modality == "CT":
            volume = img.rescale_hu(volume, slope, intercept)
            applied.append("rescale_hu")

        pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(first, "SliceThickness", 1.0) or 1.0)
        current_zooms = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))
        # Axis-aligned affine only - see module docstring's known simplification.
        affine = np.diag([current_zooms[0], current_zooms[1], current_zooms[2], 1.0])

        imaging_cfg = spec._raw.get("imaging", {}) or {}
        target_mm = tuple(imaging_cfg.get("voxel_spacing_mm", (1.0, 1.0, 1.0)))
        ct_windows = imaging_cfg.get("ct_windows", [])
        grid = tuple(imaging_cfg.get("grid", volume.shape))

        if not issues:  # only proceed through the geometry chain if the modality actually matches
            canonical, _ = img.orient_canonical(volume, affine)
            applied.append("orient_RAS")
            respaced = img.respace(canonical, current_zooms, target_mm)
            applied.append(f"respace_{target_mm}mm")

            if header_modality == "CT" and ct_windows:
                channels = [img.apply_ct_window(respaced, c, w) for c, w in ct_windows]
                applied.append(f"ct_window_x{len(ct_windows)}")
            else:
                channels = [img.apply_mr_normalize(respaced)]
                applied.append("zscore_foreground")

            channels = [img.resize_grid(ch, grid) for ch in channels]
            applied.append(f"resize_{grid}")
            stacked = np.stack(channels, axis=0)
            issues.extend(img.check_foreground(stacked))
        else:
            stacked = np.zeros((0, *volume.shape))

        record_name = raw.meta["record_name"]
        sample_id, label = _split_label_suffix(record_name)
        series_uid = str(getattr(first, "SeriesInstanceUID", sample_id))

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(str(getattr(first, "PatientID", sample_id))),
            index_time=None,
            outcome_time=None,
            site=None,
            # No standard label field in a bare DICOM series directory;
            # this build's fixtures encode it in the directory name
            # (`<id>__label<0|1>/`), same convention as the other new adapters.
            label=label,
            fields={},
            arrays={"volume": stacked},
            subgroups={},
            provenance={
                "adapter": self.name,
                "modality": header_modality,
                "series_instance_uid": series_uid,
                "sources": {"path": raw.meta.get("path"), "n_slices": len(files_sorted), "sha256": hashlib.sha256(b"".join(f.read_bytes() for f in files_sorted[:1])).hexdigest()},
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


def _dicom_files(path: Path) -> list[Path]:
    return sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".dcm")


def _hash_subject_id(raw_subject_id: str) -> str:
    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


_LABEL_SUFFIX = re.compile(r"^(?P<id>.+)__label(?P<label>[01])$")


def _split_label_suffix(record_name: str) -> tuple[str, int | None]:
    match = _LABEL_SUFFIX.match(record_name)
    if not match:
        return record_name, None
    return match.group("id"), int(match.group("label"))
