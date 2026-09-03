"""Cohort geometry audit: does this dataset actually suffer the respacing defect (D5)?

Resampling volumes to a fixed *array* grid normalises voxel counts, not physical extent. If two
subjects were acquired at different mm/voxel, they occupy different real-world fields of view in
the same tensor slot and the encoder sees anatomy at inconsistent scale. Whether that matters is a
property of the cohort, not of the code — a single-site 1 mm isotropic collection is unaffected, a
multi-vendor collection almost certainly is not.

This exists because that distinction decides how a number should be read. A result measured on a
spacing-homogeneous cohort is a clean measurement; the same result on a spacing-variable cohort was
measured under a known defect and must not be treated as a reference value to preserve. Recording
the verdict *next to the number* — and ideally inside the artifact — is what keeps the two from
being confused later.

Run it on a directory of NIfTI files, a NIfTI-bearing zip, or a list of paths:

    python -m qhealth_qml.cohort_audit /path/to/nifti_dir
    python -m qhealth_qml.cohort_audit /path/to/archive.zip --inner T1_original/
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Spacing spread below this (mm, per axis, max-minus-min across the cohort) is treated as
# homogeneous. 0.1 mm is well inside the range where resampling to a common grid preserves
# physical correspondence for brain imaging.
HOMOGENEITY_TOLERANCE_MM = 0.1


@dataclass
class CohortGeometry:
    n_scans: int
    spacings: list[list[float]]
    shapes: list[list[int]]
    fields_of_view_mm: list[list[float]]
    unreadable: list[str] = field(default_factory=list)

    @property
    def spacing_spread_mm(self) -> list[float]:
        if not self.spacings:
            return []
        array = np.asarray(self.spacings, dtype=float)
        return (array.max(axis=0) - array.min(axis=0)).round(4).tolist()

    @property
    def fov_spread_mm(self) -> list[float]:
        if not self.fields_of_view_mm:
            return []
        array = np.asarray(self.fields_of_view_mm, dtype=float)
        return (array.max(axis=0) - array.min(axis=0)).round(2).tolist()

    @property
    def is_homogeneous(self) -> bool:
        spread = self.spacing_spread_mm
        return bool(spread) and all(value <= HOMOGENEITY_TOLERANCE_MM for value in spread)

    def verdict(self) -> dict[str, Any]:
        """The line that should travel with any number measured on this cohort."""

        if not self.spacings:
            return {
                "d5_exposure": "unknown",
                "reason": "no readable headers; spacing could not be measured",
                "n_scans": self.n_scans,
            }
        if self.is_homogeneous:
            return {
                "d5_exposure": "not_affected",
                "reason": (
                    f"spacing is uniform across {self.n_scans} scans "
                    f"(per-axis spread {self.spacing_spread_mm} mm <= {HOMOGENEITY_TOLERANCE_MM} mm); "
                    "resampling to a common grid preserves physical correspondence"
                ),
                "n_scans": self.n_scans,
                "spacing_spread_mm": self.spacing_spread_mm,
                "fov_spread_mm": self.fov_spread_mm,
                "measurement_status": "clean — may be used as a reference value",
            }
        return {
            "d5_exposure": "affected",
            "reason": (
                f"spacing varies across {self.n_scans} scans "
                f"(per-axis spread {self.spacing_spread_mm} mm, FOV spread {self.fov_spread_mm} mm); "
                "resampling to a common array grid puts different physical extents in the same tensor"
            ),
            "n_scans": self.n_scans,
            "spacing_spread_mm": self.spacing_spread_mm,
            "fov_spread_mm": self.fov_spread_mm,
            "measurement_status": (
                "measured under a known defect — NOT a reference value to preserve; a respaced "
                "re-run may move the number in either direction and that move is a fix, not a regression"
            ),
        }


def _header_geometry(payload: bytes, name: str) -> tuple[list[float], list[int]] | None:
    import io

    import nibabel as nib

    try:
        if name.endswith(".gz"):
            import gzip

            payload = gzip.decompress(payload)
        holder = nib.FileHolder(fileobj=io.BytesIO(payload))
        header = nib.Nifti1Image.from_file_map({"header": holder, "image": holder}).header
        zooms = [float(z) for z in header.get_zooms()[:3]]
        shape = [int(s) for s in header.get_data_shape()[:3]]
        return zooms, shape
    except Exception:  # noqa: BLE001 - an unreadable file is data, not a crash
        return None


def audit_paths(paths: Iterable[str | Path], limit: int = 0) -> CohortGeometry:
    spacings: list[list[float]] = []
    shapes: list[list[int]] = []
    fovs: list[list[float]] = []
    unreadable: list[str] = []
    items = list(paths)
    if limit:
        items = items[:limit]
    for path in items:
        p = Path(path)
        geometry = _header_geometry(p.read_bytes(), p.name.lower())
        if geometry is None:
            unreadable.append(p.name)
            continue
        zooms, shape = geometry
        spacings.append(zooms)
        shapes.append(shape)
        fovs.append([round(z * s, 2) for z, s in zip(zooms, shape)])
    return CohortGeometry(len(items), spacings, shapes, fovs, unreadable)


def audit_archive(archive: str | Path, inner: str = "", limit: int = 0) -> CohortGeometry:
    spacings: list[list[float]] = []
    shapes: list[list[int]] = []
    fovs: list[list[float]] = []
    unreadable: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        members = [
            n
            for n in bundle.namelist()
            if n.lower().endswith((".nii", ".nii.gz")) and (not inner or inner.lower() in n.lower())
        ]
        if limit:
            members = members[:limit]
        for name in members:
            geometry = _header_geometry(bundle.read(name), name.lower())
            if geometry is None:
                unreadable.append(name)
                continue
            zooms, shape = geometry
            spacings.append(zooms)
            shapes.append(shape)
            fovs.append([round(z * s, 2) for z, s in zip(zooms, shape)])
    return CohortGeometry(len(spacings) + len(unreadable), spacings, shapes, fovs, unreadable)


def audit_directory(directory: str | Path, limit: int = 0) -> CohortGeometry:
    files = sorted(Path(directory).rglob("*.nii*"))
    return audit_paths(files, limit=limit)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--inner", default="", help="path fragment to filter inside a zip")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    target = Path(args.target)
    if target.is_dir():
        geometry = audit_directory(target, args.limit)
    elif target.suffix == ".zip":
        geometry = audit_archive(target, args.inner, args.limit)
    else:
        geometry = audit_paths([target], args.limit)
    print(json.dumps(geometry.verdict(), indent=2))
