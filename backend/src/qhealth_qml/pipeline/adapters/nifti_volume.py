"""`nifti_volume` adapter: multi-sequence NIfTI (`.nii`/`.nii.gz`) - tasks.md
T049-T055. Wraps `nibabel` rather than re-implementing NIfTI parsing.

**Source contract (a deliberate simplification of the spec's worked
example)**: the spec's own example passes a per-record `dict[str, path]`
mapping sequence name -> file directly into `Pipeline.read(source=...)`.
This build instead expects a DIRECTORY layout: either (a) one directory
per case, each containing files named `<sequence>.nii[.gz]` matching
`imaging.sequences`, with the case directory itself as `source` for a
single-record read, or (b) a root directory of such per-case
subdirectories, one Sample per subdirectory. This keeps `Pipeline.read()`
unchanged (no new dict-shaped Source kind) while preserving the same
semantics: one Sample is built from several co-registered sequences, and
the RECIPE (not the caller) dictates channel order.

Harmonisation follows the fixed chain (FR-057 through FR-064): decode ->
orient -> respace -> intensity -> grid -> stack, respacing strictly before
grid resizing (FR-059)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import imaging as img
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source

_NIFTI_SUFFIXES = (".nii.gz", ".nii")


class NiftiVolumeAdapter:
    name = "nifti_volume"
    modalities = ("mr", "ct", "angio")
    formats = (".nii", ".nii.gz")

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            if _nifti_files(path):
                return 0.8
            if any(_nifti_files(sub) for sub in path.iterdir() if sub.is_dir()):
                return 0.7
            return 0.0
        if path.name.endswith(_NIFTI_SUFFIXES):
            return 0.6  # a lone file is readable but the adapter wants a case directory
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        path = _as_path(source)
        if path is None:
            raise ValueError(f"nifti_volume adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if not path.is_dir():
            raise ValueError(
                f"nifti_volume expects a case directory of named .nii(.gz) sequence files (or a "
                f"root directory of such case-subdirectories), not a single file: {path}"
            )

        if _nifti_files(path):
            case_dirs = [path]
        else:
            case_dirs = sorted(sub for sub in path.iterdir() if sub.is_dir() and _nifti_files(sub))
        if not case_dirs:
            raise ValueError(f"no NIfTI case directories found under {path}")

        sequences: list[str] = list(spec._raw.get("imaging", {}).get("sequences", ()))

        for case_dir in case_dirs:
            files_by_sequence: dict[str, Path] = {}
            for f in _nifti_files(case_dir):
                stem = f.name
                for suffix in _NIFTI_SUFFIXES:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                for seq in sequences:
                    if stem.lower() == seq.lower():
                        files_by_sequence[seq] = f
                        break

            yield RawRecord(
                source=source,
                payload={"files_by_sequence": files_by_sequence},
                meta={"record_name": case_dir.name, "path": str(case_dir)},
            )

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        import nibabel as nib

        imaging_cfg = spec._raw.get("imaging", {}) or {}
        sequences: list[str] = list(imaging_cfg.get("sequences", ()))
        required_sequences = set(imaging_cfg.get("required_sequences", ()))
        target_mm = tuple(imaging_cfg.get("voxel_spacing_mm", (1.0, 1.0, 1.0)))

        files_by_sequence: dict[str, Path] = raw.payload["files_by_sequence"]
        if not sequences:
            sequences = sorted(files_by_sequence)
        issues: list[Issue] = []
        applied: list[str] = []
        source_provenance: dict[str, dict] = {}
        channel_arrays: dict[str, np.ndarray] = {}
        target_shape: tuple[int, int, int] | None = None

        for seq, f in files_by_sequence.items():
            img_obj = nib.load(str(f))
            data = np.asarray(img_obj.dataobj, dtype=float)
            zooms = tuple(float(z) for z in img_obj.header.get_zooms()[:3])
            canonical, _ = img.orient_canonical(data, img_obj.affine)
            respaced = img.respace(canonical, zooms, target_mm)
            normalized = img.apply_mr_normalize(respaced)
            channel_arrays[seq] = normalized
            if target_shape is None:
                target_shape = normalized.shape
            source_provenance[seq] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "zooms": list(zooms)}

        applied.extend(["orient_RAS", f"respace_{target_mm}mm", "zscore_foreground"])

        grid = tuple(imaging_cfg.get("grid", target_shape or (64, 64, 64)))
        channel_arrays = {seq: img.resize_grid(arr, grid) for seq, arr in channel_arrays.items()}
        applied.append(f"resize_{grid}")

        stacked, stack_issues = img.stack_channels(channel_arrays, sequences, required_sequences)
        issues.extend(stack_issues)

        if stacked is not None:
            issues.extend(img.check_foreground(stacked))
        else:
            stacked = np.zeros((0, *grid))

        record_name = raw.meta["record_name"]
        sample_id, label = _split_label_suffix(record_name)

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(sample_id),
            index_time=None,
            outcome_time=None,
            site=None,
            # No standard label field in a bare NIfTI case directory; this
            # build's fixtures encode it in the directory name
            # (`<id>__label<0|1>/`), same convention as edf_eeg/gait_txt.
            label=label,
            fields={},
            arrays={"volume": stacked},
            subgroups={},
            provenance={
                "adapter": self.name,
                "modality": spec.modality.upper(),
                "channel_order": sequences,
                "sources": source_provenance,
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


def _nifti_files(path: Path) -> list[Path]:
    return [f for f in path.iterdir() if f.is_file() and f.name.endswith(_NIFTI_SUFFIXES)]


def _hash_subject_id(raw_subject_id: str) -> str:
    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


_LABEL_SUFFIX = re.compile(r"^(?P<id>.+)__label(?P<label>[01])$")


def _split_label_suffix(record_name: str) -> tuple[str, int | None]:
    match = _LABEL_SUFFIX.match(record_name)
    if not match:
        return record_name, None
    return match.group("id"), int(match.group("label"))
