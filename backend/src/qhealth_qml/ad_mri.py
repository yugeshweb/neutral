"""Alzheimer's structural MRI ingestion (Zenodo 3935636) with streaming extraction.

The platform's existing P5 entry trains on eight tabular fields (age, education, SES, MMSE
and three FreeSurfer-derived volumetrics). The raw-image tiers of OASIS-1/3, ADNI and PPMI
are all gated behind institutional data-use agreements, so this uses the one open,
peer-reviewed alternative found: Zenodo record 3935636, CC BY 4.0, direct HTTP, T1
structural MRI for 78 subjects labelled `ea` (Alzheimer's, 35), `crl` (control, 19) and
`tb` (bipolar, 24). The default task is the clean **AD vs healthy control** binary, n=54.

Deliberately *streaming*: subjects are extracted from the archive one at a time, resampled
to a small grid and released, so peak disk stays at roughly one subject's file rather than
the whole 5.3 GB tree. That is what makes this ingestible on a nearly-full disk at all.

Honest scope note carried into the model card: n=54 is thin for a 3-D CNN, and the cohort is
single-site 1.5 T with no scanner diversity. This is a demonstrable pipeline on real
radiologist-grade imaging, not a validated clinical result -- expect wide confidence
intervals and treat any number from it accordingly.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .imaging_hybrid import VolumeDataset, normalize_volume

GROUP_LABELS = {"ea": "alzheimers", "crl": "control", "tb": "bipolar"}


def read_subject_groups(clinical_csv: str | Path) -> dict[str, str]:
    """Map subject id -> group code from clinical_data_id_age_gender.csv."""

    mapping: dict[str, str] = {}
    with Path(clinical_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject_id = str(row.get("id", "")).strip()
            group = str(row.get("Subject", "")).strip().lower()
            if subject_id and group:
                mapping[subject_id] = group
    if not mapping:
        raise ValueError(f"no subject/group rows parsed from {clinical_csv}")
    return mapping


def _load_nifti_bytes(payload: bytes, suffix: str) -> np.ndarray:
    """Read a NIfTI volume held in memory (avoids materialising the archive on disk)."""

    import nibabel as nib

    if suffix.endswith(".gz"):
        import gzip

        payload = gzip.decompress(payload)
    holder = nib.FileHolder(fileobj=__import__("io").BytesIO(payload))
    image = nib.Nifti1Image.from_file_map({"header": holder, "image": holder})
    data = np.asarray(image.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    return data


def _subject_id_from_name(name: str) -> str | None:
    match = re.search(r"(\d{3,4})", Path(name).name)
    return match.group(1) if match else None


def _group_from_name(name: str) -> str | None:
    """Group code from the filename prefix, e.g. 'T1_original/ea_217.nii.gz' -> 'ea'.

    The archive encodes the diagnosis directly in the filename, which is more reliable
    than matching a numeric id against the clinical table; the CSV is kept only as a
    fallback for any file that does not follow the convention.
    """

    stem = Path(name).name.lower()
    for code in GROUP_LABELS:
        if stem.startswith(f"{code}_"):
            return code
    return None


def load_ad_mri_streaming(
    archive_path: str | Path,
    clinical_csv: str | Path,
    target_shape: tuple[int, int, int] = (64, 64, 64),
    include_bipolar: bool = False,
    limit: int = 0,
    subdir: str = "T1_original/",
) -> VolumeDataset:
    """Stream T1 volumes out of the Zenodo archive, one subject at a time."""

    groups = read_subject_groups(clinical_csv)
    archive = Path(archive_path)

    volumes: list[np.ndarray] = []
    labels: list[int] = []
    row_ids: list[str] = []
    group_codes: list[str] = []
    skipped: list[str] = []

    with zipfile.ZipFile(archive) as bundle:
        # T1_original/ holds the raw structural scans; the sibling DARTEL/SPM trees are
        # processed derivatives (grey-matter maps), which is the kind of pre-summarised
        # input this whole pass exists to move away from.
        members = [
            name
            for name in bundle.namelist()
            if name.lower().endswith((".nii", ".nii.gz"))
            and not name.startswith("__MACOSX")
            and subdir.lower() in name.lower()
        ]
        members.sort()
        if limit:
            members = members[:limit]
        if not members:
            raise ValueError(f"no NIfTI members found in {archive}")

        for index, name in enumerate(members, start=1):
            subject_id = _subject_id_from_name(name)
            group = _group_from_name(name) or groups.get(subject_id or "")
            if group is None:
                skipped.append(f"{name}(no clinical row)")
                continue
            if group == "tb" and not include_bipolar:
                continue
            try:
                payload = bundle.read(name)
                volume = _load_nifti_bytes(payload, name.lower())
            except Exception as exc:  # noqa: BLE001 - report and continue, do not abort the run
                skipped.append(f"{name}({type(exc).__name__})")
                continue
            del payload  # release the compressed member before the next one

            volumes.append(normalize_volume(volume, target_shape)[None, ...])
            labels.append(1 if group == "ea" else 0)
            row_ids.append(f"{subject_id}-{group}")
            group_codes.append(group)
            if index % 10 == 0 or index == len(members):
                print(f"streamed AD-MRI subject {index}/{len(members)}", flush=True)

    if not volumes:
        raise ValueError("no usable subjects were streamed from the archive")

    return VolumeDataset(
        volumes=np.asarray(volumes, dtype=np.float32),
        y=np.asarray(labels, dtype=int),
        groups=np.asarray(row_ids, dtype=str),  # one scan per subject
        row_ids=np.asarray(row_ids, dtype=str),
        channel_names=["t1"],
        name="zenodo3935636-ad-vs-control",
        positive_label="alzheimers_disease",
        negative_label="healthy_control" if not include_bipolar else "non_alzheimers",
        positive_definition=(
            "clinical diagnosis group 'ea' (Alzheimer's disease) in the study's clinical table; "
            "negatives are 'crl' healthy controls"
            + (" plus 'tb' bipolar subjects" if include_bipolar else "")
        ),
        provenance={
            "source": "Zenodo 3935636 (Alzheimer's vs bipolar vs control MRI), CC BY 4.0",
            "raw_imaging": True,
            "modality": "T1-weighted structural MRI (preprocessed)",
            "target_shape": list(target_shape),
            "group_counts": {
                name: int(sum(1 for code in group_codes if code == code_key))
                for code_key, name in GROUP_LABELS.items()
                for name in [GROUP_LABELS[code_key]]
            },
            "include_bipolar": include_bipolar,
            "skipped": skipped,
            "cohort_caveat": (
                "n=54 for AD vs control, single-site 1.5 T, no scanner diversity -- "
                "demonstration of a real imaging pipeline, not a validated clinical result"
            ),
        },
    )


def summarize(dataset: VolumeDataset) -> dict[str, Any]:
    return {
        "subjects": int(len(dataset.y)),
        "alzheimers": int(dataset.y.sum()),
        "controls": int((1 - dataset.y).sum()),
        "grid": list(dataset.volumes.shape[2:]),
        "skipped": len(dataset.provenance.get("skipped", [])),
    }


if __name__ == "__main__":
    import sys

    ds = load_ad_mri_streaming(
        sys.argv[1], sys.argv[2], limit=int(sys.argv[3]) if len(sys.argv) > 3 else 0
    )
    print(json.dumps(summarize(ds), indent=2))
