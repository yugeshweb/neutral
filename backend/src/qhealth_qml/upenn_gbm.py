"""UPenn-GBM ingestion: multi-parametric MRI -> MGMT methylation status, streamed from TCIA.

The platform's earlier P3 attempt collapsed each scan into hand-crafted CaPTk radiomics
features and landed at chance (classical 0.474, QSVC 0.451 across a 24-configuration sweep).
This path keeps the voxels: the four structural sequences a neuro-oncology radiologist reads
(T1, T1-post-contrast, T2, FLAIR) are pulled per patient, resampled to a small grid and
stacked as channels for the shared 3-D encoder.

Access note: TCIA's NBIA REST API serves this collection over plain HTTP with no credentials,
contrary to the assumption that NBIA Data Retriever or Aspera is required -- `getSeries` lists
a patient's series and `getImage` returns one series as a DICOM ZIP. Licence is CC BY 4.0.

Streamed by construction: one series is fetched, converted, downsampled and deleted before the
next is requested, so peak disk stays at roughly one series regardless of cohort size. MGMT
labels come from UPENN-GBM_clinical_info_v2.1.csv (291 of 671 subjects have a definite
Methylated/Unmethylated call; 'Indeterminate' and 'Not Available' are excluded).
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .imaging_hybrid import VolumeDataset, normalize_volume

TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "UPENN-GBM"

# Series descriptions vary by scanner, and the naive substring rules collide: "t2_Flair_axial"
# contains "t2", and "t1 axial stealth-post" contains "t1 axial". Sequences are therefore
# matched in a fixed priority order with explicit exclusions, most specific first, and each
# series can only be claimed once.
SEQUENCE_RULES = (
    ("flair", ("flair",), ()),                       # claim FLAIR before any bare "t2" rule
    ("t1_post", ("post", "stealth-post", "+c", "gd"), ("flair",)),
    ("t1", ("t1",), ("post", "flair", "stealth-post")),
    ("t2", ("t2",), ("flair", "t1")),                # e.g. "Axial T2 tse: Processed_CaPTk"
)
CHANNEL_ORDER = ("t1", "t1_post", "t2", "flair")


def read_mgmt_labels(clinical_csv: str | Path) -> dict[str, int]:
    """Subject id -> 1 (methylated) / 0 (unmethylated); indeterminate/missing dropped."""

    labels: dict[str, int] = {}
    with Path(clinical_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("MGMT", "")).strip().lower()
            subject = str(row.get("ID", "")).strip()
            # The clinical table suffixes a timepoint ("UPENN-GBM-00022_11") that TCIA's
            # PatientID does not carry; without stripping it every API lookup returns empty.
            subject = subject.split("_")[0]
            if not subject:
                continue
            if status == "methylated":
                labels[subject] = 1
            elif status == "unmethylated":
                labels[subject] = 0
    if not labels:
        raise ValueError(f"no definite MGMT calls parsed from {clinical_csv}")
    return labels


def _get_json(url: str, timeout: int = 90) -> Any:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_patient_series(patient_id: str) -> list[dict[str, Any]]:
    return _get_json(f"{TCIA_API}/getSeries?Collection={COLLECTION}&PatientID={patient_id}")


def _pick_series(series: list[dict[str, Any]]) -> dict[str, str]:
    """Choose one SeriesInstanceUID per structural sequence, preferring smaller stacks."""

    chosen: dict[str, str] = {}
    ordered = sorted(series, key=lambda s: int(s.get("ImageCount") or 10**6))
    for channel, includes, excludes in SEQUENCE_RULES:
        for entry in ordered:
            if str(entry.get("Modality", "")).upper() != "MR":
                continue
            description = str(entry.get("SeriesDescription", "")).lower()
            if not any(token in description for token in includes):
                continue
            if any(token in description for token in excludes):
                continue
            uid = entry.get("SeriesInstanceUID")
            if uid and uid not in chosen.values():
                chosen[channel] = uid
                break
    return chosen


def fetch_series_volume(series_uid: str, timeout: int = 300) -> np.ndarray:
    """Download one series as a DICOM ZIP and assemble it into a 3-D array, in memory."""

    import urllib.request

    import pydicom

    url = f"{TCIA_API}/getImage?SeriesInstanceUID={series_uid}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()

    slices = []
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for name in bundle.namelist():
            if not name.lower().endswith(".dcm"):
                continue
            dataset = pydicom.dcmread(io.BytesIO(bundle.read(name)), force=True)
            if not hasattr(dataset, "pixel_array"):
                continue
            position = getattr(dataset, "InstanceNumber", None)
            slices.append((int(position) if position is not None else len(slices),
                           dataset.pixel_array.astype(np.float32)))
    if not slices:
        raise ValueError(f"series {series_uid} produced no readable slices")
    slices.sort(key=lambda item: item[0])
    return np.stack([array for _, array in slices], axis=-1)


def load_upenn_gbm_streaming(
    clinical_csv: str | Path,
    target_shape: tuple[int, int, int] = (64, 64, 64),
    limit: int = 0,
    balanced: bool = True,
    progress_every: int = 5,
) -> VolumeDataset:
    """Stream UPenn-GBM structural MRI from TCIA, one patient at a time."""

    labels = read_mgmt_labels(clinical_csv)
    subjects = sorted(labels)
    if balanced:
        positive = [s for s in subjects if labels[s] == 1]
        negative = [s for s in subjects if labels[s] == 0]
        take = min(len(positive), len(negative))
        if limit:
            take = min(take, max(1, limit // 2))
        subjects = sorted(positive[:take] + negative[:take])
    elif limit:
        subjects = subjects[:limit]

    volumes: list[np.ndarray] = []
    y: list[int] = []
    row_ids: list[str] = []
    skipped: list[str] = []

    channel_names = list(CHANNEL_ORDER)
    for index, subject in enumerate(subjects, start=1):
        try:
            series = list_patient_series(subject)
            picked = _pick_series(series)
            missing = [name for name in channel_names if name not in picked]
            if missing:
                skipped.append(f"{subject}(missing {missing})")
                continue
            channels = []
            for name in channel_names:
                raw = fetch_series_volume(picked[name])
                channels.append(normalize_volume(raw, target_shape))
                del raw
            volumes.append(np.stack(channels, axis=0))
            y.append(labels[subject])
            row_ids.append(subject)
        except Exception as exc:  # noqa: BLE001 - one bad patient must not kill the run
            skipped.append(f"{subject}({type(exc).__name__}: {exc})")
        if index % progress_every == 0 or index == len(subjects):
            print(f"streamed UPenn-GBM {index}/{len(subjects)} (kept {len(volumes)})", flush=True)

    if not volumes:
        raise ValueError("no UPenn-GBM patients were successfully streamed")

    return VolumeDataset(
        volumes=np.asarray(volumes, dtype=np.float32),
        y=np.asarray(y, dtype=int),
        groups=np.asarray(row_ids, dtype=str),
        row_ids=np.asarray(row_ids, dtype=str),
        channel_names=channel_names,
        name="upenn-gbm-mgmt",
        positive_label="mgmt_methylated",
        negative_label="mgmt_unmethylated",
        positive_definition=(
            "MGMT promoter methylation reported as 'Methylated' in "
            "UPENN-GBM_clinical_info_v2.1.csv (molecular pathology, not imaging-derived)"
        ),
        provenance={
            "source": "UPenn-GBM via TCIA NBIA REST API, CC BY 4.0",
            "raw_imaging": True,
            "modality": "multi-parametric structural MRI (T1, T1-post, T2, FLAIR)",
            "target_shape": list(target_shape),
            "skipped": skipped,
            "prior_attempt": (
                "hand-crafted CaPTk radiomics on this cohort scored at chance "
                "(classical 0.474 / QSVC 0.451, 24-config sweep)"
            ),
        },
    )


def summarize(dataset: VolumeDataset) -> dict[str, Any]:
    return {
        "subjects": int(len(dataset.y)),
        "methylated": int(dataset.y.sum()),
        "unmethylated": int((1 - dataset.y).sum()),
        "channels": dataset.channel_names,
        "grid": list(dataset.volumes.shape[2:]),
        "skipped": len(dataset.provenance.get("skipped", [])),
    }


if __name__ == "__main__":
    import sys

    ds = load_upenn_gbm_streaming(
        sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else 0
    )
    print(json.dumps(summarize(ds), indent=2))
