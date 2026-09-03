"""CHB-MIT pre-ictal ingestion: predicting a seizure BEFORE it happens.

This is the platform's genuine **early-detection** task. Every other imaging condition built
so far answers an *acute* or *characterisation* question — is there a bleed, how large is the
infarct core, is this tumour methylated — all of which describe an event that has already
occurred. Pre-ictal prediction asks whether a seizure is *coming*, which is the temporal
framing the wider project is actually about.

Task definition (the standard one in the seizure-prediction literature):

    ... interictal ...  |  PRE-ICTAL  |  SPH  | seizure onset
                        <-- 30 min --><-5min->

  * **pre-ictal** (positive): windows from 35 to 5 minutes before a seizure onset. The 5-minute
    gap is the *seizure prediction horizon* (SPH) -- a warning is only useful if it arrives with
    enough lead time to act on, so the minutes immediately before onset are deliberately excluded.
  * **interictal** (negative): windows drawn from recordings containing no seizure at all, and
    kept at least `interictal_guard_s` away from any annotated event.
  * **ictal and post-ictal windows are discarded** -- including them would turn a prediction task
    back into a detection task, which is the mistake this module exists to avoid.

Splitting is by **patient**, never by window: a subject's own pre-ictal physiology is highly
self-similar, so mixing one patient's windows across train and test would leak badly and report
an inflated number. This also makes the evaluation a genuine held-out-subject test, which the
platform's imaging conditions do not yet have.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import LoadedDataset

SAMPLING_RATE_HZ = 256.0
PREICTAL_START_S = 35 * 60  # window opens 35 min before onset
PREICTAL_END_S = 5 * 60     # ...and closes 5 min before (the SPH gap)
INTERICTAL_GUARD_S = 60 * 60


@dataclass(frozen=True)
class SeizureRecord:
    file_name: str
    start_s: float
    end_s: float


def parse_summary(summary_path: str | Path) -> dict[str, list[SeizureRecord]]:
    """Parse a chbNN-summary.txt into {file_name: [SeizureRecord, ...]}."""

    text = Path(summary_path).read_text(encoding="utf-8", errors="ignore")
    per_file: dict[str, list[SeizureRecord]] = {}
    current: str | None = None
    pending_start: float | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            per_file.setdefault(current, [])
            pending_start = None
        elif "Seizure" in line and "Start Time" in line and current:
            match = re.search(r"(\d+)\s*seconds", line)
            if match:
                pending_start = float(match.group(1))
        elif "Seizure" in line and "End Time" in line and current and pending_start is not None:
            match = re.search(r"(\d+)\s*seconds", line)
            if match:
                per_file[current].append(
                    SeizureRecord(current, pending_start, float(match.group(1)))
                )
                pending_start = None
    return per_file


def seizure_files(per_file: dict[str, list[SeizureRecord]]) -> list[str]:
    return [name for name, events in per_file.items() if events]


def seizure_free_files(per_file: dict[str, list[SeizureRecord]]) -> list[str]:
    return [name for name, events in per_file.items() if not events]


def _read_edf(path: str | Path) -> tuple[np.ndarray, float]:
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"])


def _window_features(window: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Per-channel band powers + simple dynamics, averaged across channels.

    Kept deliberately compact: the point of this module is the *temporal framing*, and a
    band-power representation is the standard baseline for seizure prediction. A learned
    encoder can be dropped in later via the same LoadedDataset bridge.
    """

    from scipy.signal import welch

    freqs, power = welch(window, fs=sampling_rate, nperseg=min(256, window.shape[-1]), axis=-1)
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50))
    total = power.sum(axis=-1, keepdims=True) + 1e-12
    features: list[float] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        relative = power[..., mask].sum(axis=-1) / total[..., 0]
        features.extend([float(relative.mean()), float(relative.std())])
    derivative = np.diff(window, axis=-1)
    features.extend(
        [
            float(np.mean(np.std(window, axis=-1))),
            float(np.mean(np.std(derivative, axis=-1))),
            float(np.mean(np.abs(window))),
            # line length: a standard, cheap seizure-prediction feature
            float(np.mean(np.sum(np.abs(derivative), axis=-1))),
        ]
    )
    return np.asarray(features, dtype=float)


FEATURE_NAMES = [
    f"{name}_{stat}"
    for name in ("delta", "theta", "alpha", "beta", "gamma")
    for stat in ("mean", "std")
] + ["amplitude_std", "derivative_std", "mean_abs", "line_length"]


def build_preictal_dataset(
    data_root: str | Path,
    patients: list[str],
    window_s: float = 30.0,
    max_interictal_per_patient: int = 60,
    max_preictal_per_patient: int = 60,
) -> LoadedDataset:
    """Build a patient-grouped pre-ictal vs interictal dataset from local CHB-MIT EDFs."""

    root = Path(data_root)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    row_ids: list[str] = []
    provenance_counts: dict[str, dict[str, int]] = {}

    for patient in patients:
        summary = root / f"{patient}-summary.txt"
        if not summary.exists():
            continue
        per_file = parse_summary(summary)
        counts = {"preictal": 0, "interictal": 0}

        # --- pre-ictal windows, from files that contain a seizure ---
        for file_name in seizure_files(per_file):
            edf = root / file_name
            if not edf.exists():
                continue
            signal, sampling_rate = _read_edf(edf)
            samples_per_window = int(window_s * sampling_rate)
            for event in per_file[file_name]:
                lo = event.start_s - PREICTAL_START_S
                hi = event.start_s - PREICTAL_END_S
                if hi <= 0:
                    continue  # seizure too early in the file to have a usable lead-up
                lo = max(lo, 0.0)
                start_sample = int(lo * sampling_rate)
                end_sample = int(hi * sampling_rate)
                for begin in range(start_sample, end_sample - samples_per_window, samples_per_window):
                    if counts["preictal"] >= max_preictal_per_patient:
                        break
                    window = signal[:, begin : begin + samples_per_window]
                    if window.shape[-1] < samples_per_window:
                        continue
                    rows.append(_window_features(window, sampling_rate))
                    labels.append(1)
                    groups.append(patient)
                    row_ids.append(f"{file_name}:pre:{begin}")
                    counts["preictal"] += 1
            del signal

        # --- interictal windows, from files with no annotated seizure at all ---
        for file_name in seizure_free_files(per_file):
            if counts["interictal"] >= max_interictal_per_patient:
                break
            edf = root / file_name
            if not edf.exists():
                continue
            signal, sampling_rate = _read_edf(edf)
            samples_per_window = int(window_s * sampling_rate)
            guard = int(INTERICTAL_GUARD_S * sampling_rate)
            begin = guard if signal.shape[-1] > 2 * guard else 0
            while begin + samples_per_window < signal.shape[-1]:
                if counts["interictal"] >= max_interictal_per_patient:
                    break
                window = signal[:, begin : begin + samples_per_window]
                rows.append(_window_features(window, sampling_rate))
                labels.append(0)
                groups.append(patient)
                row_ids.append(f"{file_name}:inter:{begin}")
                counts["interictal"] += 1
                begin += samples_per_window * 4  # decorrelate consecutive samples
            del signal

        provenance_counts[patient] = counts
        print(f"{patient}: preictal={counts['preictal']} interictal={counts['interictal']}", flush=True)

    if not rows:
        raise ValueError("no CHB-MIT windows were built; check that EDF files are present")

    return LoadedDataset(
        name="chbmit-preictal-prediction",
        X=np.asarray(rows, dtype=float),
        y=np.asarray(labels, dtype=int),
        feature_names=list(FEATURE_NAMES),
        positive_label="preictal_seizure_imminent",
        negative_label="interictal_baseline",
        provenance={
            "source": "CHB-MIT Scalp EEG Database (PhysioNet, ODC-By)",
            "task": "early detection / seizure prediction",
            "preictal_window": f"{PREICTAL_START_S/60:.0f}-{PREICTAL_END_S/60:.0f} min before onset",
            "seizure_prediction_horizon_min": PREICTAL_END_S / 60,
            "window_seconds": window_s,
            "per_patient_counts": provenance_counts,
            "ictal_and_postictal_excluded": True,
        },
        groups=np.asarray(groups, dtype=str),
        row_ids=np.asarray(row_ids, dtype=str),
        task_profile={
            "task_type": "binary_classification",
            "endpoint": "preictal_seizure_imminent",
            "temporal_framing": "prediction",
            "positive_definition": (
                f"EEG window {PREICTAL_START_S/60:.0f}-{PREICTAL_END_S/60:.0f} minutes BEFORE an "
                "annotated seizure onset (a lead-time warning, not detection of an ongoing event)"
            ),
            "negative_definition": "EEG window from a recording with no annotated seizure",
            "unit_of_analysis": f"one {window_s:.0f}-second multi-channel scalp EEG window",
            "lead_time_minutes": PREICTAL_END_S / 60,
        },
    )


if __name__ == "__main__":
    import sys

    dataset = build_preictal_dataset(sys.argv[1], sys.argv[2].split(","))
    print(
        json.dumps(
            {
                "windows": int(len(dataset.y)),
                "preictal": int(dataset.y.sum()),
                "interictal": int((1 - dataset.y).sum()),
                "patients": sorted(set(dataset.groups.tolist())),
                "features": len(dataset.feature_names),
            },
            indent=2,
        )
    )
