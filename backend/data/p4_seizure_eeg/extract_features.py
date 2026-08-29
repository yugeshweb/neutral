"""Adapter: turn raw CHB-MIT EDF recordings into a numeric CSV the existing
qhealth_qml tabular engine can train on directly (reuse-order rule 1/4 — this is
the classical feature-extraction boundary the spec requires before any quantum
encoding: "Feeding raw EEG streams... directly into a quantum circuit" is out of
scope; band-power features are the compact, validated signal representation).

Source: CHB-MIT Scalp EEG Database, PhysioNet, https://physionet.org/content/chbmit/1.0.0/
No account/DUA required (confirmed: public HTTP 200 downloads). Files used:
chb01_01.edf (no seizure), chb01_03.edf (1 seizure, 2996-3036s),
chb01_04.edf (1 seizure, 1467-1494s), per chb01-summary.txt.

Method: non-overlapping 4-second windows per file (no overlap, so no
autocorrelation-leakage risk across a chronological split); per-channel Welch
band power in 5 standard EEG bands (delta/theta/alpha/beta/gamma) via
scipy.signal.welch (already a pinned dependency via scikit-learn/numpy stack;
mne is used only for reading the EDF header/signal, per spec's named P4 evidence
source). A window is labeled seizure=1 if it overlaps the annotated seizure
interval at all, else 0. `start_sec` accumulates across files so a single
chronological split over the concatenated timeline is leak-free.
"""

import csv
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

HERE = Path(__file__).parent
WINDOW_SECONDS = 4.0
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# (filename, [(seizure_start_sec, seizure_end_sec), ...]) from chb01-summary.txt
FILES = [
    ("chb01_01.edf", []),
    ("chb01_03.edf", [(2996, 3036)]),
    ("chb01_04.edf", [(1467, 1494)]),
]


def band_power(window: np.ndarray, sfreq: float) -> dict[str, float]:
    freqs, psd = welch(window, fs=sfreq, nperseg=min(len(window), int(sfreq * 2)))
    powers = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        powers[band] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def main() -> None:
    rows: list[dict[str, object]] = []
    channel_names: list[str] | None = None
    global_offset = 0.0

    for filename, seizures in FILES:
        raw = mne.io.read_raw_edf(HERE / filename, preload=True, verbose="ERROR")
        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # (n_channels, n_samples)
        if channel_names is None:
            channel_names = list(raw.ch_names)
        elif channel_names != raw.ch_names:
            raise ValueError(f"{filename} has different channels than {channel_names}")

        window_samples = int(WINDOW_SECONDS * sfreq)
        n_windows = data.shape[1] // window_samples
        for w in range(n_windows):
            start_sample = w * window_samples
            end_sample = start_sample + window_samples
            window_start_sec = w * WINDOW_SECONDS
            window_end_sec = window_start_sec + WINDOW_SECONDS

            seizure = int(
                any(s_start < window_end_sec and s_end > window_start_sec for s_start, s_end in seizures)
            )

            row: dict[str, object] = {
                "window_id": f"{filename}:{w}",
                "file": filename,
                "start_sec": global_offset + window_start_sec,
            }
            for ch_index, ch_name in enumerate(channel_names):
                channel_window = data[ch_index, start_sample:end_sample]
                powers = band_power(channel_window, sfreq)
                for band, value in powers.items():
                    row[f"{ch_name}_{band}"] = value
            row["seizure"] = seizure
            rows.append(row)

        global_offset += data.shape[1] / sfreq

    feature_columns = [f"{ch}_{band}" for ch in channel_names for band in BANDS]
    fieldnames = ["window_id", "file", "start_sec", *feature_columns, "seizure"]
    out_path = HERE / "seizure_window_features.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_positive = sum(r["seizure"] for r in rows)
    print(f"wrote {out_path}: {len(rows)} windows, {len(feature_columns)} feature columns, "
          f"{n_positive} positive ({100 * n_positive / len(rows):.2f}%)")


if __name__ == "__main__":
    main()
