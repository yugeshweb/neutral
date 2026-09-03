"""`gait_txt` adapter: plain delimited gait time-series files (e.g.
accelerometer/force-plate columns over time) - tasks.md T045. No external
format library needed; this is ordinary CSV/TSV, decoded into
`(channels, samples)` the same shape `wfdb_ecg`/`edf_eeg` produce, then run
through the identical `..signal` harmonisation chain.

**Source contract**: `source` is a directory containing one or more
`.txt`/`.csv` files, one Sample per file, each a header row of channel
names followed by one row per time sample. A single file path also works."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import signal as sig
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source


class GaitTxtAdapter:
    name = "gait_txt"
    modalities = ("gait",)
    formats = (".txt", ".csv")

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            return 0.5 if any(path.glob("*.txt")) or any(path.glob("*.csv")) else 0.0
        if path.suffix.lower() in (".txt", ".csv"):
            return 0.5
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        path = _as_path(source)
        if path is None:
            raise ValueError(f"gait_txt adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")

        files = sorted(list(path.glob("*.txt")) + list(path.glob("*.csv"))) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"no .txt/.csv files found under {path}")

        for file_path in files:
            text = file_path.read_text(encoding="utf-8-sig")
            delimiter = "\t" if "\t" in text.splitlines()[0] else ","
            reader = csv.reader(text.splitlines(), delimiter=delimiter)
            rows = list(reader)
            if not rows:
                continue
            header, data_rows = rows[0], rows[1:]
            columns = [c.strip() for c in header]
            matrix = np.full((len(columns), len(data_rows)), np.nan)
            for t, row in enumerate(data_rows):
                for c, value in enumerate(row[: len(columns)]):
                    try:
                        matrix[c, t] = float(value)
                    except ValueError:
                        pass
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            fs_cfg = spec._raw.get("signal", {}).get("native_hz")
            yield RawRecord(
                source=source,
                payload={"signal": matrix, "fs": float(fs_cfg) if fs_cfg else 100.0, "channel_names": columns},
                meta={"record_name": file_path.stem, "path": str(file_path), "sha256": digest},
            )

    def harmonize(self, raw: RawRecord, spec: SourceSpec) -> Sample:
        payload = raw.payload
        native: np.ndarray = payload["signal"]
        native_fs: float = payload["fs"]
        native_names: list[str] = payload["channel_names"]

        signal_cfg = spec._raw.get("signal", {}) or {}
        declared_channels: list[str] = list(signal_cfg.get("channels", native_names))
        target_hz = float(signal_cfg.get("resample_to_hz", native_fs))
        target_samples = int(signal_cfg.get("target_samples", native.shape[1]))
        length_norm_policy = signal_cfg.get("length_norm", "crop_center")
        normalize_method = signal_cfg.get("per_record_normalize", "median_std")

        issues: list[Issue] = []
        applied: list[str] = []

        name_index = {n.strip().lower(): i for i, n in enumerate(native_names)}
        missing_channels = [ch for ch in declared_channels if ch.strip().lower() not in name_index]
        for ch in missing_channels:
            issues.append(Issue(IssueCode.CHANNEL_MISSING, "reject", f"channel '{ch}' is not present in this record.", field=ch))

        if missing_channels:
            array = np.full((len(declared_channels), native.shape[1]), np.nan)
        else:
            order = [name_index[ch.strip().lower()] for ch in declared_channels]
            array = native[order, :]
            applied.append("channel_remap")

            if target_hz != native_fs:
                array = sig.resample_channels(array, native_fs, target_hz)
                applied.append(f"resample_{native_fs:g}->{target_hz:g}Hz")

            if array.shape[1] != target_samples:
                array = sig.normalize_length(array, target_samples, length_norm_policy)
                applied.append(f"length_norm_{length_norm_policy}")

            array = sig.normalize_per_record(array, normalize_method)
            applied.append(f"per_record_{normalize_method}")

            issues.extend(sig.detect_artifacts(array, declared_channels))

        record_name = raw.meta["record_name"]
        sample_id, label = _split_label_suffix(record_name)

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(sample_id),
            index_time=None,
            outcome_time=None,
            site=None,
            # No standard label field in a plain gait time-series file;
            # this build's fixtures encode it in the filename
            # (`<id>__label<0|1>.txt`), same convention as edf_eeg.
            label=label,
            fields={},
            arrays={"signal": array},
            subgroups={},
            provenance={
                "adapter": self.name,
                "source": {"path": raw.meta.get("path"), "sha256": raw.meta.get("sha256")},
                "channels": declared_channels,
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


def _hash_subject_id(raw_subject_id: str) -> str:
    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


_LABEL_SUFFIX = re.compile(r"^(?P<id>.+)__label(?P<label>[01])$")


def _split_label_suffix(record_name: str) -> tuple[str, int | None]:
    match = _LABEL_SUFFIX.match(record_name)
    if not match:
        return record_name, None
    return match.group("id"), int(match.group("label"))
