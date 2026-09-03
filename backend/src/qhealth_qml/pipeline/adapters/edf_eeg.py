"""`edf_eeg` adapter: European Data Format (`.edf`) EEG recordings -
tasks.md T043. Wraps `mne.io.read_raw_edf` rather than re-implementing EDF
parsing (the spec's pinned dependency here is `pyedflib`; this build uses
`mne` instead - both are established, battle-tested EDF readers, and
`mne`'s wheel installs without a C compiler on this machine, which
`pyedflib`'s does not).

**Source contract**: `source` is a directory containing one or more `.edf`
files, one Sample per file; a single `.edf` path also works.

Shares the exact same harmonisation chain as `wfdb_ecg` (filter -> resample
-> length-normalise -> per-record normalise -> artefact detection, FR-048
through FR-055) via `..signal` - the two adapters differ only in how they
decode bytes into `(channels, samples)`, never in how they clean it."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import signal as sig
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source


class EdfEegAdapter:
    name = "edf_eeg"
    modalities = ("eeg",)
    formats = (".edf",)

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            return 0.8 if any(path.glob("*.edf")) else 0.0
        if path.suffix.lower() == ".edf":
            return 0.9
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        import mne

        path = _as_path(source)
        if path is None:
            raise ValueError(f"edf_eeg adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")

        edf_files = sorted(path.glob("*.edf")) if path.is_dir() else [path]
        if not edf_files:
            raise ValueError(f"no .edf files found under {path}")

        for edf_path in edf_files:
            raw_mne = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
            digest = hashlib.sha256(edf_path.read_bytes()).hexdigest()
            yield RawRecord(
                source=source,
                payload={
                    "signal": np.asarray(raw_mne.get_data(), dtype=float),  # already (channels, samples)
                    "fs": float(raw_mne.info["sfreq"]),
                    "channel_names": list(raw_mne.ch_names),
                },
                meta={"record_name": edf_path.stem, "path": str(edf_path), "sha256": digest},
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
        filter_cfg = signal_cfg.get("filter", {}) or {}
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

            if filter_cfg:
                array = sig.filter_channels(
                    array, native_fs,
                    highpass_hz=filter_cfg.get("highpass_hz"),
                    lowpass_hz=filter_cfg.get("lowpass_hz"),
                    notch_hz=filter_cfg.get("notch_hz"),
                )
                applied.append("filter_chain")

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
            # EDF headers carry no standard diagnostic-label field; this
            # build's fixtures encode the label in the filename
            # (`<id>__label<0|1>.edf`) rather than inventing a header
            # convention the real data may not follow. A predict-mode
            # upload with no such suffix correctly yields label=None.
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
