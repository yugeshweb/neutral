"""`wfdb_ecg` adapter: PhysioNet WFDB-format records (`.hea` + `.dat` pairs)
- tasks.md T036, T037, T038, T039, T040, T041, T042. Wraps the `wfdb`
library's own reader rather than re-implementing WFDB parsing.

**Source contract (a deliberate simplification from the spec's full
generality)**: `source` is a directory containing one or more WFDB record
pairs; each `.hea` file becomes one `Sample`. A single `.hea` file path
also works for a one-record read. This covers PhysioNet-style cohorts
(e.g. PTB-XL) without needing a separate manifest format.

Harmonisation order matches the declared chain (FR-048 through FR-055):
channel mapping -> filter -> resample -> length-normalise -> per-record
normalise -> artefact detection. `harmonize()` stays stateless (FR-004):
every step is a pure function of this one record plus the spec."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .. import signal as sig
from ..spec import SourceSpec
from ..types import Issue, IssueCode, QCVerdict, RawRecord, Sample, Source


class WfdbEcgAdapter:
    name = "wfdb_ecg"
    modalities = ("ecg",)
    formats = (".hea", ".dat")

    def sniff(self, source: Source) -> float:
        path = _as_path(source)
        if path is None:
            return 0.0
        if path.is_dir():
            return 0.8 if any(path.glob("*.hea")) else 0.0
        if path.suffix == ".hea":
            return 0.9
        return 0.0

    def read(self, source: Source, spec: SourceSpec) -> Iterator[RawRecord]:
        import wfdb

        path = _as_path(source)
        if path is None:
            raise ValueError(f"wfdb_ecg adapter cannot read source kind {source.kind!r}")
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")

        hea_files = sorted(path.glob("*.hea")) if path.is_dir() else [path]
        if not hea_files:
            raise ValueError(f"no .hea files found under {path}")

        for hea_path in hea_files:
            record_base = str(hea_path.with_suffix(""))
            record = wfdb.rdrecord(record_base)
            digest = hashlib.sha256(hea_path.read_bytes()).hexdigest()
            yield RawRecord(
                source=source,
                payload={
                    "signal": np.asarray(record.p_signal, dtype=float).T,  # (samples, channels) -> (channels, samples)
                    "fs": float(record.fs),
                    "channel_names": list(record.sig_name or []),
                    "comments": list(record.comments or []),
                },
                meta={"record_name": hea_path.stem, "path": str(hea_path), "sha256": digest},
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

        # Channel mapping by NAME, case-insensitively - never by position
        # (FR-049). A record missing a declared lead is refused, naming it;
        # the remaining leads are never shifted up to fill the gap.
        name_index = {n.strip().lower(): i for i, n in enumerate(native_names)}
        missing_channels = [ch for ch in declared_channels if ch.strip().lower() not in name_index]
        for ch in missing_channels:
            issues.append(Issue(IssueCode.CHANNEL_MISSING, "reject", f"lead '{ch}' is not present in this record.", field=ch))

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

        covariates = _parse_comments(payload["comments"])

        target_raw = None
        target_column = spec.target_column
        if target_column and target_column in covariates:
            target_raw = covariates.pop(target_column)
        label = None
        if target_raw is not None:
            # _parse_comments already coerced a numeric-looking comment
            # value to float (e.g. "1" -> 1.0) - comparing str(1.0) against
            # a declared positive_label of "1" would silently mismatch
            # every record, so numeric equality is tried first.
            try:
                label = 1 if float(target_raw) == float(spec.positive_label) else 0
            except (TypeError, ValueError):
                label = 1 if str(target_raw) == spec.positive_label else 0

        sample_id = raw.meta["record_name"]

        return Sample(
            sample_id=sample_id,
            subject_id=_hash_subject_id(sample_id),
            index_time=None,
            outcome_time=None,
            site=None,
            label=label,
            fields=covariates,
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
        # Modality-specific checks (channel_missing, artefacts) are already
        # folded into sample.issues by harmonize() - nothing further here.
        return QCVerdict(status="accept", issues=[])


def _as_path(source: Source) -> Path | None:
    if source.kind == "path":
        return Path(source.locator)
    return None


def _hash_subject_id(raw_subject_id: str) -> str:
    return "sha256:" + hashlib.sha256(raw_subject_id.encode("utf-8")).hexdigest()[:16]


def _parse_comments(comments: list[str]) -> dict[str, Any]:
    """WFDB header comments often carry ` key: value` covariates (e.g.
    `Age: 56`, `Sex: Male`) - parsed best-effort; anything that doesn't
    match `key: value` is ignored rather than raising."""

    fields: dict[str, Any] = {}
    for line in comments:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not key or not value:
            continue
        try:
            fields[key] = float(value)
        except ValueError:
            fields[key] = value
    return fields
