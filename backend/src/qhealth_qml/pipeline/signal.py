"""Signal harmonisation chain: resample -> filter -> length-normalise ->
per-record normalise -> artefact detection (design.md §9.2.6, tasks.md
Phase 2, FR-050 through FR-055). Every function here is stateless - a pure
function of one record's array plus declared config, never cross-record
statistics (FR-004; FR-054 specifically requires per-record, not
per-batch, normalisation statistics)."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal as _sp_signal

from .types import Issue, IssueCode


def resample_channels(array: np.ndarray, orig_hz: float, target_hz: float) -> np.ndarray:
    """(C, T) -> (C, T'). Anti-aliased polyphase resampling (FR-050)."""

    if array.size == 0 or orig_hz == target_hz:
        return array
    g = gcd(int(round(orig_hz)), int(round(target_hz)))
    up, down = int(round(target_hz)) // g, int(round(orig_hz)) // g
    return _sp_signal.resample_poly(array, up, down, axis=-1)


def filter_channels(
    array: np.ndarray, hz: float, *,
    highpass_hz: float | None = None, lowpass_hz: float | None = None, notch_hz: float | None = None,
) -> np.ndarray:
    """Butterworth band edges plus an IIR notch, applied in a FIXED order:
    highpass, then lowpass, then notch (FR-051 - order is asserted by a
    test, not merely by source-reading order)."""

    if array.size == 0:
        return array
    out = array.astype(float)
    nyquist = hz / 2.0
    if highpass_hz and 0 < highpass_hz < nyquist:
        b, a = _sp_signal.butter(4, highpass_hz / nyquist, btype="highpass")
        out = _sp_signal.filtfilt(b, a, out, axis=-1)
    if lowpass_hz and 0 < lowpass_hz < nyquist:
        b, a = _sp_signal.butter(4, lowpass_hz / nyquist, btype="lowpass")
        out = _sp_signal.filtfilt(b, a, out, axis=-1)
    if notch_hz and 0 < notch_hz < nyquist:
        b, a = _sp_signal.iirnotch(notch_hz / nyquist, Q=30)
        out = _sp_signal.filtfilt(b, a, out, axis=-1)
    return out


def normalize_length(array: np.ndarray, target_samples: int, policy: str = "crop_center") -> np.ndarray:
    """(C, T) -> (C, target_samples). `crop_center` | `crop_start` |
    `pad_zero` | `pad_edge` (FR-052). Deliberately no stochastic-crop
    policy: FR-053's inference-time refusal for a stochastic policy only
    matters once one exists, and this build does not implement one."""

    c, t = array.shape
    if t == target_samples:
        return array
    if t > target_samples:
        if policy == "crop_center":
            start = (t - target_samples) // 2
            return array[:, start:start + target_samples]
        return array[:, :target_samples]  # crop_start
    pad_total = target_samples - t
    pad_left = pad_total // 2 if policy == "crop_center" else 0
    pad_right = pad_total - pad_left
    mode = "edge" if policy == "pad_edge" else "constant"
    return np.pad(array, ((0, 0), (pad_left, pad_right)), mode=mode)


def normalize_per_record(array: np.ndarray, method: str = "median_std") -> np.ndarray:
    """Per-CHANNEL statistics computed from THIS record alone (FR-054) -
    never from the batch it happens to be scored in. `median_std` |
    `zscore`."""

    if array.size == 0:
        return array
    out = np.zeros_like(array, dtype=float)
    for c in range(array.shape[0]):
        chan = array[c]
        if method == "zscore":
            center, scale = float(np.mean(chan)), float(np.std(chan))
        else:
            center, scale = float(np.median(chan)), float(np.std(chan))
        scale = scale if scale > 1e-9 else 1.0
        out[c] = (chan - center) / scale
    return out


def detect_artifacts(
    array: np.ndarray, channel_names: list[str], *,
    flatline_std: float = 1e-6, saturation_run: int = 20,
) -> list[Issue]:
    """Flatline, saturation and non-finite runs (FR-055) - one Issue per
    offending channel, naming it, never a bare non-success."""

    issues: list[Issue] = []
    for i, name in enumerate(channel_names[: array.shape[0]]):
        chan = array[i]
        if not np.isfinite(chan).all():
            issues.append(Issue(IssueCode.NAN_RUN, "reject", f"channel '{name}' contains non-finite samples.", field=name))
            continue
        if float(np.std(chan)) < flatline_std:
            issues.append(Issue(IssueCode.FLATLINE_CHANNEL, "warn", f"channel '{name}' variance is below the artefact floor.", field=name))
        peak = float(np.abs(chan).max()) if chan.size else 0.0
        if peak > 0:
            extreme = np.abs(chan) >= peak * 0.999
            if _longest_run(extreme) >= saturation_run:
                issues.append(
                    Issue(IssueCode.SATURATION, "warn", f"channel '{name}' has a sustained run at its extreme value (clipping signature).", field=name)
                )
    return issues


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for v in mask:
        current = current + 1 if v else 0
        longest = max(longest, current)
    return longest
