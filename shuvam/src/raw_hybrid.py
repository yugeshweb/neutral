"""Raw biosignal and hybrid neural network utilities."""

from __future__ import annotations

from typing import Any
import numpy as np


def _require_torch():
    """Raise ImportError if PyTorch is not installed."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for this operation: pip install torch") from exc


def device_for_training(device: str = "auto") -> str:
    """Return appropriate compute device string ('cuda', 'mps', or 'cpu')."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def normalize_raw_ecg(signal: np.ndarray, target_samples: int = 2500) -> np.ndarray:
    """Normalize raw 12-lead ECG signal array to leads-first tensor (leads, target_samples)."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    # Signal shape is (samples, leads)
    curr_len = arr.shape[0]
    num_leads = arr.shape[1]

    if curr_len != target_samples:
        x_old = np.linspace(0, 1, curr_len)
        x_new = np.linspace(0, 1, target_samples)
        resampled = np.zeros((target_samples, num_leads), dtype=np.float32)
        for col in range(num_leads):
            resampled[:, col] = np.interp(x_new, x_old, arr[:, col])
        arr = resampled

    # Transpose to leads-first: (leads, samples)
    arr = arr.T

    # Per-lead median centering and scaling
    medians = np.median(arr, axis=1, keepdims=True)
    stds = np.std(arr, axis=1, keepdims=True)
    stds[stds < 1e-6] = 1.0

    normalized = (arr - medians) / stds
    return normalized.astype(np.float32)
