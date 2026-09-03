"""Raw 3-D medical-image encoder for the platform's imaging conditions.

Same contract as `raw_hybrid.py` (ECG) and `gait_hybrid.py` (gait): a small CNN
consumes the actual voxel data -- the scan a radiologist reads -- and emits a
compact latent vector that the existing classical and Qiskit heads consume through
the usual `LoadedDataset` bridge. Nothing downstream needs to know the input was an
MRI rather than a table of hand-computed radiomics.

This exists because the platform's prior imaging attempt (P3 glioma) collapsed each
multi-parametric MRI into a handful of hand-engineered radiomics statistics before it
ever reached a model, and landed at chance (0.474 classical / 0.451 QSVC). Feeding the
volumes through a learned encoder instead is the "richer input, not more
hand-engineering" lever identified in this session's literature research.

Volumes are aggressively downsampled (default 64^3) on ingest and cached as float16,
so a multi-GB DICOM/NIfTI collection is reduced to a few hundred MB of arrays and the
raw files can be discarded as they are processed -- peak disk stays small regardless
of collection size.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .experiment import LoadedDataset, classification_metrics, select_threshold
from .raw_hybrid import _require_torch, device_for_training

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class VolumeDataset:
    """Downsampled multi-channel 3-D volumes plus case-level labels."""

    volumes: np.ndarray  # [cases, channels, D, H, W], float32
    y: np.ndarray
    groups: np.ndarray  # patient/case id -- one scan per patient here, but kept explicit
    row_ids: np.ndarray
    channel_names: list[str]
    name: str
    positive_label: str
    negative_label: str
    positive_definition: str
    provenance: dict[str, Any]

    def as_loaded_dataset(self, X: np.ndarray, feature_prefix: str = "latent") -> LoadedDataset:
        return LoadedDataset(
            name=f"{self.name}-{feature_prefix}",
            X=np.asarray(X, dtype=float),
            y=self.y.copy(),
            feature_names=[f"{feature_prefix}_{i + 1}" for i in range(X.shape[1])],
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            provenance=dict(self.provenance),
            groups=self.groups.copy(),
            row_ids=self.row_ids.copy(),
            task_profile={
                "task_type": "binary_classification",
                "endpoint": self.positive_label,
                "positive_definition": self.positive_definition,
                "unit_of_analysis": "one patient scan (multi-channel 3-D volume)",
                "raw_imaging": True,
                "channels": list(self.channel_names),
            },
        )


def normalize_volume(volume: np.ndarray, target_shape: tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    """Resample one 3-D volume to a fixed grid and robustly intensity-normalize it.

    Uses percentile clipping (not min/max) so a single bright artefact voxel cannot
    crush the useful dynamic range -- standard practice for MR intensity, which has no
    absolute scale.
    """

    from scipy.ndimage import zoom

    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    factors = tuple(target / current for target, current in zip(target_shape, values.shape))
    if values.shape != target_shape:
        values = zoom(values, factors, order=1).astype(np.float32, copy=False)
        # zoom can be off by a voxel from rounding; crop/pad to the exact grid.
        fixed = np.zeros(target_shape, dtype=np.float32)
        slices = tuple(slice(0, min(t, c)) for t, c in zip(target_shape, values.shape))
        fixed[slices] = values[slices]
        values = fixed

    brain = values[values > 0]
    if brain.size:
        low, high = np.percentile(brain, [1.0, 99.0])
        values = np.clip(values, low, high)
        span = max(float(high - low), 1e-5)
        values = (values - low) / span
    return values.astype(np.float32, copy=False)


def load_nifti_volume(path: str | Path) -> np.ndarray:
    """Read one NIfTI file to a raw 3-D array (no resampling)."""

    import nibabel as nib

    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4:  # some sequences carry a trailing singleton or time axis
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path} has unsupported shape {data.shape}")
    return data


if nn is not None:

    class VolumeEncoder(nn.Module):
        """Small 3-D CNN sized to train on CPU or a 4 GB GPU at 64^3 input."""

        def __init__(self, in_channels: int, latent_dim: int = 4) -> None:
            super().__init__()
            if latent_dim < 2:
                raise ValueError("latent_dim must be at least 2")
            self.latent_dim = latent_dim
            self.features = nn.Sequential(
                nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(16),
                nn.SiLU(),
                nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(32),
                nn.SiLU(),
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(64),
                nn.SiLU(),
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d(1),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(64, latent_dim),
            )

        def forward(self, values: Any) -> Any:
            return self.projection(self.features(values))

    class VolumeClassifier(nn.Module):
        def __init__(self, in_channels: int, latent_dim: int = 4) -> None:
            super().__init__()
            self.encoder = VolumeEncoder(in_channels, latent_dim)
            self.head = nn.Linear(latent_dim, 1)

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return latent, self.head(latent).squeeze(-1)

else:

    class VolumeEncoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class VolumeClassifier:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module = _require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def stratified_split(
    y: np.ndarray, seed: int, test_size: float = 0.2, validation_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation/test indices (one scan per patient, so no grouping needed)."""

    from .experiment import _split_indices

    train_val, test = _split_indices(y, test_size, seed, groups=None, times=None, split_name="test")
    inner_train, inner_validation = _split_indices(
        y[train_val], validation_size, seed + 1, groups=None, times=None, split_name="validation"
    )
    return train_val[inner_train], train_val[inner_validation], test


def _probabilities(model: Any, volumes: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    torch_module = _require_torch()
    loader = DataLoader(
        TensorDataset(torch_module.from_numpy(np.asarray(volumes, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    values: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for (batch,) in loader:
            _, logits = model(batch.to(device, non_blocking=True))
            values.append(torch_module.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def encode_volumes(model: Any, volumes: np.ndarray, device: Any, batch_size: int = 4) -> np.ndarray:
    """Freeze the trained encoder and return one compact vector per scan."""

    torch_module = _require_torch()
    if isinstance(device, str):
        device = device_for_training(device)
    loader = DataLoader(
        TensorDataset(torch_module.from_numpy(np.asarray(volumes, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    values: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for (batch,) in loader:
            latent = model.encoder(batch.to(device, non_blocking=True))
            values.append(latent.detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, model.encoder.latent_dim), dtype=float)


def train_volume_encoder(
    dataset: VolumeDataset,
    latent_dim: int = 4,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "auto",
    seed: int = 7,
    target_sensitivity: float = 0.8,
) -> dict[str, Any]:
    """Train the 3-D encoder with a stratified split and a validation-selected threshold."""

    torch_module = _require_torch()
    seed_everything(seed)
    resolved_device = device_for_training(device)
    train, validation, test = stratified_split(dataset.y, seed)
    in_channels = int(dataset.volumes.shape[1])
    model = VolumeClassifier(in_channels, latent_dim).to(resolved_device)

    positive = max(float(dataset.y[train].sum()), 1.0)
    negative = max(float(len(train) - dataset.y[train].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch_module.tensor(negative / positive, device=resolved_device)
    )
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch_module.from_numpy(dataset.volumes[train]),
            torch_module.from_numpy(dataset.y[train].astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=resolved_device.type == "cuda",
    )

    best_state: dict[str, Any] | None = None
    best_validation_auc = -np.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch.to(resolved_device, non_blocking=True))
            loss = loss_fn(logits, labels.to(resolved_device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probabilities = _probabilities(
            model, dataset.volumes[validation], resolved_device, batch_size
        )
        validation_metrics = classification_metrics(
            dataset.y[validation],
            (validation_probabilities >= 0.5).astype(int),
            validation_probabilities,
            probability_score=True,
        )
        validation_auc = float(validation_metrics["roc_auc"] or -np.inf)
        history.append(
            {"epoch": float(epoch), "loss": float(np.mean(losses)), "validation_auroc": validation_auc}
        )
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:3d} loss={np.mean(losses):.4f} val_auroc={validation_auc:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("volume encoder did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = _probabilities(
        model, dataset.volumes[validation], resolved_device, batch_size
    )
    threshold = select_threshold(
        dataset.y[validation],
        validation_probabilities,
        policy="target_sensitivity",
        target_sensitivity=target_sensitivity,
    )
    test_probabilities = _probabilities(model, dataset.volumes[test], resolved_device, batch_size)
    test_predictions = (test_probabilities >= threshold["threshold"]).astype(int)
    return {
        "model": model,
        "device": str(resolved_device),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "latent_dim": int(latent_dim),
        "history": history,
        "threshold": threshold,
        "validation_metrics": classification_metrics(
            dataset.y[validation],
            (validation_probabilities >= threshold["threshold"]).astype(int),
            validation_probabilities,
            probability_score=True,
        ),
        "test_metrics": classification_metrics(
            dataset.y[test], test_predictions, test_probabilities, probability_score=True
        ),
        "train_indices": train,
        "validation_indices": validation,
        "test_indices": test,
    }
