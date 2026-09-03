"""Raw-waveform GPU encoder for the PhysioNet Gait-in-Parkinson's-Disease benchmark.

Mirrors the raw_hybrid.py (ECG) pattern: a small 1-D CNN consumes the raw
vertical-ground-reaction-force signal directly and produces a compact latent
vector that is handed to the existing Qiskit/classical heads via the same
LoadedDataset contract. This is closer to an actual clinical gait assessment
(bradykinesia / gait-pattern asymmetry) than the platform's prior 22-feature
voice dataset, and does not require hand-engineered gait-cycle features -- the
encoder learns its own representation from the raw force curves, following
the "richer input, not more hand-engineering" lesson from this session's
literature research.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

N_CHANNELS = 18  # 8 left + 8 right force sensors + 2 per-foot sums
SAMPLING_RATE_HZ = 100.0


@dataclass(frozen=True)
class RawGaitDataset:
    """Raw normalized gait force-plate recordings and subject metadata."""

    signals: np.ndarray  # [rows, 18, samples]
    y: np.ndarray  # 1 = Parkinson's patient, 0 = control
    groups: np.ndarray  # subject id -- prevents a subject's repeat trials crossing splits
    row_ids: np.ndarray
    study: np.ndarray  # Ga / Ju / Si sub-study origin, kept as a reported subgroup
    target_samples: int
    name: str = "gaitpdb-raw-force"

    def as_loaded_dataset(self, X: np.ndarray, feature_prefix: str = "latent") -> LoadedDataset:
        return LoadedDataset(
            name=f"{self.name}-{feature_prefix}",
            X=np.asarray(X, dtype=float),
            y=self.y.copy(),
            feature_names=[f"{feature_prefix}_{i + 1}" for i in range(X.shape[1])],
            positive_label="parkinsons_gait_pattern",
            negative_label="control_gait_pattern",
            provenance={"source": "PhysioNet gaitpdb 1.0.0 raw force-plate signal", "raw_waveform": True},
            groups=self.groups.copy(),
            row_ids=self.row_ids.copy(),
            subgroups={"study": self.study.copy()},
            task_profile={
                "task_type": "binary_classification",
                "endpoint": "parkinsons_gait_pattern",
                "positive_definition": "diagnosed Parkinson's disease (Pt subject code)",
                "negative_definition": "healthy control (Co subject code)",
                "unit_of_analysis": "one ~2-minute walking trial, 16-sensor force-plate insoles",
                "raw_waveform": True,
            },
        )


def normalize_raw_gait(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """Robustly normalize one [samples, 18] or [18, samples] gait recording."""

    from scipy.signal import resample

    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("raw gait signal must be a 2-D array")
    if values.shape[1] == N_CHANNELS:
        values = values.T
    if values.shape[0] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} gait channels, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[1] != target_samples:
        if values.shape[1] > target_samples:
            values = values[:, :target_samples]
        else:
            values = resample(values, target_samples, axis=1).astype(np.float32, copy=False)
    median = np.median(values, axis=1, keepdims=True)
    scale = np.std(values, axis=1, keepdims=True)
    return ((values - median) / np.maximum(scale, 1e-5)).astype(np.float32, copy=False)


def load_raw_gaitpdb(
    root: str | Path,
    target_samples: int = 12000,
    first_trial_only: bool = False,
) -> RawGaitDataset:
    """Read gaitpdb .txt recordings directly.

    Each file is 19 columns: time, 8 left-foot sensors, 8 right-foot sensors, and
    the two per-foot totals (format.txt) -- 18 signal channels at 100 Hz.
    Subjects contribute several walking trials; all are kept by default because the
    split is subject-grouped anyway, which both maximises the (small) sample count
    and keeps a subject's trials from straddling train and test.
    """

    root = Path(root)
    pattern = "*_01.txt" if first_trial_only else "*.txt"
    signal_files = sorted(
        path
        for path in root.glob(pattern)
        # keep only actual recordings: <study><group><subject>_<trial>.txt
        if re.fullmatch(r"(Ga|Ju|Si)(Co|Pt)\d+_\d+", path.stem)
    )
    if not signal_files:
        raise ValueError(f"no gaitpdb recordings found under {root}")

    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    row_ids: list[str] = []
    studies: list[str] = []
    for index, path in enumerate(signal_files, start=1):
        stem = path.stem  # e.g. GaPt03_01
        subject_code = stem.split("_")[0]  # e.g. GaPt03
        study = subject_code[:2]  # Ga / Ju / Si
        is_patient = "Pt" in subject_code
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] < N_CHANNELS + 1:
            raise ValueError(f"unexpected column count in {path}: {raw.shape}")
        # column 0 is time in seconds; columns 1-16 are the 8+8 sensors;
        # columns 17-18 are the left/right sums (format.txt-documented layout).
        sensor_block = raw[:, 1 : N_CHANNELS + 1]
        signals.append(normalize_raw_gait(sensor_block, target_samples))
        labels.append(int(is_patient))
        groups.append(subject_code)
        row_ids.append(stem)
        studies.append(study)
        if index % 25 == 0 or index == len(signal_files):
            print(f"loaded raw gait {index}/{len(signal_files)}", flush=True)

    return RawGaitDataset(
        signals=np.asarray(signals, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=str),
        row_ids=np.asarray(row_ids, dtype=str),
        study=np.asarray(studies, dtype=str),
        target_samples=target_samples,
    )


if nn is not None:

    class RawGaitEncoder(nn.Module):
        """Small 1-D CNN sized for a 4-16 GB GPU and ~2-minute, 18-channel force-plate signals."""

        def __init__(self, latent_dim: int = 4) -> None:
            super().__init__()
            if latent_dim < 2:
                raise ValueError("latent_dim must be at least 2")
            self.latent_dim = latent_dim
            self.features = nn.Sequential(
                nn.Conv1d(N_CHANNELS, 32, kernel_size=25, stride=4, padding=12, bias=False),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7, bias=False),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7, bias=False),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(64, latent_dim),
            )

        def forward(self, values: Any) -> Any:
            return self.projection(self.features(values))

    class RawGaitClassifier(nn.Module):
        def __init__(self, latent_dim: int = 4) -> None:
            super().__init__()
            self.encoder = RawGaitEncoder(latent_dim)
            self.head = nn.Linear(latent_dim, 1)

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return latent, self.head(latent).squeeze(-1)

else:

    class RawGaitEncoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class RawGaitClassifier:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module = _require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _probabilities(model: Any, signals: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    torch_module = _require_torch()
    loader = DataLoader(
        TensorDataset(torch_module.from_numpy(signals)),
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


def encode_raw_gait(model: Any, signals: np.ndarray, device: Any, batch_size: int = 16) -> np.ndarray:
    """Freeze the trained encoder and return one compact vector per trial."""

    torch_module = _require_torch()
    if isinstance(device, str):
        device = device_for_training(device)
    loader = DataLoader(
        TensorDataset(torch_module.from_numpy(np.asarray(signals, dtype=np.float32))),
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


def split_indices_grouped(dataset: RawGaitDataset, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-grouped train/validation/test split (no official folds exist for gaitpdb)."""

    from .experiment import _split_indices

    all_index = np.arange(len(dataset.y))
    train_val, test = _split_indices(
        dataset.y, 0.2, seed, groups=dataset.groups, times=None, split_name="test"
    )
    train, validation = _split_indices(
        dataset.y[train_val],
        0.2,
        seed + 1,
        groups=dataset.groups[train_val],
        times=None,
        split_name="validation",
    )
    train = train_val[train]
    validation = train_val[validation]
    if set(dataset.groups[train]).intersection(dataset.groups[validation]):
        raise ValueError("subject overlap between train and validation")
    if set(dataset.groups[train]).intersection(dataset.groups[test]):
        raise ValueError("subject overlap between train and test")
    if set(dataset.groups[validation]).intersection(dataset.groups[test]):
        raise ValueError("subject overlap between validation and test")
    return train, validation, test


def train_raw_gait_encoder(
    dataset: RawGaitDataset,
    latent_dim: int = 4,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    device: str = "auto",
    seed: int = 7,
) -> dict[str, Any]:
    """Train the raw gait encoder with a subject-grouped split and validation-selected threshold."""

    torch_module = _require_torch()
    seed_everything(seed)
    resolved_device = device_for_training(device)
    train, validation, test = split_indices_grouped(dataset, seed)
    model = RawGaitClassifier(latent_dim).to(resolved_device)
    positive = max(float(dataset.y[train].sum()), 1.0)
    negative = max(float(len(train) - dataset.y[train].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch_module.tensor(negative / positive, device=resolved_device))
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch_module.from_numpy(dataset.signals[train]),
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
        validation_probabilities = _probabilities(model, dataset.signals[validation], resolved_device, batch_size)
        validation_metrics = classification_metrics(
            dataset.y[validation],
            (validation_probabilities >= 0.5).astype(int),
            validation_probabilities,
            probability_score=True,
        )
        validation_auc = float(validation_metrics["roc_auc"] or -np.inf)
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses)), "validation_auroc": validation_auc})
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("raw gait encoder did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = _probabilities(model, dataset.signals[validation], resolved_device, batch_size)
    threshold = select_threshold(
        dataset.y[validation], validation_probabilities, policy="target_sensitivity", target_sensitivity=0.8
    )
    test_probabilities = _probabilities(model, dataset.signals[test], resolved_device, batch_size)
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
        "test_probabilities": test_probabilities,
        "test_indices": test,
        "train_indices": train,
        "validation_indices": validation,
    }
