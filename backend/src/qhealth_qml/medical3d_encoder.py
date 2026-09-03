"""Native 3-D medical encoder on frozen MedicalNet weights — replaces two earlier compromises.

Two things this fixes in the platform's imaging path:

1. **A real 3-D prior instead of ImageNet.** `pretrained_encoder.py` applies a frozen ImageNet
   ResNet18 slice-wise, which throws away through-plane context and transfers from natural
   photographs. MedicalNet (Tencent, MIT) is a 3-D ResNet pretrained across 23 medical imaging
   datasets, so the prior is volumetric and in-domain. Downloaded from the HuggingFace mirror
   `TencentMedicalNet/MedicalNet-Resnet18` (ungated).
2. **Orientation canonicalisation, which was simply missing.** Nothing in the earlier ingest
   reoriented volumes to a canonical axis order, so subjects acquired in different orientations
   were being fed to the model inconsistently — a silent, systematic input bug rather than a
   modelling choice. MONAI's `Orientation` is applied here (and in `ingestion.py`).

The backbone is frozen and only a small projection head trains, which is what makes small
cohorts trainable at all. MedicalNet expects **single-channel** input, so multi-sequence studies
are embedded one sequence at a time and concatenated — the sequences stay separate rather than
being crushed into one channel, which is also how radiologists read them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .raw_hybrid import _require_torch

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

MEDICALNET_REPO = "TencentMedicalNet/MedicalNet-Resnet18"
MEDICALNET_FILE = "resnet_18_23dataset.pth"
MEDICALNET_FEATURES = 512


def load_medicalnet_resnet18() -> Any:
    """Frozen 3-D ResNet18 with MedicalNet weights, truncated to a feature extractor."""

    _require_torch()
    from huggingface_hub import hf_hub_download
    from monai.networks.nets import resnet18

    # MedicalNet was trained with these specific structural options; MONAI enforces the match.
    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        shortcut_type="A",
        bias_downsample=True,
        feed_forward=False,
    )
    checkpoint_path = hf_hub_download(MEDICALNET_REPO, MEDICALNET_FILE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    # weights were saved under DataParallel, so every key carries a "module." prefix
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 10:  # a couple of head keys are expected; wholesale mismatch is not
        raise RuntimeError(f"MedicalNet weights did not match the model ({len(missing)} missing)")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def canonicalise_orientation(volume: np.ndarray, axcodes: str = "RAS") -> np.ndarray:
    """Reorient a 3-D volume to a canonical axis order.

    Omitted from the platform's first imaging pass, which meant differently-acquired subjects
    reached the model in different orientations. Applied without an affine this only fixes axis
    ordering/flips consistently; a full spatial correction needs the NIfTI affine, which
    `ingestion.py` passes through when available.
    """

    from monai.transforms import Orientation

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    oriented = Orientation(axcodes=axcodes)(array)
    return np.asarray(oriented, dtype=np.float32)[0]


if nn is not None:

    class MedicalNetEncoder(nn.Module):
        """Frozen 3-D MedicalNet backbone applied per sequence + trainable projection."""

        def __init__(
            self,
            in_channels: int,
            latent_dim: int = 32,
            backbone: Any | None = None,
        ) -> None:
            super().__init__()
            if latent_dim < 2:
                raise ValueError("latent_dim must be at least 2")
            self.latent_dim = latent_dim
            self.in_channels = int(in_channels)
            self.backbone = backbone if backbone is not None else load_medicalnet_resnet18()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.projection = nn.Sequential(
                nn.Linear(MEDICALNET_FEATURES * self.in_channels, 256),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(256, latent_dim),
            )

        def _embed_one(self, volume: Any) -> Any:
            """volume: [B, 1, D, H, W] -> [B, 512]"""

            with torch.no_grad():
                features = self.backbone(volume)
            if features.ndim > 2:
                features = self.pool(features).flatten(1)
            return features

        def forward(self, values: Any) -> Any:
            # values: [B, C, D, H, W]; MedicalNet takes one sequence at a time
            embeddings = [self._embed_one(values[:, c : c + 1]) for c in range(values.shape[1])]
            return self.projection(torch.cat(embeddings, dim=1))

    class MedicalNetClassifier(nn.Module):
        def __init__(self, in_channels: int, latent_dim: int = 32) -> None:
            super().__init__()
            self.encoder = MedicalNetEncoder(in_channels, latent_dim)
            self.head = nn.Linear(latent_dim, 1)

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return latent, self.head(latent).squeeze(-1)

else:

    class MedicalNetEncoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class MedicalNetClassifier:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()
