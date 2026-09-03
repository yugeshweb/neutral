"""Frozen pretrained encoder for volumetric medical imaging — the transfer-learning half
of Mari et al. that the platform's first imaging pass skipped.

Why this exists: the from-scratch 3-D CNN in `imaging_hybrid.py` trains ~300k parameters on
47-192 samples, and every small-cohort condition failed the same way (training loss collapsing
while validation AUROC stayed flat). Mari et al.'s "quantum transfer learning" result depends
on a *frozen pretrained backbone* — only the small dressed quantum circuit is trained. Taking
their quantum layer without their backbone was the central architectural error of that pass.

Design:
  volume [C, D, H, W]
    -> per-slice RGB triplets fed through a FROZEN ImageNet ResNet18 (the exact backbone
       Mari et al. use), no gradients, no fine-tuning
    -> 512-dim per-slice embeddings, aggregated over slices (mean + max) per channel
    -> a small trainable projection head to `latent_dim` (default 32, NOT 4)

The wide latent matters: the previous pipeline compressed an entire MRI to 4 numbers *inside*
the CNN because `n_qubits=4`, discarding information before the circuit ever saw it. Here the
representation stays wide and is projected to qubit count only at the quantum boundary, which
is what the reference architecture actually does.

Slice-wise 2-D transfer is used rather than a 3-D backbone deliberately: ImageNet ResNet18
weights are reliably fetchable and this is the published configuration, whereas 3-D medical
checkpoints (MedicalNet, Models Genesis) live behind ad-hoc Drive links of varying availability.
The trade-off — losing through-plane context — is recorded honestly in the model card.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .raw_hybrid import _require_torch

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_FEATURES = 512  # ResNet18 penultimate width


def load_frozen_resnet18() -> Any:
    """Return a frozen, eval-mode ResNet18 truncated before its classifier."""

    _require_torch()
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


if nn is not None:

    class PretrainedVolumeEncoder(nn.Module):
        """Frozen 2-D backbone applied slice-wise + a small trainable projection.

        Only `projection` carries gradients; the backbone is frozen, which is the entire
        point — it is what lets a 50-subject cohort train at all.
        """

        def __init__(
            self,
            in_channels: int,
            latent_dim: int = 32,
            slice_stride: int = 2,
            backbone: Any | None = None,
        ) -> None:
            super().__init__()
            if latent_dim < 2:
                raise ValueError("latent_dim must be at least 2")
            self.latent_dim = latent_dim
            self.in_channels = int(in_channels)
            self.slice_stride = max(1, int(slice_stride))
            self.backbone = backbone if backbone is not None else load_frozen_resnet18()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

            self.register_buffer(
                "_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
            )
            # mean+max pooling over slices, per input channel
            pooled = BACKBONE_FEATURES * 2 * self.in_channels
            self.projection = nn.Sequential(
                nn.Linear(pooled, 256),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(256, latent_dim),
            )

        def _embed_channel(self, volume: Any) -> Any:
            """volume: [B, D, H, W] -> [B, 2*512] via frozen backbone over sampled slices."""

            batch, depth, height, width = volume.shape
            indices = torch.arange(0, depth, self.slice_stride, device=volume.device)
            sampled = volume.index_select(1, indices)  # [B, S, H, W]
            slices = sampled.reshape(batch * len(indices), 1, height, width)
            slices = slices.expand(-1, 3, -1, -1)  # grey -> RGB for an ImageNet backbone
            if height < 64 or width < 64:
                slices = torch.nn.functional.interpolate(
                    slices, size=(64, 64), mode="bilinear", align_corners=False
                )
            slices = (slices - self._mean) / self._std
            with torch.no_grad():
                features = self.backbone(slices)  # [B*S, 512]
            features = features.view(batch, len(indices), BACKBONE_FEATURES)
            return torch.cat([features.mean(dim=1), features.amax(dim=1)], dim=1)

        def forward(self, values: Any) -> Any:
            # values: [B, C, D, H, W]
            per_channel = [self._embed_channel(values[:, c]) for c in range(values.shape[1])]
            return self.projection(torch.cat(per_channel, dim=1))

    class PretrainedVolumeClassifier(nn.Module):
        """Frozen-backbone encoder + a linear head (the classical control arm)."""

        def __init__(self, in_channels: int, latent_dim: int = 32, slice_stride: int = 2) -> None:
            super().__init__()
            self.encoder = PretrainedVolumeEncoder(in_channels, latent_dim, slice_stride)
            self.head = nn.Linear(latent_dim, 1)

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return latent, self.head(latent).squeeze(-1)

else:

    class PretrainedVolumeEncoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class PretrainedVolumeClassifier:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def trainable_parameter_report(model: Any) -> dict[str, int]:
    """Frozen vs trainable split — the number that explains why this works at small n."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }
