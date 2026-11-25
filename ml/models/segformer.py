"""SegFormer (Transformers, b0) — implémentation optionnelle via HuggingFace.

Installez `transformers` pour l'utiliser :
    pip install transformers
Puis initialisez avec `pretrained=True`.
"""

from typing import Optional

import torch
import torch.nn as nn


class SegFormerFlood(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 3,
                 pretrained: bool = False, encoder_name: str = "google/segformer-b0-512-512",
                 _unused: Optional[None] = None):
        super().__init__()
        try:
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerConfig)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SegFormer nécessite 'transformers'. "
                "Installez-le : pip install transformers"
            ) from exc

        config = SegformerConfig.from_pretrained(encoder_name)
        config.num_channels = in_channels
        config.num_labels = num_classes
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name, config=config, ignore_mismatched_sizes=True
            )
        else:
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        out = self.net(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits
