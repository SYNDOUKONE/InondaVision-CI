"""Inférence sur scènes complètes (tuiles à cheval avec moyennage des softmax).

Gère des scènes de n'importe quelle taille : découpage en fenêtres 512x512
avec recouvrement `overlap`, softmax par fenêtre, moyenne pondérée dans les
zones de recouvrement, puis argmax. Passe en mode uint8 {0,1,2}.
"""

import numpy as np
import torch


@torch.no_grad()
def sliding_window_inference(model, image: torch.Tensor, num_classes: int = 3,
                             window: int = 512, overlap: int = 128,
                             device: str | torch.device = "cpu") -> torch.Tensor:
    """image : (C, H, W) normalisé. Retour : logits (num_classes, H, W)."""
    model.eval()
    C, H, W = image.shape
    stride = window - overlap

    logits = torch.zeros(num_classes, H, W)
    counts = torch.zeros(1, H, W)
    image = image.unsqueeze(0).to(device)

    ys = list(range(0, max(H - window, 0) + 1, stride))
    xs = list(range(0, max(W - window, 0) + 1, stride))
    if ys[-1] + window < H:
        ys.append(H - window)
    if xs[-1] + window < W:
        xs.append(W - window)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + window, H), min(x0 + window, W)
            tile = image[:, :, y0:y1, x0:x1]
            out = model(tile)  # (1, C, h, w)
            out = torch.nn.functional.interpolate(
                out, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False
            )
            logits[:, y0:y1, x0:x1] += out[0].cpu()
            counts[:, y0:y1, x0:x1] += 1

    logits /= counts.clamp(min=1)
    return logits


@torch.no_grad()
def predict_mask(model, image: torch.Tensor, window: int = 512, overlap: int = 128,
                 device: str | torch.device = "cpu") -> np.ndarray:
    """Retourne le masque (H, W) uint8 : 0 background, 1 permanent, 2 flood."""
    logits = sliding_window_inference(model, image, 3, window, overlap, device)
    return logits.argmax(dim=0).numpy().astype(np.uint8)


def load_checkpoint(path, device="cpu"):
    """Charge un checkpoint sauvegardé par ml.train."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    from ml.models import build_model

    model = build_model(ckpt.get("model", "unet"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
