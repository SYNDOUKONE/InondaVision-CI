"""Service d'inférence : chargement du modèle + détection d'inondations.

Charge le checkpoint une seule fois puis sert /api/v1/detect. En l'absence de
checkpoint, le mode DEV utilise une baseline (seuillage radar) pour que la
démo fonctionne sans GPU ni modèle entraîné.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio as rio
import torch

from ml import baseline
from ml.inference import load_checkpoint, predict_mask

logger = logging.getLogger(__name__)


class FloodDetectionService:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self._model = None
        self._ckpt_meta = None
        if self.checkpoint_path.exists():
            self._load()
        else:
            logger.warning(
                "Checkpoint introuvable (%s) — mode baseline (seuillage radar).",
                self.checkpoint_path,
            )

    def _load(self):
        self._model, self._ckpt_meta = load_checkpoint(self.checkpoint_path, self.device)
        logger.info("Modèle chargé : %s", self.checkpoint_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def read_scene(self, path: str | Path) -> tuple[np.ndarray, dict]:
        """Lit un GeoTIFF S1 -> (image (2,H,W) flottant dB, profil rasterio)."""
        with rio.open(path) as src:
            img = src.read().astype(np.float32)
            if img.shape[0] == 1:
                img = np.repeat(img, 2, axis=0)
            profile = {
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "tags": src.tags(),
            }
        return img, profile

    def normalize(self, img: np.ndarray) -> torch.Tensor:
        img = np.clip(img, -30.0, 5.0)
        img = (img + 30.0) / 35.0
        return torch.from_numpy(img)

    def detect(self, pre_path: str | Path, post_path: str | Path) -> dict:
        """Renvoie masque 3 classes + profils pre/post + méthode utilisée."""
        pre, pre_profile = self.read_scene(pre_path)
        post, post_profile = self.read_scene(post_path)

        if self._model is not None:
            tensor = self.normalize(post)
            mask = predict_mask(self._model, tensor, device=self.device)
            method = f"deeplearning:{self._ckpt_meta.get('model', 'unet')}"
            mask_pre = predict_mask(
                self._model, self.normalize(pre), device=self.device
            )
        else:
            perm = baseline.water_mask_pre(pre[0])
            mask = baseline.water_mask_post(pre[0], post[0], permanent_water=perm)
            mask_pre = baseline.water_mask_post(pre[0], pre[0], permanent_water=perm)
            method = "baseline:otsu-threshold"

        return {
            "mask": mask,
            "mask_pre": mask_pre,
            "pre_profile": pre_profile,
            "post_profile": post_profile,
            "method": method,
        }
