"""Jeux de données de segmentation d'inondations (Sen1Floods11 et formats compatibles).

Convention de layout attendue (compatible Sen1Floods11 / production interne) :

    <data_dir>/s1/<scene_id>_pre_cis.tif      2 canaux VV/VH en dB, float32
    <data_dir>/s1/<scene_id>_post_cis.tif     idem
    <data_dir>/labels/<scene_id>_label.tif    uint8 : 0 land, 1 permanent_water,
                                              2 flood_water, 255 no_data/ignore

Chaque scène appartient à un « événement » (ex : abidjan-2024-06). Les splits
 train/val/test sont définis PAR ÉVÉNEMENT (fichier JSON), jamais par tuile
 tirée au hasard — cela évite la fuite de données entre tuiles d'une même scène,
 qui surestimerait les métriques.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NODATA_LABEL = 255
PRE_SUFIX = "_pre_cis.tif"
POST_SUFFIX = "_post_cis.tif"


@dataclass
class FloodSample:
    scene_id: str
    event: str
    image_path: Path
    label_path: Path | None
    stack: str  # "pre" ou "post"


def load_split(data_dir: str | Path, split: str) -> list[FloodSample]:
    """Charge les échantillons d'un split ('train', 'val', 'test').

    Le fichier <data_dir>/splits.json mappe event -> {"train": [...], ...}
    avec des identifiants de SCÈNES (pas de tuiles).
    """
    data_dir = Path(data_dir)
    split_file = data_dir / "splits.json"
    if split_file.exists():
        mapping = json.loads(split_file.read_text())
        scenes = mapping.get(split, [])
    else:
        scenes = sorted(p.name for p in (data_dir / "s1").glob(f"*{POST_SUFFIX}"))
        logger.warning("splits.json absent, tous les scènes vont dans %s", split)

    samples = []
    for scene_id in scenes:
        img = data_dir / "s1" / f"{scene_id}{POST_SUFFIX}"
        lab = data_dir / "labels" / f"{scene_id}_label.tif"
        samples.append(FloodSample(scene_id, scene_id.split("_")[0], img, lab, "post"))
    return samples


class FloodDataset(Dataset):
    """Tuiles Sentinel-1 VV/VH (2 canaux, dB) + masque de segmentation 3 classes.

    - crop aléatoire `tile_size` pour l'entraînement, image entière sinon.
    - les pixels dont le label = 255 ne contribuent pas à la perte/aux métriques.
    """

    def __init__(self, data_dir: str | Path, split: str = "train",
                 tile_size: int | None = 512, normalize: bool = True):
        self.samples = load_split(data_dir, split)
        self.tile_size = tile_size
        self.split = split
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        with rasterio_open(path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
            self.transform = src.transform
            self.crs = src.crs
        return img

    @staticmethod
    def _load_label(path: Path | None, shape) -> np.ndarray:
        if path is None or not path.exists():
            return np.zeros(shape, dtype=np.int64)
        with rasterio_open(path) as src:
            lab = src.read(1).astype(np.int64)
        lab[lab == NODATA_LABEL] = NODATA_LABEL
        return lab

    def __getitem__(self, idx: int):
        import rasterio as rio

        s = self.samples[idx]
        with rio.open(s.image_path) as src:
            img = src.read().astype(np.float32)
            profile = {k: getattr(src, k) for k in ("transform", "crs")}
        label = self._load_label(s.label_path, img.shape[1:])

        if img.shape[0] == 1:
            img = np.repeat(img, 2, axis=0)

        if self.normalize:
            img = np.clip(img, -30.0, 5.0)
            img = (img + 30.0) / 35.0

        if self.tile_size and self.split == "train":
            img, label = random_crop(img, label, self.tile_size)

        valid = (label != NODATA_LABEL).astype(np.float32)  # (H, W)
        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(np.where(label == NODATA_LABEL, 0, label)),
            "valid": torch.from_numpy(valid),
            "scene_id": s.scene_id,
            "image_path": str(s.image_path),
            "label_path": str(s.label_path),
        }


def random_crop(img: np.ndarray, label: np.ndarray, size: int):
    _, H, W = img.shape
    if H <= size or W <= size:
        pad_h, pad_w = max(size - H, 0), max(size - W, 0)
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))
        label = np.pad(label, ((0, pad_h), (0, pad_w)), constant_values=NODATA_LABEL)
        _, H, W = img.shape
    top, left = np.random.randint(0, H - size + 1), np.random.randint(0, W - size + 1)
    return img[:, top:top + size, left:left + size], label[top:top + size, left:left + size]


def rasterio_open(path):
    import rasterio as rio
    return rio.open(path)
