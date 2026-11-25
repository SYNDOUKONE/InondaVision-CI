# fmt: off
"""Évaluation finale sur le split TEST (événements jamais vus).

Produit evaluation/metrics.json + matrice de confusion + cartes PNG des
prédictions sur un échantillon de scènes, et compare avec la baseline
seuillage/détection de changement.

Usage :
    python -m ml.evaluate --data data/processed/synthetic \
        --ckpt models/checkpoints/unet/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ml import baseline
from ml.data.dataset import FloodDataset
from ml.inference import load_checkpoint
from ml.metrics import confusion_matrix_from_logits, metrics_from_confusion, summarize

logger = logging.getLogger(__name__)


def _bincount_cm(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred = pred[valid]
    label = label[valid]
    flat = label * 3 + pred
    return np.bincount(flat.ravel(), minlength=9).reshape(3, 3)


def evaluate_split(model, data_dir, device, batch_size=4) -> dict:
    ds = FloodDataset(data_dir, "test", tile_size=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    cm = np.zeros((3, 3), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            cm += confusion_matrix_from_logits(logits, batch["label"].numpy(),
                                               batch["valid"].numpy())
    return metrics_from_confusion(cm)


def _read_pre_scene(image_path: Path) -> np.ndarray | None:
    """Lit la scène pré-événement jumelle (<scene_id>_pre_cis.tif)."""
    import rasterio as rio

    pre_path = Path(str(image_path).replace("_post_cis.tif", "_pre_cis.tif"))
    if not pre_path.exists():
        return None
    with rio.open(pre_path) as src:
        img = src.read().astype(np.float32)
        return img[0] if img.shape[0] == 1 else img[0]


def evaluate_baseline(data_dir) -> dict:
    ds = FloodDataset(data_dir, "test", tile_size=None, normalize=False)
    cm = np.zeros((3, 3), dtype=np.int64)

    for i in range(len(ds)):
        sample = ds[i]
        img = sample["image"].numpy()
        label = sample["label"].numpy()
        valid = sample["valid"].numpy().astype(bool)

        pre_vv = _read_pre_scene(sample["image_path"])
        post_vv = img[0] if img.shape[0] == 1 else img[0]

        perm = baseline.water_mask_pre(pre_vv if pre_vv is not None else post_vv)
        pred = baseline.water_mask_post(pre_vv if pre_vv is not None else post_vv,
                                        post_vv, permanent_water=perm)
        cm += _bincount_cm(pred, label, valid)
    return metrics_from_confusion(cm)


def save_scene_maps(model, data_dir, out_dir: Path, device, max_scenes: int = 3):
    """PNG couleur des masques prédits pour lecture rapide."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = np.array([(0, 0, 0), (0, 102, 204), (255, 80, 80)], dtype=np.uint8)

    ds = FloodDataset(data_dir, "test", tile_size=None)
    with torch.no_grad():
        for i, sample in enumerate(ds):
            if i >= max_scenes:
                break
            mask = predict_single(model, sample["image"], device)
            rgb = colors[mask]
            Image.fromarray(rgb).save(out_dir / f"{sample['scene_id']}_pred.png")


def predict_single(model, image: torch.Tensor, device) -> np.ndarray:
    from ml.inference import predict_mask

    return predict_mask(model, image, device=device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    model, ckpt = load_checkpoint(args.ckpt, device)
    logger.info("checkpoint époque %s (flood_iou val %.4f)",
                ckpt.get("epoch"), ckpt.get("val_metrics", {}).get("flood_iou", -1))

    dl_metrics = evaluate_split(model, args.data, device)
    logger.info("Deep Learning (test, événements jamais vus)\n%s", summarize(dl_metrics))

    bs_metrics = evaluate_baseline(args.data)
    logger.info("Baseline seuillage (test)\n%s", summarize(bs_metrics))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"deep_learning": dl_metrics, "baseline_threshold": bs_metrics}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        cm = np.array(dl_metrics["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), ["bg", "perm", "flood"])
        ax.set_yticks(range(3), ["bg", "perm", "flood"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xlabel("prédit")
        ax.set_ylabel("réel")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=120)
    except ImportError:
        logger.warning("matplotlib absent : matrice de confusion PNG ignorée")

    save_scene_maps(model, args.data, out / "segmentation_results", device)
    logger.info("Résultats écrits dans %s", out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Évaluation test InondaVision")
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="evaluation")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(parse_args())
