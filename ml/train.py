# fmt: off
"""Entraînement : U-Net / DeepLabV3+ / SegFormer sur masques d'inondation.

Usage :
    python -m ml.train --data-demo data/processed/synthetic --model unet \
        --epochs 20 --split-file data/processed/synthetic/splits.json

Perte : CrossEntropy pondérée par fréquence inverse des classes (le fond
domine très largement). Les pixels valid=0 (label 255) sont exclus.
Le split est défini PAR ÉVÉNEMENT via data/splits.json (voir dataset.py).
"""
# fmt: on

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.dataset import FloodDataset
from ml.metrics import confusion_matrix_from_logits, metrics_from_confusion, summarize
from ml.models import build_model

logger = logging.getLogger(__name__)


def class_weights(dataset: FloodDataset, num_classes: int = 3, max_pixels: int = 2 ** 24):
    """Poids de classes en fréquence inverse, calculés sur un échantillon."""
    counts = np.ones(num_classes, dtype=np.float64)
    total = 0
    for i in range(len(dataset)):
        if total > max_pixels:
            break
        lab = dataset[i]["label"].numpy().ravel()
        lab = lab[lab < 255]
        counts += np.bincount(lab, minlength=num_classes)[:num_classes]
        total += lab.size
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights / weights.sum() * num_classes, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3) -> dict:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        img = batch["image"].to(device)
        logits = model(img)
        cm += confusion_matrix_from_logits(logits, batch["label"].numpy(),
                                           batch["valid"].numpy(), num_classes)
    return metrics_from_confusion(cm, num_classes)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    train_ds = FloodDataset(args.data, "train", tile_size=args.tile)
    val_ds = FloodDataset(args.data, "val", tile_size=args.tile)
    logger.info("train=%d  val=%d", len(train_ds), len(val_ds))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch, num_workers=args.workers)

    model = build_model(args.model, in_channels=2, num_classes=3).to(device)
    weights = class_weights(FloodDataset(args.data, "train", tile_size=None))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(train_dl), 1))
    scaler = torch.amp.GradScaler(device.type)

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_flood_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            img = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = batch["valid"]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(img)
                loss = criterion(logits, label)

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            running += loss.item()

        val_metrics = evaluate(model, val_dl, device)
        history.append({
            "epoch": epoch,
            "train_loss": running / max(len(train_dl), 1),
            **{k: val_metrics[k] for k in
               ("macro_iou", "macro_dice", "flood_iou", "flood_dice")},
        })
        logger.info("epoch %d  loss=%.4f  macro_iou=%.4f  flood_iou=%.4f",
                    epoch, running / max(len(train_dl), 1),
                    val_metrics["macro_iou"], val_metrics["flood_iou"])

        if val_metrics["flood_iou"] > best_flood_iou:
            best_flood_iou = val_metrics["flood_iou"]
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": {k: val_metrics[k] for k in
                                ("macro_iou", "flood_iou", "flood_dice")},
                "class_weights": weights.tolist(),
            }, out_dir / "best.pt")
            logger.info("meilleur flood_iou=%.4f sauvegardé", best_flood_iou)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "final_metrics.json").write_text(
        json.dumps(evaluate(model, val_dl, device), indent=2)
    )
    logger.info("Terminé. Métriques in %s", out_dir)
    return out_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entraînement InondaVision")
    p.add_argument("--data", required=True, help="dossier dataset (s1/, labels/)")
    p.add_argument("--model", default="unet",
                   choices=["unet", "deeplabv3", "segformer"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="models/checkpoints")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())


def _train_tiny(data_dir, epochs: int = 1):
    """Entraînement minimal pour tests de fumée (CPU, quelques secondes)."""
    import shutil
    from types import SimpleNamespace

    out_dir = Path(data_dir.parent) / "ckpt_smoke"
    args = SimpleNamespace(
        data=str(data_dir), model="unet", epochs=epochs, batch=1, eval_batch=1,
        tile=128, lr=1e-3, workers=0, out=str(out_dir),
    )
    return train(args)


# cleanup utilitaire pour tests : évite d'alourdir le dépôt
def _cleanup(path):
    shutil.rmtree(path, ignore_errors=True)
