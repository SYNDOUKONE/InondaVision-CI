"""Métriques de segmentation : IoU, Dice, Precision, Recall, F1.

Attention : jamais d'accuracy globale seule — la terre domine largement les
pixels inondés sur Sentinel-1 (déséquilibre de classes extrême).

Les pixels avec valid=0 (label 255) sont exclus de tous les calculs.
"""

import numpy as np
import torch

CLASS_NAMES = ["background", "permanent_water", "flood_water"]


@torch.no_grad()
def confusion_matrix_from_logits(logits: torch.Tensor, labels: torch.Tensor,
                                 valid: torch.Tensor, num_classes: int = 3
                                 ) -> np.ndarray:
    """Matrice de confusion (num_classes x num_classes) cumulée sur le batch.

    logits (B, C, H, W), labels (B, H, W) int64, valid (B, H, W) {0,1}.
    """
    preds = logits.argmax(dim=1)  # (B, H, W)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    preds_np = to_np(preds)
    labels_np = to_np(labels)
    valid_np = to_np(valid).astype(bool)

    preds_np = preds_np[valid_np]
    labels_np = labels_np[valid_np]

    flat = labels_np * num_classes + preds_np
    counts = np.bincount(flat.flatten(), minlength=num_classes ** 2)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def metrics_from_confusion(cm: np.ndarray, eps: float = 1e-9) -> dict:
    """IoU / Dice / Precision / Recall / F1 par classe + moyennes (macro et flood)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    support = cm.sum(axis=1).astype(np.int64).tolist()

    results = {
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support_pixels": support[i],
            }
            for i, name in enumerate(CLASS_NAMES[: cm.shape[0]])
        },
        "macro_iou": float(iou.mean()),
        "macro_dice": float(dice.mean()),
        "flood_iou": float(iou[2]) if cm.shape[0] > 2 else None,
        "flood_dice": float(dice[2]) if cm.shape[0] > 2 else None,
    }

    bg_count = cm[0].sum()
    water_count = cm.sum() - bg_count
    overall_acc = tp.sum() / max(cm.sum(), 1)
    results["overall_accuracy"] = float(overall_acc)
    results["water_pixel_ratio"] = float(water_count / max(cm.sum(), 1))
    return results


def summarize(metrics: dict) -> str:
    pc = metrics["per_class"]
    lines = [
        f"IoU macro  : {metrics['macro_iou']:.4f}",
        f"Dice macro : {metrics['macro_dice']:.4f}",
    ]
    for name in CLASS_NAMES:
        if name in pc:
            m = pc[name]
            lines.append(
                f"  {name:<16} IoU={m['iou']:.4f} Dice={m['dice']:.4f} "
                f"P={m['precision']:.4f} R={m['recall']:.4f}"
            )
    lines.append(
        f"flood IoU  : {metrics['flood_iou']:.4f}  "
        f"flood Dice : {metrics['flood_dice']:.4f}"
    )
    return "\n".join(lines)
