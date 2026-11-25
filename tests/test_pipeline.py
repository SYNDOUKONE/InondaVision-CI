"""Tests de fumée du pipeline InondaVision (rapides, sans GPU).

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.metrics import confusion_matrix_from_logits, metrics_from_confusion  # noqa: E402
from ml.models import build_model  # noqa: E402


def test_models_forward():
    for name in ("unet", "deeplabv3"):
        model = build_model(name, in_channels=2, num_classes=3)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 2, 64, 64))
        assert out.shape == (1, 3, 64, 64), name


def test_metrics_perfect_prediction():
    labels = torch.randint(0, 3, (1, 16, 16))
    valid = torch.ones_like(labels, dtype=torch.float32)
    logits = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 10
    cm = confusion_matrix_from_logits(logits, labels, valid)
    m = metrics_from_confusion(cm)
    assert m["macro_iou"] > 0.99
    assert m["flood_iou"] > 0.99


def test_metrics_ignores_nodata():
    labels = torch.ones(1, 8, 8, dtype=torch.int64)
    valid = torch.ones(1, 8, 8, dtype=torch.float32)
    valid[:, :, :4] = 0  # moitié ignorée
    preds = torch.full((1, 8, 8), 1)
    logits = torch.zeros(1, 3, 8, 8)
    logits[:, 1] = 10
    cm = confusion_matrix_from_logits(logits, labels, valid)
    assert cm.sum() == 32


def test_baseline_synthetic():
    from ml import baseline

    rng = np.random.default_rng(0)
    water = np.full((64, 64), -22.0) + rng.normal(0, 1, (64, 64))
    land = np.full((64, 64), -9.0) + rng.normal(0, 1, (64, 64)) * 0.5

    pre = land.copy()
    post = land.copy()
    post[:, :32] = water[:, :32]  # moitié de la scène inondée dans l'intervalle
    mask = baseline.water_mask_post(pre, post)
    assert mask.shape == (64, 64)
    flood_ratio = (mask == 2).mean()
    assert 0.35 < flood_ratio < 0.55, flood_ratio
    assert (mask == 1).mean() == 0.0  # pas d'eau permanente déclarée


def test_dataset_and_train_smoke(tmp_path):
    from scripts.make_synthetic_dataset import main as make_data
    from ml.train import _train_tiny  # helper défini pour les tests

    data_dir = tmp_path / "ds"
    make_data(data_dir, n_events=3, tiles_per_event=1, size=128)
    _train_tiny(data_dir, epochs=1)
