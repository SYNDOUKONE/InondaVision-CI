"""Configuration du backend (variables d'environnement, valeurs par défaut)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("INONDA_DATA_DIR", BASE_DIR / "data" / "processed"))
CHECKPOINTS_DIR = Path(os.getenv("INONDA_CHECKPOINTS", BASE_DIR / "models" / "checkpoints"))
RESULTS_DIR = Path(os.getenv("INONDA_RESULTS", BASE_DIR / "evaluation" / "segmentation_results"))

DEFAULT_MODEL = os.getenv("INONDA_MODEL", "unet")
DEFAULT_CKPT = os.getenv("INONDA_CKPT", str(CHECKPOINTS_DIR / "unet" / "best.pt"))

DEV = os.getenv("INONDA_DEV", "true").lower() in ("1", "true", "yes")

class_to_rgb = {
    0: (120, 120, 120),   # background - gris
    1: (0, 102, 204),     # permanent_water - bleu
    2: (255, 80, 80),     # flood_water - rouge
}
