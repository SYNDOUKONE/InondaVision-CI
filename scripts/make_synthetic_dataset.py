# fmt: off
"""Données synthétiques Sentinel-1 pour la démonstration et les tests.

Génère des scènes post-événement VV/VH (dB) + masques 3 classes, avec lagune
permanente, écoulements de crue, zones urbaines à forte réponse et bruit
speckle - fidèle à la physique du backscatter radar (eau sombre, ville
brillante). Sert de « Dataset 0 » avant l'obtention de vraies scènes
Copernicus et permet d'exécuter tout le pipeline (train + eval + API) sans GPU.

Usage : python scripts/make_synthetic_dataset.py --out data/processed/synthetic
"""

import argparse
import json
from pathlib import Path

import numpy as np


def write_raster(path: Path, arr: np.ndarray, dtype="float32", crs="EPSG:32630"):
    import rasterio as rio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = arr.shape[-2:]
    arr = arr if arr.ndim == 3 else arr[None]
    with rio.open(path, "w", driver="GTiff", height=H, width=W,
                  count=arr.shape[0], dtype=dtype, crs=crs,
                  transform=from_origin(430000, 630000, 10, 10)) as dst:
        dst.write(arr.astype(dtype))


def synth_scene(seed: int, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (image (2,H,W) en dB, label (H,W) uint8 0/1/2)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    label = np.zeros((size, size), dtype=np.uint8)

    # Lagune permanente (bas-gauche)
    cx, cy = size * 0.25, size * 0.75
    lagune = ((xx - cx) ** 2 + (yy - cy) ** 2) < (size * 0.18) ** 2

    # Cours d'eau serpentin de gauche à droite
    band = np.sin(xx / size * 6) * size * 0.05 + size * 0.35
    river = np.abs(yy - band) < size * 0.25 * (0.4 + 0.6 * (xx / size))

    # Surfaces inondées temporaires (blobs répartis en aval)
    blobs = np.zeros((size, size), dtype=bool)
    for _ in range(3):
        cxb, cyb = rng.uniform(0.45, 0.9, 2) * size
        r = rng.uniform(0.05, 0.12) * size
        blobs |= ((xx - cxb) ** 2 + (yy - cyb) ** 2) < r ** 2

    label[lagune | (river & blobs)] = 1
    label[river & ~lagune & ~blobs] = 2

    # Backscatter : eau sombre (-22 dB), terre autour de -9 dB, villes claires
    base_vv = np.where(label == 0, -9.0, -22.0)
    base_vv += rng.normal(0, 1.1, (size, size))
    base_vh = base_vv - rng.normal(12, 1.5, (size, size))

    urban = rng.random((size, size)) < 0.04
    urban &= label == 0
    base_vv[urban] = rng.normal(-4, 2, int(urban.sum()))
    base_vh[urban] = rng.normal(-16, 2, int(urban.sum()))

    return np.stack([base_vv, base_vh]).astype(np.float32), label


def main(out: Path, n_events: int = 3, tiles_per_event: int = 3, size: int = 512):
    scenes = []
    for ev in range(n_events):
        for k in range(tiles_per_event):
            scene_id = f"synthetic-ev{ev}_{k:02d}"
            img, lab = synth_scene(seed=ev * 100 + k, size=size)
            write_raster(out / "s1" / f"{scene_id}_post_cis.tif", img)
            write_raster(out / "labels" / f"{scene_id}_label.tif", lab[None], dtype="uint8")

            # scène pré-événement : l'eau d'inondation (classe 2) est encore de la terre
            pre = img.copy()
            flooded = lab == 2
            pre[0][flooded] = -9.0
            pre[1][flooded] = -21.0
            write_raster(out / "s1" / f"{scene_id}_pre_cis.tif", pre)
            scenes.append(scene_id)

    # Split PAR ÉVÉNEMENT : les tuiles d'un même événement restent dans le même split
    events = sorted({s.rsplit("_", 1)[0] for s in scenes})
    split = {"train": [], "val": [], "test": []}
    for i, ev in enumerate(events):
        key = "train" if i < n_events - 2 else ("val" if i == n_events - 2 else "test")
        split[key] += [s for s in scenes if s.rsplit("_", 1)[0] == ev]

    (out / "splits.json").write_text(json.dumps(split, indent=2))
    print(f"OK : {len(scenes)} scènes écrites dans {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed/synthetic")
    p.add_argument("--events", type=int, default=3)
    p.add_argument("--tiles-per-event", type=int, default=3)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    main(Path(a.out), a.events, a.tiles_per_event, a.size)
