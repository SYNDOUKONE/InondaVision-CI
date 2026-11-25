# fmt: off
r"""Baseline non neuronale : seuillage radar + détection de changement.

Référence pour prouver que le Deep Learning apporte une amélioration réelle.
Méthode (classique en télédétection Sentinel-1, cf. protocole UN-SPIDER) :

1. L'imagerie VV (et VH) en dB est seuillée : l'eau présente un backscatter
   faible (< seuil, typ. -14 à -18 dB selon le site et la polarisation).
2. L'eau "temporaire" = eau post-événement & non-eau pré-événement & non-eau
   permanente (référence JRC Global Surface Water via masque fourni).
3. Option : seuillage adaptatif d'Otsu calculé par scène, plus robuste
   qu'un seuil fixe.

Entrées : GeoTIFF mono-canal en dB. Sortie : masque uint8 {0,1,2}.
"""
# fmt: on

import numpy as np


def otsu_threshold(dB: np.ndarray, nbins: int = 512) -> float:
    """Seuil d'Otsu sur l'histogramme des valeurs en dB."""
    finite = dB[np.isfinite(dB)]
    if finite.size == 0:
        return -15.0
    lo, hi = np.percentile(finite, [1, 99])
    hist, edges = np.histogram(finite, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2

    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    if (w0 <= 0).any() or (w1 <= 0).any():
        return float(center_lo := centers[len(centers) // 2])

    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    total = (hist * centers).sum()
    m1 = (total - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    return float(centers[int(np.argmax(between))])


def water_mask_pre(dB_vv: np.ndarray, threshold: float | None = None,
                   min_area_px: int = 3) -> np.ndarray:
    """Masque d'eau binaire (1 = eau) sur une scène pré-événement."""
    if threshold is None:
        threshold = otsu_threshold(dB_vv)
    water = (dB_vv < threshold).astype(np.uint8)
    return despeckle(water, min_area_px)


def water_mask_post(dB_vv_pre: np.ndarray, dB_vv_post: np.ndarray,
                    permanent_water: np.ndarray | None = None,
                    threshold: float | None = None,
                    min_area_px: int = 3) -> np.ndarray:
    """Masque 3 classes {0,1,2} post-événement par seuillage + changement.

    flood = eau détectée en post & non détectée en pré & non permanente.
    """
    if threshold is None:
        threshold = otsu_threshold(dB_vv_post)
    water_post = (dB_vv_post < threshold).astype(np.uint8)
    water_pre = (dB_vv_pre < threshold).astype(np.uint8)

    if permanent_water is None:
        permanent_water = np.zeros_like(water_post)

    flood = water_post & (1 - water_pre) & (1 - permanent_water.astype(np.uint8))
    flood = despeckle(flood, min_area_px)

    mask = np.zeros_like(water_post)
    mask[permanent_water > 0] = 1
    mask[flood > 0] = 2
    mask[dB_vv_post == -9999] = 255  # no_data -> ignoré dans les métriques
    return mask


def despeckle(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Supprime les taches de moins de `min_area_px` pixels (bruit speckle)."""
    if min_area_px <= 1:
        return mask
    try:
        from scipy.ndimage import label, sum_labels
    except ImportError:
        return mask

    labelled, n = label(mask)
    if n == 0:
        return mask
    sizes = sum_labels(np.ones_like(mask), labelled, index=np.arange(1, n + 1))
    small = np.isin(labelled, np.nonzero(sizes < min_area_px)[0] + 1)
    return np.where(small, 0, mask).astype(mask.dtype)
