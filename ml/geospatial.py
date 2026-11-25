"""Traitement géospatial : vectorisation du masque, superficies, changement.

Entrée : masque 3 classes {0 land, 1 permanent_water, 2 flood_water}
        + transform affine (rasterio) du GeoTIFF source.

Sorties : GeoJSON des polygones, statistiques d'aire (km² / ha) et
        carte d'évolution entre deux dates (gains / pertes d'eau).
"""

import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.ops import unary_union

CLASS_NAMES = {1: "permanent_water", 2: "flood_water"}


def pixel_area_m2(transform) -> float:
    """Aire d'un pixel en m² à partir de l'affine rasterio ((a, b, d, e, ...))."""
    a, d = transform.a, transform.e
    return abs(a * d)


def make_valid(geom):
    """Buffer(0) pour réparer les géométries invalides issues de la vectorisation."""
    return geom if geom.is_valid else geom.buffer(0)


def mask_to_polygons(mask: np.ndarray, transform, target_class: int = 2,
                     min_area_m2: float = 500.0) -> list[dict]:
    """Vectorise la classe `target_class` en polygones GeoJSONfiltrés par aire."""
    binary = (mask == target_class).astype(np.uint8)
    if binary.sum() == 0:
        return []

    shapes = rasterio.features.shapes(binary, mask=binary == 1, transform=transform)
    polys = []
    for geom, value in shapes:
        poly = make_valid(shape(geom))
        if not poly.is_empty and poly.area >= min_area_m2:
            polys.append(poly)

    merged = unary_union(polys)
    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    return [g.__geo_interface__ for g in geoms if g.area >= min_area_m2]


def area_stats(mask: np.ndarray, transform) -> dict:
    """Superficies par classe, en m², km² et hectares."""
    px = pixel_area_m2(transform)
    stats = {}
    for cls, name in CLASS_NAMES.items():
        n = int((mask == cls).sum())
        m2 = n * px
        stats[name] = {
            "pixels": n,
            "area_m2": m2,
            "area_km2": round(m2 / 1e6, 4),
            "area_hectares": round(m2 / 1e4, 2),
        }
    stats["pixel_size_m"] = round(abs(transform.a), 3)
    stats["confusion_note"] = (
        "superficie estimée par pixel de 10 m : précision limitée, "
        "voir §13 (limites scientifiques)"
    )
    return stats


def change_map(mask_pre: np.ndarray, mask_post: np.ndarray) -> dict:
    """Évolution entre deux acquisitions.

    Retourne (H, W) uint8 : 0 stable-terre, 1 nouveau flood, 2 eau disparue,
    3 eau persistante, plus des stats en % de pixels.
    """
    flood_pre = mask_pre == 2
    flood_post = mask_post == 2

    change = np.zeros_like(mask_pre, dtype=np.uint8)
    change[(~flood_pre) & flood_post] = 1     # nouvelle inondation
    change[flood_pre & (~flood_post)] = 2     # eau retirée
    change[flood_pre & flood_post] = 3        # flood persistant

    total = change.size
    return {
        "map": change,
        "stats": {
            "new_flood_pct": round(float((change == 1).sum() / total * 100), 3),
            "receded_pct": round(float((change == 2).sum() / total * 100), 3),
            "persistent_pct": round(float((change == 3).sum() / total * 100), 3),
        },
    }


def export_geotiff(mask: np.ndarray, path, transform, crs):
    import rasterio as rio

    with rio.open(
        path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", transform=transform, crs=crs,
    ) as dst:
        dst.write(mask, 1)
