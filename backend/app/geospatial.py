"""Analyse d'exposition : superpose le masque d'inondation aux infrastructures.

Entrées (optionnelles, GeoJSON WGS84) : buildings (Polygons), roads (LineStrings).
Sortie : nb bâtiments dans la zone inondée, longueur de routes exposées (km),
identifiants des routes touchées. Il s'agit d'une EXPOSITION GÉOGRAPHIQUE
estimée, pas de dommages confirmés (cf. §6 et §13 du projet).
"""

import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.ops import unary_union


def flood_polygons(mask: np.ndarray, transform):
    """Polygones (shapely, CRS du raster) des pixels flood (classe 2)."""
    binary = (mask == 2).astype(np.uint8)
    if binary.sum() == 0:
        return None, 0
    shapes = rasterio.features.shapes(binary, mask=binary == 1, transform=transform)
    polys = [shape(g) for g, _ in shapes]
    merged = unary_union(polys)
    merged = merged if merged.is_valid else merged.buffer(0)
    count = 1 if merged.geom_type == "Polygon" else len(merged.geoms)
    return merged, count


def exposure_assessment(mask, transform, crs, buildings_geojson=None,
                        roads_geojson=None) -> dict:
    """Évalue l'exposition des bâtiments et routes au masque d'inondation.

    Hypothèse : le raster est en CRS métrique (UTM ⇒ unités = mètres) et les
    GeoJSON d'entrée sont dans le même CRS, ou les distances sont comparées
    telles quelles. Sinon, reprojection à faire en amont (note dans `caveats`).
    """
    flood, n_polys = flood_polygons(mask, transform)
    if flood is None:
        return {
            "flood_polys": 0,
            "buildings_exposed": 0,
            "buildings_total": 0,
            "roads_exposed_km": 0.0,
            "roads_total_km": 0.0,
            "note": "aucune zone inondée détectée dans ce masque",
        }

    buildings = (buildings_geojson or {}).get("features", [])
    roads = (roads_geojson or {}).get("features", [])

    exposed_buildings = [
        f.get("id") or f.get("properties", {}).get("id", i)
        for i, f in enumerate(buildings)
        if flood.intersects(shape(f["geometry"]))
    ]

    roads_exposed_km = 0.0
    roads_total_km = 0.0
    exposed_road_ids = []
    for i, f in enumerate(roads):
        geom = shape(f["geometry"])
        roads_total_km += geom.length
        if flood.intersects(geom):
            roads_exposed_km += flood.intersection(geom).length
            exposed_road_ids.append(f.get("id") or f.get("properties", {}).get("id", i))

    return {
        "flood_polys": n_polys,
        "buildings_exposed": len(exposed_buildings),
        "buildings_total": len(buildings),
        "exposed_building_ids": exposed_buildings,
        "roads_exposed_km": round(roads_exposed_km / 1000, 2),
        "roads_total_km": round(roads_total_km / 1000, 2),
        "exposed_road_ids": exposed_road_ids,
        "pixel_size_m": round(abs(transform.a), 3),
        "crs": str(crs),
        "note": (
            "Exposition GÉOGRAPHIQUE estimée (superposition satellitaire), "
            "pas de dommages confirmés ni d'accessibilité routière garantie."
        ),
    }
