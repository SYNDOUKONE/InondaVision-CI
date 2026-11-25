"""Analyse d'exposition : superpose le masque d'inondation aux infrastructures.

Entrées (optionnelles, GeoJSON WGS84) : buildings (Polygons), roads (LineStrings).
Sortie : nb bâtiments dans la zone inondée, longueur de routes exposées (km),
identifiants des routes touchées. Il s'agit d'une EXPOSITION GÉOGRAPHIQUE
estimée, pas de dommages confirmés (cf. §6 et §13 du projet).
"""

import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

WGS84 = "EPSG:4326"


def to_raster_crs(geom, raster_crs, input_crs=WGS84):
    """Reprojette une géométrie shapely du CRS d'entrée vers le CRS du raster."""
    from pyproj import CRS, Transformer

    src, dst = CRS.from_user_input(input_crs), CRS.from_user_input(raster_crs)
    if src == dst:
        return geom
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return shp_transform(lambda x, y: transformer.transform(x, y), geom)


def to_wgs84(geom, raster_crs):
    """Reprojette une géométrie du CRS du raster vers WGS84."""
    return to_raster_crs(geom, WGS84, input_crs=raster_crs)


def flood_bbox_wgs84(mask, transform, raster_crs, pad_fraction: float = 0.02) -> tuple:
    """Bbox WGS84 contenant le raster (marge 2 % par défaut) pour interroger Overpass.

    On prend l'emprise entière du masque (et non la seule zone inondée) :
    simple et robuste, et les features potentiellement exposées autour de la
    zone inondée restent utiles pour l'analyse de changement.
    """
    import rasterio.warp
    from rasterio.transform import array_bounds

    left, bottom, right, top = array_bounds(mask.shape[0], mask.shape[1], transform)
    minlon, minlat, maxlon, maxlat = rasterio.warp.transform_bounds(
        raster_crs, WGS84, left, bottom, right, top)
    pad_x = (maxlon - minlon) * pad_fraction
    pad_y = (maxlat - minlat) * pad_fraction
    return (minlon - pad_x, minlat - pad_y, maxlon + pad_x, maxlat + pad_y)


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
                        roads_geojson=None, input_crs=WGS84) -> dict:
    """Évalue l'exposition des bâtiments et routes au masque d'inondation.

    Le raster doit être en CRS métrique (UTM ⇒ unités = mètres) pour que les
    aires et longueurs soient correctes. Les GeoJSON d'entrée sont en WGS84
    par défaut (OpenStreetMap) et reprojetés vers le CRS du raster ; passez
    `input_crs=crs` s'ils sont déjà dans ce CRS.
    """
    import rasterio.crs

    raster_crs = rasterio.crs.CRS.from_user_input(crs)

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
        if flood.intersects(to_raster_crs(shape(f["geometry"]), raster_crs,
                                          input_crs))
    ]

    roads_exposed_km = 0.0
    roads_total_km = 0.0
    exposed_road_ids = []
    for i, f in enumerate(roads):
        geom = to_raster_crs(shape(f["geometry"]), raster_crs, input_crs)
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
        "crs": str(raster_crs),
        "note": (
            "Exposition GÉOGRAPHIQUE estimée (superposition satellitaire), "
            "pas de dommages confirmés ni d'accessibilité routière garantie."
        ),
    }
