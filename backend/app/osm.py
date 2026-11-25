"""Récupération OpenStreetMap via Overpass API : bâtiments et routes.

Entrée : bbox WGS84 (minlon, minlat, maxlon, maxlat).
Sortie : deux FeatureCollections GeoJSON (WGS84) — buildings (Polygons) et
roads (LineStrings), filtrées sur les catégories routières utiles.

Les données OSM sont sous licence ODbL : mentionner la source
("© OpenStreetMap contributors") dans toute publication.

Le résultat est mis en cache sur disque (une requête par bbox arrondie à
3 décimales ≈ 110 m) pour éviter de marteler l'API Overpass publique.
"""

import hashlib
import json
import logging
from pathlib import Path

import httpx
from shapely.geometry import LineString, Polygon

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

ROAD_FILTER = (
    '["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential'
    "|unclassified|service|living_street|pedestrian)$\"]"
)


def build_query(bbox: tuple[float, float, float, float]) -> str:
    minlon, minlat, maxlon, maxlat = bbox
    bbox_q = f"{minlat},{minlon},{maxlat},{maxlon}"
    return (
        "[out:json][timeout:60];\n(\n"
        f'  way["building"]({bbox_q});\n'
        f"  {ROAD_FILTER}({bbox_q});\n"
        ");\nout body geom;"
    )


def parse_overpass_elements(elements: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convertis les éléments Overpass (out body geom) en features GeoJSON.

    Les ways fermées avec un tag `building` deviennent des Polygons ; les ways
    `highway` deviennent des LineStrings. Les relations sont ignorées (les
    bâtiments composés restent marginaux et l'API explose en complexité).
    """
    buildings, roads = [], []

    for el in elements:
        if el.get("type") != "way":
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
        if len(coords) < 2:
            continue

        tags = el.get("tags") or {}
        props = {"osm_id": el.get("id"), "tags": {k: v for k, v in tags.items()
                                                  if k in ("building", "highway", "name")}}

        if tags.get("building"):
            if len(coords) >= 4 and coords[0] == coords[-1]:
                buildings.append({"type": "Feature", "id": f"way/{el['id']}",
                                  "properties": props,
                                  "geometry": Polygon(coords).__geo_interface__})
        elif tags.get("highway"):
            roads.append({"type": "Feature", "id": f"way/{el['id']}",
                          "properties": props,
                          "geometry": LineString(coords).__geo_interface__})

    return buildings, roads


def _cache_path(bbox, cache_dir: Path) -> Path:
    key = hashlib.sha1(
        ",".join(f"{c:.3f}" for c in bbox).encode()).hexdigest()[:16]
    return cache_dir / f"osm_{key}.json"


def _default_client(timeout: int) -> httpx.Client:
    """Client httpx par défaut (honore HTTPS_PROXY/HTTP_PROXY si définis)."""
    return httpx.Client(timeout=timeout)


def fetch_osm(bbox: tuple[float, float, float, float], timeout: int = 90,
              cache_dir: str | Path | None = None,
              client: httpx.Client | None = None) -> dict:
    """Renvoie {'buildings': FeatureCollection, 'roads': FeatureCollection} (WGS84)."""
    minlon, minlat, maxlon, maxlat = bbox
    if not (-180 <= minlon <= 180 and -90 <= minlat <= 90
            and minlon < maxlon and minlat < maxlat):
        raise ValueError(f"bbox WGS84 invalide : {bbox}")

    cache_file = _cache_path(bbox, Path(cache_dir)) if cache_dir else None
    if cache_file and cache_file.exists():
        data = json.loads(cache_file.read_text())
        logger.info("OSM depuis le cache : %s", cache_file.name)
        return data

    query = build_query(bbox)
    own_client = client is None
    c = client or _default_client(timeout)
    try:
        resp = c.post(
            OVERPASS_URL, content=query,
            headers={"Content-Type": "text/plain;charset=utf-8"})
    finally:
        if own_client:
            c.close()
    resp.raise_for_status()
    payload = resp.json()

    buildings, roads = parse_overpass_elements(payload.get("elements", []))
    result = {
        "buildings": {"type": "FeatureCollection", "features": buildings},
        "roads": {"type": "FeatureCollection", "features": roads},
        "source": "© OpenStreetMap contributors (ODbL), via Overpass API",
        "bbox_wgs84": list(bbox),
    }

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))
    logger.info("OSM : %d bâtiments, %d routes", len(buildings), len(roads))
    return result
