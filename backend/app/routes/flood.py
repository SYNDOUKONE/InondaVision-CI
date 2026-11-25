"""Routes API d'InondaVision CI.

Endpoints :
    GET  /api/v1/scenarios       — scènes disponibles (démo)
    POST /api/v1/detect          — détection d'inondation (masque + GeoJSON + aires)
    POST /api/v1/change          — évolution entre deux acquisitions
    GET  /api/v1/exposure        — bâtiments/routes exposés (fichier scenes local)
"""

import json

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.geospatial import exposure_assessment
from app.inference import FloodDetectionService
from ml.geospatial import area_stats, change_map, export_geotiff

router = APIRouter(prefix="/api/v1", tags=["flood"])

service: FloodDetectionService | None = None


def get_service() -> FloodDetectionService:
    global service
    if service is None:
        from app.config import DEFAULT_CKPT

        service = FloodDetectionService(DEFAULT_CKPT)
    return service


@router.get("/health")
def health():
    return {"status": "ok", "model_ready": get_service().ready}


@router.post("/detect")
async def detect(
    pre: UploadFile = File(..., description="GeoTIFF S1 pré-événement (VV/VH dB)"),
    post: UploadFile = File(..., description="GeoTIFF S1 post-événement (VV/VH dB)"),
    min_area_m2: float = Query(500, ge=0),
):
    """Détecte les zones inondées et renvoie masque + polygones GeoJSON + superficies."""
    import tempfile
    from pathlib import Path

    from ml.geospatial import mask_to_polygons

    svc = get_service()
    with tempfile.TemporaryDirectory() as tmp:
        pre_path = Path(tmp) / "pre.tif"
        post_path = Path(tmp) / "post.tif"
        pre_path.write_bytes(await pre.read())
        post_path.write_bytes(await post.read())

        try:
            result = svc.detect(pre_path, post_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Traitement impossible : {exc}")

    post_profile = result["post_profile"]
    mask = result["mask"]
    transform, crs = post_profile["transform"], post_profile["crs"]

    return {
        "method": result["method"],
        "mask_shape": list(mask.shape),
        "areas": area_stats(mask, transform),
        "flood_geojson": {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": g, "properties": {"class": "flood_water"},
                 "id": i}
                for i, g in enumerate(
                    mask_to_polygons(mask, transform, target_class=2, min_area_m2=min_area_m2)
                )
            ],
        },
        "permanent_geojson": {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": g, "properties": {"class": "permanent_water"},
                 "id": i}
                for i, g in enumerate(
                    mask_to_polygons(mask, transform, target_class=1, min_area_m2=min_area_m2)
                )
            ],
        },
        "crs": str(crs),
    }


@router.post("/change")
async def change_analysis(
    mask_pre: UploadFile = File(..., description="GeoTIFF masque pré (uint8 0-2, 255=nodata)"),
    mask_post: UploadFile = File(..., description="GeoTIFF masque post (uint8 0-2, 255=nodata)"),
):
    """Évolution de l'étendue d'eau entre deux dates (nouveau / retiré / persistant)."""
    import tempfile
    from pathlib import Path

    import numpy as np
    import rasterio as rio

    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = Path(tmp) / "pre.tif", Path(tmp) / "post.tif"
        p1.write_bytes(await mask_pre.read())
        p2.write_bytes(await mask_post.read())

        with rio.open(p1) as s1:
            m1 = s1.read(1)
            transform, crs = s1.transform, s1.crs
        with rio.open(p2) as s2:
            m2 = s2.read(1)

        if m1.shape != m2.shape:
            raise HTTPException(status_code=422, detail="masques de tailles différentes")

        result = change_map(m1, m2)
        out_path = Path(tmp) / "change.tif"
        export_geotiff(result["map"], out_path, transform, crs)

    stats = result["stats"]
    stats["crs"] = str(crs)
    return {"change_map": stats, "classes":
            {"0": "stable_land", "1": "new_flood", "2": "water_receded",
             "3": "persistent_flood"}}


@router.post("/exposure")
async def exposure(
    mask: UploadFile = File(..., description="GeoTIFF masque flood (3 classes)"),
    buildings: UploadFile | None = File(None, description="GeoJSON bâtiments (WGS84)"),
    roads: UploadFile | None = File(None, description="GeoJSON routes (WGS84)"),
):
    """Bâtiments situés dans la zone inondée + linéaire routier exposé."""
    import tempfile
    from pathlib import Path

    import rasterio as rio

    with tempfile.TemporaryDirectory() as tmp:
        mask_path = Path(tmp) / "mask.tif"
        mask_path.write_bytes(await mask.read())
        with rio.open(mask_path) as src:
            m = src.read(1)
            transform, crs = src.transform, src.crs

        buildings = json.loads(await buildings.read()) if buildings else None
        roads = json.loads(await roads.read()) if roads else None

    return exposure_assessment(m, transform, crs, buildings, roads)
