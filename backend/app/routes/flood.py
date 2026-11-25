"""Routes API d'InondaVision CI.

Endpoints :
    GET  /api/v1/health                  — état du service
    GET  /api/v1/scenarios               — scènes pré/post disponibles (serveur)
    GET  /api/v1/scenes/{id}/preview.png — aperçu image d'une scène (bande VV ou VH)
    POST /api/v1/detect                  — détection d'inondation (masque + GeoJSON + aires)
    POST /api/v1/change                  — évolution entre deux masques
    POST /api/v1/exposure                — bâtiments/routes exposés (upload ou OSM)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.geospatial import exposure_assessment
from app.inference import FloodDetectionService
from ml.geospatial import area_stats, change_map, export_geotiff, mask_to_polygons

router = APIRouter(prefix="/api/v1", tags=["flood"])

service: FloodDetectionService | None = None
PRE_SUFFIX = "_pre_cis.tif"
POST_SUFFIX = "_post_cis.tif"


def get_service() -> FloodDetectionService:
    global service
    if service is None:
        from app.config import DEFAULT_CKPT

        service = FloodDetectionService(DEFAULT_CKPT)
    return service


def _scene_paths(scene_id: str) -> tuple[Path, Path]:
    from app.config import DATA_DIR

    for post in DATA_DIR.glob(f"**/*{POST_SUFFIX}"):
        if post.name.replace(POST_SUFFIX, "") == scene_id:
            pre = post.parent / f"{scene_id}{PRE_SUFFIX}"
            return pre, post
    raise HTTPException(status_code=404, detail=f"scène inconnue : {scene_id}")


@router.get("/health")
def health():
    return {"status": "ok", "model_ready": get_service().ready}


@router.get("/scenarios")
def scenarios():
    """Liste des scènes disponibles côté serveur, pour le Satellite Explorer.

    Recherche récursive dans DATA_DIR (s1/, processed/synthetic/s1/, etc.) ;
    le champ dataset indique le sous-dossier racine du fichier trouvé.
    """
    from app.config import DATA_DIR

    scenes = []
    for post in sorted(DATA_DIR.glob(f"**/*{POST_SUFFIX}")):
        scene_id = post.name.replace(POST_SUFFIX, "")
        event = scene_id.rsplit("_", 1)[0]
        s1_dir = post.parent
        scenes.append({
            "scene_id": scene_id,
            "event": event,
            "dataset": str(post.parent.relative_to(DATA_DIR).parts[0])
                       if post.parent != DATA_DIR else ".",
            "has_pre": (s1_dir / f"{scene_id}{PRE_SUFFIX}").exists(),
            "has_label": (post.parent.parent / "labels" / f"{scene_id}_label.tif").exists(),
            "post_size_mb": round(post.stat().st_size / 1e6, 1),
        })
    return {"count": len(scenes), "scenes": scenes}


@router.get("/scenes/{scene_id}/preview.png")
def scene_preview(scene_id: str, band: int = Query(0, ge=0, le=1)):
    """Aperçu PNG (échelle de gris, dB normalisé) de la bande VV ou VH d'une scène."""
    from PIL import Image

    _, post_path = _scene_paths(scene_id)
    import rasterio as rio
    import rasterio.warp

    with rio.open(post_path) as src:
        arr = src.read(min(band + 1, src.count)).astype(np.float32)
        if arr.shape[0] > 1:
            arr = arr[0]
        wgs_left, wgs_bottom, wgs_right, wgs_top = rasterio.warp.transform_bounds(
            src.crs, "EPSG:4326", *src.bounds)

    lo, hi = np.nanpercentile(arr, [2, 98])
    norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    pil = Image.fromarray((norm * 255).astype(np.uint8))

    buf = tempfile.NamedTemporaryFile(suffix=".png")
    pil.save(buf.name)
    payload = Path(buf.name).read_bytes()
    Path(buf.name).unlink(missing_ok=True)

    return Response(payload, media_type="image/png",
                    headers={"X-Scene-Bounds": json.dumps({
                        "minlon": wgs_left, "minlat": wgs_bottom,
                        "maxlon": wgs_right, "maxlat": wgs_top})})


@router.post("/detect")
async def detect(
    pre: UploadFile | None = File(None, description="GeoTIFF S1 pré-événement (VV/VH dB)"),
    post: UploadFile | None = File(None, description="GeoTIFF S1 post-événement (VV/VH dB)"),
    scene_id: str | None = Query(None, description="scène serveur (alternative aux uploads)"),
    min_area_m2: float = Query(500, ge=0),
):
    """Détecte les zones inondées : masque, polygones GeoJSON et superficies.

    Mode dashboard : scene_id → paires pré/post du serveur. Sinon : uploads.
    """
    if scene_id:
        pre_path, post_path = _scene_paths(scene_id)
        if not pre_path.exists():
            raise HTTPException(422, f"pré-événement manquant pour {scene_id}")
    else:
        if pre is None or post is None:
            raise HTTPException(422, "fournir les fichiers pre et post, ou scene_id")
        tmp = tempfile.mkdtemp()
        pre_path = Path(tmp) / "pre.tif"
        post_path = Path(tmp) / "post.tif"
        pre_path.write_bytes(await pre.read())
        post_path.write_bytes(await post.read())

    try:
        result = get_service().detect(pre_path, post_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Traitement impossible : {exc}")

    transform, crs = result["post_profile"]["transform"], result["post_profile"]["crs"]
    mask, mask_pre = result["mask"], result["mask_pre"]
    transform_pre = result["pre_profile"]["transform"]

    return {
        "method": result["method"],
        "mask_shape": list(mask.shape),
        "areas": area_stats(mask, transform),
        "areas_pre": area_stats(mask_pre, transform_pre),
        "flood_geojson": _feature_collection(mask, transform, 2, min_area_m2),
        "permanent_geojson": _feature_collection(mask, transform, 1, min_area_m2),
        "crs": str(crs),
    }


def _feature_collection(mask, transform, target_class: int, min_area_m2: float) -> dict:
    features = [
        {"type": "Feature", "geometry": g,
         "properties": {"class": "flood_water" if target_class == 2 else "permanent_water"},
         "id": i}
        for i, g in enumerate(mask_to_polygons(mask, transform,
                                               target_class=target_class,
                                               min_area_m2=min_area_m2))
    ]
    return {"type": "FeatureCollection", "features": features}


@router.post("/change")
async def change_analysis(
    mask_pre: UploadFile = File(..., description="GeoTIFF masque pré (uint8 0-2, 255=nodata)"),
    mask_post: UploadFile = File(..., description="GeoTIFF masque post (uint8 0-2, 255=nodata)"),
):
    """Évolution de l'étendue d'eau entre deux dates (nouveau / retiré / persistant)."""
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
        export_geotiff(result["map"], Path(tmp) / "change.tif", transform, crs)

    stats = result["stats"]
    stats["crs"] = str(crs)
    return {"change_map": stats,
            "classes": {"0": "stable_land", "1": "new_flood", "2": "water_receded",
                        "3": "persistent_flood"}}


@router.post("/exposure")
async def exposure(
    mask: UploadFile = File(..., description="GeoTIFF masque flood (3 classes)"),
    buildings: UploadFile | None = File(None, description="GeoJSON bâtiments (WGS84)"),
    roads: UploadFile | None = File(None, description="GeoJSON routes (WGS84)"),
    osm: bool = Query(False, description="Récupérer bâtiments/routes OpenStreetMap"),
    bbox: str | None = Query(
        None, description="minlon,minlat,maxlon,maxlat WGS84 — requis si osm=true "
                          "(par défaut : emprise du masque)"),
):
    """Bâtiments situés dans la zone inondée + linéaire routier exposé.

    Sources : fichiers GeoJSON uploadés, ou OpenStreetMap (Overpass) avec osm=true.
    """
    import rasterio as rio

    with tempfile.TemporaryDirectory() as tmp:
        mask_path = Path(tmp) / "mask.tif"
        mask_path.write_bytes(await mask.read())
        with rio.open(mask_path) as src:
            m = src.read(1)
            transform, crs = src.transform, src.crs

        buildings_fc = json.loads(await buildings.read()) if buildings else None
        roads_fc = json.loads(await roads.read()) if roads else None

    source = "upload"
    if osm:
        from app.config import OSM_CACHE_DIR, OSM_TIMEOUT
        from app.geospatial import flood_bbox_wgs84
        from app.osm import fetch_osm

        try:
            bbox_wgs84 = (
                tuple(float(v) for v in bbox.split(",")) if bbox
                else flood_bbox_wgs84(m, transform, crs)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            osm_data = fetch_osm(bbox_wgs84, timeout=OSM_TIMEOUT,
                                 cache_dir=OSM_CACHE_DIR)
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"Overpass indisponible : {exc}")

        if buildings_fc is None:
            buildings_fc = osm_data["buildings"]
        if roads_fc is None:
            roads_fc = osm_data["roads"]
        source = osm_data["source"]

    result = exposure_assessment(m, transform, crs, buildings_fc, roads_fc)
    result["osm_source"] = source if osm else None
    return result
