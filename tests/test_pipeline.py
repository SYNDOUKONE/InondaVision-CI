"""Tests de fumée du pipeline InondaVision (rapides, sans GPU).

    pytest tests/ -v
"""

import json
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


def test_osm_parse_overpass_elements():
    from backend.app.osm import parse_overpass_elements

    elements = [
        {"type": "way", "id": 1, "tags": {"building": "yes"},
         "geometry": [{"lat": 5.36, "lon": -3.98}, {"lat": 5.361, "lon": -3.98},
                      {"lat": 5.361, "lon": -3.979}, {"lat": 5.36, "lon": -3.979},
                      {"lat": 5.36, "lon": -3.98}]},
        {"type": "way", "id": 2, "tags": {"highway": "residential"},
         "geometry": [{"lat": 5.36, "lon": -3.98}, {"lat": 5.365, "lon": -3.978}]},
        {"type": "node", "id": 3, "tags": {}, "geometry": [{"lat": 5.36, "lon": -3.98}]},
    ]
    buildings, roads = parse_overpass_elements(elements)
    assert len(buildings) == 1 and len(roads) == 1
    assert buildings[0]["geometry"]["type"] == "Polygon"
    assert roads[0]["geometry"]["type"] == "LineString"


def _abidjan_utm() -> tuple[float, float]:
    """Coordonnées UTM 30N du centre de Cocody (Abidjan)."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:32630",
                                always_xy=True).transform(-3.98, 5.36)


def test_exposure_with_wgs84_features_over_utm_mask():
    from rasterio.transform import from_origin

    from backend.app.geospatial import exposure_assessment

    x0, y0 = _abidjan_utm()
    transform = from_origin(x0 + 1280, y0 + 1280, 10, 10)  # tuile 1280 m x 1280 m
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:150, 100:160] = 2  # ~500 m x 600 m inondés

    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)

    # centre du bloc inondé (pixels 100:150, 100:160) en WGS84
    cx = x0 + 1280 + 130 * 10
    cy = y0 + 1280 - 125 * 10
    clon, clat = to_wgs84.transform(cx, cy)

    def ring(lon0_, lat0_, lon1_, lat1_):
        return [[[lon0_, lat0_], [lon1_, lat0_], [lon1_, lat1_], [lon0_, lat1_],
                 [lon0_, lat0_]]]

    buildings = {"type": "FeatureCollection", "features": [
        {"id": "B1", "geometry": {"type": "Polygon",
                                 "coordinates": ring(clon - 0.001, clat - 0.001,
                                                      clon + 0.001, clat + 0.001)}},
    ]}
    roads = {"type": "FeatureCollection", "features": [
        {"id": "R1", "geometry": {"type": "LineString", "coordinates": [
            [clon - 0.005, clat - 0.005], [clon + 0.005, clat + 0.005]]}},
    ]}

    res = exposure_assessment(mask, transform, "EPSG:32630", buildings, roads)
    assert res["buildings_exposed"] == 1
    assert res["roads_exposed_km"] > 0.1
    assert res["crs"] == "EPSG:32630"


def test_flood_bbox_wgs84():
    from rasterio.transform import from_origin

    from backend.app.geospatial import flood_bbox_wgs84

    mask = np.zeros((256, 256), dtype=np.uint8)
    x0, y0 = _abidjan_utm()
    transform = from_origin(x0, y0, 10, 10)
    bbox = flood_bbox_wgs84(mask, transform, "EPSG:32630")

    assert bbox[0] < -3.98 < bbox[2]
    assert bbox[1] < 5.36 < bbox[3]


def test_api_dashboard_endpoints(tmp_path):
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "backend"))

    from scripts.make_synthetic_dataset import main as make_data

    make_data(tmp_path / "ds", n_events=2, tiles_per_event=1, size=128)

    import app.config as cfg
    from fastapi.testclient import TestClient
    from app.main import app

    cfg.DEFAULT_CKPT = str(REPO_ROOT / "models" / "checkpoints" / "unet" / "best.pt")
    client = TestClient(app)

    r = client.get("/api/v1/scenarios")
    assert r.status_code == 200 and r.json()["count"] >= 2
    sid = r.json()["scenes"][0]["scene_id"]
    scene = next(s for s in r.json()["scenes"] if s["scene_id"] == sid)
    assert scene["has_pre"] and scene["has_label"]

    r = client.get(f"/api/v1/scenes/{sid}/preview.png?band=1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/png")
    bounds = json.loads(r.headers["X-Scene-Bounds"])
    assert bounds["minlon"] < bounds["maxlon"]

    r = client.post(f"/api/v1/detect?scene_id={sid}&min_area_m2=100")
    assert r.status_code == 200
    d = r.json()
    assert {"background", "permanent_water", "flood_water"} <= set(d["areas"])
    assert "areas_pre" in d

    r = client.post("/api/v1/detect?min_area_m2=100")
    assert r.status_code == 422
