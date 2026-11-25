"""Point d'entrée FastAPI d'InondaVision CI.

Lancement :
    uvicorn app.main:app --reload --port 8000    (depuis backend/)
    docker compose up backend                    (à la racine)

La page d'accueil sert le dashboard React (frontend/dist).
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.routes import flood  # noqa: E402

app = FastAPI(
    title="InondaVision CI",
    description=(
        "Détection d'inondations à Abidjan par Deep Learning sur Sentinel-1. "
        "Classes : background / permanent_water / flood_water."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(flood.router)

DIST_DIR = REPO_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="dashboard")
else:
    @app.get("/")
    def index():
        return {
            "message": "Dashboard non compilé. Exécutez : cd frontend && npm install && npm run build",
            "api_docs": "/docs",
        }
