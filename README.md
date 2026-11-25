# InondaVision CI

**Détection d'inondations à Abidjan par Deep Learning sur images Sentinel-1.**

Le projet analyse des paires d'images radar Sentinel-1 (pré/post événement pluvieux),
détecte les surfaces recouvertes d'eau, sépare l'eau permanente de l'eau d'inondation
et produit un masque cartographique + superficies estimées + zones potentiellement
exposées (bâtiments, routes).

> ⚠️ **Lecture des résultats** — outil de cartographie et d'aide à la décision, avec
> validation humaine : ce n'est **pas** un système d'alerte d'urgence et il ne conclut
> pas qu'une route est praticable ou qu'un bâtiment est détruit. Les pixels de 10 m
> ne permettent pas de mesurer la profondeur d'eau ou les dégâts à l'échelle d'une
> habitation. Voir « Limites » plus bas.

## Architecture

```
Sources satellite            Moteur IA                    Exposition & API
Copernicus Sentinel-1   →    U-Net / DeepLabV3+ /    →    FastAPI (detect, change,
pré/post (VV/VH dB)          SegFormer (+ baseline        exposure, change), mappe
                             seuillage Otsu)              Leaflet, PostGIS prévu
```

| Composant         | Techno                                     |
| ----------------- | ------------------------------------------ |
| Deep Learning     | PyTorch — U-Net, DeepLabV3+, SegFormer     |
| Raster / vecteur  | Rasterio, Shapely (GDAL), PostGIS ( prévu) |
| API               | FastAPI + Uvicorn                          |
| Carto             | React + Tailwind + Leaflet, Docker         |

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Données synthétiques de démonstration (Dataset 0, sans téléchargement)
python scripts/make_synthetic_dataset.py --out data/processed/synthetic

# 2. Entraîner le U-Net (CPU ok sur données synthétiques)
python -m ml.train --data data/processed/synthetic --model unet --epochs 20

# 3. Évaluer sur le split test (événements jamais vus) + baseline seuillage
python -m ml.evaluate --data data/processed/synthetic --ckpt models/checkpoints/unet/best.pt

# 4. Lancer l'API + carte de démonstration
uvicorn app.main:app --app-dir backend --port 8000
# → http://localhost:8000 (carte) — http://localhost:8000/docs (API)
```

### Docker

```bash
docker compose up --build backend   # API sur :8000
docker compose up db                # PostGIS sur :5432
```

## Dashboard (React + Tailwind)

Interface web en 4 vues (au-dessus de l'API) :

| Vue | Rôle |
| --- | ---- |
| 🛰 Satellite Explorer | scènes serveur (pré/post), aperçu VV/VH superposé à la carte |
| 🤖 AI Detection | lancement de la détection (`scene_id` ou uploads), masques flood + eau permanente |
| 🌊 Flood Map | carte Leaflet filtrable, export GeoJSON des polygones |
| 📊 Analytics | superficies avant/après, évolution Δ, répartition des classes |

```bash
# développement (proxy /api -> :8000)
cd frontend && npm install && npm run dev

# production : build puis servi par FastAPI ou nginx
cd frontend && npm run build          # -> frontend/dist
uvicorn app.main:app --app-dir backend --port 8000   # sert dist auto
docker compose up --build backend frontend           # nginx sur :8080 (proxy /api)
```

Le backend ajoute des endpoints pour le dashboard :
`GET /api/v1/scenarios`, `GET /api/v1/scenes/{id}/preview.png`,
`POST /api/v1/detect?scene_id=...`.

## API (extraits)

| Endpoint                | Rôle                                                        |
| ----------------------- | ----------------------------------------------------------- |
| `POST /api/v1/detect`   | 2 GeoTIFF VV/VH (pré/post) → masque 3 classes, GeoJSON, hectares |
| `POST /api/v1/change`   | 2 masques → évolution (nouveau / retiré / persistant)        |
| `POST /api/v1/exposure` | masque + bâtiments/routes GeoJSON, ou OSM via `?osm=true&bbox=minlon,minlat,maxlon,maxlat` |
| `GET  /api/v1/scenarios` | scènes pré/post disponibles côté serveur                    |
| `GET  /api/v1/scenes/{id}/preview.png` | aperçu PNG d'une scène (bandes VV/VH)           |
| `GET  /api/v1/health`   | état du service                                             |

En l'absence de checkpoint entraîné, le backend bascule automatiquement sur la
**baseline seuillage Otsu + détection de changement** (référence non neuronale
obligatoire pour démontrer l'apport du Deep Learning).

## Astuce Sentinel-1 (3 classes)

```
0 background        terre non inondée
1 permanent_water   lagunes / cours d'eau permanents (réf. JRC GSW)
2 flood_water       eau temporaire = eau post − eau permanente
255 no_data         pixels ignorés dans la perte et les métriques
```

## Données

| Source | Rôle |
| ------ | ---- |
| [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | benchmark d'entraînement (4 831 tuiles, 11 crues). Vérifier la licence avant redistribution |
| [WorldFloods V2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2) | complément Sentinel-2, CC BY-NC 4.0 (non commercial) |
| Copernicus Data Space | scènes Sentinel-1 locales (juin 2023 / juin 2024, Abidjan) |
| JRC Global Surface Water | masque d'eau permanente |
| Rapports ONPC CI | événements documentés (13-14 juin 2024, etc.) |

## Tests

```bash
pytest tests/ -v
```

## Limites (section 13 du projet)

- Réponses radar complexes sur bâti dense (double rebond).
- Sols humides confondus avec des surfaces inondées.
- Eau permanente vs temporaire : nécessite une référence externe (JRC GSW).
- Résolution 10 m : pas de mesure au niveau d'une habitation.
- Fréquence de revisite : pas d'observation immédiate après chaque pluie.
- Superposition géographique = **exposition estimée**, pas dégâts confirmés.

## Feuille de route (8 semaines)

1. Cadrage + exploration — 2. Préparation Sentinel-1 — 3. Masques/annotations —
4. U-Net — 5. Benchmark DeepLabV3+/SegFormer — 6. Validation événements ivoiriens —
7. API + carto — 8. MVP documentation/démo.

## Licence

Code : à définir (MIT recommandé). Données : respecter les licences des datasets
utilisés (voir tableau ci-dessus).
