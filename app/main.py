"""
app/main.py

Application entrypoint. Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   (dev)
    uvicorn app.main:app --host 0.0.0.0 --port $PORT           (Render)

STARTUP SEQUENCE (see `lifespan` below)
----------------------------------------
1. If model.pkl is missing and AUTO_TRAIN_IF_MISSING=true (default), train
   it from data/accidents_data.csv right now. This is what makes "clone
   the repo and run it" work even before anyone has trained anything on
   Colab -- and it's also the safety net if a Colab-trained model.pkl ever
   gets deleted.
2. Load accidents_data.csv, road_segments.geojson and places.json into
   memory once (DataStore).
3. Build the routable road graph (RoadNetwork) from the GeoJSON.
4. Load model.pkl into a RiskPredictor (falls back to a transparent
   heuristic if, for some reason, there is still no model).
5. Wire the RiskPredictor + RoadNetwork into a RouteEngine.
All four objects are attached to `app.state` so every router can reach
them through `app/deps.py` without any global mutable state.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make sure the project root (parent of this `app/` package) is importable
# regardless of the working directory uvicorn was launched from, so
# `import train_model` below always resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import __version__
from app.config import settings
from app.ml.predictor import RiskPredictor
from app.routing.graph_builder import RoadNetwork
from app.routing.route_engine import RouteEngine
from app.services.data_store import DataStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("accident_api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    model_path = settings.path(settings.model_path)

    if not model_path.exists() and settings.auto_train_if_missing:
        logger.info("model.pkl not found at %s -- training from CSV now...", model_path)
        try:
            from train_model import train as train_model_fn
            train_model_fn(settings.path(settings.accidents_csv))
        except Exception:
            logger.exception("Automatic training failed -- API will fall back to a heuristic risk model.")

    data_store = DataStore(
        accidents_csv=settings.path(settings.accidents_csv),
        road_segments_geojson=settings.path(settings.road_segments_geojson),
        places_json=settings.path(settings.places_json),
    )
    road_network = RoadNetwork(data_store.geojson)
    predictor = RiskPredictor(model_path, road_network)
    route_engine = RouteEngine(road_network, predictor)

    app.state.data_store = data_store
    app.state.road_network = road_network
    app.state.predictor = predictor
    app.state.route_engine = route_engine

    logger.info("Startup complete in %.2fs (model_loaded=%s)", time.time() - start, predictor.is_loaded)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Peddapalli Accident Risk & Safest-Route API",
    description=(
        "Predicts road-accident risk in Peddapalli district (Telangana, India) "
        "and computes the safest route between two points using a scikit-learn "
        "model trained on synthetic-but-realistic historical accident data."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


from app.routers import analytics, health, hotspots, places, predict  # noqa: E402

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(hotspots.router)
app.include_router(analytics.router)
app.include_router(places.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "name": "Peddapalli Accident Risk & Safest-Route API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/predict/risk", "/predict/route", "/hotspots", "/analytics", "/places"],
    }
