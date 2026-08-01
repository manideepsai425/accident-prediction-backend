from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.deps import get_predictor, get_route_engine
from app.ml.predictor import RiskPredictor
from app.routing.route_engine import RouteEngine, RouteNotFoundError
from app.schemas import RiskPredictionRequest, RiskPredictionResponse, RouteRequest, RouteResponse

logger = logging.getLogger("accident_api.routers.predict")
router = APIRouter(prefix="/predict", tags=["predict"])


def _check_bounds(lat: float, lon: float, label: str) -> None:
    if not (settings.lat_min <= lat <= settings.lat_max and settings.lon_min <= lon <= settings.lon_max):
        raise HTTPException(
            status_code=400,
            detail=f"{label} ({lat}, {lon}) is outside the supported Peddapalli district area "
                   f"(lat {settings.lat_min}-{settings.lat_max}, lon {settings.lon_min}-{settings.lon_max}).",
        )


@router.post("/risk", response_model=RiskPredictionResponse)
def predict_risk(
    body: RiskPredictionRequest,
    predictor: RiskPredictor = Depends(get_predictor),
) -> RiskPredictionResponse:
    _check_bounds(body.latitude, body.longitude, "Point")
    result = predictor.predict_point(
        latitude=body.latitude,
        longitude=body.longitude,
        time_of_day=body.time_of_day,
        weather_condition=body.weather_condition,
        traffic_density=body.traffic_density,
        is_peak_hour=body.is_peak_hour,
    )
    return RiskPredictionResponse(**result)


@router.post("/route", response_model=RouteResponse)
def predict_route(
    body: RouteRequest,
    route_engine: RouteEngine = Depends(get_route_engine),
) -> RouteResponse:
    _check_bounds(body.origin_lat, body.origin_lng, "Origin")
    _check_bounds(body.dest_lat, body.dest_lng, "Destination")
    try:
        result = route_engine.find_routes(
            origin_lat=body.origin_lat, origin_lng=body.origin_lng,
            dest_lat=body.dest_lat, dest_lng=body.dest_lng,
            preferred_time=body.preferred_time, weather_condition=body.weather_condition,
            max_alternatives=body.max_alternatives,
        )
    except RouteNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        logger.exception("Unexpected error computing route")
        raise HTTPException(status_code=500, detail="Internal error computing route.") from None
    return RouteResponse(**result)
