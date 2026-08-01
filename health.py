from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.deps import get_data_store, get_predictor, get_road_network
from app.ml.predictor import RiskPredictor
from app.routing.graph_builder import RoadNetwork
from app.schemas import HealthResponse
from app.services.data_store import DataStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    data_store: DataStore = Depends(get_data_store),
    road_network: RoadNetwork = Depends(get_road_network),
    predictor: RiskPredictor = Depends(get_predictor),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_trained_at=predictor.trained_at,
        accidents_loaded=len(data_store.accidents_df),
        road_segments_loaded=len(road_network.segments),
        graph_nodes=road_network.graph.number_of_nodes(),
        graph_edges=road_network.graph.number_of_edges(),
        version=__version__,
    )
