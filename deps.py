"""
app/deps.py

Thin FastAPI `Depends()` wrappers around the singletons created once in
main.py's lifespan (data store, road network, predictor, route engine).
Keeping these in one place means routers never touch `request.app.state`
directly -- they just declare what they need.
"""

from __future__ import annotations

from fastapi import Request

from app.ml.predictor import RiskPredictor
from app.routing.graph_builder import RoadNetwork
from app.routing.route_engine import RouteEngine
from app.services.data_store import DataStore


def get_data_store(request: Request) -> DataStore:
    return request.app.state.data_store


def get_road_network(request: Request) -> RoadNetwork:
    return request.app.state.road_network


def get_predictor(request: Request) -> RiskPredictor:
    return request.app.state.predictor


def get_route_engine(request: Request) -> RouteEngine:
    return request.app.state.route_engine
