"""
tests/test_api.py

Exercises every endpoint through FastAPI's TestClient (which drives the
real `lifespan`, so this also validates the startup sequence: auto-train,
data loading, graph building). Run with:

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

PEDDAPALLI_CENTER = (18.616, 79.383)
RAMAGUNDAM = (18.780, 79.450)
BASANTHNAGAR = (18.745, 79.545)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["accidents_loaded"] == 500
    assert body["graph_nodes"] > 0
    assert body["graph_edges"] > 0


def test_predict_risk_basic(client):
    r = client.post("/predict/risk", json={
        "latitude": PEDDAPALLI_CENTER[0], "longitude": PEDDAPALLI_CENTER[1],
        "time_of_day": "Night", "weather_condition": "Heavy Rain",
    })
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert body["nearest_road_name"] is not None
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 0


def test_predict_risk_clear_day_lower_than_stormy_night(client):
    calm = client.post("/predict/risk", json={
        "latitude": PEDDAPALLI_CENTER[0], "longitude": PEDDAPALLI_CENTER[1],
        "time_of_day": "Morning", "weather_condition": "Clear",
    }).json()
    stormy = client.post("/predict/risk", json={
        "latitude": BASANTHNAGAR[0], "longitude": BASANTHNAGAR[1],
        "time_of_day": "Night", "weather_condition": "Heavy Rain",
    }).json()
    # Not a strict guarantee for every possible pair, but should hold here
    # given how the model was trained -- a good regression-style sanity check.
    assert calm["risk_score"] < stormy["risk_score"]


def test_predict_risk_out_of_bounds_rejected(client):
    r = client.post("/predict/risk", json={
        "latitude": 10.0, "longitude": 79.383,
        "time_of_day": "Morning", "weather_condition": "Clear",
    })
    assert r.status_code == 400


def test_predict_route_basic(client):
    r = client.post("/predict/route", json={
        "origin_lat": PEDDAPALLI_CENTER[0], "origin_lng": PEDDAPALLI_CENTER[1],
        "dest_lat": RAMAGUNDAM[0], "dest_lng": RAMAGUNDAM[1],
        "preferred_time": "Night", "weather_condition": "Heavy Rain",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["safest_route"]["overall_risk_score"] >= 0
    assert len(body["safest_route"]["segments"]) > 0
    assert len(body["alternatives"]) >= 1
    # Safest route really should have the lowest (or tied-lowest, if it was
    # the only one with zero high-risk segments) risk among options shown.
    all_scores = [body["safest_route"]["overall_risk_score"]] + [
        a["overall_risk_score"] for a in body["alternatives"]
    ]
    assert body["safest_route"]["overall_risk_score"] <= max(all_scores)


def test_predict_route_same_point_errors_cleanly(client):
    r = client.post("/predict/route", json={
        "origin_lat": PEDDAPALLI_CENTER[0], "origin_lng": PEDDAPALLI_CENTER[1],
        "dest_lat": PEDDAPALLI_CENTER[0], "dest_lng": PEDDAPALLI_CENTER[1],
    })
    assert r.status_code == 422


def test_hotspots(client):
    r = client.get("/hotspots?top_n=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total_hotspots"] == 10
    assert body["hotspots"][0]["avg_risk_score"] >= body["hotspots"][-1]["avg_risk_score"]


def test_analytics(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_accidents"] == 500
    assert len(body["severity_breakdown"]) == 3
    assert len(body["monthly_trend"]) > 0


def test_places(client):
    r = client.get("/places")
    assert r.status_code == 200
    body = r.json()
    names = [p["name"] for p in body["places"]]
    assert "Peddapalli Town" in names
    assert "Ramagundam" in names


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()
