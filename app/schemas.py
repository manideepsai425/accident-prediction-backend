"""
app/schemas.py

Every request/response shape the API uses, in one file so the contract is
easy to read end-to-end. All models are Pydantic v2.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

TimeOfDay = Literal["Morning", "Afternoon", "Evening", "Night"]
Weather = Literal["Clear", "Rain", "Fog", "Heavy Rain"]
Traffic = Literal["Low", "Medium", "High"]
RoadType = Literal["Highway", "Arterial", "Local"]
RiskLevel = Literal["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_trained_at: Optional[str] = None
    accidents_loaded: int
    road_segments_loaded: int
    graph_nodes: int
    graph_edges: int
    version: str


# ---------------------------------------------------------------------------
# /predict/risk
# ---------------------------------------------------------------------------
class RiskPredictionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude within the Peddapalli district")
    longitude: float = Field(..., description="Longitude within the Peddapalli district")
    time_of_day: TimeOfDay
    weather_condition: Weather
    traffic_density: Optional[Traffic] = Field(
        None, description="If omitted, inferred from road type + peak hour"
    )
    is_peak_hour: Optional[bool] = Field(
        None, description="If omitted, inferred from time_of_day"
    )


class RiskPredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    nearest_road_name: Optional[str] = None
    nearest_segment_id: Optional[str] = None
    distance_to_road_km: Optional[float] = None
    features_used: dict = Field(default_factory=dict)
    contributing_factors: list[str] = Field(default_factory=list)
    explanation: str


# ---------------------------------------------------------------------------
# /predict/route
# ---------------------------------------------------------------------------
class RouteRequest(BaseModel):
    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    preferred_time: TimeOfDay = "Afternoon"
    weather_condition: Weather = "Clear"
    max_alternatives: int = Field(3, ge=1, le=5)


class RouteSegmentInfo(BaseModel):
    segment_id: str
    road_name: str
    start: tuple[float, float]  # (lat, lon)
    end: tuple[float, float]    # (lat, lon)
    length_km: float
    risk_score: float
    risk_level: RiskLevel
    warning: Optional[str] = None


class RouteOption(BaseModel):
    route_id: str
    label: str
    rank: int
    coordinates: list[tuple[float, float]]  # full polyline, (lat, lon) pairs
    overall_risk_score: float
    total_distance_km: float
    est_travel_time_min: float
    segments: list[RouteSegmentInfo]
    high_risk_segment_count: int
    medium_risk_segment_count: int
    is_safest: bool
    is_shortest: bool
    summary: str


class RouteResponse(BaseModel):
    origin: tuple[float, float]
    destination: tuple[float, float]
    safest_route: RouteOption
    alternatives: list[RouteOption]
    warnings: list[str] = Field(default_factory=list)
    explanation: str


# ---------------------------------------------------------------------------
# /hotspots
# ---------------------------------------------------------------------------
class Hotspot(BaseModel):
    rank: int
    road_segment_id: str
    road_name: str
    latitude: float
    longitude: float
    accident_count: int
    avg_risk_score: float
    risk_level: RiskLevel
    dominant_severity: str
    dominant_weather: str
    dominant_time_of_day: str


class HotspotsResponse(BaseModel):
    total_hotspots: int
    hotspots: list[Hotspot]


# ---------------------------------------------------------------------------
# /analytics
# ---------------------------------------------------------------------------
class CategoryBreakdown(BaseModel):
    category: str
    count: int
    percentage: float


class MonthlyTrendPoint(BaseModel):
    year_month: str
    accident_count: int
    avg_risk_score: float


class AnalyticsResponse(BaseModel):
    total_accidents: int
    date_range: tuple[str, str]
    avg_risk_score: float
    severity_breakdown: list[CategoryBreakdown]
    weather_breakdown: list[CategoryBreakdown]
    time_of_day_breakdown: list[CategoryBreakdown]
    road_type_breakdown: list[CategoryBreakdown]
    top_risky_roads: list[CategoryBreakdown]
    monthly_trend: list[MonthlyTrendPoint]
    peak_hour_share: float
    intersection_share: float
    curve_share: float


# ---------------------------------------------------------------------------
# /places
# ---------------------------------------------------------------------------
class Place(BaseModel):
    name: str
    latitude: float
    longitude: float


class PlacesResponse(BaseModel):
    places: list[Place]
