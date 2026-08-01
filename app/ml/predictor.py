"""
app/ml/predictor.py

The single place that turns "a point on the map + a few conditions" into
a risk_score. Used directly by /predict/risk, and called once per
candidate road segment by the route engine for /predict/route.

FEATURE ASSEMBLY
----------------
The trained pipeline expects exactly the columns in train_model.FEATURE_COLUMNS:
    weather_condition, time_of_day, traffic_density, road_type   (categorical)
    latitude, longitude, num_lanes, has_intersection, has_curve, is_peak_hour (numeric)

The API only ever asks the caller for latitude, longitude, time_of_day and
weather_condition (matching the brief). Everything else is filled in here:
    - road_type / num_lanes / has_curve / has_intersection come from the
      nearest road segment in the RoadNetwork (a real, physical road
      nearby "owns" those attributes -- the caller shouldn't have to know
      how many lanes the nearest highway has).
    - is_peak_hour, if not supplied, is inferred from time_of_day.
    - traffic_density, if not supplied, is inferred from road_type + peak hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from app.config import risk_label, settings
from app.routing.graph_builder import RoadNetwork, Segment

logger = logging.getLogger("accident_api.predictor")

FEATURE_COLUMNS = [
    "weather_condition", "time_of_day", "traffic_density", "road_type",
    "latitude", "longitude", "num_lanes", "has_intersection", "has_curve", "is_peak_hour",
]

# Default traffic density by (road_type, is_peak_hour) when the caller
# doesn't specify one -- mirrors the tendencies baked into the synthetic
# training data's sample_traffic_density().
_DEFAULT_TRAFFIC = {
    ("Highway", True): "High", ("Highway", False): "Medium",
    ("Arterial", True): "Medium", ("Arterial", False): "Medium",
    ("Local", True): "Medium", ("Local", False): "Low",
}


def infer_is_peak_hour(time_of_day: str) -> bool:
    return time_of_day in ("Morning", "Evening")


def infer_traffic_density(road_type: str, is_peak_hour: bool) -> str:
    return _DEFAULT_TRAFFIC.get((road_type, is_peak_hour), "Medium")


class RiskPredictor:
    def __init__(self, model_path: Path, road_network: RoadNetwork):
        self.model_path = model_path
        self.road_network = road_network
        self.pipeline = None
        self.trained_at: str | None = None
        self._load_or_none()

    def _load_or_none(self) -> None:
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
            logger.info("Loaded trained model from %s", self.model_path)
            meta_path = self.model_path.parent / "model_metadata.json"
            if meta_path.exists():
                import json
                self.trained_at = json.loads(meta_path.read_text()).get("trained_at")
        else:
            logger.warning("No model.pkl found at %s -- will use fallback heuristic", self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    # ------------------------------------------------------------------
    def resolve_segment_attributes(self, lat: float, lon: float) -> tuple[Segment | None, float | None]:
        try:
            seg, km = self.road_network.nearest_segment(lat, lon)
            return seg, km
        except Exception:
            logger.exception("Nearest-segment lookup failed for (%s, %s)", lat, lon)
            return None, None

    def assemble_features(
        self, *, latitude: float, longitude: float, time_of_day: str, weather_condition: str,
        traffic_density: str | None = None, is_peak_hour: bool | None = None,
        segment: Segment | None = None,
    ) -> dict:
        if is_peak_hour is None:
            is_peak_hour = infer_is_peak_hour(time_of_day)

        if segment is not None:
            road_type, num_lanes = segment.road_type, segment.num_lanes
            has_curve, has_intersection = segment.has_curve, segment.has_intersection
        else:
            road_type, num_lanes, has_curve, has_intersection = "Arterial", 2, False, False

        if traffic_density is None:
            traffic_density = infer_traffic_density(road_type, is_peak_hour)

        return {
            "weather_condition": weather_condition,
            "time_of_day": time_of_day,
            "traffic_density": traffic_density,
            "road_type": road_type,
            "latitude": latitude,
            "longitude": longitude,
            "num_lanes": num_lanes,
            "has_intersection": int(has_intersection),
            "has_curve": int(has_curve),
            "is_peak_hour": int(is_peak_hour),
        }

    def predict_from_features(self, features: dict) -> float:
        if self.pipeline is not None:
            row = pd.DataFrame([{k: features[k] for k in FEATURE_COLUMNS}])
            score = float(self.pipeline.predict(row)[0])
            return max(0.05, min(0.98, score))
        return self._fallback_heuristic(features)

    @staticmethod
    def _fallback_heuristic(features: dict) -> float:
        """Only used if model.pkl genuinely can't be loaded/trained -- a
        simple, transparent stand-in so the API still returns something
        sensible instead of erroring."""
        weather_risk = {"Clear": 0.10, "Rain": 0.55, "Fog": 0.60, "Heavy Rain": 0.85}
        time_risk = {"Morning": 0.25, "Afternoon": 0.30, "Evening": 0.55, "Night": 0.70}
        road_risk = {"Highway": 0.55, "Arterial": 0.35, "Local": 0.20}
        score = (
            0.35 * road_risk.get(features["road_type"], 0.35)
            + 0.30 * weather_risk.get(features["weather_condition"], 0.3)
            + 0.20 * time_risk.get(features["time_of_day"], 0.3)
            + 0.08 * (1 if features["has_curve"] else 0)
            + 0.07 * (1 if features["has_intersection"] else 0)
        )
        return max(0.05, min(0.95, score))

    # ------------------------------------------------------------------
    def explain(self, features: dict, risk_score: float) -> tuple[list[str], str]:
        """Rule-based, human-readable explanation -- not SHAP, but fast,
        deterministic, and cheap enough to run on every request (SHAP's
        overhead isn't worth it for a synthetic model this small; see
        README "Future work")."""
        factors = []
        if features["weather_condition"] in ("Rain", "Fog", "Heavy Rain"):
            factors.append(f"{features['weather_condition']} weather")
        if features["time_of_day"] in ("Evening", "Night"):
            factors.append(f"{features['time_of_day']} conditions (lower visibility)")
        if features["road_type"] == "Highway":
            factors.append("highway-speed road")
        if features["has_curve"]:
            factors.append("road curve nearby")
        if features["has_intersection"]:
            factors.append("intersection nearby")
        if features["is_peak_hour"]:
            factors.append("peak-hour traffic")
        if features["traffic_density"] == "High":
            factors.append("high traffic density")

        level = risk_label(risk_score)
        if not factors:
            explanation = f"{level} risk ({risk_score:.2f}) -- clear, low-traffic conditions on this stretch."
        else:
            explanation = f"{level} risk ({risk_score:.2f}) driven mainly by: " + ", ".join(factors) + "."
        return factors, explanation

    # ------------------------------------------------------------------
    def predict_point(
        self, *, latitude: float, longitude: float, time_of_day: str, weather_condition: str,
        traffic_density: str | None = None, is_peak_hour: bool | None = None,
    ) -> dict:
        segment, distance_km = self.resolve_segment_attributes(latitude, longitude)
        features = self.assemble_features(
            latitude=latitude, longitude=longitude, time_of_day=time_of_day,
            weather_condition=weather_condition, traffic_density=traffic_density,
            is_peak_hour=is_peak_hour, segment=segment,
        )
        score = self.predict_from_features(features)
        factors, explanation = self.explain(features, score)
        return {
            "risk_score": round(score, 4),
            "risk_level": risk_label(score),
            "nearest_road_name": segment.road_name if segment else None,
            "nearest_segment_id": segment.segment_id if segment else None,
            "distance_to_road_km": round(distance_km, 4) if distance_km is not None else None,
            "features_used": features,
            "contributing_factors": factors,
            "explanation": explanation,
        }


_predictor: RiskPredictor | None = None


def get_predictor(model_path: Path | None = None, road_network: RoadNetwork | None = None) -> RiskPredictor:
    global _predictor
    if _predictor is None:
        if model_path is None or road_network is None:
            raise RuntimeError("RiskPredictor not initialised yet")
        _predictor = RiskPredictor(model_path, road_network)
    return _predictor
