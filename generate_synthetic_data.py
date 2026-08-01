"""
generate_synthetic_data.py

Builds the two data files the whole project depends on:

    data/road_segments.geojson   -> the road network graph (routing)
    data/accidents_data.csv      -> 500 historical accident records (training)
    data/places.json             -> named place -> lat/lng lookup (for the API)

WHY ONE SCRIPT BUILDS BOTH FILES
--------------------------------
The route engine later needs to say "segment RS_014 has historically had a
0.7 average risk" -- that only works if the segments in the GeoJSON and the
`road_segment_id` values in the CSV refer to the exact same physical roads.
So instead of inventing the CSV and the GeoJSON separately, we first design
ONE small road network (a set of named waypoints + named roads connecting
them), then mechanically derive both files from it. Consistency is free
this way; it would not be if the two files were generated independently.

WHY risk_score IS NOT RANDOM NOISE
-----------------------------------
`compute_risk_score()` below builds risk_score as a weighted sum of the
row's own features (road type, weather, time of day, traffic, curve,
intersection, peak hour) plus a small amount of Gaussian noise. This is
what makes the ML step meaningful: a model trained on this CSV should
recover feature importances that roughly match the weights used here,
which is a good built-in sanity check that training actually worked,
rather than a model that memorised noise.

Run:
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Peddapalli district bounding box, as specified for this project.
LAT_MIN, LAT_MAX = 18.50, 18.80
LON_MIN, LON_MAX = 79.20, 79.60

N_ACCIDENTS = 500
DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2026, 4, 1, 23, 59, 59)

# ---------------------------------------------------------------------------
# 1. THE ROAD NETWORK (source of truth for both output files)
# ---------------------------------------------------------------------------
# Coordinates are (lat, lon). They are illustrative placements of real
# Peddapalli-district places, kept inside the bounding box above -- this is
# a synthetic dataset, not a surveyed map.
WAYPOINTS: dict[str, tuple[float, float]] = {
    "peddapalli_center":    (18.616, 79.383),
    "peddapalli_north":     (18.645, 79.395),
    "peddapalli_bypass_jn": (18.605, 79.355),
    "godavarikhani":        (18.760, 79.470),
    "ramagundam":           (18.780, 79.450),
    "basanthnagar":         (18.745, 79.545),
    "manthani":             (18.680, 79.590),
    "dharmaram":            (18.535, 79.260),
    "karimnagar_bypass_w":  (18.560, 79.205),
    "mid_sh1_1":            (18.680, 79.420),
    "mid_sh1_2":            (18.715, 79.470),
    "mid_bypass_1":         (18.590, 79.300),
    "mid_karimnagar_sh1":   (18.585, 79.245),
    "mid_manthani_1":       (18.640, 79.470),
    "mid_dharmaram_1":      (18.575, 79.320),
    "sh7_jn":               (18.660, 79.330),
    "sh7_north":            (18.700, 79.350),
    # Slightly offset from the straight SH-1 line between ramagundam and
    # basanthnagar on purpose: without this, "Basanthnagar Road" and the
    # SH-1 stretch between the same two towns interpolate to identical
    # coordinates and collapse into a single graph edge, silently erasing
    # that route alternative.
    "basanthnagar_local_mid": (18.734, 79.508),
}

# Waypoints exposed to the API as human-friendly named places.
PLACE_LABELS: dict[str, str] = {
    "peddapalli_center": "Peddapalli Town",
    "godavarikhani": "Godavarikhani",
    "ramagundam": "Ramagundam",
    "basanthnagar": "Basanthnagar",
    "manthani": "Manthani",
    "dharmaram": "Dharmaram",
    "karimnagar_bypass_w": "Karimnagar Road (District Edge)",
}

# Each road = an ordered walk over WAYPOINTS + static physical attributes +
# a base_risk in [0, 1] (a structural prior -- e.g. a fast, narrow highway
# is just riskier than a slow local road, independent of weather/time).
# `extra_risk_near` adds a bonus to edges touching a specific waypoint,
# used for the brief's "higher accidents on SH-1 near Basanthnagar".
ROADS: list[dict] = [
    {
        "name": "Rajiv Highway (SH-1)",
        "waypoints": ["peddapalli_center", "mid_sh1_1", "mid_sh1_2", "basanthnagar", "ramagundam"],
        "road_type": "Highway", "lanes": (4, 6), "base_risk": 0.62,
        "extra_risk_near": {"basanthnagar": 0.18},
    },
    {
        "name": "SH-1 Peddapalli-Karimnagar",
        "waypoints": ["peddapalli_center", "mid_karimnagar_sh1", "karimnagar_bypass_w"],
        "road_type": "Highway", "lanes": (4, 4), "base_risk": 0.50,
    },
    {
        "name": "Karimnagar-Peddapalli Bypass",
        "waypoints": ["peddapalli_bypass_jn", "mid_bypass_1", "karimnagar_bypass_w"],
        "road_type": "Highway", "lanes": (4, 6), "base_risk": 0.48,
    },
    {
        "name": "SH-7",
        "waypoints": ["peddapalli_center", "sh7_jn", "sh7_north"],
        "road_type": "Highway", "lanes": (4, 4), "base_risk": 0.45,
    },
    {
        "name": "Godavarikhani Road",
        "waypoints": ["peddapalli_north", "sh7_north", "godavarikhani"],
        "road_type": "Arterial", "lanes": (2, 4), "base_risk": 0.40,
    },
    {
        "name": "Ramagundam Road",
        "waypoints": ["godavarikhani", "ramagundam"],
        "road_type": "Arterial", "lanes": (2, 4), "base_risk": 0.42,
    },
    {
        "name": "Basanthnagar Road",
        "waypoints": ["ramagundam", "basanthnagar_local_mid", "basanthnagar"],
        "road_type": "Arterial", "lanes": (2, 2), "base_risk": 0.47,
    },
    {
        "name": "Manthani Road",
        "waypoints": ["peddapalli_center", "mid_manthani_1", "manthani"],
        "road_type": "Arterial", "lanes": (2, 2), "base_risk": 0.30,
    },
    {
        "name": "Dharmaram Road",
        "waypoints": ["peddapalli_bypass_jn", "mid_dharmaram_1", "dharmaram"],
        "road_type": "Local", "lanes": (2, 2), "base_risk": 0.24,
    },
    {
        "name": "Peddapalli Ring Road",
        "waypoints": ["peddapalli_center", "peddapalli_bypass_jn"],
        "road_type": "Arterial", "lanes": (2, 4), "base_risk": 0.35,
    },
    {
        "name": "Peddapalli North Link",
        "waypoints": ["peddapalli_center", "peddapalli_north"],
        "road_type": "Arterial", "lanes": (2, 2), "base_risk": 0.33,
    },
]

# ---------------------------------------------------------------------------
# Categorical option sets + the "risk story" behind each one
# ---------------------------------------------------------------------------
WEATHER_OPTIONS = ["Clear", "Rain", "Fog", "Heavy Rain"]
TIME_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]

WEATHER_RISK = {"Clear": 0.05, "Rain": 0.55, "Fog": 0.60, "Heavy Rain": 0.85}
TIME_RISK = {"Morning": 0.20, "Afternoon": 0.25, "Evening": 0.60, "Night": 0.75}
TRAFFIC_RISK = {"Low": 0.15, "Medium": 0.45, "High": 0.65}
ROAD_TYPE_RISK = {"Highway": 0.55, "Arterial": 0.35, "Local": 0.20}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def sample_weather(month: int) -> str:
    """Telangana has a real seasonal pattern: SW monsoon (Jun-Sep) brings
    heavy rain, and interior winters (Nov-Jan) bring morning fog. Baking
    this in gives the data a believable seasonal shape."""
    if month in (6, 7, 8, 9):
        weights = [0.35, 0.45, 0.05, 0.15]
    elif month in (11, 12, 1):
        weights = [0.45, 0.10, 0.35, 0.10]
    else:
        weights = [0.70, 0.15, 0.10, 0.05]
    return random.choices(WEATHER_OPTIONS, weights=weights, k=1)[0]


def sample_time_of_day() -> str:
    weights = [0.20, 0.20, 0.32, 0.28]  # Evening/Night weighted up, per brief
    return random.choices(TIME_OPTIONS, weights=weights, k=1)[0]


def sample_is_peak_hour(time_of_day: str) -> bool:
    p = {"Morning": 0.55, "Afternoon": 0.25, "Evening": 0.60, "Night": 0.10}[time_of_day]
    return random.random() < p


def sample_traffic_density(road_type: str, is_peak_hour: bool) -> str:
    if road_type == "Highway":
        weights = [0.20, 0.40, 0.40] if is_peak_hour else [0.35, 0.45, 0.20]
    elif road_type == "Arterial":
        weights = [0.25, 0.45, 0.30] if is_peak_hour else [0.40, 0.45, 0.15]
    else:
        weights = [0.55, 0.35, 0.10] if is_peak_hour else [0.70, 0.25, 0.05]
    return random.choices(TRAFFIC_OPTIONS, weights=weights, k=1)[0]


def sample_has_curve(road_name: str) -> bool:
    p = {"Manthani Road": 0.45, "Dharmaram Road": 0.40}.get(road_name, 0.20)
    return random.random() < p


def sample_has_intersection(road_type: str) -> bool:
    p = {"Highway": 0.25, "Arterial": 0.40, "Local": 0.20}[road_type]
    return random.random() < p


def compute_risk_score(
    *, road_type: str, weather: str, time_of_day: str, traffic_density: str,
    has_curve: bool, has_intersection: bool, is_peak_hour: bool,
    road_base_risk: float, extra_bonus: float = 0.0,
) -> float:
    """Transparent weighted-sum ground truth (see module docstring)."""
    score = (
        0.30 * road_base_risk
        + 0.15 * ROAD_TYPE_RISK[road_type]
        + 0.20 * WEATHER_RISK[weather]
        + 0.15 * TIME_RISK[time_of_day]
        + 0.10 * TRAFFIC_RISK[traffic_density]
        + (0.10 if has_curve else 0.0)
        + (0.07 if has_intersection else 0.0)
        + (0.05 if is_peak_hour else 0.0)
        + extra_bonus
    )
    score += np.random.normal(0, 0.06)
    return float(np.clip(score, 0.1, 0.95))


def sample_severity(risk_score: float) -> str:
    if risk_score >= 0.65:
        weights = [0.15, 0.35, 0.50]
    elif risk_score >= 0.40:
        weights = [0.30, 0.45, 0.25]
    else:
        weights = [0.55, 0.35, 0.10]
    return random.choices(["Low", "Medium", "High"], weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# 2. Build road segments from the network above
# ---------------------------------------------------------------------------
def build_segments() -> list[dict]:
    """Walk every road's waypoint sequence and cut each edge into 2-4
    sub-segments (longer edges get more pieces), so the network ends up
    with realistic, medium-length segments instead of a handful of very
    long ones."""
    segments = []
    seg_counter = 1
    for road in ROADS:
        wps = road["waypoints"]
        for i in range(len(wps) - 1):
            a_name, b_name = wps[i], wps[i + 1]
            a, b = WAYPOINTS[a_name], WAYPOINTS[b_name]
            edge_len = haversine_km(a, b)
            n_parts = int(np.clip(round(edge_len / 2.5), 2, 4))
            pts = [
                (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                for t in np.linspace(0, 1, n_parts + 1)
            ]
            bonus = 0.0
            for place_name, amt in road.get("extra_risk_near", {}).items():
                if a_name == place_name or b_name == place_name:
                    bonus = max(bonus, amt)

            for j in range(n_parts):
                seg_id = f"RS_{seg_counter:03d}"
                seg_counter += 1
                start, end = pts[j], pts[j + 1]
                # has_curve / has_intersection are FIXED, physical properties
                # of this stretch of road -- decided once here, not resampled
                # per accident -- so the route engine can look them up
                # deterministically for any segment.
                segments.append({
                    "segment_id": seg_id,
                    "road_name": road["name"],
                    "road_type": road["road_type"],
                    "num_lanes": random.randint(*road["lanes"]),
                    "base_risk": road["base_risk"],
                    "extra_bonus": bonus,
                    "has_curve": sample_has_curve(road["name"]),
                    "has_intersection": sample_has_intersection(road["road_type"]),
                    "start_node": a_name if j == 0 else None,
                    "end_node": b_name if j == n_parts - 1 else None,
                    "start": start,
                    "end": end,
                    "length_km": haversine_km(start, end),
                })
    return segments


def write_geojson(segments: list[dict]) -> None:
    """NOTE: GeoJSON coordinate order is [longitude, latitude], the
    opposite of how we store points internally as (lat, lon). Converting
    at this one boundary (and again, symmetrically, in the app's graph
    builder) keeps that mix-up from leaking anywhere else."""
    features = []
    for s in segments:
        features.append({
            "type": "Feature",
            "properties": {
                "segment_id": s["segment_id"],
                "road_name": s["road_name"],
                "road_type": s["road_type"],
                "num_lanes": s["num_lanes"],
                "length_km": round(s["length_km"], 3),
                "has_curve": s["has_curve"],
                "has_intersection": s["has_intersection"],
                "start_node": s["start_node"],
                "end_node": s["end_node"],
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(s["start"][1], 6), round(s["start"][0], 6)],
                    [round(s["end"][1], 6), round(s["end"][0], 6)],
                ],
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    with open(DATA_DIR / "road_segments.geojson", "w") as f:
        json.dump(fc, f, indent=2)


def write_places() -> None:
    places = {
        label: {"lat": WAYPOINTS[key][0], "lng": WAYPOINTS[key][1], "node_id": key}
        for key, label in PLACE_LABELS.items()
    }
    with open(DATA_DIR / "places.json", "w") as f:
        json.dump(places, f, indent=2)


# ---------------------------------------------------------------------------
# 3. Generate the 500 accident rows from the segments above
# ---------------------------------------------------------------------------
def random_datetime_in_range(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def set_hour_for_time_of_day(dt: datetime, time_of_day: str) -> datetime:
    hour_ranges = {"Morning": (6, 11), "Afternoon": (12, 16), "Evening": (17, 20)}
    if time_of_day == "Night":
        hour = random.randint(0, 5) if random.random() < 0.4 else random.randint(21, 23)
    else:
        lo, hi = hour_ranges[time_of_day]
        hour = random.randint(lo, hi)
    return dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def generate_accidents(segments: list[dict], n: int = N_ACCIDENTS) -> pd.DataFrame:
    weights = np.array([s["base_risk"] + s["extra_bonus"] + 0.15 for s in segments], dtype=float)
    weights = weights / weights.sum()
    seg_indices = np.random.choice(len(segments), size=n, p=weights)

    rows = []
    for i, seg_idx in enumerate(seg_indices, start=1):
        seg = segments[seg_idx]

        t = random.random()
        lat = seg["start"][0] + (seg["end"][0] - seg["start"][0]) * t + np.random.normal(0, 0.0015)
        lon = seg["start"][1] + (seg["end"][1] - seg["start"][1]) * t + np.random.normal(0, 0.0015)
        lat = float(np.clip(lat, LAT_MIN, LAT_MAX))
        lon = float(np.clip(lon, LON_MIN, LON_MAX))

        dt = random_datetime_in_range(DATE_START, DATE_END)
        weather = sample_weather(dt.month)
        time_of_day = sample_time_of_day()
        dt = set_hour_for_time_of_day(dt, time_of_day)
        is_peak = sample_is_peak_hour(time_of_day)
        traffic = sample_traffic_density(seg["road_type"], is_peak)
        has_curve = seg["has_curve"]
        has_intersection = seg["has_intersection"]

        risk = compute_risk_score(
            road_type=seg["road_type"], weather=weather, time_of_day=time_of_day,
            traffic_density=traffic, has_curve=has_curve, has_intersection=has_intersection,
            is_peak_hour=is_peak, road_base_risk=seg["base_risk"], extra_bonus=seg["extra_bonus"],
        )

        rows.append({
            "accident_id": i,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "road_segment_id": seg["segment_id"],
            "road_name": seg["road_name"],
            "accident_severity": sample_severity(risk),
            "weather_condition": weather,
            "time_of_day": time_of_day,
            "traffic_density": traffic,
            "road_type": seg["road_type"],
            "num_lanes": seg["num_lanes"],
            "has_intersection": has_intersection,
            "has_curve": has_curve,
            "is_peak_hour": is_peak,
            "risk_score": round(risk, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date_time").reset_index(drop=True)
    df["accident_id"] = range(1, len(df) + 1)
    df.to_csv(DATA_DIR / "accidents_data.csv", index=False)
    return df


if __name__ == "__main__":
    segments = build_segments()
    write_geojson(segments)
    write_places()
    df = generate_accidents(segments, n=N_ACCIDENTS)

    print(f"Roads: {len(ROADS)}  |  Segments: {len(segments)}  |  Accident rows: {len(df)}")
    print("\nRisk score summary:")
    print(df["risk_score"].describe().round(3))
    print("\nRows per road:")
    print(df["road_name"].value_counts())
    print("\nSeverity breakdown:")
    print(df["accident_severity"].value_counts())
