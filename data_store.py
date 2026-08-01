"""
app/services/data_store.py

Loads the CSV, GeoJSON and places lookup exactly once (at app startup) and
keeps them in memory. Every router reads through this class instead of
re-reading files off disk on every request.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from app.config import risk_label, settings

logger = logging.getLogger("accident_api.data_store")


class DataStore:
    def __init__(self, accidents_csv: Path, road_segments_geojson: Path, places_json: Path):
        self.accidents_df: pd.DataFrame = self._load_accidents(accidents_csv)
        self.geojson: dict = self._load_geojson(road_segments_geojson)
        self.places: dict = self._load_places(places_json)

        # Historical per-segment stats, used both by /hotspots and as a
        # prior the route engine can fall back on if the ML model is ever
        # unavailable.
        self.segment_stats: pd.DataFrame = self._build_segment_stats()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_accidents(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date_time"])
        bool_cols = ["has_intersection", "has_curve", "is_peak_hour"]
        for col in bool_cols:
            if df[col].dtype != bool:
                df[col] = df[col].astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
        logger.info("Loaded %d accident records from %s", len(df), path)
        return df

    @staticmethod
    def _load_geojson(path: Path) -> dict:
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded %d road segment features from %s", len(data["features"]), path)
        return data

    @staticmethod
    def _load_places(path: Path) -> dict:
        if not path.exists():
            logger.warning("places.json not found at %s -- /places will be empty", path)
            return {}
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def _build_segment_stats(self) -> pd.DataFrame:
        grp = self.accidents_df.groupby("road_segment_id")
        stats = grp.agg(
            road_name=("road_name", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            accident_count=("accident_id", "count"),
            avg_risk_score=("risk_score", "mean"),
        ).reset_index()

        def _mode(series: pd.Series) -> str:
            m = series.mode()
            return str(m.iloc[0]) if not m.empty else ""

        stats["dominant_severity"] = grp["accident_severity"].agg(_mode).values
        stats["dominant_weather"] = grp["weather_condition"].agg(_mode).values
        stats["dominant_time_of_day"] = grp["time_of_day"].agg(_mode).values
        return stats.sort_values("avg_risk_score", ascending=False).reset_index(drop=True)

    def hotspots(self, top_n: int = 15) -> list[dict]:
        top = self.segment_stats.sort_values(
            ["avg_risk_score", "accident_count"], ascending=[False, False]
        ).head(top_n)
        out = []
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            out.append({
                "rank": rank,
                "road_segment_id": row["road_segment_id"],
                "road_name": row["road_name"],
                "latitude": round(float(row["latitude"]), 6),
                "longitude": round(float(row["longitude"]), 6),
                "accident_count": int(row["accident_count"]),
                "avg_risk_score": round(float(row["avg_risk_score"]), 4),
                "risk_level": risk_label(float(row["avg_risk_score"])),
                "dominant_severity": row["dominant_severity"],
                "dominant_weather": row["dominant_weather"],
                "dominant_time_of_day": row["dominant_time_of_day"],
            })
        return out

    def analytics(self) -> dict:
        df = self.accidents_df

        def breakdown(col: str, top_n: int | None = None) -> list[dict]:
            counts = df[col].value_counts()
            if top_n:
                counts = counts.head(top_n)
            total = len(df)
            return [
                {"category": str(idx), "count": int(cnt), "percentage": round(100 * cnt / total, 2)}
                for idx, cnt in counts.items()
            ]

        monthly = (
            df.assign(year_month=df["date_time"].dt.strftime("%Y-%m"))
            .groupby("year_month")
            .agg(accident_count=("accident_id", "count"), avg_risk_score=("risk_score", "mean"))
            .reset_index()
            .sort_values("year_month")
        )
        monthly_trend = [
            {
                "year_month": row["year_month"],
                "accident_count": int(row["accident_count"]),
                "avg_risk_score": round(float(row["avg_risk_score"]), 4),
            }
            for _, row in monthly.iterrows()
        ]

        top_risky_roads = (
            df.groupby("road_name")["risk_score"].mean().sort_values(ascending=False).head(5)
        )
        top_risky_roads_out = [
            {"category": name, "count": int((df["road_name"] == name).sum()), "percentage": round(float(score), 4)}
            for name, score in top_risky_roads.items()
        ]

        return {
            "total_accidents": len(df),
            "date_range": (
                df["date_time"].min().strftime("%Y-%m-%d"),
                df["date_time"].max().strftime("%Y-%m-%d"),
            ),
            "avg_risk_score": round(float(df["risk_score"].mean()), 4),
            "severity_breakdown": breakdown("accident_severity"),
            "weather_breakdown": breakdown("weather_condition"),
            "time_of_day_breakdown": breakdown("time_of_day"),
            "road_type_breakdown": breakdown("road_type"),
            "top_risky_roads": top_risky_roads_out,
            "monthly_trend": monthly_trend,
            "peak_hour_share": round(float(df["is_peak_hour"].mean()), 4),
            "intersection_share": round(float(df["has_intersection"].mean()), 4),
            "curve_share": round(float(df["has_curve"].mean()), 4),
        }


_data_store: DataStore | None = None


def get_data_store() -> DataStore:
    global _data_store
    if _data_store is None:
        _data_store = DataStore(
            accidents_csv=settings.path(settings.accidents_csv),
            road_segments_geojson=settings.path(settings.road_segments_geojson),
            places_json=settings.path(settings.places_json),
        )
    return _data_store
