"""
app/config.py

One place for every path and every environment-configurable value. Nothing
else in the codebase should hardcode a file path or read os.environ directly
-- if it needs configuring, it belongs here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = one level above the `app/` package.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- CORS ---
    # Comma-separated origins, e.g. "https://my-app.vercel.app,http://localhost:5173"
    # "*" allows any origin (fine for this project: no cookies/auth are used).
    cors_origins: str = "*"

    # --- Data & model paths (relative to BASE_DIR unless absolute) ---
    data_dir: str = "data"
    model_path: str = "model.pkl"
    accidents_csv: str = "data/accidents_data.csv"
    road_segments_geojson: str = "data/road_segments.geojson"
    places_json: str = "data/places.json"

    # If model.pkl is missing at startup, train it from accidents_csv
    # instead of crashing. Keeps the "it just works" promise even on a
    # fresh clone with no committed model.
    auto_train_if_missing: bool = True

    # --- Domain constants (Peddapalli district bounding box) ---
    lat_min: float = 18.45
    lat_max: float = 18.85
    lon_min: float = 79.15
    lon_max: float = 79.65

    # Risk-score band edges used everywhere a score is turned into a label.
    risk_medium_threshold: float = 0.40
    risk_high_threshold: float = 0.70

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*" or not raw:
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def path(self, relative: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else BASE_DIR / p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def risk_label(score: float) -> str:
    """Turn a numeric risk score into the Low / Medium / High band used
    consistently across every endpoint's response."""
    if score > settings.risk_high_threshold:
        return "High"
    if score > settings.risk_medium_threshold:
        return "Medium"
    return "Low"
