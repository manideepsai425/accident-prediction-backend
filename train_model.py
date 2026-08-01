"""
train_model.py

Trains the risk_score regression model used by /predict/risk and
/predict/route, and saves it to model.pkl (plus a small model_metadata.json
next to it).

WHY THESE FEATURES
------------------
FEATURE_COLUMNS below is exactly the set of values the API can assemble at
prediction time: 2 numbers the caller always gives us (latitude, longitude),
2 categories the caller always gives us (weather_condition, time_of_day),
and 4 more values that either the caller can override or the backend fills
in by looking up the nearest road segment (traffic_density, road_type,
num_lanes, has_curve, has_intersection, is_peak_hour). We deliberately do
NOT use road_segment_id, road_name, accident_id or date_time as features:
the IDs are effectively unique labels the model could "memorise" instead of
generalising from (a classic overfitting trap called target leakage /
high-cardinality ID leakage), and raw date_time is already summarised by
time_of_day + weather_condition (which is what the seasonal weather pattern
in the data is a proxy for anyway).

WHY COMPARE TWO MODELS
-----------------------
RandomForestRegressor (bagging: many independent trees, averaged) and
GradientBoostingRegressor (boosting: trees built one after another, each
correcting the previous one's mistakes) fail in different ways on small
tabular datasets. Trying both and keeping whichever cross-validates better
is cheap (dataset is only 500 rows) and is standard practice rather than
guessing up front.

USAGE
-----
Local / Render build step:
    python train_model.py

Google Colab:
    1. Upload accidents_data.csv (or the whole `data/` folder) and this
       script to your Colab session (or `git clone` the repo).
    2. !pip install scikit-learn pandas numpy joblib
    3. !python train_model.py
    4. Download model.pkl (and model_metadata.json) from the Colab file
       browser and drop them into this project's root folder, replacing
       the committed ones.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "accidents_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

TARGET = "risk_score"
CATEGORICAL_FEATURES = ["weather_condition", "time_of_day", "traffic_density", "road_type"]
NUMERIC_FEATURES = ["latitude", "longitude", "num_lanes", "has_intersection", "has_curve", "is_peak_hour"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42


def load_data(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["has_intersection", "has_curve", "is_peak_hour"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df[col] = df[col].astype(int)
    return df


def build_pipeline(regressor) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


CANDIDATES = {
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE,
    ),
}


def cross_validate_candidates(X: pd.DataFrame, y: pd.Series) -> dict:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, regressor in CANDIDATES.items():
        pipe = build_pipeline(regressor)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=None,
        )
        results[name] = {
            "r2_mean": float(np.mean(scores["test_r2"])),
            "r2_std": float(np.std(scores["test_r2"])),
            "mae_mean": float(-np.mean(scores["test_mae"])),
            "rmse_mean": float(-np.mean(scores["test_rmse"])),
        }
    return results


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    ohe: OneHotEncoder = fitted_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + NUMERIC_FEATURES


def train(csv_path: Path = DATA_CSV, verbose: bool = True) -> dict:
    df = load_data(csv_path)
    X, y = df[FEATURE_COLUMNS], df[TARGET]

    if verbose:
        print(f"Loaded {len(df)} rows, {len(FEATURE_COLUMNS)} features -> target '{TARGET}'")
        print("\n=== 5-fold cross-validation ===")

    cv_results = cross_validate_candidates(X, y)
    if verbose:
        for name, r in cv_results.items():
            print(f"  {name:>18s}:  R2 = {r['r2_mean']:.3f} (+/-{r['r2_std']:.3f})   "
                  f"MAE = {r['mae_mean']:.4f}   RMSE = {r['rmse_mean']:.4f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["r2_mean"])
    if verbose:
        print(f"\nSelected model: {best_name} (highest mean CV R2)")

    # Held-out split purely for a human-readable final report; the model we
    # SHIP is refit on all 500 rows afterwards (see below).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    report_pipeline = build_pipeline(CANDIDATES[best_name])
    report_pipeline.fit(X_train, y_train)
    y_pred = report_pipeline.predict(X_test)
    holdout_metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    if verbose:
        print(f"\n=== Held-out 20% test report ({best_name}) ===")
        print(f"  R2 = {holdout_metrics['r2']:.3f}   MAE = {holdout_metrics['mae']:.4f}   "
              f"RMSE = {holdout_metrics['rmse']:.4f}")

    # Refit the chosen model type on the FULL dataset -- standard practice
    # once the architecture is chosen via CV: more training data => a
    # better production model, and the held-out score above already tells
    # us what to expect out-of-sample.
    final_pipeline = build_pipeline(CANDIDATES[best_name])
    final_pipeline.fit(X, y)

    feature_importances = None
    fitted_regressor = final_pipeline.named_steps["regressor"]
    if hasattr(fitted_regressor, "feature_importances_"):
        names = get_feature_names(final_pipeline)
        importances = fitted_regressor.feature_importances_
        feature_importances = sorted(
            zip(names, [float(i) for i in importances]), key=lambda t: -t[1]
        )
        if verbose:
            print(f"\n=== Feature importances ({best_name}) ===")
            for name, imp in feature_importances[:12]:
                print(f"  {name:<28s} {imp:.4f}")

    joblib.dump(final_pipeline, MODEL_PATH)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": best_name,
        "n_training_rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "cv_results": cv_results,
        "holdout_metrics": holdout_metrics,
        "feature_importances": feature_importances,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nSaved model -> {MODEL_PATH}")
        print(f"Saved metadata -> {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_CSV
    train(csv_arg)
