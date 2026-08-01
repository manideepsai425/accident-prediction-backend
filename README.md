# Peddapalli Accident Risk & Safest-Route API

FastAPI + scikit-learn backend that predicts road-accident risk in Peddapalli
district (Telangana, India) and computes the safest route between two points,
avoiding high-risk stretches where possible.

Live docs (once deployed): `https://<your-render-url>/docs`

---

## 1. Architecture

```
accident-prediction-backend/
├── app/
│   ├── main.py              # FastAPI app, lifespan startup, CORS, routers
│   ├── config.py            # All settings (env-var driven), risk_label()
│   ├── schemas.py            # Pydantic v2 request/response models
│   ├── deps.py               # Depends() wrappers around app.state
│   ├── ml/
│   │   └── predictor.py      # Feature assembly + prediction + explanation
│   ├── routing/
│   │   ├── graph_builder.py  # GeoJSON -> networkx graph + nearest-segment lookup
│   │   └── route_engine.py   # k-shortest-paths + per-segment risk -> safest route
│   ├── services/
│   │   └── data_store.py     # Loads CSV/GeoJSON/places once; hotspot & analytics aggregation
│   └── routers/               # health, predict, hotspots, analytics, places
├── data/
│   ├── accidents_data.csv    # 500 synthetic-but-realistic accident records
│   ├── road_segments.geojson # 61 road segments across 11 named roads
│   └── places.json           # Named-place -> lat/lng lookup
├── scripts/
│   └── generate_synthetic_data.py  # Regenerate the two files above from scratch
├── tests/test_api.py          # pytest + TestClient, covers every endpoint
├── train_model.py             # Train / retrain the risk model (local, Render, or Colab)
├── model.pkl                  # Shipped, pre-trained model (see "Retraining on Colab")
├── model_metadata.json        # CV scores, holdout metrics, feature importances
├── Dockerfile
├── render.yaml
└── requirements.txt
```

**Data flow:** `scripts/generate_synthetic_data.py` is the *only* source of
`accidents_data.csv` and `road_segments.geojson` — both are derived from one
hand-designed road network so a `road_segment_id` always means the same
physical stretch of road in both files. `train_model.py` reads the CSV and
produces `model.pkl`. The running API reads the CSV, GeoJSON and `model.pkl`
— it never re-runs the generator.

---

## 2. How the safest-route algorithm works

1. **Snap** origin and destination lat/lng to the nearest node in the road
   graph (built from `road_segments.geojson` via `networkx`).
2. **Generate candidates**: `nx.shortest_simple_paths` (Yen's algorithm)
   returns several *genuinely different* simple paths between the two
   nodes, in increasing order of distance — this is what gives real route
   alternatives instead of one path with cosmetic variations.
3. **Score every segment** of every candidate with the trained scikit-learn
   model (weather + time-of-day from the request; road type / lanes / curve
   / intersection looked up from that segment's own static attributes).
4. **Pick the safest**: prefer any candidate with **zero** segments above
   the high-risk threshold (0.7); among those, take the lowest
   length-weighted average risk. If *every* candidate has at least one
   high-risk segment, fall back to "fewest high-risk segments, then lowest
   overall risk" and explicitly flag the unavoidable segment(s) — this
   matches the brief's "if the safest route still has risk, highlight it."
5. **Alternatives**: the remaining candidates, ranked by risk, with the
   shortest-distance option guaranteed to appear even if it isn't picked,
   so you always get to see a genuine "faster but riskier" tradeoff
   (`"Route 2: Faster - Risk 0.68"`).

Overall route risk = **length-weighted average** of segment risk scores
(a 200m high-risk stretch shouldn't count the same as a 5km one).

---

## 3. API Reference

### `GET /health`
Returns whether the model is loaded, how many accidents/segments/graph
nodes are in memory, and the app version.

### `POST /predict/risk`
```json
{
  "latitude": 18.745, "longitude": 79.545,
  "time_of_day": "Night", "weather_condition": "Heavy Rain"
}
```
`traffic_density` and `is_peak_hour` are optional — inferred if omitted.
Returns `risk_score` (0-1), `risk_level`, nearest road/segment, and a
plain-English `explanation`.

### `POST /predict/route`
```json
{
  "origin_lat": 18.616, "origin_lng": 79.383,
  "dest_lat": 18.780, "dest_lng": 79.450,
  "preferred_time": "Night", "weather_condition": "Heavy Rain"
}
```
Returns `safest_route` (full polyline + per-segment risk + warnings) and
`alternatives` (each labelled like `"Route 1: Safest - Risk 0.32"`).

### `GET /hotspots?top_n=15`
Top historical accident hotspots by road segment, ranked by average risk.

### `GET /analytics`
Severity/weather/time-of-day/road-type breakdowns, monthly trend, top risky
roads, peak-hour/intersection/curve shares — everything a dashboard needs.

### `GET /places`
Named places (`"Peddapalli Town"`, `"Ramagundam"`, etc.) with lat/lng, for a
frontend dropdown.

Full interactive schema at `/docs` (Swagger) or `/redoc`.

---

## 4. The ML model

`train_model.py` trains on 10 features (`weather_condition`, `time_of_day`,
`traffic_density`, `road_type`, `latitude`, `longitude`, `num_lanes`,
`has_intersection`, `has_curve`, `is_peak_hour`) to predict `risk_score`.
`road_name` / `road_segment_id` / `accident_id` are deliberately excluded —
they're identifiers, not causes, and a model can "memorise" them instead of
generalising (a classic high-cardinality leakage trap).

It **compares two model families via 5-fold cross-validation**
(`RandomForestRegressor` vs `GradientBoostingRegressor`) and keeps whichever
scores higher, then refits that architecture on the full 500-row dataset for
the shipped model. Current result (see `model_metadata.json` for the exact
numbers from your last training run):

| Model | CV R² | CV MAE |
|---|---|---|
| Gradient Boosting (selected) | 0.765 | 0.057 |
| Random Forest | 0.716 | 0.062 |

Held-out test R² = 0.814, MAE = 0.053. Top feature importances land on latitude/longitude
(spatial risk clusters like Basanthnagar), `weather_condition_Clear`, and
`road_type_Highway` — i.e. the model recovered the same structure the
synthetic data was built from, which is a good sanity check.

---

## 5. Local setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# -> http://localhost:8000/docs
```

If `model.pkl` is missing, the app trains one automatically on first startup
(a few seconds — see `AUTO_TRAIN_IF_MISSING` in `.env.example`).

Run tests:
```bash
pytest tests/ -v
```

Regenerate the dataset (different seed, different row count, etc.):
```bash
python scripts/generate_synthetic_data.py
```

---

## 6. Retraining on Google Colab

1. In Colab: `!pip install scikit-learn pandas numpy joblib`
2. Upload `train_model.py` and `data/accidents_data.csv` (or clone your repo).
3. `!python train_model.py`
4. Download `model.pkl` and `model_metadata.json` from Colab's file browser.
5. Replace the two files in your local project root with the downloaded ones.
6. Commit + push (or redeploy) — Render will serve the new model immediately;
   no code changes needed since `train_model.py` and the API always agree
   on the feature contract.

> **Version note:** if Colab's scikit-learn version differs from
> `requirements.txt`, pin them to match — joblib-pickled sklearn pipelines
> are not guaranteed to load across major version jumps.

---

## 7. Deploying to Render

**Option A — Blueprint (`render.yaml`)**: Render dashboard → New → Blueprint
→ point at your repo → it reads `render.yaml` automatically.

**Option B — Manual web service**:
- Environment: `Python 3`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add an env var `CORS_ORIGINS` = your Vercel URL once the frontend is live.

**Option C — Docker**: Render environment = `Docker`; it will use the
included `Dockerfile` as-is.

First request after a free-tier cold start will take a few seconds longer
(startup loads the CSV + trains if needed) — subsequent requests are fast.

---

## 8. Environment variables

See `.env.example` for the full list with defaults. None are required to run
locally; on Render, set at minimum `CORS_ORIGINS` to your Vercel domain once
the frontend exists.

---

## 9. Limitations / future work

- Road network is a **synthetic, hand-designed graph** (11 roads, 61
  segments) representative of Peddapalli district, not an OSM/real-map
  extraction — swapping in a real road graph (e.g. via OSMnx) would be a
  natural upgrade and wouldn't require changing the route engine's interface.
- Explanations are rule-based (fast, deterministic) rather than SHAP-based;
  SHAP would give exact per-feature attributions at the cost of extra
  compute per request — worth adding if you move past a free-tier deploy.
- `has_curve` / `has_intersection` are fixed per road segment (a real
  physical property); everything else (weather, traffic, peak hour) is
  situational and supplied per request.

---

## License

MIT — see `LICENSE`.
