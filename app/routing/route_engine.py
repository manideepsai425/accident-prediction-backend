"""
app/routing/route_engine.py

Implements the "safest route" feature end to end:

  1. Snap origin/destination to the road graph.
  2. Ask networkx for several *distinct* candidate paths (Yen's algorithm,
     via nx.shortest_simple_paths) ordered by distance.
  3. Score every segment of every candidate with the trained ML model.
  4. Prefer a candidate with NO segment above the high-risk threshold
     (0.7). Only if every candidate has at least one such segment do we
     fall back to "fewest high-risk segments, then lowest overall risk" --
     and in that case we explicitly flag the unavoidable segment(s), per
     the brief.
  5. Package the result as one safest route + a few labelled alternatives.

WHY YEN'S ALGORITHM (nx.shortest_simple_paths)
-----------------------------------------------
A plain shortest-path call only ever gives you ONE route. Yen's algorithm
generates simple paths (no repeated nodes) in strictly increasing order of
total weight, which is exactly "give me several genuinely different ways
to get from A to B, cheapest first" -- the natural way to get real
alternatives out of a graph instead of writing custom detour logic.
"""

from __future__ import annotations

import logging
from itertools import islice

import networkx as nx

from app.config import risk_label, settings
from app.ml.predictor import RiskPredictor, infer_is_peak_hour, infer_traffic_density
from app.routing.graph_builder import RoadNetwork, Segment

logger = logging.getLogger("accident_api.route_engine")

# Rough average speeds by road type, used only for the est_travel_time_min
# estimate shown to the user -- not used anywhere in the risk calculation.
AVG_SPEED_KMH = {"Highway": 55.0, "Arterial": 35.0, "Local": 25.0}

MAX_CANDIDATE_PATHS = 8


class RouteNotFoundError(Exception):
    pass


class RouteEngine:
    def __init__(self, road_network: RoadNetwork, predictor: RiskPredictor):
        self.road_network = road_network
        self.predictor = predictor

    # ------------------------------------------------------------------
    def _edge_weight_factory(self):
        by_id = self.road_network.by_id

        def weight(u, v, data) -> float:
            return by_id[data["segment_id"]].length_km

        return weight

    def _candidate_paths(self, src, dst) -> list[list]:
        try:
            gen = nx.shortest_simple_paths(self.road_network.graph, src, dst, weight=self._edge_weight_factory())
            return list(islice(gen, MAX_CANDIDATE_PATHS))
        except nx.NetworkXNoPath:
            return []

    def _segments_for_path(self, path: list) -> list[Segment]:
        segs = []
        for u, v in zip(path[:-1], path[1:]):
            seg_id = self.road_network.graph[u][v]["segment_id"]
            segs.append(self.road_network.by_id[seg_id])
        return segs

    def _score_segment(self, seg: Segment, preferred_time: str, weather_condition: str) -> float:
        mid_lat = (seg.start[0] + seg.end[0]) / 2
        mid_lon = (seg.start[1] + seg.end[1]) / 2
        is_peak = infer_is_peak_hour(preferred_time)
        traffic = infer_traffic_density(seg.road_type, is_peak)
        features = self.predictor.assemble_features(
            latitude=mid_lat, longitude=mid_lon, time_of_day=preferred_time,
            weather_condition=weather_condition, traffic_density=traffic,
            is_peak_hour=is_peak, segment=seg,
        )
        return self.predictor.predict_from_features(features)

    def _build_candidate(self, path: list, preferred_time: str, weather_condition: str) -> dict:
        segs = self._segments_for_path(path)
        seg_scores = [self._score_segment(s, preferred_time, weather_condition) for s in segs]
        total_km = sum(s.length_km for s in segs) or 1e-6
        overall_risk = sum(sc * s.length_km for sc, s in zip(seg_scores, segs)) / total_km
        travel_time_min = sum((s.length_km / AVG_SPEED_KMH[s.road_type]) * 60 for s in segs)
        high_risk_n = sum(1 for sc in seg_scores if sc > settings.risk_high_threshold)
        medium_risk_n = sum(
            1 for sc in seg_scores
            if settings.risk_medium_threshold < sc <= settings.risk_high_threshold
        )
        return {
            "path": path, "segments": segs, "segment_scores": seg_scores,
            "total_km": total_km, "overall_risk": overall_risk,
            "travel_time_min": travel_time_min,
            "high_risk_n": high_risk_n, "medium_risk_n": medium_risk_n,
        }

    # ------------------------------------------------------------------
    def find_routes(
        self, *, origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float,
        preferred_time: str, weather_condition: str, max_alternatives: int = 3,
    ) -> dict:
        src = self.road_network.nearest_node(origin_lat, origin_lng)
        dst = self.road_network.nearest_node(dest_lat, dest_lng)
        if src == dst:
            raise RouteNotFoundError("Origin and destination resolve to the same point on the road network.")

        raw_paths = self._candidate_paths(src, dst)
        if not raw_paths:
            raise RouteNotFoundError("No route could be found between these two points on the known road network.")

        candidates = [self._build_candidate(p, preferred_time, weather_condition) for p in raw_paths]

        # Prefer candidates with zero segments above the high-risk threshold;
        # only fall back to "fewest high-risk segments" if none exist.
        clean = [c for c in candidates if c["high_risk_n"] == 0]
        pool = clean if clean else candidates
        pool_sorted = sorted(pool, key=lambda c: (c["high_risk_n"], c["overall_risk"]))
        safest = pool_sorted[0]

        shortest_by_distance = min(candidates, key=lambda c: c["total_km"])

        # Build the alternative pool: every other candidate, ranked by risk,
        # but make sure the shortest/fastest option is represented so the
        # user actually sees a "faster but riskier" tradeoff when one exists.
        remaining = [c for c in candidates if c is not safest]
        remaining_sorted = sorted(remaining, key=lambda c: c["overall_risk"])
        alt_pool: list[dict] = []
        if shortest_by_distance is not safest:
            alt_pool.append(shortest_by_distance)
        for c in remaining_sorted:
            if c not in alt_pool and len(alt_pool) < max_alternatives:
                alt_pool.append(c)
        alt_pool = alt_pool[:max_alternatives]

        warnings: list[str] = []
        safest_option = self._to_route_option(
            safest, rank=1, route_id="route_1", is_safest=True,
            is_shortest=(safest is shortest_by_distance),
            label_prefix="Route 1: Safest",
        )
        if safest["high_risk_n"] > 0:
            warnings.append(
                f"Even the safest available route has {safest['high_risk_n']} high-risk "
                f"segment(s) (risk > {settings.risk_high_threshold}) that could not be avoided "
                "given the known road network -- see the flagged segments below."
            )

        alt_options = []
        for i, c in enumerate(alt_pool, start=2):
            if c is shortest_by_distance and c["overall_risk"] > safest["overall_risk"]:
                prefix = f"Route {i}: Faster"
            else:
                prefix = f"Route {i}: Alternative"
            alt_options.append(self._to_route_option(
                c, rank=i, route_id=f"route_{i}", is_safest=False,
                is_shortest=(c is shortest_by_distance), label_prefix=prefix,
            ))

        explanation = self._build_explanation(safest, safest_option, candidates)

        return {
            "origin": (origin_lat, origin_lng),
            "destination": (dest_lat, dest_lng),
            "safest_route": safest_option,
            "alternatives": alt_options,
            "warnings": warnings,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    def _to_route_option(self, c: dict, *, rank: int, route_id: str, is_safest: bool, is_shortest: bool, label_prefix: str) -> dict:
        seg_infos = []
        for seg, score in zip(c["segments"], c["segment_scores"]):
            level = risk_label(score)
            warning = None
            if score > settings.risk_high_threshold:
                warning = f"High-risk stretch on {seg.road_name} -- consider extra caution or an alternative time."
            elif score > settings.risk_medium_threshold:
                warning = f"Moderate risk on {seg.road_name}."
            seg_infos.append({
                "segment_id": seg.segment_id, "road_name": seg.road_name,
                "start": seg.start, "end": seg.end, "length_km": round(seg.length_km, 3),
                "risk_score": round(score, 4), "risk_level": level, "warning": warning,
            })

        coordinates = [c["segments"][0].start] + [s.end for s in c["segments"]]
        roads_touched = list(dict.fromkeys(s.road_name for s in c["segments"]))
        summary = (
            f"{round(c['total_km'], 1)} km via " + ", ".join(roads_touched[:3])
            + ("..." if len(roads_touched) > 3 else "")
        )

        return {
            "route_id": route_id,
            "label": f"{label_prefix} - Risk {c['overall_risk']:.2f}",
            "rank": rank,
            "coordinates": coordinates,
            "overall_risk_score": round(c["overall_risk"], 4),
            "total_distance_km": round(c["total_km"], 2),
            "est_travel_time_min": round(c["travel_time_min"], 1),
            "segments": seg_infos,
            "high_risk_segment_count": c["high_risk_n"],
            "medium_risk_segment_count": c["medium_risk_n"],
            "is_safest": is_safest,
            "is_shortest": is_shortest,
            "summary": summary,
        }

    @staticmethod
    def _build_explanation(safest: dict, safest_option: dict, all_candidates: list[dict]) -> str:
        roads = ", ".join(dict.fromkeys(s.road_name for s in safest["segments"]))
        parts = [
            f"Compared {len(all_candidates)} candidate route(s); selected the one via {roads} "
            f"as safest with an overall risk of {safest['overall_risk']:.2f}."
        ]
        if safest["high_risk_n"] == 0:
            parts.append("This route has no segments above the high-risk threshold.")
        else:
            flagged = [s.road_name for s, sc in zip(safest["segments"], safest["segment_scores"])
                       if sc > 0.7]
            parts.append(
                "It still passes through " + ", ".join(dict.fromkeys(flagged)) +
                " where risk could not be fully avoided -- see flagged segments for details."
            )
        return " ".join(parts)
