"""
app/routing/graph_builder.py

Turns data/road_segments.geojson into:
  1. A list of `Segment` records (with a shapely LineString each), used for
     nearest-road lookups (e.g. "what road is closest to this lat/lng?").
  2. A networkx.Graph you can run shortest-path / k-shortest-paths on.

COORDINATE CONVENTION
----------------------
GeoJSON stores coordinates as [longitude, latitude]. Everywhere else in
this codebase (schemas, the ML model, the CSV) we use (latitude, longitude)
because that is the order humans read map coordinates in. This file is the
ONE place that does the [lon, lat] -> (lat, lon) conversion, so the rest of
the app never has to think about it.

GRAPH NODE IDENTITY
-------------------
Two segments that share a physical waypoint must produce the exact same
node key so the graph is actually connected there. We round every
coordinate to 6 decimal places (about 11 cm of precision) before using it
as a dict/graph key, which absorbs floating-point noise from the
generation step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
from shapely.geometry import LineString, Point

logger = logging.getLogger("accident_api.graph_builder")

COORD_PRECISION = 6


def _node_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, COORD_PRECISION), round(lon, COORD_PRECISION))


@dataclass
class Segment:
    segment_id: str
    road_name: str
    road_type: str
    num_lanes: int
    length_km: float
    has_curve: bool
    has_intersection: bool
    start: tuple[float, float]  # (lat, lon)
    end: tuple[float, float]    # (lat, lon)
    line: LineString            # shapely, built in (lon, lat) x/y order


class RoadNetwork:
    """Everything about the road network the rest of the app needs:
    per-segment static attributes, a routable graph, and nearest-segment
    lookups."""

    def __init__(self, geojson: dict):
        self.segments: list[Segment] = []
        self.by_id: dict[str, Segment] = {}
        self.graph: nx.Graph = nx.Graph()

        for feature in geojson["features"]:
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]  # [[lon,lat], [lon,lat]]
            start = (coords[0][1], coords[0][0])
            end = (coords[-1][1], coords[-1][0])

            seg = Segment(
                segment_id=props["segment_id"],
                road_name=props["road_name"],
                road_type=props["road_type"],
                num_lanes=int(props["num_lanes"]),
                length_km=float(props["length_km"]),
                has_curve=bool(props["has_curve"]),
                has_intersection=bool(props["has_intersection"]),
                start=start,
                end=end,
                line=LineString(coords),  # shapely uses (x=lon, y=lat)
            )
            self.segments.append(seg)
            self.by_id[seg.segment_id] = seg

            u, v = _node_key(*start), _node_key(*end)
            self.graph.add_node(u)
            self.graph.add_node(v)
            # Parallel edges (two different roads sharing endpoints) are rare
            # here, but if one exists, keep the shorter/first one simple by
            # just overwriting -- fine for a synthetic network this size.
            self.graph.add_edge(u, v, segment_id=seg.segment_id)

        n_components = nx.number_connected_components(self.graph)
        if n_components > 1:
            logger.warning(
                "Road graph has %d disconnected components -- some routes "
                "may be unreachable.", n_components
            )
        logger.info(
            "Built road graph: %d nodes, %d edges, %d segments, %d component(s)",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
            len(self.segments), n_components,
        )

    # ------------------------------------------------------------------
    def nearest_segment(self, lat: float, lon: float) -> tuple[Segment, float]:
        """Return (closest Segment, distance_km) to an arbitrary point.
        Brute-force over ~60-100 segments is trivial at this scale --
        no spatial index needed."""
        point = Point(lon, lat)  # shapely: (x=lon, y=lat)
        best_seg, best_deg = None, float("inf")
        for seg in self.segments:
            d = seg.line.distance(point)  # degrees (good enough to *rank* candidates)
            if d < best_deg:
                best_deg, best_seg = d, seg
        # Convert the winning candidate's degree-distance to km using the
        # nearest point on the line, so the reported distance is accurate
        # (not just "smallest in degrees", which can be a poor proxy for km
        # near non-equatorial latitudes -- negligible here, but we do it
        # properly anyway).
        nearest_point_on_line = best_seg.line.interpolate(best_seg.line.project(point))
        km = _haversine_km(lat, lon, nearest_point_on_line.y, nearest_point_on_line.x)
        return best_seg, km

    def nearest_node(self, lat: float, lon: float) -> tuple[float, float]:
        """Snap an arbitrary point to the nearest graph node (used to turn
        an origin/destination lat-lng into a routable graph node)."""
        best_node, best_d = None, float("inf")
        for node in self.graph.nodes:
            d = _haversine_km(lat, lon, node[0], node[1])
            if d < best_d:
                best_d, best_node = d, node
        return best_node

    def node_for_place(self, node_id_lat: float, node_id_lon: float) -> tuple[float, float]:
        return _node_key(node_id_lat, node_id_lon)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


_road_network: RoadNetwork | None = None


def get_road_network(geojson: dict | None = None) -> RoadNetwork:
    global _road_network
    if _road_network is None:
        if geojson is None:
            raise RuntimeError("RoadNetwork not initialised yet -- pass geojson on first call")
        _road_network = RoadNetwork(geojson)
    return _road_network
