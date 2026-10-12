"""Real track geometry: route every modeled edge along OpenStreetMap rails.

The station graph only knows each edge's two endpoint coordinates, so drawn
naively every segment is a straight chord -- which cuts corners on every
curve (Mahim, Kurla, Thane creek, the Vashi bridge approach). This module
fetches Mumbai's actual railway centerlines from OpenStreetMap once (Overpass
API, cached to data/raw/osm_rail.json), builds a track-level graph from
them, snaps each station to its nearest track node, and runs Dijkstra on
the *track* graph between the two snapped nodes. The result is a polyline
that follows the physical alignment, plus its true along-track length.

That length is also an analytics output in its own right: track_km /
distance_km is the segment's *circuity* (how much the real alignment
detours relative to the straight line), which the dashboard reports.

Everything degrades gracefully: no cache + no network -> straight chords,
exactly the previous behaviour, with a warning.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import config

logger = config.get_logger(__name__)

# Generous bbox around the modeled network, outer branches included.
_BBOX = (18.70, 72.70, 20.05, 73.45)  # south, west, north, east (Dahanu, Kasara, Khopoli included)
_OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
# Mainline running tracks only -- yards/sidings would let Dijkstra take
# shortcuts through depots that no passenger train uses.
_QUERY = (
    "[out:json][timeout:180];"
    'way["railway"="rail"]["service"!~"yard|siding|spur|crossover"]({s},{w},{n},{e});'
    "(._;>;);out skel qt;"
)
_SNAP_MAX_M = 2500.0      # some outer-branch source coords are ~2 km off; beyond this, do not snap
_MAX_CIRCUITY = 2.2       # a track path this much longer than the chord is rejected


def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine in metres (accepts scalars or numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * config.EARTH_RADIUS_KM * 1000 * np.arcsin(np.sqrt(a))


def _ssl_trust_os_store() -> None:
    """Use the OS certificate store when `truststore` is installed.

    Campus/corporate networks often re-sign HTTPS with a local root CA that
    Windows trusts but Python's bundled CA list does not; truststore fixes
    that *without* disabling verification.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def fetch_osm_rail(force: bool = False) -> dict | None:
    """Return raw Overpass JSON for Mumbai's rail network, cached on disk."""
    if config.OSM_RAIL_JSON.exists() and not force:
        return json.loads(config.OSM_RAIL_JSON.read_text(encoding="utf-8"))

    _ssl_trust_os_store()
    s, w, n, e = _BBOX
    body = urllib.parse.urlencode({"data": _QUERY.format(s=s, w=w, n=n, e=e)}).encode()
    headers = {"User-Agent": "mumbai-transit-project/1.0 (academic analytics; track geometry)"}

    for attempt in range(6):
        url = _OVERPASS_ENDPOINTS[attempt % len(_OVERPASS_ENDPOINTS)]
        try:
            logger.info("Fetching OSM rail geometry from %s (attempt %d)...", url, attempt + 1)
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=200) as resp:
                data = json.loads(resp.read())
            config.OSM_RAIL_JSON.write_text(json.dumps(data), encoding="utf-8")
            logger.info("Cached %d OSM elements to %s", len(data.get("elements", [])), config.OSM_RAIL_JSON.name)
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Overpass request failed: %s", exc)
            time.sleep(min(30, 4 * (attempt + 1)))
    logger.warning("Could not fetch OSM rail data -- map will fall back to straight segments.")
    return None


def _build_track_graph(osm: dict) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    graph = nx.Graph()
    for el in osm["elements"]:
        if el["type"] != "way":
            continue
        refs = [r for r in el["nodes"] if r in nodes]
        for a, b in zip(refs, refs[1:]):
            (la, lo), (lb, lob) = nodes[a], nodes[b]
            graph.add_edge(a, b, w=float(_haversine_m(la, lo, lb, lob)))
    nx.set_node_attributes(graph, {nid: nodes[nid] for nid in graph.nodes}, "xy")
    ids = np.array([nid for nid in graph.nodes])
    coords = np.array([nodes[nid] for nid in ids])

    # Parallel running lines (fast/slow, up/down) are separate OSM ways that
    # only meet at mapped crossovers, which the query excludes. Without a
    # link, two neighbouring stations snapped onto different parallel tracks
    # would be routed via a junction kilometres away. Bridge any nodes within
    # 25 m of each other with a small penalty so paths prefer staying on one
    # track but can hop across where real trains would.
    k = math.cos(math.radians(19.1))
    tree = cKDTree(np.column_stack([coords[:, 0], coords[:, 1] * k]))
    for i, j in tree.query_pairs(r=25 / 111_320):
        a, b = ids[i], ids[j]
        if not graph.has_edge(a, b):
            d = float(_haversine_m(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]))
            graph.add_edge(a, b, w=d * 1.5 + 5)

    # Drop isolated fragments (dead-end stubs mapped without connections).
    main = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()
    ids = np.array([nid for nid in graph.nodes])
    coords = np.array([nodes[nid] for nid in ids])
    return graph, ids, coords


def compute_track_paths(
    stations_df: pd.DataFrame, edges_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route every edge along real track.

    Returns two frames:
      * edges: edges_df + `path` ([lat, lon] list), `track_km`, `circuity`, `on_track`
      * snaps: one row per station -- `snap_lat`, `snap_lon`, `snap_m` (how far the
        recorded coordinate sits from the nearest running track; a data-quality
        signal: anything over ~150 m means the source coordinate is imprecise).

    Only the OSM download is cached; routing takes about a second, so it is
    simply recomputed every run.
    """
    out = edges_df.copy()
    coords = stations_df.set_index("station_id")[["latitude", "longitude"]]
    snaps = pd.DataFrame({
        "station_id": coords.index, "snap_lat": coords["latitude"].values,
        "snap_lon": coords["longitude"].values, "snap_m": np.nan,
    }).set_index("station_id")

    def chord(row):
        a, b = snaps.loc[row.source_id], snaps.loc[row.target_id]
        return [[a.snap_lat, a.snap_lon], [b.snap_lat, b.snap_lon]]

    osm = fetch_osm_rail()
    if not osm:
        out["path"] = [chord(r) for r in out.itertuples(index=False)]
        out["track_km"] = out["distance_km"]
        out["circuity"] = 1.0
        out["on_track"] = False
        return out, snaps.reset_index()

    graph, ids, node_coords = _build_track_graph(osm)
    # KD-tree in a locally-scaled lat/lon space (lon shrunk by cos(lat)) is
    # accurate to well under a metre at Mumbai's latitude for snapping.
    k = math.cos(math.radians(19.1))
    tree = cKDTree(np.column_stack([node_coords[:, 0], node_coords[:, 1] * k]))

    snapped_idx = {}
    for sid, row in coords.iterrows():
        _, idx = tree.query([row.latitude, row.longitude * k])
        lat, lon = node_coords[idx]
        dist_m = float(_haversine_m(row.latitude, row.longitude, lat, lon))
        snaps.loc[sid, "snap_m"] = round(dist_m)
        if dist_m <= _SNAP_MAX_M:
            snapped_idx[sid] = idx
            snaps.loc[sid, ["snap_lat", "snap_lon"]] = [lat, lon]

    paths, track_km, on_track = [], [], []
    for row in out.itertuples(index=False):
        ia, ib = snapped_idx.get(row.source_id), snapped_idx.get(row.target_id)
        path, km, ok = chord(row), row.distance_km, False
        if ia is not None and ib is not None and ia != ib:
            try:
                # Several tracks run side by side (fast/slow, freight corridor);
                # the single nearest node may sit on one not linked to the
                # neighbour's. Let Dijkstra choose among every track node near
                # each station via virtual source/sink nodes.
                g = graph
                src, dst = ("_src",), ("_dst",)
                for tag, sid, idx in ((src, row.source_id, ia), (dst, row.target_id, ib)):
                    for c in tree.query_ball_point(node_coords[idx] * [1, k], r=(80 / 111_320)):
                        off = float(_haversine_m(*node_coords[idx], *node_coords[c]))
                        g.add_edge(tag, ids[c], w=off * 2)
                node_path = nx.shortest_path(g, src, dst, weight="w")[1:-1]
                g.remove_nodes_from([src, dst])
                pts = np.array([graph.nodes[n]["xy"] for n in node_path])
                length_km = float(_haversine_m(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]).sum()) / 1000
                if length_km <= max(row.distance_km, 0.3) * _MAX_CIRCUITY:
                    path, km, ok = pts.round(6).tolist(), round(length_km, 3), True
            except nx.NetworkXNoPath:
                graph.remove_nodes_from([("_src",), ("_dst",)])
        paths.append(path)
        track_km.append(km)
        on_track.append(ok)

    out["path"] = paths
    out["track_km"] = track_km
    # Circuity vs. the chord between the *snapped* endpoints, so imprecise
    # source coordinates do not masquerade as straight (or impossible <1) track.
    chord_km = [float(_haversine_m(p[0][0], p[0][1], p[-1][0], p[-1][1])) / 1000 for p in paths]
    out["circuity"] = (out["track_km"] / np.clip(chord_km, 0.05, None)).round(3)
    out["on_track"] = on_track
    logger.info("Track-following geometry for %d/%d edges (rest fall back to chords).", sum(on_track), len(out))
    return out, snaps.reset_index()
