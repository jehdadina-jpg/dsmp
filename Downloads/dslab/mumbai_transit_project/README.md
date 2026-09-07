# Mumbai Suburban Rail Network — Efficiency & Accessibility Tracker

A Pandas + Graph + GIS analytics project over a **real, curated subset of the
Mumbai Suburban Railway** (Western, Central, Harbour, plus one real
Trans-Harbour link). It builds a weighted network graph, algorithmically
surfaces the busiest interchange hubs, finds optimal routes across lines,
models pedestrian accessibility around interchanges, and renders both an
interactive dark-mode map and a standalone analytics dashboard.

> **Real data, not simulated.** Every station is a real, named station with
> coordinates sourced from its own Wikipedia infobox, and every edge is a
> real adjacency (or, for Thane↔Vashi, a real direct Trans-Harbour service).
> See [Data Sources & Methodology](#data-sources--methodology) for exactly
> what is cited fact vs. documented estimate.

---

## What it does

| Layer | Tooling | Output |
|---|---|---|
| Data engineering | pandas, numpy | cleaned, haversine-enriched station/edge tables |
| Graph analytics | networkx | shortest routes (by time or distance), degree & betweenness centrality, Neo4j Cypher export |
| Spatial / GIS | geopandas, shapely, pyproj | 1 km pedestrian buffers around interchange hubs, QGIS-ready GeoJSON |
| Visualization | folium, three.js | interactive dark-mode Leaflet map + a standalone WebGL analytics dashboard |

## Project structure

```
mumbai_transit_project/
├── data/
│   ├── raw/                 # curated real source extract (generated on first run)
│   └── processed/           # cleaned CSVs + GeoJSON layers + Cypher script
├── src/
│   ├── config.py            # paths, constants, logging (single source of truth)
│   ├── data_loader.py       # real dataset + pandas cleaning + haversine
│   ├── graph_analytics.py   # networkx graph, routing, centrality, Cypher/Neo4j
│   ├── spatial_analysis.py  # geopandas buffers, GeoJSON export, Folium map
│   └── dashboard.py         # standalone HTML dashboard (presentation only)
├── output/
│   ├── maps/mumbai_rail_map.html
│   └── dashboard.html
├── tests/                   # pytest sanity checks (data shape, routing, centrality)
├── main.py                  # interactive CLI + `--full` non-interactive mode
├── requirements.txt
├── requirements-dev.txt     # + pytest
└── README.md
```

`data/`, `tests/`, `requirements-dev.txt` and `src/config.py` go beyond the
original brief's file list — they're the difference between "a script that
works" and a project that's safe to hand to someone else. Everything asked
for in the brief is still exactly where it was specified.

## Setup

Requires **Python 3.11–3.13** (geopandas's compiled dependencies — pyproj,
shapely, fiona — lag behind the very newest CPython releases; 3.14 did not
have wheels available at the time this was built).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
# Interactive menu: pick actions one at a time
python main.py

# Non-interactive: run the whole pipeline once (what CI/grading should call)
python main.py --full

# Override the example route (defaults to Virar -> Panvel)
python main.py --full --origin Andheri --destination Kalyan

# Also push the graph into a live Neo4j instance
python main.py --full --push-neo4j --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password <pw>
```

Running the full pipeline produces:

* `data/processed/stations_clean.csv`, `edges_clean.csv` — cleaned, enriched tables
* `data/processed/stations.geojson`, `rail_lines.geojson`, `interchange_buffers.geojson` — open these directly in QGIS
* `data/processed/neo4j_ingest.cypher` — a ready-to-run Cypher script (`cypher-shell -f neo4j_ingest.cypher`)
* `output/maps/mumbai_rail_map.html` — interactive dark-mode map
* `output/dashboard.html` — the analytics dashboard (open this one first)

### Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Neo4j (optional)

`generate_cypher_queries` / `export_cypher_script` need no running database —
they just emit a `.cypher` file. Loading it into an actual graph database is
optional and separate:

1. Run a local instance: `docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/your_password neo4j:5` (or use Neo4j Desktop).
2. `python main.py --full --push-neo4j --neo4j-password your_password`, or set
   `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` env vars instead of flags.
3. Statements use `MERGE ... ON CREATE SET` rather than a bare `CREATE`, so
   re-running the ingestion is idempotent instead of duplicating every node —
   see the comment in `graph_analytics.generate_cypher_queries` for why.
4. No Neo4j instance was available in the environment this was built in, so
   Cypher *generation* is tested (`tests/test_graph_analytics.py`), but the
   live `push_to_neo4j` path could not be exercised against a real server —
   verify it against your own instance before relying on it.

## Data Sources & Methodology

**Topology & line membership** — Wikipedia: [Western line](https://en.wikipedia.org/wiki/Western_line_(Mumbai_Suburban_Railway)),
[Central line](https://en.wikipedia.org/wiki/Central_line_(Mumbai_Suburban_Railway)),
[Harbour line](https://en.wikipedia.org/wiki/Harbour_line_(Mumbai_Suburban_Railway)),
[List of Mumbai Suburban Railway stations](https://en.wikipedia.org/wiki/List_of_Mumbai_Suburban_Railway_stations).
Line colors (Western=green, Central=red, Harbour=blue) match the commonly
published Mumbai Suburban Railway route map convention; Trans-Harbour has no
single universally-published color, so amber was chosen purely for map
legibility.

**Coordinates** — each station's own Wikipedia infobox (WGS84), fetched
individually per station. Precision matches what each infobox publishes
(most to ~1m, a couple — Sion, Vikhroli — only to ~1km).

**Scope decision** — this project curates **25 major/interchange stations**
rather than all ~150 suburban stops, so the graph stays tractable for a
mini-project. Edges connect consecutive *modeled* stations in real
geographic order, which means some edges are "express" hops that skip real
intermediate stops (e.g. Churchgate→Mumbai Central skips the real Marine
Lines/Charni Road/Grant Road stations). One genuine Trans-Harbour edge
(Thane↔Vashi) was added beyond the three named lines because it is a real,
direct passenger service and is exactly what makes Thane a true multi-line
junction — without it, Thane would appear far less central than it actually
is, which would work against the project's own stated goal of
algorithmically finding hubs like Thane.

**distance_km** — a real haversine great-circle calculation over the
coordinates above (see `data_loader._haversine_km`), computed once per edge
for the two *modeled* endpoints. It is exact for that straight-line
computation, but is not the official railway-timetable distance for the
segment (real track curvature typically adds 5–15% versus straight-line
distance).

**travel_time_mins** — derived as `distance_km / 38 km/h`, where 38 km/h is
a documented approximation of average Mumbai suburban EMU speed including
station halts. This is a modeling assumption, not a scraped timetable — see
`config.AVG_SUBURBAN_SPEED_KMH`. Swap in real CR/WR timetable data for
production use.

**daily_footfall** — two values are directly cited public figures:
**Kalyan ≈ 800,000/day** (also FY24's highest official annual outward
footfall in India, 83.7M) and **Borivali ≈ 287,000/day** (the busiest
station on the Western Line). Every other station's figure is an analytical
estimate, ordered to be consistent with that station's documented
real-world role (terminus / major interchange / major hub / standard stop)
and with published line-level ridership totals (Western ≈2.66M/day,
Central ≈3.9M/day, network-wide ≈7.5M/day across ~150 stations) — **these
are not official station-wise counts**, because Indian Railways does not
publish per-station ridership as a single open dataset. `data_loader.py`'s
`_STATION_RECORDS` is the one place to replace these with an authoritative
source (an RTI disclosure, a GTFS feed, or a railway open-data API) — every
downstream module (graph, buffers, map, dashboard) consumes whatever is in
the processed CSV and does not care where the numbers came from.

**Interchange hub flag** — `interchange_hub=True` iff a station joins ≥2
lines *within this project's modeled network*. Real-world junctions outside
that scope (e.g. Kalyan's Kasara/Khopoli lines) are intentionally not
flagged, since the graph in this project doesn't model those lines either.

## Design notes

* **Undirected graph** — suburban trains run both directions on the same
  station pairs, so there's no one-way semantic to preserve (unlike, say,
  one-way street routing).
* **Buffering in a projected CRS** — the 1 km pedestrian buffers are computed
  after reprojecting to UTM zone 43N (`EPSG:32643`), not in raw lat/long,
  because buffering in degrees would distort a circle into an ellipse away
  from the equator. They're reprojected back to WGS84 afterward for
  Folium/Leaflet, which only understands lat/long.
* **`src/dashboard.py` has no networkx/geopandas import** — it only takes
  already-computed DataFrames and dicts, so the presentation layer can be
  restyled without touching analytics code, and vice versa.
* **Frontend stack** — the dashboard's 3D graph is vanilla three.js (loaded
  from cdnjs) driven by a real `networkx.spring_layout(dim=3)` computed in
  Python, not a placeholder animation. (Threlte, which was suggested during
  scoping, wraps three.js in a Svelte component-and-build pipeline; adding a
  second, JS-framework build toolchain purely for one decorative element
  would work against "production ready" for what is fundamentally a Python
  data project, so vanilla three.js gets the same visual result without it.
  Jitter is a browser-based motion-design tool with no API to call
  programmatically — the route-finder's "train" animation and the
  leaderboard's reveal motion are hand-authored SVG/CSS instead.)

## Known limitations

* 25-station curated network, not the full ~150-station suburban system.
* `daily_footfall` is mostly modeled, not officially sourced per-station (see above).
* `push_to_neo4j` is implemented against the documented driver API but was not tested against a live server in this environment.
* Travel times are speed-model estimates, not scraped timetables.
