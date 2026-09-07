"""Central configuration: filesystem paths, constants and logging.

Keeping every path and magic number in one module means the rest of the
codebase never hardcodes a string literal for a file location, a CRS code,
or a modeling assumption -- change it once here and it propagates everywhere.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = BASE_DIR / "output"
MAPS_DIR: Path = OUTPUT_DIR / "maps"

for _dir in (RAW_DIR, PROCESSED_DIR, MAPS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Raw (source-of-truth extract) files
STATIONS_RAW_CSV: Path = RAW_DIR / "stations_raw.csv"
EDGES_RAW_CSV: Path = RAW_DIR / "edges_raw.csv"

# Cleaned / enriched files
STATIONS_CLEAN_CSV: Path = PROCESSED_DIR / "stations_clean.csv"
EDGES_CLEAN_CSV: Path = PROCESSED_DIR / "edges_clean.csv"

# Spatial exports (QGIS-ready)
STATIONS_GEOJSON: Path = PROCESSED_DIR / "stations.geojson"
RAIL_LINES_GEOJSON: Path = PROCESSED_DIR / "rail_lines.geojson"
BUFFERS_GEOJSON: Path = PROCESSED_DIR / "interchange_buffers.geojson"

# Neo4j ingestion script
CYPHER_SCRIPT: Path = PROCESSED_DIR / "neo4j_ingest.cypher"

# HTML outputs
MAP_HTML: Path = MAPS_DIR / "mumbai_rail_map.html"
DASHBOARD_HTML: Path = OUTPUT_DIR / "dashboard.html"

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------
# Mean Earth radius (km) used for the haversine great-circle formula.
EARTH_RADIUS_KM: float = 6371.0

# Documented average Mumbai suburban EMU speed *including station halts*,
# used to derive travel_time_mins from a real haversine distance for the
# curated (non-adjacent-in-reality) hops in this dataset. Actual scheduled
# timetables vary by service type (fast/slow/AC) -- see README "Methodology".
AVG_SUBURBAN_SPEED_KMH: float = 38.0
MIN_TRAVEL_TIME_MINS: int = 2  # floor, so adjacent curated stops never round to 0

# Pedestrian catchment radius used for interchange buffer zones (metres).
BUFFER_RADIUS_M: float = 1000.0

# CRS codes: WGS84 for storage/display, UTM 43N (covers Mumbai, 72-78 deg E)
# for any operation that needs true planar distance (buffering!).
GEOGRAPHIC_CRS: str = "EPSG:4326"
PROJECTED_CRS: str = "EPSG:32643"

# Official Mumbai Suburban Railway map colours (Western=green, Central=red,
# Harbour=blue). Trans-Harbour has no single universally-published colour on
# CR/WR diagrams, so amber is used purely for on-map legibility.
LINE_COLORS: dict[str, str] = {
    "Western": "#22C55E",
    "Central": "#EF4444",
    "Harbour": "#3B82F6",
    "Trans-Harbour": "#F59E0B",
    "Interchange": "#FFD24C",
}

DEFAULT_ORIGIN: str = "VIR"        # Virar
DEFAULT_DESTINATION: str = "PNV"   # Panvel

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
