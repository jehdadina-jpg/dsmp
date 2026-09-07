"""GeoPandas spatial analysis + a styled, interactive Folium map.

Covers the GIS half of the brief: turning tabular lat/long into real
GeoDataFrames, buffering interchange hubs to model pedestrian catchment
areas, exporting QGIS-ready GeoJSON, and rendering everything as a single
dark-themed interactive HTML map.
"""

from __future__ import annotations

import math

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from src import config

logger = config.get_logger(__name__)


def stations_to_geodataframe(stations_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Points, in WGS84 (lat/long) -- the natural CRS for GPS-style data."""
    gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
        crs=config.GEOGRAPHIC_CRS,
    )
    return gdf


def edges_to_geodataframe(edges_df: pd.DataFrame, stations_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """One LineString per edge, joining each pair of real station coordinates."""
    coords = stations_df.set_index("station_id")[["latitude", "longitude"]]

    def _line(row) -> LineString:
        lat1, lon1 = coords.loc[row.source_id]
        lat2, lon2 = coords.loc[row.target_id]
        return LineString([(lon1, lat1), (lon2, lat2)])  # shapely wants (x=lon, y=lat)

    geometry = [_line(row) for row in edges_df.itertuples(index=False)]
    gdf = gpd.GeoDataFrame(edges_df.copy(), geometry=geometry, crs=config.GEOGRAPHIC_CRS)
    return gdf


def compute_interchange_buffers(stations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """1 km pedestrian-catchment buffer polygons around every interchange hub.

    Buffering directly in EPSG:4326 (degrees) would distort the circle into
    an ellipse, because a degree of longitude is shorter than a degree of
    latitude everywhere off the equator -- so we reproject to a metric,
    Mumbai-appropriate CRS (UTM zone 43N), buffer in real metres, then
    reproject back to WGS84 so Folium/Leaflet (which only understands
    lat/long) can draw it.
    """
    hubs = stations_gdf[stations_gdf["interchange_hub"]].copy()
    if hubs.empty:
        logger.warning("No interchange hubs flagged -- returning an empty buffer layer.")
        return gpd.GeoDataFrame(columns=[*stations_gdf.columns, "buffer_area_km2"], geometry=[], crs=config.GEOGRAPHIC_CRS)

    projected = hubs.to_crs(config.PROJECTED_CRS)
    projected["geometry"] = projected.geometry.buffer(config.BUFFER_RADIUS_M)
    projected["buffer_area_km2"] = (projected.geometry.area / 1_000_000).round(3)

    buffers = projected.to_crs(config.GEOGRAPHIC_CRS)
    logger.info(
        "Computed %d interchange buffers (%.1f km2 total pedestrian-catchment area).",
        len(buffers), buffers["buffer_area_km2"].sum(),
    )
    return buffers


def export_geojson_layers(
    stations_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame, buffers_gdf: gpd.GeoDataFrame
) -> None:
    """Persist all three layers as GeoJSON for direct import into QGIS."""
    try:
        stations_gdf.to_file(config.STATIONS_GEOJSON, driver="GeoJSON")
        edges_gdf.to_file(config.RAIL_LINES_GEOJSON, driver="GeoJSON")
        if not buffers_gdf.empty:
            buffers_gdf.to_file(config.BUFFERS_GEOJSON, driver="GeoJSON")
        logger.info(
            "GeoJSON exported: %s, %s, %s",
            config.STATIONS_GEOJSON.name, config.RAIL_LINES_GEOJSON.name, config.BUFFERS_GEOJSON.name,
        )
    except OSError as exc:
        logger.error("Failed to write GeoJSON output: %s", exc)
        raise


def _line_color(line: str) -> str:
    """First matching colour for a (possibly multi-line, '/'-joined) station/edge."""
    for token in line.split("/"):
        if token in config.LINE_COLORS:
            return config.LINE_COLORS[token]
    return config.LINE_COLORS["Interchange"]


def _marker_radius(footfall: int, *, min_r: float = 4.0, max_r: float = 16.0) -> float:
    """Square-root scaling so *marker area*, not radius, tracks footfall."""
    scaled = math.sqrt(max(footfall, 0)) / math.sqrt(800_000) * max_r
    return max(min_r, min(max_r, scaled))


_THEME_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

  .leaflet-container { background:#0b0e14 !important; font-family:'Inter',sans-serif; }
  .leaflet-control-attribution { background:rgba(11,14,20,.75) !important; color:#8b93a7 !important; }
  .leaflet-control-attribution a { color:#c8cede !important; }
  .leaflet-control-zoom a {
    background:#151a24 !important; color:#e6e8ee !important; border:1px solid rgba(255,255,255,.08) !important;
  }
  .leaflet-control-zoom a:hover { background:#1f2633 !important; }
  .leaflet-control-layers {
    background:rgba(17,21,30,.92) !important; color:#e6e8ee !important;
    border:1px solid rgba(255,255,255,.08) !important; border-radius:12px !important;
    font-family:'Inter',sans-serif; backdrop-filter: blur(10px);
  }
  .leaflet-control-layers-list label { padding:2px 0; }

  .leaflet-popup-content-wrapper {
    background:#12161f; color:#e6e8ee; border-radius:14px;
    border:1px solid rgba(255,255,255,.08); box-shadow:0 12px 32px rgba(0,0,0,.55);
  }
  .leaflet-popup-tip { background:#12161f; }
  .station-card { font-family:'Inter',sans-serif; min-width:180px; }
  .station-card h4 {
    margin:0 0 6px; font-family:'Space Grotesk',sans-serif; font-size:15px; color:#fff;
  }
  .station-card .badge {
    display:inline-block; font-size:10px; font-weight:600; letter-spacing:.04em;
    padding:2px 8px; border-radius:999px; margin:2px 4px 2px 0; text-transform:uppercase;
  }
  .station-card .stat {
    font-family:'JetBrains Mono',monospace; font-size:13px; color:#9fe870; margin-top:4px;
  }
  .station-card .muted { color:#8b93a7; font-size:11px; margin-top:2px; }

  .map-hero {
    position:absolute; top:16px; left:16px; z-index:1000; max-width:300px;
    background:rgba(15,18,26,.82); backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.09); border-radius:18px; padding:16px 18px;
    box-shadow:0 20px 50px rgba(0,0,0,.5); font-family:'Inter',sans-serif; color:#e6e8ee;
  }
  .map-hero h1 {
    font-family:'Space Grotesk',sans-serif; font-size:16px; margin:0 0 4px; color:#fff; letter-spacing:-.01em;
  }
  .map-hero p { font-size:11.5px; color:#9099ad; margin:0 0 10px; line-height:1.5; }
  .map-hero-back {
    display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:600;
    color:#22F0FF; text-decoration:none; margin-bottom:10px; letter-spacing:.02em;
  }
  .map-hero-back:hover { text-decoration:underline; }
  .map-hero .legend-row { display:flex; align-items:center; gap:8px; font-size:12px; margin:5px 0; }
  .map-hero .dot { width:10px; height:10px; border-radius:50%; flex:none; box-shadow:0 0 8px currentColor; }
  .map-hero .dot.ring { background:transparent; border:2px solid currentColor; }

  .hub-marker {
    width:16px; height:16px; border-radius:50%; background:#FFD24C;
    border:2px solid #fff8e6; box-shadow:0 0 0 rgba(255,210,76,.6);
    animation: hub-pulse 2.2s ease-out infinite;
  }
  @keyframes hub-pulse {
    0%   { box-shadow:0 0 0 0 rgba(255,210,76,.55); }
    70%  { box-shadow:0 0 0 14px rgba(255,210,76,0); }
    100% { box-shadow:0 0 0 0 rgba(255,210,76,0); }
  }
</style>
"""


def build_folium_map(
    stations_df: pd.DataFrame, edges_df: pd.DataFrame, buffers_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    """Assemble the full interactive dark-mode map: lines, stations, buffers."""
    center_lat = stations_df["latitude"].mean()
    center_lon = stations_df["longitude"].mean()

    # Both folium's built-in "CartoDB dark_matter" alias and Carto's own
    # basemaps.cartocdn.com CDN now require an API key for anonymous use --
    # every unauthenticated tile request comes back stamped "API key
    # required" instead of map imagery. Standard OpenStreetMap tiles are
    # still free/keyless, so use those and fake the dark theme with a CSS
    # filter on the tile pane (a well-known no-key dark-map trick) instead.
    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap",
        control_scale=True,
    )
    fmap.get_root().html.add_child(folium.Element(
        "<style>.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) "
        "brightness(0.95) contrast(0.9) saturate(0.6);}</style>"
    ))

    coords = stations_df.set_index("station_id")

    line_groups: dict[str, folium.FeatureGroup] = {}
    for line_name in edges_df["line_name"].unique():
        line_groups[line_name] = folium.FeatureGroup(name=f"{line_name} Line", show=True)

    for row in edges_df.itertuples(index=False):
        a, b = coords.loc[row.source_id], coords.loc[row.target_id]
        locations = [(a.latitude, a.longitude), (b.latitude, b.longitude)]
        color = _line_color(row.line_name)
        group = line_groups[row.line_name]
        # Outer wide, low-opacity stroke first -> reads as a soft neon glow
        # under the crisp inner line once both are composited by the browser.
        folium.PolyLine(locations, color=color, weight=9, opacity=0.18).add_to(group)
        folium.PolyLine(
            locations, color=color, weight=3, opacity=0.95,
            tooltip=f"{row.line_name} Line: {a['name']} - {b['name']} "
                    f"({row.distance_km:.1f} km, {row.travel_time_mins:.0f} min)",
        ).add_to(group)

    for group in line_groups.values():
        group.add_to(fmap)

    stations_group = folium.FeatureGroup(name="Stations", show=True)
    hubs_group = folium.FeatureGroup(name="Interchange Hubs", show=True)

    for row in stations_df.itertuples(index=False):
        popup_html = f"""
        <div class="station-card">
          <h4>{row.name}</h4>
          <span class="badge" style="background:{_line_color(row.line)}22;color:{_line_color(row.line)};">
            {row.line}
          </span>
          {'<span class="badge" style="background:#FFD24C22;color:#FFD24C;">Interchange Hub</span>' if row.interchange_hub else ''}
          <div class="stat">{row.daily_footfall:,} riders / day</div>
          <div class="muted">{row.latitude:.4f}, {row.longitude:.4f}</div>
        </div>
        """
        if row.interchange_hub:
            folium.Marker(
                location=[row.latitude, row.longitude],
                icon=folium.DivIcon(html='<div class="hub-marker"></div>', icon_size=(16, 16), icon_anchor=(8, 8)),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row.name,
            ).add_to(hubs_group)
        else:
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=_marker_radius(row.daily_footfall),
                color=_line_color(row.line),
                fill=True, fill_color=_line_color(row.line), fill_opacity=0.85, weight=1.5,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row.name,
            ).add_to(stations_group)

    stations_group.add_to(fmap)
    hubs_group.add_to(fmap)

    if not buffers_gdf.empty:
        buffers_group = folium.FeatureGroup(name="1 km Pedestrian Buffers", show=True)
        for row in buffers_gdf.itertuples(index=False):
            folium.GeoJson(
                data=gpd.GeoSeries([row.geometry], crs=config.GEOGRAPHIC_CRS).__geo_interface__,
                style_function=lambda _f: {
                    "color": "#FFD24C", "weight": 1.5, "fillColor": "#FFD24C",
                    "fillOpacity": 0.10, "dashArray": "4,4",
                },
                tooltip=f"{row.name}: 1 km catchment (~{row.buffer_area_km2:.2f} km2)",
            ).add_to(buffers_group)
        buffers_group.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    hero_html = """
    <div class="map-hero">
      <a href="../dashboard.html" class="map-hero-back">&larr; Back to Dashboard</a>
      <h1>Mumbai Suburban Rail</h1>
      <p>Western &middot; Central &middot; Harbour &middot; Trans-Harbour<br>
         nodes scaled by daily footfall, hubs pulse gold.</p>
      <div class="legend-row"><span class="dot" style="color:#22C55E;background:#22C55E;"></span>Western Line</div>
      <div class="legend-row"><span class="dot" style="color:#EF4444;background:#EF4444;"></span>Central Line</div>
      <div class="legend-row"><span class="dot" style="color:#3B82F6;background:#3B82F6;"></span>Harbour Line</div>
      <div class="legend-row"><span class="dot" style="color:#F59E0B;background:#F59E0B;"></span>Trans-Harbour</div>
      <div class="legend-row"><span class="dot ring" style="color:#FFD24C;"></span>Interchange hub + 1 km buffer</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(_THEME_CSS + hero_html))
    return fmap


def save_map(fmap: folium.Map) -> None:
    try:
        fmap.save(str(config.MAP_HTML))
        logger.info("Interactive map saved to %s", config.MAP_HTML)
    except OSError as exc:
        logger.error("Failed to save map HTML: %s", exc)
        raise


def run_spatial_pipeline(stations_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """End-to-end GIS step used by main.py: GeoDataFrames -> GeoJSON -> map."""
    stations_gdf = stations_to_geodataframe(stations_df)
    edges_gdf = edges_to_geodataframe(edges_df, stations_df)
    buffers_gdf = compute_interchange_buffers(stations_gdf)

    export_geojson_layers(stations_gdf, edges_gdf, buffers_gdf)

    fmap = build_folium_map(stations_df, edges_df, buffers_gdf)
    save_map(fmap)

    return {
        "buffers_gdf": buffers_gdf,
        "total_buffer_area_km2": round(buffers_gdf["buffer_area_km2"].sum(), 2) if not buffers_gdf.empty else 0.0,
        "map_path": config.MAP_HTML,
    }
