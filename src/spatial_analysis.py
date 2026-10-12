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

from src import config, track_geometry

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
        if "path" in edges_df.columns:  # real track geometry from track_geometry.py
            return LineString([(lon, lat) for lat, lon in row.path])
        lat1, lon1 = coords.loc[row.source_id]
        lat2, lon2 = coords.loc[row.target_id]
        return LineString([(lon1, lat1), (lon2, lat2)])  # shapely wants (x=lon, y=lat)

    geometry = [_line(row) for row in edges_df.itertuples(index=False)]
    gdf = gpd.GeoDataFrame(edges_df.drop(columns=["path"], errors="ignore"), geometry=geometry, crs=config.GEOGRAPHIC_CRS)
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
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@700&display=swap');

  .leaflet-container { background:#0B0B0C !important; font-family:'Archivo',system-ui,sans-serif; }
  .leaflet-control-attribution { background:rgba(11,11,12,.8) !important; color:#6B675F !important; font-size:10px; }
  .leaflet-control-attribution a { color:#A39F94 !important; }
  .leaflet-bar { border:none !important; box-shadow:none !important; }
  .leaflet-control-zoom a {
    background:#0B0B0C !important; color:#FFC72C !important; border:1px solid rgba(242,239,230,.18) !important;
    border-radius:0 !important; width:34px !important; height:34px !important; line-height:32px !important;
  }
  .leaflet-control-zoom a + a { border-top:none !important; }
  .leaflet-control-zoom a:hover { background:#FFC72C !important; color:#141100 !important; }
  .leaflet-control-scale-line { background:rgba(11,11,12,.8); color:#A39F94; border-color:#A39F94; font-family:'JetBrains Mono',monospace; font-size:10px; }
  .leaflet-control-layers {
    background:rgba(11,11,12,.94) !important; color:#F2EFE6 !important; border-radius:0 !important;
    border:1px solid rgba(242,239,230,.14) !important; border-top:3px solid #FFC72C !important;
    font-family:'Archivo',sans-serif; font-size:13px; padding:10px 14px 10px 10px !important;
    box-shadow:0 20px 50px rgba(0,0,0,.5) !important;
  }
  .leaflet-control-layers-list label { padding:3px 0; }
  .leaflet-control-layers input { accent-color:#FFC72C; }
  .leaflet-control-layers-separator { border-top-color:rgba(242,239,230,.12) !important; }

  .leaflet-popup-content-wrapper {
    background:#0B0B0C; color:#F2EFE6; border-radius:0; padding:0;
    border:1px solid rgba(242,239,230,.14); box-shadow:0 18px 40px rgba(0,0,0,.6);
  }
  .leaflet-popup-content { margin:0; }
  .leaflet-popup-tip { background:#0B0B0C; }
  .leaflet-container a.leaflet-popup-close-button { color:#141100; top:6px; right:6px; }
  .leaflet-tooltip {
    background:#FFC72C; color:#141100; border:none; border-radius:0; box-shadow:none;
    font-family:'Archivo',sans-serif; font-weight:700; font-size:12px; padding:4px 8px;
  }
  .leaflet-tooltip-top:before, .leaflet-tooltip-bottom:before,
  .leaflet-tooltip-left:before, .leaflet-tooltip-right:before { display:none; }

  .station-card { font-family:'Archivo',sans-serif; min-width:210px; }
  .station-card h4 {
    margin:0; background:#FFC72C; color:#141100; padding:9px 34px 9px 12px;
    font-weight:900; font-variation-settings:"wdth" 75; font-size:20px; text-transform:uppercase; line-height:1.05;
  }
  .station-card .body { padding:10px 12px 12px; }
  .station-card .badge {
    display:inline-flex; align-items:center; gap:6px; font-size:10px; font-weight:600; letter-spacing:.08em;
    margin:0 10px 6px 0; text-transform:uppercase; color:#A39F94;
  }
  .station-card .badge i { width:14px; height:3px; display:inline-block; }
  .station-card .stat {
    font-family:'Archivo',sans-serif; font-weight:800; font-variation-settings:"wdth" 75; font-size:26px;
    color:#F2EFE6; margin-top:2px; line-height:1; font-variant-numeric:tabular-nums;
  }
  .station-card .stat span { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:400; color:#A39F94; letter-spacing:.08em; margin-left:4px; }
  .station-card .muted { color:#6B675F; font-size:10.5px; margin-top:6px; font-family:'JetBrains Mono',monospace; }

  .map-hero {
    position:absolute; top:12px; left:58px; z-index:1000; width:270px;
    background:rgba(11,11,12,.94); border:1px solid rgba(242,239,230,.14);
    box-shadow:0 20px 50px rgba(0,0,0,.5); font-family:'Archivo',sans-serif; color:#F2EFE6;
  }
  .map-hero-board { background:#FFC72C; color:#141100; padding:4px; }
  .map-hero-board div { border:2px solid #141100; padding:8px 12px 10px; }
  .map-hero-board .dev { font-family:'Noto Sans Devanagari',sans-serif; font-weight:700; font-size:12px; line-height:1.3; }
  .map-hero h1 {
    font-weight:900; font-variation-settings:"wdth" 70; font-size:28px; line-height:.9;
    margin:2px 0 0; text-transform:uppercase;
  }
  .map-hero-body { padding:12px 14px 14px; }
  .map-hero p { font-size:12px; color:#A39F94; margin:0 0 12px; line-height:1.5; }
  .map-hero-back {
    display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:10.5px;
    letter-spacing:.1em; text-transform:uppercase; color:#FFC72C; text-decoration:none; margin-top:6px;
  }
  .map-hero-back:hover { text-decoration:underline; }
  .map-hero .legend-row { display:flex; align-items:center; gap:10px; font-size:12.5px; margin:6px 0; }
  .map-hero .swatch { width:22px; height:4px; flex:none; }
  .map-hero .hubkey {
    width:14px; height:14px; border-radius:50%; flex:none; margin:0 4px; background:#0B0B0C;
    border:3px solid #F2EFE6; box-shadow:0 0 0 4px rgba(255,199,44,.25);
  }
  @media (max-width:640px){ .map-hero { width:auto; left:60px; right:60px; top:10px; } .map-hero p, .map-hero .legend-row { display:none; } }

  .hub-marker {
    width:18px; height:18px; border-radius:50%; background:#0B0B0C; border:3.5px solid #F2EFE6;
    box-sizing:border-box; box-shadow:0 0 0 5px rgba(255,199,44,.28);
    transition:transform .2s cubic-bezier(.16,1,.3,1);
  }
  .hub-marker:hover { transform:scale(1.25); }
</style>
"""


def build_folium_map(
    stations_df: pd.DataFrame, track_df: pd.DataFrame, snaps_df: pd.DataFrame,
    buffers_gdf: gpd.GeoDataFrame,
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
        "brightness(0.82) contrast(0.92) saturate(0.25);}</style>"
    ))

    coords = stations_df.set_index("station_id")
    snaps = snaps_df.set_index("station_id")

    line_groups: dict[str, folium.FeatureGroup] = {}
    for line_name in track_df["line_name"].unique():
        line_groups[line_name] = folium.FeatureGroup(name=f"{line_name} Line", show=True)

    for row in track_df.itertuples(index=False):
        a, b = coords.loc[row.source_id], coords.loc[row.target_id]
        color = _line_color(row.line_name)
        group = line_groups[row.line_name]
        detail = (f"{row.track_km:.1f} km of track &middot; circuity {row.circuity:.2f}&times;"
                  if row.on_track else "approximate route, station location is off in the source data")
        tip = (f"<b>{row.line_name} Line</b><br>{a['name']}  to  {b['name']}<br>"
               f"{detail} &middot; {row.travel_time_mins:.0f} min")
        # Dark casing under a solid line -- the printed-rail-map convention,
        # which keeps parallel/adjacent lines legible over the basemap. Edges
        # without a verified track path are drawn dashed so they never pass
        # for surveyed geometry.
        folium.PolyLine(row.path, color="#0B0B0C", weight=8, opacity=0.9).add_to(group)
        folium.PolyLine(
            row.path, color=color, weight=4, opacity=1 if row.on_track else 0.75,
            dash_array=None if row.on_track else "6,7", tooltip=tip,
        ).add_to(group)

    for group in line_groups.values():
        group.add_to(fmap)

    stations_group = folium.FeatureGroup(name="Stations", show=True)
    hubs_group = folium.FeatureGroup(name="Interchange Hubs", show=True)

    def snap_note(sid: str) -> str:
        m = snaps.loc[sid, "snap_m"]
        return f"<br>recorded &#8776;{m / 1000:.1f} km off-track" if m == m and m > 150 else ""

    for row in stations_df.itertuples(index=False):
        # Draw stops where the track actually is (snapped), so markers sit on
        # the lines; the popup still reports the recorded coordinate.
        pos = [snaps.loc[row.station_id, "snap_lat"], snaps.loc[row.station_id, "snap_lon"]]
        popup_html = f"""
        <div class="station-card">
          <h4>{row.name}</h4>
          <div class="body">
            {''.join(f'<span class="badge"><i style="background:{_line_color(t)}"></i>{t}</span>' for t in row.line.split('/'))}
            {'<span class="badge"><i style="background:#FFC72C"></i>Interchange</span>' if row.interchange_hub else ''}
            <div class="stat">{row.daily_footfall:,}<span>RIDERS / DAY</span></div>
            <div class="muted">{row.latitude:.4f}&deg;N &nbsp;{row.longitude:.4f}&deg;E{snap_note(row.station_id)}</div>
          </div>
        </div>
        """
        if row.interchange_hub:
            folium.Marker(
                location=pos,
                icon=folium.DivIcon(html='<div class="hub-marker"></div>', icon_size=(18, 18), icon_anchor=(9, 9)),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row.name,
            ).add_to(hubs_group)
        else:
            folium.CircleMarker(
                location=pos,
                radius=_marker_radius(row.daily_footfall),
                color="#0B0B0C",
                fill=True, fill_color=_line_color(row.line), fill_opacity=1, weight=2,
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
                    "color": "#FFC72C", "weight": 1.2, "fillColor": "#FFC72C",
                    "fillOpacity": 0.07, "dashArray": "2,5",
                },
                tooltip=f"{row.name}: 1 km catchment (~{row.buffer_area_km2:.2f} km2)",
            ).add_to(buffers_group)
        buffers_group.add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    # Legend swatches read from config so they can never drift from the lines.
    LC = config.LINE_COLORS
    hero_html = f"""
    <div class="map-hero">
      <div class="map-hero-board"><div>
        <div class="dev" lang="mr">मुंबई उपनगरीय रेल्वे</div>
        <h1>Mumbai Suburban Rail</h1>
      </div></div>
      <div class="map-hero-body">
        <p>Bigger dots are busier stations. Ringed dots are interchanges, with a 1&nbsp;km walking circle around each.</p>
        <div class="legend-row"><span class="swatch" style="background:{LC['Western']}"></span>Western Line</div>
        <div class="legend-row"><span class="swatch" style="background:{LC['Central']}"></span>Central Line</div>
        <div class="legend-row"><span class="swatch" style="background:{LC['Harbour']}"></span>Harbour Line</div>
        <div class="legend-row"><span class="swatch" style="background:{LC['Trans-Harbour']}"></span>Trans-Harbour</div>
        <div class="legend-row"><span class="hubkey"></span>Interchange hub + 1 km buffer</div>
        <a href="../dashboard.html" class="map-hero-back">&larr; Back to dashboard</a>
      </div>
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
    track_df, snaps_df = track_geometry.compute_track_paths(stations_df, edges_df)
    edges_gdf = edges_to_geodataframe(track_df, stations_df)
    buffers_gdf = compute_interchange_buffers(stations_gdf)

    export_geojson_layers(stations_gdf, edges_gdf, buffers_gdf)

    fmap = build_folium_map(stations_df, track_df, snaps_df, buffers_gdf)
    save_map(fmap)

    return {
        "track_df": track_df,
        "snaps_df": snaps_df,
        "buffers_gdf": buffers_gdf,
        "total_buffer_area_km2": round(buffers_gdf["buffer_area_km2"].sum(), 2) if not buffers_gdf.empty else 0.0,
        "map_path": config.MAP_HTML,
    }
