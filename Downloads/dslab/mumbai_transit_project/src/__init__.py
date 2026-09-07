"""Mumbai Suburban Rail Network Efficiency & Accessibility Tracker.

Modules:
    config            Shared paths, constants and logging setup.
    data_loader       Real station/edge dataset, cleaning and haversine enrichment.
    graph_analytics   NetworkX graph construction, routing, centrality, Cypher export.
    spatial_analysis  GeoPandas buffers, GeoJSON export and the Folium map.
    dashboard         Standalone HTML analytics dashboard (3D graph hero + reports).
"""

__version__ = "1.0.0"
