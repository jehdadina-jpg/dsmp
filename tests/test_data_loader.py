"""Sanity checks for the data layer: shape, uniqueness, and the haversine math."""

import math

from src import data_loader


def test_load_and_clean_data_shapes():
    stations_df, edges_df = data_loader.load_and_clean_data()
    assert len(stations_df) == len(data_loader._STATION_RECORDS)
    assert stations_df["station_id"].is_unique
    assert set(stations_df.columns) >= {
        "station_id", "name", "line", "latitude", "longitude", "daily_footfall", "interchange_hub",
    }
    assert not stations_df[["latitude", "longitude"]].isnull().values.any()


def test_edges_reference_known_stations():
    stations_df, edges_df = data_loader.load_and_clean_data()
    known = set(stations_df["station_id"])
    assert set(edges_df["source_id"]) <= known
    assert set(edges_df["target_id"]) <= known
    assert (edges_df["distance_km"] > 0).all()
    assert (edges_df["travel_time_mins"] >= data_loader.config.MIN_TRAVEL_TIME_MINS).all()


def test_haversine_known_distance():
    # Churchgate to Virar (both on the Western Line) is roughly 60 km apart.
    dist = data_loader._haversine_km(18.934886, 72.827164, 19.455300, 72.812000)
    assert 55 < dist < 65


def test_haversine_zero_for_identical_points():
    assert math.isclose(data_loader._haversine_km(19.0, 72.8, 19.0, 72.8), 0.0, abs_tol=1e-9)
