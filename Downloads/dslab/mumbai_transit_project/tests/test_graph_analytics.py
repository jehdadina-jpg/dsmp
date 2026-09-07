"""Sanity checks for graph construction, routing and centrality."""

import pytest

from src import data_loader, graph_analytics


@pytest.fixture(scope="module")
def graph():
    stations_df, edges_df = data_loader.load_and_clean_data()
    return graph_analytics.build_graph(stations_df, edges_df)


def test_graph_is_connected(graph):
    assert graph.number_of_nodes() == 25
    summary = graph_analytics.network_summary(graph)
    assert summary["is_connected"] is True


def test_shortest_route_virar_to_panvel(graph):
    route = graph_analytics.shortest_route(graph, "VIR", "PNV", weight="time")
    assert route["path_ids"][0] == "VIR"
    assert route["path_ids"][-1] == "PNV"
    # Must interchange at least once: Western -> Central/Harbour is unavoidable.
    assert route["interchanges"] >= 1
    assert route["total_time_mins"] > 0


def test_shortest_route_accepts_station_names(graph):
    by_id = graph_analytics.shortest_route(graph, "DDR", "KUR", weight="distance")
    by_name = graph_analytics.shortest_route(graph, "Dadar", "Kurla", weight="distance")
    assert by_id["path_ids"] == by_name["path_ids"]


def test_shortest_route_unknown_station_raises(graph):
    with pytest.raises(KeyError):
        graph_analytics.shortest_route(graph, "NOT_A_STATION", "PNV")


def test_top_hubs_surface_known_interchanges(graph):
    top = graph_analytics.top_hubs(graph, n=5)
    # Dadar and Kurla are structural interchanges by construction; they
    # should algorithmically rank near the top of betweenness centrality.
    assert "DDR" in set(top["station_id"]) or "KUR" in set(top["station_id"])


def test_cypher_generation_is_well_formed(graph):
    stations_df, edges_df = data_loader.load_and_clean_data()
    statements = graph_analytics.generate_cypher_queries(stations_df, edges_df)
    assert any("MERGE (s:Station" in s for s in statements)
    assert any("CONNECTS_TO" in s for s in statements)
    assert len(statements) == 1 + len(stations_df) + len(edges_df)  # +1 constraint statement
