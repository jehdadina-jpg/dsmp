#!/usr/bin/env python3
"""CLI entry point for the Mumbai Suburban Rail Network Efficiency & Accessibility Tracker.

Two modes:
    python main.py            interactive menu (pick actions one at a time)
    python main.py --full     non-interactive: run the entire pipeline once and exit
                               (also what CI / grading scripts should call)

Examples:
    python main.py --full
    python main.py --full --origin Virar --destination Panvel
    python main.py --full --push-neo4j --neo4j-uri bolt://localhost:7687
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import networkx as nx
import pandas as pd

from src import config, dashboard, data_loader, graph_analytics, spatial_analysis

logger = config.get_logger("main")

# Windows terminals often default to a legacy codepage (cp1252) that cannot
# encode box-drawing characters; force UTF-8 stdout so the CLI banner never
# crashes the whole run on a plain `python main.py` from cmd.exe/PowerShell.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Pretty terminal helpers (no extra dependency -- plain formatted text)
# ---------------------------------------------------------------------------
def _rule(title: str = "") -> None:
    width = 72
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, width - len(title) - 4)}")
    else:
        print("─" * width)


def print_network_summary(summary: dict) -> None:
    _rule("NETWORK SUMMARY")
    print(f"  Stations        : {summary['num_stations']}")
    print(f"  Track segments  : {summary['num_edges']}  ({summary['total_track_km']} km total)")
    print(f"  Lines modeled   : {', '.join(summary['lines'])}")
    print(f"  Daily footfall  : {summary['total_daily_footfall']:,} riders/day (sum across stations)")
    print(f"  Interchange hubs: {', '.join(summary['interchange_hubs'])}")
    print(f"  Connected       : {summary['is_connected']}   Avg. degree: {summary['avg_degree']}")


def print_route(route: dict) -> None:
    label = "travel time" if route["weight_mode"] == "time" else "distance"
    _rule(f"ROUTE ({label}-optimal): {route['path_names'][0]} -> {route['path_names'][-1]}")
    print(f"  Path: {' -> '.join(route['path_names'])}")
    print(f"  Stops: {route['num_stations']}   Interchanges: {route['interchanges']}")
    print(f"  Total time: {route['total_time_mins']:.0f} min   Total distance: {route['total_distance_km']:.2f} km")
    print(f"  Fare (2nd class): Rs.{route['fare_inr']}   CO2 saved vs. car: {route['co2_saved_kg']:.2f} kg")
    for hop in route["hops"]:
        print(f"    {hop['from_name']:<24} -> {hop['to_name']:<24} [{hop['line']:<13}] "
              f"{hop['distance_km']:>5.2f} km / {hop['travel_time_mins']:>4.0f} min")


def print_centrality(df: pd.DataFrame, top_n: int) -> None:
    _rule(f"TOP {top_n} HUBS BY CENTRALITY")
    print(f"  {'#':<3}{'Station':<24}{'Line':<20}{'Degree':>9}{'Betweenness':>13}{'Footfall':>12}")
    for _, row in df.head(top_n).iterrows():
        print(f"  {row['rank']:<3}{row['name']:<24}{row['line']:<20}"
              f"{row['degree_centrality']:>9.3f}{row['betweenness_centrality']:>13.3f}"
              f"{row['daily_footfall']:>12,}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    origin: str,
    destination: str,
    top_n: int,
    push_neo4j: bool,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
) -> None:
    t0 = time.time()

    _rule("STEP 1/5 -- DATA LOADER (pandas)")
    stations_df, edges_df = data_loader.load_and_clean_data()

    _rule("STEP 2/5 -- GRAPH ANALYTICS (networkx)")
    graph = graph_analytics.build_graph(stations_df, edges_df)
    summary = graph_analytics.network_summary(graph)
    print_network_summary(summary)

    try:
        route_time = graph_analytics.shortest_route(graph, origin, destination, weight="time")
        route_distance = graph_analytics.shortest_route(graph, origin, destination, weight="distance")
        print_route(route_time)
        print_route(route_distance)
    except (KeyError, nx.NetworkXNoPath) as exc:
        logger.error("Routing failed: %s", exc)
        sys.exit(1)

    centrality_df = graph_analytics.compute_centrality(graph)
    print_centrality(centrality_df, top_n)

    graph_analytics.export_cypher_script(stations_df, edges_df)
    if push_neo4j:
        uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            logger.warning("No Neo4j password supplied (--neo4j-password or $NEO4J_PASSWORD) -- skipping live push.")
        else:
            graph_analytics.push_to_neo4j(uri, user, password, stations_df, edges_df)

    _rule("STEP 3/5 -- SPATIAL ANALYSIS (geopandas + folium)")
    spatial_results = spatial_analysis.run_spatial_pipeline(stations_df, edges_df)
    print(f"  Interactive map : {spatial_results['map_path']}")
    print(f"  Buffer coverage : {spatial_results['total_buffer_area_km2']} km2 across "
          f"{len(summary['interchange_hubs'])} interchange hubs")

    _rule("STEP 4/5 -- 3D LAYOUT + DASHBOARD (three.js)")
    layout_3d = graph_analytics.compute_3d_layout(graph)
    dashboard.build_dashboard(
        stations_df=stations_df,
        edges_df=edges_df,
        layout_3d=layout_3d,
        centrality_df=centrality_df,
        routes={"time": route_time, "distance": route_distance},
        network_summary=summary,
        buffer_stats=spatial_results,
        insights=graph_analytics.compute_insights(graph, centrality_df),
    )
    print(f"  Dashboard       : {config.DASHBOARD_HTML}")

    _rule("STEP 5/5 -- DONE")
    print(f"  Finished in {time.time() - t0:.1f}s\n")
    print(f"  Open in a browser:\n    {config.DASHBOARD_HTML}\n    {config.MAP_HTML}")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------
def _prompt(msg: str, default: str) -> str:
    raw = input(f"{msg} [{default}]: ").strip()
    return raw or default


def interactive_menu() -> None:
    stations_df, edges_df, graph = None, None, None
    centrality_df = None

    menu = """
  MUMBAI SUBURBAN RAIL NETWORK TRACKER
  1) Load & clean data (pandas)
  2) Build graph + show network summary (networkx)
  3) Find shortest route between two stations
  4) Show centrality report (top hubs)
  5) Export Neo4j Cypher script
  6) Push to a live Neo4j instance (optional)
  7) Run spatial analysis (GeoJSON + Folium map)
  8) Build the analytics dashboard (three.js)
  9) Run the FULL pipeline (1-8 with defaults)
  0) Exit
"""
    while True:
        print(menu)
        choice = input("  Choose an option: ").strip()

        try:
            if choice == "1":
                stations_df, edges_df = data_loader.load_and_clean_data()
                print(f"  Loaded {len(stations_df)} stations, {len(edges_df)} edges.")

            elif choice == "2":
                if stations_df is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                graph = graph_analytics.build_graph(stations_df, edges_df)
                print_network_summary(graph_analytics.network_summary(graph))

            elif choice == "3":
                if graph is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                    graph = graph_analytics.build_graph(stations_df, edges_df)
                origin = _prompt("Origin station (id or name)", config.DEFAULT_ORIGIN)
                destination = _prompt("Destination station (id or name)", config.DEFAULT_DESTINATION)
                weight = _prompt("Optimize by 'time' or 'distance'", "time")
                route = graph_analytics.shortest_route(graph, origin, destination, weight=weight)
                print_route(route)

            elif choice == "4":
                if graph is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                    graph = graph_analytics.build_graph(stations_df, edges_df)
                centrality_df = graph_analytics.compute_centrality(graph)
                top_n = int(_prompt("How many top hubs to show", "8"))
                print_centrality(centrality_df, top_n)

            elif choice == "5":
                if stations_df is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                graph_analytics.export_cypher_script(stations_df, edges_df)

            elif choice == "6":
                if stations_df is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                uri = _prompt("Neo4j URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
                user = _prompt("Neo4j user", os.environ.get("NEO4J_USER", "neo4j"))
                password = input("  Neo4j password: ").strip()
                if password:
                    graph_analytics.push_to_neo4j(uri, user, password, stations_df, edges_df)
                else:
                    print("  Skipped: no password given.")

            elif choice == "7":
                if stations_df is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                results = spatial_analysis.run_spatial_pipeline(stations_df, edges_df)
                print(f"  Map saved to {results['map_path']} "
                      f"({results['total_buffer_area_km2']} km2 buffered).")

            elif choice == "8":
                if stations_df is None:
                    stations_df, edges_df = data_loader.load_processed_data()
                if graph is None:
                    graph = graph_analytics.build_graph(stations_df, edges_df)
                if centrality_df is None:
                    centrality_df = graph_analytics.compute_centrality(graph)
                origin = _prompt("Dashboard route: origin", config.DEFAULT_ORIGIN)
                destination = _prompt("Dashboard route: destination", config.DEFAULT_DESTINATION)
                route_time = graph_analytics.shortest_route(graph, origin, destination, weight="time")
                route_distance = graph_analytics.shortest_route(graph, origin, destination, weight="distance")
                spatial_results = spatial_analysis.run_spatial_pipeline(stations_df, edges_df)
                layout_3d = graph_analytics.compute_3d_layout(graph)
                dashboard.build_dashboard(
                    stations_df, edges_df, layout_3d, centrality_df,
                    {"time": route_time, "distance": route_distance},
                    graph_analytics.network_summary(graph), spatial_results,
                    graph_analytics.compute_insights(graph, centrality_df),
                )
                print(f"  Dashboard saved to {config.DASHBOARD_HTML}")

            elif choice == "9":
                run_pipeline(
                    config.DEFAULT_ORIGIN, config.DEFAULT_DESTINATION, top_n=8,
                    push_neo4j=False, neo4j_uri=None, neo4j_user=None, neo4j_password=None,
                )

            elif choice == "0":
                print("  Goodbye.")
                break
            else:
                print("  Not a valid option, try again.")

        except (KeyError, nx.NetworkXNoPath, ValueError) as exc:
            logger.error("%s", exc)
        except Exception as exc:  # noqa: BLE001 -- CLI boundary: never crash the menu loop
            logger.error("Unexpected error: %s", exc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Run the entire pipeline once, non-interactively.")
    parser.add_argument("--origin", default=config.DEFAULT_ORIGIN, help="Origin station id or name (default: Virar).")
    parser.add_argument("--destination", default=config.DEFAULT_DESTINATION, help="Destination station id or name (default: Panvel).")
    parser.add_argument("--top-n", type=int, default=8, help="How many hubs to show in the centrality report.")
    parser.add_argument("--push-neo4j", action="store_true", help="Also push the graph into a live Neo4j instance.")
    parser.add_argument("--neo4j-uri", default=None, help="Overrides $NEO4J_URI (default bolt://localhost:7687).")
    parser.add_argument("--neo4j-user", default=None, help="Overrides $NEO4J_USER (default 'neo4j').")
    parser.add_argument("--neo4j-password", default=None, help="Overrides $NEO4J_PASSWORD.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.full:
        run_pipeline(
            args.origin, args.destination, args.top_n,
            args.push_neo4j, args.neo4j_uri, args.neo4j_user, args.neo4j_password,
        )
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
