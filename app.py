"""
OptiRoute — Multi-Objective Route Optimization (NSGA-III)
Streamlit web app port of the validated OptiRoute Colab pipeline.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)
"""

import math
import random
import time as time_module

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# ----------------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------------

ox.settings.log_console = False
random.seed(42)
np.random.seed(42)

MAX_RADIUS_M = 15000
DEFAULT_RADIUS_M = 6000
N_GEN = 25  # NSGA-III generations. Kept lower than the notebook's 30 for web
            # response time; still gives the algorithm room to converge on a
            # graph of this size. Adjust if you need faster/slower trade-offs.

PRIORITY_PRESETS = {
    "Fastest": (1.0, 0.0, 0.0, 0.0),
    "Shortest": (0.0, 1.0, 0.0, 0.0),
    "Eco-friendly": (0.0, 0.0, 1.0, 0.0),
    "Safest": (0.0, 0.0, 0.0, 1.0),
    "Balanced": (0.25, 0.25, 0.25, 0.25),
}

HIGHWAY_RISK = {
    "motorway": 0.95, "trunk": 0.85, "primary": 0.65, "secondary": 0.5,
    "tertiary": 0.35, "residential": 0.15, "living_street": 0.08, "unclassified": 0.35,
}

st.set_page_config(page_title="OptiRoute", page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------------
# Core pipeline — ported 1:1 from the Colab notebook (algorithm logic
# unchanged; only I/O and origin/destination selection are adapted since a
# real app takes user-supplied From/To locations rather than random nodes).
# ----------------------------------------------------------------------------


def highway_type(data):
    h = data.get("highway", "unclassified")
    if isinstance(h, list):
        h = h[0]
    return h


def compute_edge_weights(u, v, data, node_degree):
    length_m = data.get("length", 1.0)
    speed_kph = data.get("speed_kph", 30.0)
    speed_ms = max(speed_kph, 5.0) * 1000 / 3600

    time_s = data.get("travel_time", length_m / speed_ms)
    distance_m = length_m

    grade_proxy = {"motorway": 0.0, "trunk": 0.1, "primary": 0.2}.get(highway_type(data), 0.3)
    fuel_rate_per_km = 0.06 * (speed_kph ** 2) / 100 + 0.08 + grade_proxy * 0.02
    fuel_cost = (length_m / 1000) * fuel_rate_per_km

    junction_density = min(node_degree.get(u, 1) + node_degree.get(v, 1), 12) / 12
    lit = 0.0 if str(data.get("lit", "no")).lower() == "yes" else 1.0
    road_risk = HIGHWAY_RISK.get(highway_type(data), 0.4)
    speed_risk = min(speed_kph / 100, 1.0)
    risk_rate_per_km = 0.25 * junction_density + 0.15 * lit + 0.4 * road_risk + 0.2 * speed_risk
    accident_risk = (length_m / 1000) * risk_rate_per_km

    return time_s, distance_m, fuel_cost, accident_risk


def best_parallel_edge(G, u, v, weight="time"):
    edges = G.get_edge_data(u, v)
    return min(edges.values(), key=lambda d: d.get(weight, 1e9))


def route_cost(G, path):
    totals = np.zeros(4)
    for u, v in zip(path[:-1], path[1:]):
        d = best_parallel_edge(G, u, v, weight="time")
        totals += [d["time"], d["distance"], d["fuel"], d["risk"]]
    return totals


def weighted_edge_cost(weights):
    wt, wd, wf, wr = weights

    def _cost(u, v, data):
        d = min(data.values(), key=lambda e: e.get("time", 1e9))
        return wt * d["time"] + wd * d["distance"] + wf * d["fuel"] + wr * d["risk"]

    return _cost


def road_types_on_path(G, path):
    types = set()
    for u, v in zip(path[:-1], path[1:]):
        d = min(G.get_edge_data(u, v).values(), key=lambda e: e.get("time", 1e9))
        types.add(highway_type(d))
    return types


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --- NSGA-III custom operators (identical logic to the notebook, but the
# A*-seed baseline path is now passed in explicitly instead of read from a
# module-level global, so a fresh problem/algorithm can be built per request) ---


def random_weights():
    return tuple(np.random.dirichlet(np.ones(4)))


def random_route(G, origin, dest, tries=5):
    for _ in range(tries):
        try:
            w = random_weights()
            return nx.shortest_path(G, origin, dest, weight=weighted_edge_cost(w))
        except nx.NetworkXNoPath:
            continue
    return None


def seeded_route(G, base_path, jitter=0.3):
    if len(base_path) < 4 or random.random() > jitter:
        return list(base_path)
    i = random.randint(0, len(base_path) - 3)
    j = random.randint(i + 2, len(base_path) - 1)
    try:
        w = random_weights()
        sub = nx.shortest_path(G, base_path[i], base_path[j], weight=weighted_edge_cost(w))
        return base_path[:i] + sub + base_path[j + 1:]
    except nx.NetworkXNoPath:
        return list(base_path)


class RouteProblem(Problem):
    def __init__(self, G, origin, dest):
        super().__init__(n_var=1, n_obj=4, n_ieq_constr=0, xl=None, xu=None)
        self.G, self.origin, self.dest = G, origin, dest

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([route_cost(self.G, x[0]) for x in X])


class RouteSampling(Sampling):
    def __init__(self, baseline_path):
        super().__init__()
        self.baseline_path = baseline_path

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, 1), dtype=object)
        for i in range(n_samples):
            if i % 2 == 0:
                X[i, 0] = seeded_route(problem.G, self.baseline_path)
            else:
                r = random_route(problem.G, problem.origin, problem.dest)
                X[i, 0] = r if r is not None else list(self.baseline_path)
        return X


class RouteCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y = np.empty_like(X, dtype=object)
        for m in range(n_matings):
            p1, p2 = X[0, m, 0], X[1, m, 0]
            common = set(p1[1:-1]) & set(p2[1:-1])
            if common:
                node = random.choice(list(common))
                i1, i2 = p1.index(node), p2.index(node)
                Y[0, m, 0] = p1[:i1] + p2[i2:]
                Y[1, m, 0] = p2[:i2] + p1[i1:]
            else:
                Y[0, m, 0] = list(p1)
                Y[1, m, 0] = list(p2)
        return Y


class RouteMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i, 0] = seeded_route(problem.G, X[i, 0], jitter=0.5)
        return X


class RouteDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]


# ----------------------------------------------------------------------------
# Cached graph construction
# ----------------------------------------------------------------------------

# The main overpass-api.de instance is a shared free public service and is
# frequently overloaded / briefly refuses connections. Retry across a few
# known public mirrors with backoff before giving up, instead of failing on
# the first hiccup.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


def _download_graph_with_retries(center_lat, center_lon, radius_m, attempts_per_mirror=2):
    last_err = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        ox.settings.overpass_endpoint = mirror
        for attempt in range(attempts_per_mirror):
            try:
                return ox.graph_from_point((center_lat, center_lon), dist=radius_m,
                                            network_type="drive")
            except Exception as e:
                last_err = e
                time_module.sleep(2 * (attempt + 1))  # brief backoff before retrying
    raise ConnectionError(
        "Couldn't reach any OpenStreetMap road-data server after several tries "
        f"(last error: {last_err}). This is usually a temporary outage on their "
        "free public service — please wait a minute and try again."
    )


@st.cache_resource(show_spinner=False)
def build_graph(center_lat, center_lon, radius_m):
    """Download + weight the road graph. Cached per (rounded center, radius)
    so repeated requests for the same area don't re-hit OSM."""
    G = _download_graph_with_retries(center_lat, center_lon, radius_m)
    G = ox.routing.add_edge_speeds(G, fallback=30)  # 30 kph default for untagged roads,
                                                      # matches compute_edge_weights' own default
    G = ox.routing.add_edge_travel_times(G)

    node_degree = dict(G.degree())
    for u, v, k, data in G.edges(keys=True, data=True):
        t, d, f, r = compute_edge_weights(u, v, data, node_degree)
        data["time"] = t
        data["distance"] = d
        data["fuel"] = f
        data["risk"] = r
    return G


@st.cache_resource(show_spinner=False)
def geocode_place(query):
    last_err = None
    for attempt in range(3):
        try:
            return ox.geocode(query)
        except Exception as e:
            last_err = e
            time_module.sleep(2 * (attempt + 1))
    raise ConnectionError(
        f"Couldn't geocode '{query}' after several tries (last error: {last_err}). "
        "The geocoding service may be temporarily unavailable — try again shortly."
    )


# ----------------------------------------------------------------------------
# Optional live traffic (TomTom) — only queries edges on the Pareto routes
# ----------------------------------------------------------------------------


def apply_live_traffic(G, routes, api_key):
    """Re-fetch current speed for edges on the given routes only, and return an
    array of updated costs (one row per route, same order as `routes`).
    Never raises — falls back silently per edge on any API failure."""
    import requests

    edges_seen = set()
    for path in routes:
        for u, v in zip(path[:-1], path[1:]):
            edges_seen.add((u, v))

    updated_times = {}
    for (u, v) in edges_seen:
        d = best_parallel_edge(G, u, v, weight="time")
        lat, lon = G.nodes[u]["y"], G.nodes[u]["x"]
        try:
            resp = requests.get(
                "https://api.tomtom.com/traffic/services/4/flowSegmentData/"
                "absolute/10/json",
                params={"point": f"{lat},{lon}", "key": api_key},
                timeout=3,
            )
            resp.raise_for_status()
            js = resp.json()
            current_speed_kph = js["flowSegmentData"]["currentSpeed"]
            if current_speed_kph and current_speed_kph > 0:
                speed_ms = current_speed_kph * 1000 / 3600
                updated_times[(u, v)] = d["distance"] / speed_ms
        except Exception:
            continue  # silent fallback to static time for this edge

    new_costs = []
    for path in routes:
        totals = np.zeros(4)
        for u, v in zip(path[:-1], path[1:]):
            d = best_parallel_edge(G, u, v, weight="time")
            t = updated_times.get((u, v), d["time"])
            totals += [t, d["distance"], d["fuel"], d["risk"]]
        new_costs.append(totals)
    return np.array(new_costs), len(updated_times)


def recommend_route(pareto_costs, pareto_routes, priority_weights):
    mins, maxs = pareto_costs.min(axis=0), pareto_costs.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    normalized = (pareto_costs - mins) / ranges
    scores = normalized @ np.array(priority_weights)
    best_idx = int(np.argmin(scores))
    return best_idx, pareto_routes[best_idx], pareto_costs[best_idx]


def path_to_coords(G, path):
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


# ----------------------------------------------------------------------------
# Full pipeline run (graph -> baseline -> NSGA-III -> validated Pareto set)
# ----------------------------------------------------------------------------


def run_pipeline(from_query, to_query, radius_m):
    from_pt = geocode_place(from_query)
    to_pt = geocode_place(to_query)

    center_lat = (from_pt[0] + to_pt[0]) / 2
    center_lon = (from_pt[1] + to_pt[1]) / 2

    G = build_graph(round(center_lat, 4), round(center_lon, 4), radius_m)

    origin = ox.distance.nearest_nodes(G, X=from_pt[1], Y=from_pt[0])
    dest = ox.distance.nearest_nodes(G, X=to_pt[1], Y=to_pt[0])

    origin_gap = haversine_m(from_pt[0], from_pt[1], G.nodes[origin]["y"], G.nodes[origin]["x"])
    dest_gap = haversine_m(to_pt[0], to_pt[1], G.nodes[dest]["y"], G.nodes[dest]["x"])
    if origin_gap > radius_m * 0.5 or dest_gap > radius_m * 0.5:
        raise ValueError(
            "One of the locations falls outside the downloaded road network area. "
            "Try increasing the area radius, or pick points closer together."
        )

    if origin == dest:
        raise ValueError("Origin and destination resolved to the same road network point. "
                          "Try a more specific address.")

    if not nx.has_path(G, origin, dest):
        raise ValueError("No drivable path exists between these two points in the "
                          "downloaded area. Try increasing the area radius.")

    def euclid(u, v):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        return ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5

    baseline_path = nx.astar_path(G, origin, dest, heuristic=euclid, weight="distance")

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
    pop_size = len(ref_dirs)

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=RouteSampling(baseline_path),
        crossover=RouteCrossover(),
        mutation=RouteMutation(),
        eliminate_duplicates=RouteDuplicateElimination(),
    )
    problem = RouteProblem(G, origin, dest)
    res = minimize(problem, algorithm, ("n_gen", N_GEN), seed=1, verbose=False)

    pareto_routes, pareto_costs, seen = [], [], set()
    for x in res.X:
        path = x[0]
        key = tuple(path)
        if key in seen:
            continue
        if not all(G.has_edge(u, v) for u, v in zip(path[:-1], path[1:])):
            continue
        seen.add(key)
        pareto_routes.append(path)
        pareto_costs.append(route_cost(G, path))

    if not pareto_routes:
        # NSGA-III found nothing valid (can happen on a very sparse/small graph) —
        # fall back to the baseline as the sole "Pareto set" so the app still works.
        pareto_routes = [baseline_path]
        pareto_costs = [route_cost(G, baseline_path)]

    pareto_costs = np.array(pareto_costs)

    return {
        "G": G, "origin": origin, "dest": dest,
        "baseline_path": baseline_path,
        "pareto_routes": pareto_routes, "pareto_costs": pareto_costs,
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("🚗 OptiRoute")
st.caption("Multi-objective route optimization — no single best route, only trade-offs.")

col1, col2 = st.columns(2)
with col1:
    from_input = st.text_input("From", placeholder="e.g. Bandra West, Mumbai, India")
with col2:
    to_input = st.text_input("To", placeholder="e.g. Andheri East, Mumbai, India")

radius_m = st.slider("Area radius (meters)", min_value=1000, max_value=MAX_RADIUS_M,
                      value=DEFAULT_RADIUS_M, step=500,
                      help="Road network is downloaded around the midpoint of your two "
                           "locations, out to this radius. Larger = slower to compute.")

priority_choice = st.selectbox("Priority", list(PRIORITY_PRESETS.keys()) + ["Advanced"])

if priority_choice == "Advanced":
    st.write("Set custom weights (auto-normalized to sum to 1):")
    a1, a2, a3, a4 = st.columns(4)
    w_time = a1.slider("Time", 0.0, 1.0, 0.25, 0.05)
    w_dist = a2.slider("Distance", 0.0, 1.0, 0.25, 0.05)
    w_fuel = a3.slider("Fuel", 0.0, 1.0, 0.25, 0.05)
    w_risk = a4.slider("Risk", 0.0, 1.0, 0.25, 0.05)
    raw = np.array([w_time, w_dist, w_fuel, w_risk])
    priority_weights = tuple(raw / raw.sum()) if raw.sum() > 0 else (0.25, 0.25, 0.25, 0.25)
else:
    priority_weights = PRIORITY_PRESETS[priority_choice]

tomtom_key = st.secrets.get("TOMTOM_API_KEY", "") if hasattr(st, "secrets") else ""
use_traffic = st.checkbox(
    "Use live traffic data (TomTom)", value=False, disabled=not bool(tomtom_key),
    help="Only available when a TOMTOM_API_KEY is configured in Streamlit secrets."
         if not tomtom_key else
         "Re-fetches current speeds for edges on the Pareto routes only."
)

find_clicked = st.button("Find Route", type="primary")

# ----------------------------------------------------------------------------
# Session state cache key: only rerun NSGA-III when from/to/radius change.
# Switching priority (or the traffic toggle) just re-scores the same set.
# ----------------------------------------------------------------------------

if find_clicked:
    if not from_input or not to_input:
        st.error("Please enter both a From and To location.")
    else:
        cache_key = (from_input.strip().lower(), to_input.strip().lower(), radius_m)
        try:
            with st.spinner("Finding optimal routes... (downloading road network, "
                             "running NSGA-III — this can take 20–60s)"):
                result = run_pipeline(from_input, to_input, radius_m)
            st.session_state["optiroute_result"] = result
            st.session_state["optiroute_key"] = cache_key
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("optiroute_result", None)
        except Exception as e:
            st.error(f"Couldn't complete this route search: {e}. "
                     f"Try a more specific address or a different area.")
            st.session_state.pop("optiroute_result", None)

result = st.session_state.get("optiroute_result")

if result is not None:
    G = result["G"]
    pareto_routes = result["pareto_routes"]
    pareto_costs = result["pareto_costs"]
    baseline_path = result["baseline_path"]
    origin, dest = result["origin"], result["dest"]

    baseline_cost = route_cost(G, baseline_path)

    traffic_note = None
    if use_traffic and tomtom_key:
        try:
            with st.spinner("Fetching live traffic for candidate routes..."):
                # Include baseline in the same lookup so it's judged on the same
                # real-time basis as the Pareto routes, not left on static speed-limit time.
                all_routes = pareto_routes + [baseline_path]
                new_costs, n_updated = apply_live_traffic(G, all_routes, tomtom_key)
            pareto_costs = new_costs[:-1]
            baseline_cost = new_costs[-1]
            traffic_note = f"Live traffic applied to {n_updated} road segments (including baseline)."
        except Exception:
            traffic_note = "Live traffic lookup failed — showing static speed-limit times."

    best_idx, best_path, best_cost = recommend_route(pareto_costs, pareto_routes, priority_weights)

    st.subheader("Recommended route")
    if traffic_note:
        st.caption(traffic_note)

    m = folium.Map(location=(G.nodes[origin]["y"], G.nodes[origin]["x"]),
                    zoom_start=14, tiles="OpenStreetMap")
    for path in pareto_routes:
        folium.PolyLine(path_to_coords(G, path), color="#93c5fd", weight=3, opacity=0.5).add_to(m)
    folium.PolyLine(path_to_coords(G, baseline_path), color="#6b7280", weight=4,
                     opacity=0.9, dash_array="6,6", tooltip="Baseline (A*)").add_to(m)
    folium.PolyLine(path_to_coords(G, best_path), color="#dc2626", weight=5, opacity=1.0,
                     tooltip=f"Recommended ({priority_choice})").add_to(m)
    folium.Marker((G.nodes[origin]["y"], G.nodes[origin]["x"]), tooltip="Origin",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((G.nodes[dest]["y"], G.nodes[dest]["x"]), tooltip="Destination",
                  icon=folium.Icon(color="red")).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])

    # --- comparison table: recommended vs. baseline vs. a couple of Pareto alternatives
    order = np.argsort(pareto_costs[:, 0])  # sort alternatives by time for variety
    alt_idxs = [i for i in order if i != best_idx][:2]
    rows = [("Recommended", best_cost), ("Baseline (A*)", baseline_cost)]
    for i, idx in enumerate(alt_idxs, 1):
        rows.append((f"Alternative {i}", pareto_costs[idx]))

    df = pd.DataFrame(
        [{"Route": name, "Time (s)": round(c[0], 1), "Distance (m)": round(c[1], 1),
          "Fuel": round(c[2], 3), "Risk": round(c[3], 3)} for name, c in rows]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    # --- plain-language trade-off summary vs. the fastest route in the Pareto set
    fastest_idx = int(np.argmin(pareto_costs[:, 0]))
    fastest_cost = pareto_costs[fastest_idx]
    if best_idx != fastest_idx:
        dt = best_cost[0] - fastest_cost[0]
        dfuel_pct = (1 - best_cost[2] / fastest_cost[2]) * 100 if fastest_cost[2] > 0 else 0
        drisk_pct = (1 - best_cost[3] / fastest_cost[3]) * 100 if fastest_cost[3] > 0 else 0
        st.markdown(
            f"This route is **{abs(dt):.0f}s {'slower' if dt > 0 else 'faster'}** than the "
            f"fastest option, but uses **{dfuel_pct:.0f}% {'less' if dfuel_pct > 0 else 'more'} fuel** "
            f"and carries **{drisk_pct:.0f}% {'less' if drisk_pct > 0 else 'more'} accident risk**."
        )
    else:
        st.markdown("This route is also the fastest one in the Pareto set for this trip.")

    # --- Pareto front chart
    st.subheader("Pareto front: Time vs. Accident Risk (color = Fuel Cost)")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(pareto_costs[:, 0], pareto_costs[:, 3], c=pareto_costs[:, 2],
                     cmap="viridis", s=80, edgecolor="k")
    ax.scatter([best_cost[0]], [best_cost[3]], c="red", s=180, marker="*",
               label=f"Recommended ({priority_choice})", zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accident Risk (relative)")
    plt.colorbar(sc, label="Fuel Cost", ax=ax)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(f"{len(pareto_routes)} validated Pareto-optimal routes found across "
               f"{G.number_of_nodes()} intersections / {G.number_of_edges()} road segments.")
else:
    st.info("Enter a From and To location, then click **Find Route**.")
