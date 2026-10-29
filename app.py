"""
OptiRoute — Multi-Objective Route Optimization (NSGA-III)
Streamlit web app port of the validated OptiRoute Colab pipeline.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)
"""

import math
import os
import pickle
import random
import re
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
from pymoo.core.callback import Callback
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
ox.settings.requests_timeout = 15  # fail fast instead of hanging for minutes on geocoding
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

# Relative base crash rate per vehicle-km at the reference speed, by road type.
# ASSUMED values (illustrative, HSM-style: motorways safest per km, urban
# streets higher). Replace with local crash data if available.
HIGHWAY_BASE_RISK = {
    "motorway": 0.4, "trunk": 0.7, "primary": 0.9, "secondary": 1.0,
    "tertiary": 1.0, "unclassified": 1.0, "residential": 1.0, "living_street": 0.8,
}
REF_SPEED_KPH = 50.0
SPEED_POWER = 2.0        # Nilsson Power Model exponent for injury crashes (4 for fatal)
JUNCTION_PENALTY = 0.3   # extra relative risk per km at the busiest junction density

# Junction / signal delays (ASSUMED averages, tune for your city). Added to every
# road segment for the junction it leads into.
SIGNAL_DELAY_S = 25.0     # average wait at a traffic signal
JUNCTION_DELAY_S = 8.0    # average slowdown at a busy unsignalled junction (4+ roads)
IDLE_G_CO2_PER_S = 0.5    # approx. idling emissions, g CO2 per second waited

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


def edge_risk(length_m, speed_kph, hw, junction_density):
    """Relative accident risk: base rate by road type x Power Model speed
    factor + junction penalty, scaled by length. A relative index, not a
    probability."""
    v = max(speed_kph, 5.0)
    base = HIGHWAY_BASE_RISK.get(hw, 1.0)
    per_km = base * (v / REF_SPEED_KPH) ** SPEED_POWER + JUNCTION_PENALTY * junction_density
    return (length_m / 1000) * per_km


def compute_edge_weights(u, v, data, node_degree):
    length_m = data.get("length", 1.0)
    speed_kph = data.get("speed_kph", 30.0)
    speed_ms = max(speed_kph, 5.0) * 1000 / 3600

    time_s = data.get("travel_time", length_m / speed_ms)
    distance_m = length_m

    # COPERT-style speed-dependent emission factor (g CO2/km, petrol car).
    # Shape: high in slow traffic, lowest ~50 kph, rising at high speed.
    # NOTE: coefficients are approximate; replace with the official COPERT
    # table values for your vehicle class before citing exact numbers.
    v = max(speed_kph, 5.0)
    ef_g_per_km = 85 + 800 / v + 0.0035 * v ** 2
    fuel_cost = (length_m / 1000) * ef_g_per_km  # grams CO2 (proxy for fuel)

    junction_density = min(node_degree.get(u, 1) + node_degree.get(v, 1), 12) / 12
    data["jd"] = junction_density  # reused when traffic changes the speed
    accident_risk = edge_risk(length_m, speed_kph, highway_type(data), junction_density)

    return time_s, distance_m, fuel_cost, accident_risk


def emission_factor(speed_kph):
    """COPERT-style g CO2 per km at a given average speed (approximate)."""
    v = max(speed_kph, 5.0)
    return 85 + 800 / v + 0.0035 * v ** 2


def add_junction_delays(G):
    """Add a wait at the junction each segment leads into: traffic signals
    (tagged in OpenStreetMap) and busy junctions. Stores the pure driving time
    in d["t_move"] and the wait in d["delay"], and adds both to time and fuel."""
    deg = dict(G.degree())  # counts in + out roads, so a 4-way two-way crossing = 8
    for u, v, k, d in G.edges(keys=True, data=True):
        hw = G.nodes[v].get("highway")
        hw = " ".join(map(str, hw)) if isinstance(hw, list) else str(hw)
        if "traffic_signals" in hw:
            delay = SIGNAL_DELAY_S
        elif deg.get(v, 0) >= 8:
            delay = JUNCTION_DELAY_S
        else:
            delay = 0.0
        d["t_move"] = d["time"]
        d["delay"] = delay
        d["time"] += delay
        d["fuel"] += delay * IDLE_G_CO2_PER_S


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


def remove_loops(path):
    """Cut any cycle out of a node path (keeps first visit of each node)."""
    out, pos = [], {}
    for n in path:
        if n in pos:
            out = out[:pos[n] + 1]
            pos = {m: i for i, m in enumerate(out)}
        else:
            pos[n] = len(out)
            out.append(n)
    return out


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
        return remove_loops(base_path[:i] + sub + base_path[j + 1:])
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
                Y[0, m, 0] = remove_loops(p1[:i1] + p2[i2:])
                Y[1, m, 0] = remove_loops(p2[:i2] + p1[i1:])
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

LOCAL_GRAPH_FILE = "mumbai_graph.pkl.gz"  # made by download_mumbai.py (raw map)
READY_GRAPH_FILE = "mumbai_graph_ready.pkl"  # made by preprocess_graph.py (fast, preferred)

# This app uses ONLY the saved Mumbai road network. It never downloads road
# data from OpenStreetMap/Overpass.


@st.cache_resource(show_spinner="Loading the saved Mumbai road network (first time only)...")
def load_local_graph():
    """Load the road network once and keep node coordinates as arrays so
    cropping per trip is fast. If the pre-weighted, slimmed file made by
    preprocess_graph.py exists, load it directly (fast, low memory). Otherwise
    fall back to weighting the raw graph here (slow, memory-hungry)."""
    t0 = time_module.time()
    source = "ready file"
    if os.path.exists(READY_GRAPH_FILE):
        with open(READY_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
    else:
        import gzip
        source = "raw file"
        with gzip.open(LOCAL_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
    t_read = time_module.time() - t0

    # The ready file must already carry the four weights. If it doesn't (e.g. it
    # was made by an older script), weight it here instead of crashing later.
    sample = next(iter(G.edges(data=True)), (None, None, {}))[2]
    if not all(k in sample for k in ("time", "distance", "fuel", "risk")):
        source += " + weighted at startup"
        G = ox.routing.add_edge_speeds(G, fallback=30)
        G = ox.routing.add_edge_travel_times(G)
        node_degree = dict(G.degree())
        for u, v, k, data in G.edges(keys=True, data=True):
            t, d, f_, r = compute_edge_weights(u, v, data, node_degree)
            data["time"], data["distance"], data["fuel"], data["risk"] = t, d, f_, r
        add_junction_delays(G)
    G.graph["load_info"] = (f"Map: {source}, {G.number_of_nodes():,} nodes / "
                            f"{G.number_of_edges():,} edges, read in {t_read:.0f}s, "
                            f"ready in {time_module.time() - t0:.0f}s")
    print(G.graph["load_info"], flush=True)
    ids = np.array(list(G.nodes))
    lat = np.array([G.nodes[n]["y"] for n in ids])
    lon = np.array([G.nodes[n]["x"] for n in ids])
    return G, ids, lat, lon


def crop_local_graph(center_lat, center_lon, radius_m):
    """Cut the area around the trip out of the saved network (independent copy)."""
    G, ids, lat, lon = load_local_graph()
    dy = (lat - center_lat) * 111320.0
    dx = (lon - center_lon) * 111320.0 * math.cos(math.radians(center_lat))
    keep = ids[(dx * dx + dy * dy) <= radius_m ** 2]
    if len(keep) < 50:
        raise ValueError("This area is outside the saved Mumbai map. Pick locations "
                         "inside Mumbai, or re-run download_mumbai.py with a wider area.")
    Gc = G.subgraph(keep.tolist()).copy()
    biggest = max(nx.strongly_connected_components(Gc), key=len)
    return Gc.subgraph(biggest).copy()


# Built-in Mumbai places (lat, lon). APPROXIMATE (typically within a few hundred
# metres of the area centre); the route is snapped to the nearest saved road node.
# For an exact spot, type "lat, lon" in the custom-coordinates boxes instead.
MUMBAI_PLACES = {
    "Colaba": (18.9067, 72.8147),
    "Cuffe Parade": (18.9140, 72.8200),
    "Gateway of India": (18.9220, 72.8347),
    "Nariman Point": (18.9256, 72.8242),
    "Churchgate": (18.9322, 72.8264),
    "CSMT (Chhatrapati Shivaji Terminus)": (18.9398, 72.8355),
    "Marine Lines": (18.9435, 72.8238),
    "Girgaon Chowpatty": (18.9543, 72.8130),
    "Malabar Hill": (18.9548, 72.7985),
    "Grant Road": (18.9634, 72.8153),
    "Mazgaon": (18.9689, 72.8444),
    "Mumbai Central": (18.9690, 72.8205),
    "Byculla": (18.9766, 72.8326),
    "Lower Parel": (18.9953, 72.8305),
    "Worli": (19.0090, 72.8180),
    "Parel": (19.0090, 72.8410),
    "Prabhadevi": (19.0150, 72.8290),
    "Dadar": (19.0178, 72.8478),
    "Wadala": (19.0170, 72.8610),
    "Matunga": (19.0270, 72.8560),
    "Dharavi": (19.0400, 72.8540),
    "Sion": (19.0390, 72.8619),
    "Mahim": (19.0409, 72.8404),
    "Chembur": (19.0522, 72.9005),
    "Bandra West": (19.0596, 72.8295),
    "Bandra East": (19.0600, 72.8420),
    "Bandra Kurla Complex (BKC)": (19.0660, 72.8680),
    "Khar": (19.0720, 72.8360),
    "Kurla": (19.0728, 72.8826),
    "Santacruz": (19.0820, 72.8410),
    "Ghatkopar": (19.0860, 72.9080),
    "Mumbai Airport (T2)": (19.0896, 72.8656),
    "Vile Parle": (19.0990, 72.8440),
    "Juhu": (19.1075, 72.8263),
    "Chandivali": (19.1090, 72.8980),
    "Vikhroli": (19.1110, 72.9270),
    "Andheri East": (19.1150, 72.8690),
    "Powai": (19.1176, 72.9060),
    "Andheri Station": (19.1197, 72.8464),
    "Kanjurmarg": (19.1290, 72.9300),
    "Versova": (19.1310, 72.8100),
    "Andheri West": (19.1360, 72.8290),
    "Jogeshwari": (19.1360, 72.8490),
    "Bhandup": (19.1430, 72.9370),
    "Goregaon": (19.1663, 72.8526),
    "Mulund": (19.1726, 72.9560),
    "Malad": (19.1860, 72.8480),
    "Kandivali": (19.2040, 72.8520),
    "Borivali": (19.2307, 72.8567),
    "Dahisar": (19.2500, 72.8590),
}


def geocode_place(query):
    """Offline place lookup: 'lat, lon' text, or a name from MUMBAI_PLACES.
    No internet needed."""
    q = query.strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", q)
    if len(nums) == 2 and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*[,\s]\s*-?\d+(?:\.\d+)?\s*", q):
        lat, lon = float(nums[0]), float(nums[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"'{query}' isn't a valid latitude, longitude pair.")
        return lat, lon
    lowered = {name.lower(): name for name in MUMBAI_PLACES}
    key = q.lower()
    if key in lowered:
        return MUMBAI_PLACES[lowered[key]]
    matches = [n for low, n in lowered.items() if key and key in low]
    if len(matches) == 1:
        return MUMBAI_PLACES[matches[0]]
    hint = f" Did you mean: {', '.join(matches[:5])}?" if matches else ""
    raise ValueError(f"Couldn't find '{query}' in the built-in Mumbai places.{hint} "
                     "Pick a name from the list, or enter coordinates as 'lat, lon'.")


# ----------------------------------------------------------------------------
# Optional live traffic (TomTom) — sampled before optimization
# ----------------------------------------------------------------------------


def apply_traffic_factors(G, api_key, n_samples=40):
    """Sample TomTom at ~n_samples nodes (congestion = current / free-flow speed),
    give each edge its nearest sample, and set its speed from TomTom's free-flow
    speed x congestion BEFORE optimization. Mutates G (pass a copy)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(0)
    nodes = rng.sample(list(G.nodes), min(n_samples, G.number_of_nodes()))

    def fetch(n):
        lat, lon = G.nodes[n]["y"], G.nodes[n]["x"]
        try:
            r = requests.get(
                "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
                params={"point": f"{lat},{lon}", "key": api_key}, timeout=4)
            r.raise_for_status()
            seg = r.json()["flowSegmentData"]
            cur, free = seg["currentSpeed"], seg["freeFlowSpeed"]
            if cur > 0 and free > 0:
                return lat, lon, min(cur / free, 1.0), free
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        got = [g for g in ex.map(fetch, nodes) if g]
    if len(got) < 5:
        return "Live traffic unavailable — using speed-limit times."

    pts = np.array([(g[0], g[1]) for g in got])
    factors = np.array([g[2] for g in got])
    frees = np.array([g[3] for g in got])
    for u, v, k, d in G.edges(keys=True, data=True):
        lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        i = int(np.argmin((pts[:, 0] - lat) ** 2 + (pts[:, 1] - lon) ** 2))
        f = max(factors[i], 0.15)                      # congestion: 1 = free flow
        t_move = d.get("t_move", d["time"])
        delay = d.get("delay", 0.0)
        osm_kph = max(d["distance"] / t_move * 3.6, 5.0) if t_move > 0 else 30.0
        # TomTom's real free-flow speed replaces the OpenStreetMap limit when the
        # limit is higher (optimistic); then apply the live slowdown.
        eff_kph = max(min(osm_kph, frees[i]) * f, 4.0)
        delay_now = delay * (1 + 2 * (1 - f))          # queues at signals grow in traffic
        d["time"] = d["distance"] / (eff_kph / 3.6) + delay_now
        d["fuel"] = (d["distance"] / 1000 * emission_factor(eff_kph)
                     + delay_now * IDLE_G_CO2_PER_S)
        d["risk"] = edge_risk(d["distance"], eff_kph, highway_type(d), d.get("jd", 0.5))
    return f"Live traffic applied from {len(got)} sample points before route search."


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


class ProgressCallback(Callback):
    """Reports the NSGA-III generation number to the UI as it runs."""

    def __init__(self, progress, total):
        super().__init__()
        self.progress, self.total = progress, total

    def notify(self, algorithm):
        self.progress(f"Step 3/3: optimizing routes (NSGA-III generation "
                      f"{algorithm.n_gen}/{self.total})...")


def run_pipeline(from_query, to_query, radius_m, traffic_key=None, progress=None):
    progress = progress or (lambda msg: None)
    t_start = time_module.time()
    progress("Step 1/3: looking up the two places (offline)...")
    from_pt = geocode_place(from_query)
    to_pt = geocode_place(to_query)
    t_geo = time_module.time()
    progress("Step 2/3: loading the saved map and cropping the area...")

    center_lat = (from_pt[0] + to_pt[0]) / 2
    center_lon = (from_pt[1] + to_pt[1]) / 2

    G = crop_local_graph(center_lat, center_lon, radius_m)  # local file only, no download
    t_crop = time_module.time()
    traffic_note = None
    if traffic_key:
        G = G.copy()  # never mutate the cached graph
        traffic_note = apply_traffic_factors(G, traffic_key)

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
    t_opt0 = time_module.time()
    res = minimize(problem, algorithm, ("n_gen", N_GEN), seed=1, verbose=False,
                   callback=ProgressCallback(progress, N_GEN))
    t_opt = time_module.time()

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
        "traffic_note": traffic_note,
        "timings": (f"Timings: geocoding {t_geo - t_start:.0f}s, map load + crop "
                    f"{t_crop - t_geo:.0f}s, NSGA-III {t_opt - t_opt0:.0f}s"),
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("🚗 OptiRoute")
st.caption("Multi-objective route optimization — no single best route, only trade-offs.")

if not (os.path.exists(READY_GRAPH_FILE) or os.path.exists(LOCAL_GRAPH_FILE)):
    st.error(f"Saved map file `{LOCAL_GRAPH_FILE}` was not found next to app.py. "
             "Add it to the repo (use Git LFS if it is over 100 MB) or run "
             "download_mumbai.py to create it. This app does not download maps online.")
    st.stop()

PLACE_NAMES = sorted(MUMBAI_PLACES)
col1, col2 = st.columns(2)
with col1:
    from_sel = st.selectbox("From", PLACE_NAMES, index=None,
                            placeholder="Type to search, e.g. Bandra West")
with col2:
    to_sel = st.selectbox("To", PLACE_NAMES, index=None,
                          placeholder="Type to search, e.g. Andheri East")
with st.expander("Use exact coordinates instead (optional)"):
    cc1, cc2 = st.columns(2)
    from_custom = cc1.text_input("From (lat, lon)", placeholder="e.g. 19.0596, 72.8295")
    to_custom = cc2.text_input("To (lat, lon)", placeholder="e.g. 19.1136, 72.8697")
# A typed coordinate pair, if given, overrides the dropdown choice.
from_input = from_custom.strip() or (from_sel or "")
to_input = to_custom.strip() or (to_sel or "")

radius_m = st.slider("Area radius (meters)", min_value=1000, max_value=MAX_RADIUS_M,
                      value=DEFAULT_RADIUS_M, step=500,
                      help="Road network is cut from the saved Mumbai map around the midpoint of "
                           "your two locations, out to this radius. Larger = slower to compute.")

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

try:
    tomtom_key = st.secrets.get("TOMTOM_API_KEY", "")
except Exception:
    tomtom_key = ""
use_traffic = st.checkbox(
    "Use live traffic data (TomTom)", value=False, disabled=not bool(tomtom_key),
    help="Only available when a TOMTOM_API_KEY is configured in Streamlit secrets."
         if not tomtom_key else
         "Samples live traffic at ~40 points before the route search; click Find Route to apply."
)

find_clicked = st.button("Find Route", type="primary")

# ----------------------------------------------------------------------------
# Session state cache key: NSGA-III reruns only when Find Route is clicked.
# Changing priority just re-scores the same Pareto set; changing from/to/radius/
# traffic shows a warning until the search is rerun.
# ----------------------------------------------------------------------------

if find_clicked:
    if not from_input or not to_input:
        st.error("Please enter both a From and To location.")
    else:
        cache_key = (from_input.strip().lower(), to_input.strip().lower(), radius_m,
                     bool(use_traffic and tomtom_key))
        try:
            with st.status("Starting...", expanded=True) as status:
                result = run_pipeline(from_input, to_input, radius_m,
                                      tomtom_key if use_traffic else None,
                                      progress=lambda msg: status.update(label=msg))
                status.update(label="Done", state="complete", expanded=False)
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

if result is not None and st.session_state.get("optiroute_key") != (
        from_input.strip().lower(), to_input.strip().lower(), radius_m,
        bool(use_traffic and tomtom_key)):
    st.warning("Inputs changed since the last search — click **Find Route** to update.")

if result is not None:
    G = result["G"]
    pareto_routes = result["pareto_routes"]
    pareto_costs = result["pareto_costs"]
    baseline_path = result["baseline_path"]
    origin, dest = result["origin"], result["dest"]

    baseline_cost = route_cost(G, baseline_path)

    traffic_note = result.get("traffic_note")

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
          "Fuel (g CO2)": round(c[2], 1), "Risk": round(c[3], 3)} for name, c in rows]
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
    st.subheader("Pareto front: Time vs. Accident Risk (color = Fuel, g CO2)")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(pareto_costs[:, 0], pareto_costs[:, 3], c=pareto_costs[:, 2],
                     cmap="viridis", s=80, edgecolor="k")
    ax.scatter([best_cost[0]], [best_cost[3]], c="red", s=180, marker="*",
               label=f"Recommended ({priority_choice})", zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accident Risk (relative)")
    plt.colorbar(sc, label="Fuel (g CO2)", ax=ax)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(f"{len(pareto_routes)} validated Pareto-optimal routes found across "
               f"{G.number_of_nodes()} intersections / {G.number_of_edges()} road segments.")
    if G.graph.get("load_info"):
        st.caption(G.graph["load_info"])
    if result.get("timings"):
        st.caption(result["timings"])
else:
    st.info("Enter a From and To location, then click **Find Route**.")
