"""
spatial_graph.py
================
Constructs the spatial hub graph for the Inner Melbourne VSR Zone and
exports it as a PyTorch Geometric–compatible Data object.

VSR Zone decision (hardcoded here, referenced in thesis §4.2.3)
----------------------------------------------------------------
Zone name : "Inner Melbourne VSR Zone"
Centroid  : (-37.8136, 144.9631)  [Melbourne CBD]
Radius    : 8 km road-network radius from centroid
Rationale : Covers CBD, Fitzroy, Collingwood, Richmond, South Yarra,
            St Kilda, Docklands, North Melbourne — a realistic DNSP zone
            boundary for a VIC VSRP pilot. Hub count target: 10–20 hubs
            drawn from Open Charge Map EV stations within this boundary.

Graph structure
---------------
Nodes  : individual public V2G hubs (one node per hub)
Edges  : pairs of hubs within MAX_EDGE_DISTANCE_KM road-network km
         (default 6 km — hubs within this range draw from overlapping
         EV owner catchment areas and therefore have strategic
         interdependence that motivates the GAT encoder)
Edge weight : normalised road-network distance (0=adjacent, 1=threshold)

Node features (per hub, static — updated each env step only for dynamic
features via nem_wdr_env.HubConfig):
  loc_x, loc_y : position relative to VSR zone centroid, normalised [-1,1]
  distance_km  : road-network distance proxy from residential centroid

Data sources
------------
Hub locations : Open Charge Map public API (https://api.openchargemap.io/v3/poi/)
                Creative Commons licence, accessed 2025
Road network  : OpenStreetMap via OSMnx (Boeing 2017), ODbL licence

Offline / fallback mode
-----------------------
If the OCM API or OSMnx is unavailable (no internet, rate limit), the module
falls back to a synthetic hub set generated from the zone centroid.
Pass use_synthetic=True to HubGraphBuilder to force this mode — useful in
CI/CD and on compute clusters without outbound HTTP.

PyG compatibility
-----------------
When torch_geometric is installed, to_pyg_data() returns a real
torch_geometric.data.Data object. When it is not installed, it returns a
GraphData dataclass with the same .x, .edge_index, .edge_attr attributes
backed by numpy arrays. The GAT agent in baselines/gnn_rl/networks.py
checks for the type at import time and imports accordingly.

Usage
-----
    builder = HubGraphBuilder(zone="inner_melbourne", use_synthetic=False)
    graph = builder.build()          # GraphData or torch_geometric.data.Data
    hub_configs = builder.hub_configs  # list[HubConfig] for nem_wdr_env

    # Persist to disk (avoid re-fetching every run)
    builder.save("data/graphs/inner_melbourne.pkl")
    graph, hub_configs = HubGraphBuilder.load("data/graphs/inner_melbourne.pkl")
"""

import os
import math
import pickle
import logging
import warnings
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyG availability — resolved lazily, never at import time.
# Importing torch at module level causes a bus error on some architectures
# (e.g. ARM sandboxes with an x86-compiled torch wheel).
# ---------------------------------------------------------------------------
_HAS_PYG: bool = False   # set to True on first successful _check_pyg() call


def _check_pyg() -> bool:
    """Try to import torch + torch_geometric. Safe to call multiple times."""
    global _HAS_PYG
    if _HAS_PYG:
        return True
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return False
        import torch               # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        _HAS_PYG = True
    except Exception:
        _HAS_PYG = False
    return _HAS_PYG


# ---------------------------------------------------------------------------
# Zone registry
# ---------------------------------------------------------------------------
ZONE_REGISTRY = {
    "inner_melbourne": {
        "display_name": "Inner Melbourne VSR Zone",
        # NEM VSR zone: VIC1 region, inner Melbourne DNSP boundary
        "centroid_lat": -37.8136,
        "centroid_lon": 144.9631,
        "radius_km": 8.0,
        # Open Charge Map bounding box (lat_min, lat_max, lon_min, lon_max)
        "ocm_bbox": (-37.88, -37.75, 144.90, 145.03),
        # Connection type IDs: 25 = Type 2 CCS (DC fast), 33 = CHAdeMO, 2 = Type 2 AC
        # We filter for DC fast chargers only (bidirectional V2G capable)
        "ocm_connection_types": [25, 33, 2],
        "ocm_min_kw": 7.0,   # minimum charger power to qualify (kW)
    }
}

# Graph construction parameters — treated as sensitivity parameters in §4.3.4
MAX_EDGE_DISTANCE_KM = 6.0    # hubs within this road distance are connected
PROXIMITY_RADIUS_KM  = 10.0   # OSMnx graph download radius around centroid (km)

# Normalisation constants
COORD_NORMALISE_KM = 8.0      # VSR zone radius; used to normalise loc_x, loc_y


# ---------------------------------------------------------------------------
# GraphData fallback (numpy-backed, same interface as PyG Data)
# ---------------------------------------------------------------------------
@dataclass
class GraphData:
    """
    Numpy-backed graph data container.

    Provides the same .x, .edge_index, .edge_attr interface as
    torch_geometric.data.Data so the GAT agent can use either
    interchangeably.

    Attributes
    ----------
    x : np.ndarray, shape (H, node_feature_dim)
        Static node feature matrix. Columns: [loc_x, loc_y, distance_km_norm].
        Dynamic features (n_connected, mean_soc, p_max_kw) are added by
        nem_wdr_env.py at each step before passing to the GAT encoder.
    edge_index : np.ndarray, shape (2, E), dtype int64
        COO-format adjacency. edge_index[0] = source nodes,
        edge_index[1] = target nodes. Edges are bidirectional.
    edge_attr : np.ndarray, shape (E, 1)
        Normalised road-network distance for each edge [0, 1].
        0 = adjacent hubs, 1 = at MAX_EDGE_DISTANCE_KM threshold.
    hub_ids : list[int]
        Hub IDs in the same order as rows of x.
    n_nodes : int
        Number of hubs (H).
    n_edges : int
        Number of directed edges (bidirectional, so 2 × undirected count).
    zone_name : str
    """
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    hub_ids: list
    n_nodes: int
    n_edges: int
    zone_name: str = "inner_melbourne"

    def to_pyg(self):
        """Convert to torch_geometric.data.Data if PyG is available."""
        if not _check_pyg():
            raise ImportError(
                "torch_geometric not installed or not loadable. "
                "Run: conda install pyg -c pyg"
            )
        import torch
        from torch_geometric.data import Data as PyGData
        return PyGData(
            x=torch.tensor(self.x, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# HubConfig import (avoid circular import — redefine the minimal fields here
# and import the full version from nem_wdr_env when available)
# ---------------------------------------------------------------------------
@dataclass
class HubConfig:
    """
    Static configuration for one V2G hub.
    Mirrors nem_wdr_env.HubConfig exactly — kept here to avoid circular imports.
    nem_wdr_env.py imports from this module at runtime.
    """
    hub_id: int
    distance_km: float       # road-network distance proxy from residential centroid
    loc_x: float             # position relative to VSR centroid, normalised [-1,1]
    loc_y: float
    n_chargers: int = 4
    charger_max_kw: float = 22.0
    lat: float = 0.0         # WGS84 latitude (for OSMnx routing)
    lon: float = 0.0         # WGS84 longitude

    @property
    def p_max_kw(self) -> float:
        return self.n_chargers * self.charger_max_kw


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS84 points (km).
    Used as a fast pre-filter before road-network shortest path.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_to_local_xy(lat: float, lon: float,
                        ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """
    Convert (lat, lon) to approximate local Cartesian (km east, km north)
    relative to a reference point.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(ref_lat))
    x_km = (lon - ref_lon) * km_per_deg_lon   # east positive
    y_km = (lat - ref_lat) * km_per_deg_lat   # north positive
    return x_km, y_km


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class HubGraphBuilder:
    """
    Builds the spatial hub graph for a named VSR zone.

    Parameters
    ----------
    zone : str
        Zone key in ZONE_REGISTRY. Default "inner_melbourne".
    max_edge_distance_km : float
        Maximum road-network distance for an edge between two hubs.
    use_synthetic : bool
        If True, skip API calls and generate a synthetic hub set.
        Useful for offline development and CI testing.
    ocm_api_key : str, optional
        Open Charge Map API key. If None, uses unauthenticated access
        (rate-limited to ~100 req/hr — sufficient for thesis development).
    rng : np.random.Generator, optional
        RNG for synthetic hub generation. Seeded for reproducibility.
    n_synthetic_hubs : int
        Number of synthetic hubs to generate when use_synthetic=True.

    Attributes (populated after build())
    ------------------------------------
    hub_configs : list[HubConfig]
        One HubConfig per hub, in the same order as graph nodes.
    road_graph : networkx.MultiDiGraph
        OSMnx road network for the VSR zone (None if use_synthetic=True).
    distance_matrix_km : np.ndarray, shape (H, H)
        Pairwise road-network distances between hubs.
    """

    def __init__(
        self,
        zone: str = "inner_melbourne",
        max_edge_distance_km: float = MAX_EDGE_DISTANCE_KM,
        use_synthetic: bool = False,
        ocm_api_key: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        n_synthetic_hubs: int = 15,
    ):
        if zone not in ZONE_REGISTRY:
            raise ValueError(f"Unknown zone '{zone}'. Available: {list(ZONE_REGISTRY)}")

        self.zone_cfg = ZONE_REGISTRY[zone]
        self.zone_name = zone
        self.max_edge_distance_km = max_edge_distance_km
        self.use_synthetic = use_synthetic
        self.ocm_api_key = ocm_api_key
        self._rng = rng if rng is not None else np.random.default_rng(42)
        self.n_synthetic_hubs = n_synthetic_hubs

        # Populated by build()
        self.hub_configs: list[HubConfig] = []
        self.road_graph = None
        self.distance_matrix_km: Optional[np.ndarray] = None
        self._graph_data: Optional[GraphData] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> GraphData:
        """
        Build the hub graph end-to-end.

        Steps:
          1. Fetch or generate hub locations
          2. Download OSMnx road network (or skip for synthetic)
          3. Compute pairwise road-network distances
          4. Construct graph edges within threshold
          5. Assemble GraphData / PyG Data object

        Returns
        -------
        GraphData (or torch_geometric.data.Data if PyG available)
        """
        logger.info(f"Building hub graph for zone: {self.zone_cfg['display_name']}")

        # Step 1: hub locations
        raw_hubs = (
            self._generate_synthetic_hubs()
            if self.use_synthetic
            else self._fetch_ocm_hubs()
        )

        if len(raw_hubs) == 0:
            warnings.warn(
                "No hubs found from OCM API — falling back to synthetic hubs. "
                "Check your internet connection or set use_synthetic=True.",
                RuntimeWarning,
            )
            raw_hubs = self._generate_synthetic_hubs()

        logger.info(f"  {len(raw_hubs)} hub locations acquired")

        # Step 2: road network
        if not self.use_synthetic and len(raw_hubs) > 0:
            self.road_graph = self._download_road_network()
        else:
            self.road_graph = None

        # Step 3: pairwise distances
        self.distance_matrix_km = self._compute_distance_matrix(raw_hubs)

        # Step 4: build hub configs
        self.hub_configs = self._build_hub_configs(raw_hubs)

        # Step 5: assemble graph
        self._graph_data = self._build_graph_data()

        logger.info(
            f"  Graph: {self._graph_data.n_nodes} nodes, "
            f"{self._graph_data.n_edges} directed edges "
            f"(threshold={self.max_edge_distance_km} km)"
        )

        return self._graph_data

    def save(self, path: str) -> None:
        """Persist the built graph and hub configs to disk."""
        if self._graph_data is None:
            raise RuntimeError("Call build() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "graph_data": self._graph_data,
                "hub_configs": self.hub_configs,
                "distance_matrix_km": self.distance_matrix_km,
                "zone_name": self.zone_name,
                "max_edge_distance_km": self.max_edge_distance_km,
            }, f)
        logger.info(f"Graph saved to {path}")

    @staticmethod
    def load(path: str) -> tuple["GraphData", list[HubConfig]]:
        """Load a previously saved graph from disk."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        logger.info(
            f"Loaded graph: {d['graph_data'].n_nodes} hubs, zone={d['zone_name']}"
        )
        return d["graph_data"], d["hub_configs"]

    # ------------------------------------------------------------------
    # Step 1: Hub location acquisition
    # ------------------------------------------------------------------

    def _fetch_ocm_hubs(self) -> list[dict]:
        """
        Fetch EV charging station locations from Open Charge Map API.

        Filters for stations within the zone bounding box that have at
        least one fast charger (≥7 kW) — proxy for V2G-capable hardware.

        Returns list of dicts with keys: lat, lon, n_chargers, max_kw.

        API reference: https://openchargemap.org/site/develop/api
        Data licence: Creative Commons Attribution 4.0
        """
        try:
            import requests
        except ImportError:
            warnings.warn("requests not installed — using synthetic hubs")
            return []

        cfg = self.zone_cfg
        lat_min, lat_max, lon_min, lon_max = cfg["ocm_bbox"]

        params = {
            "output": "json",
            "countrycode": "AU",
            "latitude": cfg["centroid_lat"],
            "longitude": cfg["centroid_lon"],
            "distance": cfg["radius_km"],
            "distanceunit": "km",
            "maxresults": 100,
            "compact": True,
            "verbose": False,
            "connectiontypeid": ",".join(str(x) for x in cfg["ocm_connection_types"]),
        }
        if self.ocm_api_key:
            params["key"] = self.ocm_api_key

        url = "https://api.openchargemap.io/v3/poi/"

        try:
            logger.info(f"  Fetching OCM hubs from {url}")
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            stations = resp.json()
        except Exception as e:
            warnings.warn(f"OCM API request failed ({e}) — using synthetic hubs")
            return []

        hubs = []
        for station in stations:
            addr = station.get("AddressInfo", {})
            lat = addr.get("Latitude")
            lon = addr.get("Longitude")
            if lat is None or lon is None:
                continue

            # Only include stations inside the bounding box
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue

            # Count fast-charge connections
            connections = station.get("Connections", [])
            fast_connections = [
                c for c in connections
                if (c.get("PowerKW") or 0) >= cfg["ocm_min_kw"]
            ]
            if not fast_connections:
                continue

            n_chargers = len(fast_connections)
            max_kw = max((c.get("PowerKW") or 7.0) for c in fast_connections)

            hubs.append({
                "lat": float(lat),
                "lon": float(lon),
                "n_chargers": min(n_chargers, 8),   # cap at 8 chargers per hub
                "max_kw": float(min(max_kw, 50.0)), # cap at 50 kW per charger
                "name": addr.get("Title", f"Hub_{len(hubs)}"),
            })

        logger.info(f"  OCM returned {len(hubs)} qualifying hubs")
        return hubs

    def _generate_synthetic_hubs(self) -> list[dict]:
        """
        Generate synthetic hub locations within the VSR zone.

        Used when OCM API is unavailable or use_synthetic=True.
        Hubs are placed by perturbing the centroid with Gaussian noise
        scaled to the zone radius, then clipping to the bounding box.

        The synthetic positions are deterministic given the RNG seed,
        ensuring reproducibility across runs.
        """
        cfg = self.zone_cfg
        ref_lat = cfg["centroid_lat"]
        ref_lon = cfg["centroid_lon"]
        radius_km = cfg["radius_km"]
        n = self.n_synthetic_hubs

        # Degrees per km at this latitude
        deg_per_km_lat = 1.0 / 111.32
        deg_per_km_lon = 1.0 / (111.32 * math.cos(math.radians(ref_lat)))

        # Sample hub positions: Gaussian in local km coords, then convert to lat/lon
        # Use radius/2 as std so most hubs fall within the zone
        std_km = radius_km / 2.5
        x_km = self._rng.normal(0, std_km, n)  # east offset
        y_km = self._rng.normal(0, std_km, n)  # north offset

        # Clip to zone radius (discard hubs outside the circle)
        distances = np.sqrt(x_km**2 + y_km**2)
        keep = distances <= radius_km
        x_km, y_km = x_km[keep], y_km[keep]

        # If we discarded too many, pad with uniform samples
        while len(x_km) < n:
            extra = int((n - len(x_km)) * 1.5)
            xe = self._rng.normal(0, std_km, extra)
            ye = self._rng.normal(0, std_km, extra)
            de = np.sqrt(xe**2 + ye**2)
            xe, ye = xe[de <= radius_km], ye[de <= radius_km]
            x_km = np.concatenate([x_km, xe])[:n]
            y_km = np.concatenate([y_km, ye])[:n]

        x_km, y_km = x_km[:n], y_km[:n]

        hubs = []
        for i in range(len(x_km)):
            lat = ref_lat + y_km[i] * deg_per_km_lat
            lon = ref_lon + x_km[i] * deg_per_km_lon
            hubs.append({
                "lat": float(lat),
                "lon": float(lon),
                "n_chargers": int(self._rng.integers(2, 7)),
                "max_kw": float(self._rng.choice([7.0, 11.0, 22.0, 50.0])),
                "name": f"SyntheticHub_{i:02d}",
            })

        logger.info(f"  Generated {len(hubs)} synthetic hubs")
        return hubs

    # ------------------------------------------------------------------
    # Step 2: Road network download
    # ------------------------------------------------------------------

    def _download_road_network(self):
        """
        Download OSMnx road network centred on the VSR zone.

        Uses the 'drive' network type (driveable roads) since EV owners
        travel by car to reach hubs. The network is used exclusively for
        computing pairwise shortest-path distances.

        Returns networkx.MultiDiGraph or None on failure.
        """
        try:
            import osmnx as ox
            cfg = self.zone_cfg
            logger.info(
                f"  Downloading OSMnx road network "
                f"(centre: {cfg['centroid_lat']:.4f}, {cfg['centroid_lon']:.4f}, "
                f"radius: {PROXIMITY_RADIUS_KM} km)..."
            )
            G = ox.graph_from_point(
                (cfg["centroid_lat"], cfg["centroid_lon"]),
                dist=PROXIMITY_RADIUS_KM * 1000,  # metres
                network_type="drive",
                simplify=True,
            )
            logger.info(
                f"  Road network: {len(G.nodes)} nodes, {len(G.edges)} edges"
            )
            return G
        except Exception as e:
            warnings.warn(
                f"OSMnx road network download failed ({e}). "
                "Falling back to Haversine (straight-line) distances. "
                "Road-network distances will be unavailable for this run."
            )
            return None

    # ------------------------------------------------------------------
    # Step 3: Pairwise distance computation
    # ------------------------------------------------------------------

    def _compute_distance_matrix(self, hubs: list[dict]) -> np.ndarray:
        """
        Compute pairwise road-network distances (km) between all hub pairs.

        If the road network is available, projects hub coordinates to the
        nearest OSM node and runs Dijkstra shortest path on edge lengths.
        Falls back to Haversine great-circle distance if the road network
        is unavailable or if a hub cannot be projected.

        Returns
        -------
        np.ndarray, shape (H, H)
            Symmetric matrix. Entry [i, j] = road-network km from hub i to j.
            Diagonal is 0.
        """
        H = len(hubs)
        D = np.zeros((H, H))

        if self.road_graph is not None:
            D = self._road_network_distances(hubs)
        else:
            # Haversine fallback
            logger.info("  Using Haversine distances (no road network)")
            for i in range(H):
                for j in range(i + 1, H):
                    d = haversine_km(
                        hubs[i]["lat"], hubs[i]["lon"],
                        hubs[j]["lat"], hubs[j]["lon"],
                    )
                    D[i, j] = D[j, i] = d

        return D

    def _road_network_distances(self, hubs: list[dict]) -> np.ndarray:
        """
        Project hubs to nearest OSM nodes and run pairwise Dijkstra.

        OSMnx edge lengths are in metres; we convert to km.
        If shortest path is not found (disconnected graph), falls back
        to Haversine for that pair.
        """
        try:
            import osmnx as ox
        except ImportError:
            return self._haversine_matrix(hubs)

        G = self.road_graph
        H = len(hubs)
        D = np.zeros((H, H))

        # Project each hub to its nearest OSM node
        logger.info(f"  Projecting {H} hubs to road network nodes...")
        osm_nodes = []
        for hub in hubs:
            try:
                node = ox.nearest_nodes(G, hub["lon"], hub["lat"])
                osm_nodes.append(node)
            except Exception:
                osm_nodes.append(None)

        # Pairwise shortest paths using networkx Dijkstra on 'length' (metres)
        for i in range(H):
            for j in range(i + 1, H):
                ni, nj = osm_nodes[i], osm_nodes[j]
                if ni is None or nj is None:
                    # Fallback to Haversine for this pair
                    d = haversine_km(
                        hubs[i]["lat"], hubs[i]["lon"],
                        hubs[j]["lat"], hubs[j]["lon"],
                    )
                else:
                    try:
                        length_m = nx.shortest_path_length(
                            G, ni, nj, weight="length"
                        )
                        d = length_m / 1000.0
                    except nx.NetworkXNoPath:
                        d = haversine_km(
                            hubs[i]["lat"], hubs[i]["lon"],
                            hubs[j]["lat"], hubs[j]["lon"],
                        )
                D[i, j] = D[j, i] = d

        return D

    def _haversine_matrix(self, hubs: list[dict]) -> np.ndarray:
        H = len(hubs)
        D = np.zeros((H, H))
        for i in range(H):
            for j in range(i + 1, H):
                d = haversine_km(
                    hubs[i]["lat"], hubs[i]["lon"],
                    hubs[j]["lat"], hubs[j]["lon"],
                )
                D[i, j] = D[j, i] = d
        return D

    # ------------------------------------------------------------------
    # Step 4: HubConfig construction
    # ------------------------------------------------------------------

    def _build_hub_configs(self, hubs: list[dict]) -> list[HubConfig]:
        """
        Convert raw hub dicts to HubConfig objects.

        loc_x, loc_y are normalised positions relative to zone centroid:
            loc = local_xy_km / COORD_NORMALISE_KM
            ∈ [-1, 1] for hubs within the zone radius

        distance_km is the road-network distance from the zone centroid
        (used as the travel distance proxy d_i in ρ(c, d, s)).
        """
        cfg = self.zone_cfg
        ref_lat = cfg["centroid_lat"]
        ref_lon = cfg["centroid_lon"]

        hub_configs = []
        for i, hub in enumerate(hubs):
            x_km, y_km = latlon_to_local_xy(hub["lat"], hub["lon"], ref_lat, ref_lon)

            # Normalise to [-1, 1] using zone radius
            loc_x = float(np.clip(x_km / COORD_NORMALISE_KM, -1.0, 1.0))
            loc_y = float(np.clip(y_km / COORD_NORMALISE_KM, -1.0, 1.0))

            # Distance from centroid = proxy for residential travel cost d_i
            distance_from_centroid_km = math.sqrt(x_km**2 + y_km**2)

            hub_configs.append(HubConfig(
                hub_id=i,
                distance_km=round(distance_from_centroid_km, 2),
                loc_x=loc_x,
                loc_y=loc_y,
                n_chargers=hub.get("n_chargers", 4),
                charger_max_kw=hub.get("max_kw", 22.0),
                lat=hub["lat"],
                lon=hub["lon"],
            ))

        return hub_configs

    # ------------------------------------------------------------------
    # Step 5: Graph assembly
    # ------------------------------------------------------------------

    def _build_graph_data(self) -> GraphData:
        """
        Assemble the GraphData object from hub configs and distance matrix.

        Node features (static, 3 features per node):
            [loc_x, loc_y, distance_km_norm]
            where distance_km_norm = distance_from_centroid / zone_radius

        Dynamic features (n_connected, mean_soc, p_max_kw_norm) are NOT
        included here — they are injected by nem_wdr_env.py each step
        before passing to the GAT encoder. The static node features provide
        the spatial prior; the dynamic features carry the operational state.

        Edge construction:
            Connect hubs i↔j if distance_matrix[i,j] <= max_edge_distance_km.
            Edges are bidirectional (both (i→j) and (j→i) added).
            Edge weight = 1 - d_ij / max_edge_distance_km
            (1 = adjacent hubs, 0 = at threshold — closer = higher weight)
        """
        H = len(self.hub_configs)
        D = self.distance_matrix_km
        zone_radius = self.zone_cfg["radius_km"]

        # --- Node feature matrix ---
        x = np.zeros((H, 3), dtype=np.float32)
        for i, hc in enumerate(self.hub_configs):
            x[i, 0] = hc.loc_x
            x[i, 1] = hc.loc_y
            x[i, 2] = float(np.clip(hc.distance_km / zone_radius, 0.0, 1.0))

        # --- Edge construction ---
        src_list, dst_list, weight_list = [], [], []
        for i in range(H):
            for j in range(H):
                if i == j:
                    continue
                d_ij = D[i, j]
                if d_ij <= self.max_edge_distance_km:
                    src_list.append(i)
                    dst_list.append(j)
                    # Weight: closer hubs → higher weight (more catchment overlap)
                    w = 1.0 - d_ij / self.max_edge_distance_km
                    weight_list.append(w)

        if len(src_list) == 0:
            # Fallback: connect every hub to its nearest neighbour
            # (ensures the graph is connected even with sparse hubs)
            warnings.warn(
                f"No edges found within {self.max_edge_distance_km} km. "
                "Connecting each hub to its nearest neighbour.",
                RuntimeWarning,
            )
            for i in range(H):
                row = D[i].copy()
                row[i] = np.inf
                j = int(np.argmin(row))
                src_list += [i, j]
                dst_list += [j, i]
                d_ij = D[i, j]
                w = 1.0 - min(d_ij, self.max_edge_distance_km) / self.max_edge_distance_km
                weight_list += [w, w]

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(weight_list, dtype=np.float32).reshape(-1, 1)

        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            hub_ids=list(range(H)),
            n_nodes=H,
            n_edges=len(src_list),
            zone_name=self.zone_name,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def graph_summary(self) -> dict:
        """Return summary statistics for logging and thesis reporting."""
        if self._graph_data is None:
            return {}
        D = self.distance_matrix_km
        H = self._graph_data.n_nodes
        if H < 2:
            return {"n_hubs": H}

        # Pairwise distances (upper triangle only)
        upper = D[np.triu_indices(H, k=1)]
        edge_weights = self._graph_data.edge_attr.flatten()
        nx_G = self._to_networkx()

        return {
            "n_hubs": H,
            "n_directed_edges": self._graph_data.n_edges,
            "n_undirected_edges": self._graph_data.n_edges // 2,
            "avg_degree": self._graph_data.n_edges / H,
            "mean_pairwise_dist_km": float(upper.mean()),
            "max_pairwise_dist_km": float(upper.max()),
            "min_pairwise_dist_km": float(upper.min()),
            "mean_edge_weight": float(edge_weights.mean()),
            "is_connected": nx.is_weakly_connected(nx_G),
            "zone": self.zone_cfg["display_name"],
            "edge_threshold_km": self.max_edge_distance_km,
        }

    def _to_networkx(self) -> nx.DiGraph:
        """Convert graph data to networkx DiGraph (for analysis)."""
        G = nx.DiGraph()
        gd = self._graph_data
        G.add_nodes_from(range(gd.n_nodes))
        for k in range(gd.n_edges):
            src = int(gd.edge_index[0, k])
            dst = int(gd.edge_index[1, k])
            w = float(gd.edge_attr[k, 0])
            G.add_edge(src, dst, weight=w)
        return G
