"""
build_graph.py
Run once to fetch Melbourne hub locations from Open Charge Map
and cache the graph to disk. After this, training loads from cache.

Usage:
    python build_graph.py                    # real OCM data
    python build_graph.py --synthetic        # offline/synthetic fallback
"""
import sys
from nem_env.spatial_graph import HubGraphBuilder

use_synthetic = "--synthetic" in sys.argv

builder = HubGraphBuilder(
    zone="inner_melbourne",
    use_synthetic=False,
    # n_synthetic_hubs=15,
    ocm_api_key="f99a32f1-30fb-46d4-8127-486d1c9faade",
)


graph = builder.build()
builder.save("data/graphs/inner_melbourne.pkl")

print(builder.graph_summary())



# ── Save human-readable config ──────────────────────────────
config = {
    "zone": "inner_melbourne",
    "n_hubs": graph.n_nodes,
    "n_directed_edges": graph.n_edges,
    "hub_configs": [
        {
            "hub_id": hc.hub_id,
            "distance_km": hc.distance_km,
            "loc_x": round(hc.loc_x, 4),
            "loc_y": round(hc.loc_y, 4),
            "n_chargers": hc.n_chargers,
            "charger_max_kw": hc.charger_max_kw,
            "p_max_kw": hc.p_max_kw,
            "lat": round(hc.lat, 6),
            "lon": round(hc.lon, 6),
        }
        for hc in builder.hub_configs
    ]
}
with open("data/graphs/inner_melbourne_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to data/graphs/inner_melbourne_config.json")