"""
build_graph.py
Fetch hub locations from Open Charge Map for a given zone and cache the
resulting graph to disk. After this, training loads from cache.

Usage:
    python build_graph.py --zone inner_melbourne
    python build_graph.py --zone greater_melbourne
    python build_graph.py --zone inner_melbourne --synthetic

Fixes applied (this revision)
-------------------------------
The previous version of this script had no real CLI (a bare
`"--synthetic" in sys.argv` check, everything else hardcoded to
"inner_melbourne"), was missing `import json` (crashed after building
the graph but before writing the config file), and always wrote to the
same fixed output path regardless of zone. Together this meant running
the script with ANY argument (including an unrecognised one like
`--help`) silently regenerated the inner_melbourne graph and overwrote
data/graphs/inner_melbourne.pkl in place — discovered when a bare
`--help` invocation did exactly this and had to be reverted with
`git checkout`. Output filenames are now zone-derived
(data/graphs/<zone>.pkl), so a mistaken invocation for one zone can
never silently overwrite another zone's graph file, and --help now
works correctly via argparse instead of falling through to the default
build behaviour.
"""
import json
import argparse
import logging

from nem_env.spatial_graph import HubGraphBuilder, ZONE_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and cache a hub graph for a given zone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zone", type=str, default="inner_melbourne",
        choices=list(ZONE_REGISTRY.keys()),
        help="Zone key from ZONE_REGISTRY (nem_env/spatial_graph.py).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic/offline hub generation instead of real "
             "OpenChargeMap data (useful if OCM/OSMnx network access "
             "is unavailable, e.g. on some compute-node configurations).",
    )
    parser.add_argument(
        "--ocm_api_key", type=str,
        default="f99a32f1-30fb-46d4-8127-486d1c9faade",
        help="OpenChargeMap API key.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/graphs",
        help="Output directory. Files are written as "
             "<out_dir>/<zone>.pkl and <out_dir>/<zone>_config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_pkl = f"{args.out_dir}/{args.zone}.pkl"
    out_json = f"{args.out_dir}/{args.zone}_config.json"

    logger.info(f"Building graph for zone='{args.zone}' (synthetic={args.synthetic})")
    logger.info(f"Output: {out_pkl}")

    builder = HubGraphBuilder(
        zone=args.zone,
        use_synthetic=args.synthetic,
        ocm_api_key=args.ocm_api_key,
    )
    graph = builder.build()
    builder.save(out_pkl)

    summary = builder.graph_summary()
    print(summary)
    logger.info(
        f"Built {summary.get('n_hubs', '?')} hubs for zone '{args.zone}' "
        f"— if this doesn't look right (e.g. far from an intended hub "
        f"count for a scaling experiment), adjust the zone's radius_km / "
        f"ocm_bbox in ZONE_REGISTRY and rebuild before using this graph "
        f"for training."
    )

    # ── Save human-readable config ──────────────────────────────
    config = {
        "zone": args.zone,
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
    with open(out_json, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {out_json}")


if __name__ == "__main__":
    main()
