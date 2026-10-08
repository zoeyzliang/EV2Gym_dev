"""
build_cre21.py
==============
Convert the CRE21 network (Team-Nando/MV-LV-Networks, BSD-3, AusNet Services
data) into this repo's network format for the CRE21 robustness study
(spec §6, "CRE21 network study"):

  ev2gym/data/network_data/node_cre21/
    Nodes_cre21.csv   NODES, Tb, PD, QD, Pct, Ict, Zct      (slack = bus 1)
    Lines_cre21.csv   FROM, TO, R, X, B, STATUS, TAP, RATING_KW
    Sites_cre21.csv   transformer LV busbars: NODE, KVA, TYPE, CUSTOMERS, RATING_KW
    SOURCE.md         provenance, licence, modelling choices

Balanced positive-sequence model of the 22 kV network (as in HetGPS):
  - slack = 22 kV side of the zone substation (bus 111);
  - MV lines: linecode r1, x1 (ohm/km) × length; parallel duplicate rows
    combined (admittances added, ratings summed); line charging ignored;
    rating = √3 × 22 kV × Ampacity1 × 0.95 (kW);
  - each distribution transformer: a branch from its MV bus to a new LV
    busbar node, R = loadloss% and X = xhl% on its kVA base referred to
    22 kV; rating = kVA × 0.95 (kW);
  - load at each LV busbar: residential = 99th percentile of the aggregate
    of `Customers` household profiles drawn (seeded) from the CRE21 profile
    pool; C&I = 0.5 × kVA × 0.95 kW (assumption); Q at pf 0.95.
LV networks below the transformers are not modelled.

Usage:
  python tools/build_cre21.py <path to MV-LV-Networks checkout>
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

V_KV = 22.0
PF = 0.95
SEED = 2026
OUT = Path(__file__).resolve().parent.parent / "ev2gym" / "data" / "network_data" / "node_cre21"


def main(src: str):
    src = Path(src)
    xl = pd.ExcelFile(src / "Network_4_Urban_CRE21.xlsx")
    lines = pd.read_excel(xl, "lines")
    codes = pd.read_excel(xl, "linecodes").set_index("Linecode_ID")
    tx = pd.read_excel(xl, "lvtx")
    res = np.load(src / "prof" / "Res_load_data_30min_res.npy")          # (houses, 365, 48) kW

    # ---- MV lines, parallel duplicates combined -------------------------
    code = lines["Linecode"].astype(str) + "-" + lines["Phases"].astype(str) + "ph"
    r = codes.loc[code, "r1"].to_numpy() * lines["Length"].to_numpy()
    x = codes.loc[code, "x1"].to_numpy() * lines["Length"].to_numpy()
    amp = codes.loc[code, "Ampacity1"].to_numpy()
    mv = pd.DataFrame({"u": lines["Start_Node"].astype(str), "v": lines["End_Node"].astype(str),
                       "y": 1.0 / (r + 1j * x), "rating": np.sqrt(3) * V_KV * amp * PF})
    mv["key"] = [tuple(sorted(e)) for e in zip(mv.u, mv.v)]
    mv = mv.groupby("key").agg(y=("y", "sum"), rating=("rating", "sum"), n=("y", "size")).reset_index()

    g = nx.Graph()
    g.add_edges_from(mv["key"])
    assert nx.is_tree(g), "CRE21 MV network is expected to be radial"
    root = "111"

    # ---- transformers as branches to new LV busbar nodes ------------------
    rows = []
    for k, t in tx.reset_index(drop=True).iterrows():
        lv = f"LV{k}"
        zb = V_KV ** 2 * 1000.0 / float(t["kvas_primary"])                # ohm at 22 kV
        z = (float(t["loadloss"]) + 1j * float(t["xhl"])) / 100.0 * zb
        rows.append({"key": (str(int(t["Bus1"])), lv), "y": 1.0 / z,
                     "rating": float(t["kvas_primary"]) * PF, "n": 1})
        g.add_edge(str(int(t["Bus1"])), lv)
    br = pd.concat([mv, pd.DataFrame(rows)], ignore_index=True)

    # ---- number buses: slack first, then BFS order ------------------------
    order = [root] + [v for _, v in nx.bfs_edges(g, root)]
    num = {b: i + 1 for i, b in enumerate(order)}
    parent = {v: u for u, v in nx.bfs_edges(g, root)}

    # ---- loads at LV busbars ----------------------------------------------
    rng = np.random.default_rng(SEED)
    flat = res.reshape(res.shape[0], -1)                                   # (houses, 17520)
    pd_kw = {}
    sites = []
    for k, t in tx.reset_index(drop=True).iterrows():
        n = int(t["Customers"])
        if t["Type"] == "RES":
            idx = rng.choice(flat.shape[0], size=n, replace=n > flat.shape[0])
            p = float(np.percentile(flat[idx].sum(axis=0), 99))
        else:
            p = 0.5 * float(t["kvas_primary"]) * PF
        pd_kw[f"LV{k}"] = p
        sites.append({"NODE": num[f"LV{k}"], "KVA": float(t["kvas_primary"]), "TYPE": t["Type"],
                      "CUSTOMERS": n, "RATING_KW": float(t["kvas_primary"]) * PF, "PEAK_KW": round(p, 3)})

    tanphi = np.tan(np.arccos(PF))
    nodes = pd.DataFrame({"NODES": [num[b] for b in order],
                          "Tb": [1] + [0] * (len(order) - 1),
                          "PD": [round(pd_kw.get(b, 0.0), 4) for b in order],
                          "QD": [round(pd_kw.get(b, 0.0) * tanphi, 4) for b in order],
                          "Pct": 1, "Ict": 0, "Zct": 0})

    out = []
    for _, b in br.iterrows():
        u, v = b["key"]
        f, t_ = (u, v) if parent.get(v) == u else (v, u)
        z = 1.0 / b["y"]
        out.append({"FROM": num[f], "TO": num[t_], "R": round(z.real, 6), "X": round(z.imag, 6),
                    "B": 0, "STATUS": 1, "TAP": 1, "RATING_KW": round(b["rating"], 2)})
    lines_out = pd.DataFrame(out).sort_values("TO").reset_index(drop=True)
    assert len(lines_out) == len(nodes) - 1

    OUT.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUT / "Nodes_cre21.csv", index=False)
    lines_out.to_csv(OUT / "Lines_cre21.csv", index=False)
    pd.DataFrame(sites).to_csv(OUT / "Sites_cre21.csv", index=False)
    (OUT / "SOURCE.md").write_text(
        "CRE21 urban 22 kV network converted by tools/build_cre21.py from\n"
        "https://github.com/Team-Nando/MV-LV-Networks (BSD 3-Clause; data courtesy of AusNet Services;\n"
        "cite Ochoa et al.). Balanced positive-sequence MV model with distribution transformers as branches;\n"
        "LV networks not modelled. See the docstring of tools/build_cre21.py for all modelling choices.\n")
    print(f"nodes {len(nodes)}, branches {len(lines_out)} (MV {len(mv)}, of which parallel-combined "
          f"{int((mv.n > 1).sum())}; transformers {len(tx)})")
    print(f"total residential + C&I peak {nodes.PD.sum():.0f} kW; per-customer (res) "
          f"{sum(s['PEAK_KW'] for s in sites if s['TYPE'] == 'RES') / max(1, sum(s['CUSTOMERS'] for s in sites if s['TYPE'] == 'RES')):.2f} kW")


if __name__ == "__main__":
    main(sys.argv[1])
