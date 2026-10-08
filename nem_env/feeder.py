"""
feeder.py
=========
Distribution-feeder model for the feeder-grounded DOE environment
(docs/env_redesign_spec.md §4.1–4.3).

* Network: the 34-node radial feeder shipped with EV2Gym (from RL-ADN),
  solved with the same tensor power flow as EV2Gym's GridTensor
  (re-implemented in nem_env/powerflow.py, without numba/pandapower).
* Hubs are placed on feeder buses by a stylised, disclosed rule: hubs further
  from the zone centre go to buses electrically further from the substation.
* Background load and rooftop PV per bus follow AEMO regional shapes for the
  same dates as prices (nem_env/grid_profiles.py).
* DOEs are computed the way a DNSP would, from power flow: each hub's
  individual hosting capacity (others at baseline), then scaled by a common
  factor so that all hubs at their DOE are jointly within voltage limits under
  the forecast ("scaled individual hosting capacity"). Thermal limits are not
  modelled (the feeder data has no line ratings).

Sign convention: hub net site flow p > 0 is import (adds to bus load), p < 0
is export. All powers in kW.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .powerflow import TensorPowerFlow

_NET_DIR = Path(__file__).resolve().parent.parent / "ev2gym" / "data" / "network_data"


@dataclass
class FeederConfig:
    """Feeder parameters (spec §4.2–4.3); defaults are the thesis settings."""
    network: str = "node_34"
    v_slack: float = 1.03          # substation set point (pu)
    v_min: float = 0.95            # voltage limits enforced by the DOE (pu)
    v_max: float = 1.05
    kappa_load: float = 1.0        # scale on the feeder's nominal bus loads (calibrated)
    pv_penetration: float = 0.6    # PV peak per bus as a fraction of that bus's peak load
    doe_block: int = 6             # steps per DOE publication (30 min)
    graph_k_hops: int = 3          # electrical graph: hubs within k feeder hops
    thermal_margin: float = 1.0    # section rating = margin × downstream design peak
                                   # (or × the network file's RATING_KW, where given)
    v_base: float = 11.0           # kV of the network files (node_34: 11; node_cre21: 22)
    siting: str = "base"           # hub siting on networks with a Sites file: base | constrained


class Feeder:
    """
    Feeder network, hub placement, background load, power flow and DOEs.

    Parameters
    ----------
    hub_configs : list of HubConfig
        Real hubs (OpenChargeMap); uses loc_x/loc_y and p_max_kw.
    cfg : FeederConfig
    """

    def __init__(self, hub_configs: list, cfg: FeederConfig = None):
        self.cfg = cfg or FeederConfig()
        n = self.cfg.network
        nodes_csv = _NET_DIR / n / f"Nodes_{n.split('_')[1]}.csv"
        lines_csv = _NET_DIR / n / f"Lines_{n.split('_')[1]}.csv"
        self.nodes = pd.read_csv(nodes_csv)
        self.lines = pd.read_csv(lines_csv)
        # Same algorithm/model as EV2Gym's GridTensor, without its numba /
        # pandapower dependencies (see nem_env/powerflow.py).
        self.pf = TensorPowerFlow(self.nodes, self.lines, v_base=self.cfg.v_base)
        sites_csv = _NET_DIR / n / f"Sites_{n.split('_')[1]}.csv"
        self.sites = pd.read_csv(sites_csv) if sites_csv.exists() else None

        # Non-slack buses (power-flow vectors exclude the slack bus 1)
        self.bus_ids = self.nodes["NODES"].to_numpy()[1:]
        self.nb = len(self.bus_ids)
        self.pd_nom = self.nodes["PD"].to_numpy(dtype=float)[1:]
        self.qd_nom = self.nodes["QD"].to_numpy(dtype=float)[1:]

        self._build_topology()

        self.hubs = hub_configs
        self.H = len(hub_configs)
        self.hub_cap = np.array([hc.p_max_kw for hc in hub_configs], dtype=float)
        self.hub_bus = self._map_hubs()          # index into the nb non-slack buses

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def _build_topology(self):
        import networkx as nx
        g = nx.Graph()
        for _, r in self.lines.iterrows():
            z = float(np.hypot(r["R"], r["X"]))
            g.add_edge(int(r["FROM"]), int(r["TO"]), z=z)
        self.tree = g
        slack = int(self.nodes["NODES"].iloc[0])
        dist = nx.single_source_dijkstra_path_length(g, slack, weight="z")
        self.elec_dist = np.array([dist[int(b)] for b in self.bus_ids])
        self.hops = dict(nx.all_pairs_shortest_path_length(g))
        self.zdist = dict(nx.all_pairs_dijkstra_path_length(g, weight="z"))

        # Thermal sections. The feeder is radial, so the (lossless) flow on
        # each section is the sum of bus powers downstream of it: flow = P @ M.T
        # with M[l, b] = 1 if bus b is downstream of section l. Section
        # ratings follow a design rule (no ratings in the network data):
        # rating = margin × design peak load downstream.
        col = {int(b): i for i, b in enumerate(self.bus_ids)}
        rooted = nx.bfs_tree(g, slack)
        self.sections = list(rooted.edges())
        self.M = np.zeros((len(self.sections), self.nb))
        for l, (_, v) in enumerate(self.sections):
            for n in nx.descendants(rooted, v) | {v}:
                self.M[l, col[int(n)]] = 1.0
        if "RATING_KW" in self.lines.columns:
            # real ratings (CRE21: line ampacity, transformer kVA), scaled by the margin
            r = {(int(a), int(b)): float(k) for a, b, k in
                 zip(self.lines["FROM"], self.lines["TO"], self.lines["RATING_KW"])}
            self.rating = self.cfg.thermal_margin * np.array(
                [r.get((int(u), int(v)), r.get((int(v), int(u)), np.nan)) for u, v in self.sections])
            if np.isnan(self.rating).any():
                raise ValueError("a section has no RATING_KW")
        else:
            self.rating = self.cfg.thermal_margin * (self.M @ self.pd_nom)

    def section_flows(self, P: np.ndarray) -> np.ndarray:
        """Section flows (kW, + towards loads) for bus powers P (..., nb)."""
        return P @ self.M.T

    def _map_hubs(self) -> np.ndarray:
        """
        Stylised placement (disclosed): rank hubs by radial distance from the
        zone centre, rank loaded buses by electrical distance from the
        substation, choose H buses at evenly spaced electrical-distance
        quantiles, and assign in rank order.
        """
        if self.sites is not None:
            return self._map_hubs_to_sites()
        loaded = np.where(self.pd_nom > 0)[0]
        if len(loaded) < self.H:
            raise ValueError(f"{self.H} hubs but only {len(loaded)} loaded buses")
        order = loaded[np.argsort(self.elec_dist[loaded], kind="stable")]
        pick = np.unique(np.round(np.linspace(0, len(order) - 1, self.H)).astype(int))
        buses = order[pick]
        radius = np.array([np.hypot(hc.loc_x, hc.loc_y) for hc in self.hubs])
        hub_rank = np.argsort(radius, kind="stable")
        hub_bus = np.empty(self.H, dtype=int)
        hub_bus[hub_rank] = buses
        return hub_bus

    def _map_hubs_to_sites(self) -> np.ndarray:
        """
        Hubs at distribution-transformer LV busbars (spec §6, CRE21 study),
        each at a distinct transformer whose rating ≥ the hub capacity.
          base: H targets at evenly spaced electrical-distance quantiles of all
                transformer busbars; hubs in radius rank order (as in the
                default rule) take the feasible free busbar nearest their target.
          constrained: hubs in decreasing capacity take the smallest feasible
                free transformer (ties: nearer the substation).
        """
        col = {int(b): i for i, b in enumerate(self.bus_ids)}
        site_bus = np.array([col[int(n)] for n in self.sites["NODE"]])
        site_cap = self.sites["RATING_KW"].to_numpy(float)
        free = np.ones(len(site_bus), bool)
        hub_bus = np.empty(self.H, dtype=int)
        d = self.elec_dist[site_bus]
        if self.cfg.siting == "base":
            targets = np.quantile(d, np.linspace(0, 1, self.H))
            radius = np.array([np.hypot(hc.loc_x, hc.loc_y) for hc in self.hubs])
            for k, h in enumerate(np.argsort(radius, kind="stable")):
                ok = free & (site_cap >= self.hub_cap[h])
                if not ok.any():
                    raise ValueError(f"no free transformer can host hub {h} ({self.hub_cap[h]} kW)")
                j = np.where(ok)[0][np.argmin(np.abs(d[ok] - targets[k]))]
                hub_bus[h] = site_bus[j]; free[j] = False
        elif self.cfg.siting == "constrained":
            for h in np.argsort(-self.hub_cap, kind="stable"):
                ok = np.where(free & (site_cap >= self.hub_cap[h]))[0]
                if not len(ok):
                    raise ValueError(f"no free transformer can host hub {h} ({self.hub_cap[h]} kW)")
                j = ok[np.lexsort((d[ok], site_cap[ok]))[0]]
                hub_bus[h] = site_bus[j]; free[j] = False
        else:
            raise ValueError(f"siting must be 'base' or 'constrained', got {self.cfg.siting!r}")
        return hub_bus

    def electrical_graph(self, base_graph):
        """
        GraphData over hubs: edge if the hubs' buses are within k hops in the
        feeder tree. Edge attribute = path impedance, normalised to [0, 1].
        Node features (x) and metadata are copied from the road graph.
        """
        from .spatial_graph import GraphData
        src, dst, w = [], [], []
        for i in range(self.H):
            for j in range(self.H):
                if i == j:
                    continue
                bi, bj = int(self.bus_ids[self.hub_bus[i]]), int(self.bus_ids[self.hub_bus[j]])
                if self.hops[bi][bj] <= self.cfg.graph_k_hops:
                    src.append(i); dst.append(j); w.append(self.zdist[bi][bj])
        w = np.array(w, dtype=np.float32)
        w = w / w.max() if len(w) and w.max() > 0 else w
        return GraphData(
            x=base_graph.x,
            edge_index=np.array([src, dst], dtype=np.int64).reshape(2, -1),
            edge_attr=w.reshape(-1, 1),
            hub_ids=base_graph.hub_ids,
            n_nodes=self.H,
            n_edges=len(src),
            zone_name=f"{base_graph.zone_name}_electrical_k{self.cfg.graph_k_hops}",
        )

    # ------------------------------------------------------------------
    # Background load and PV
    # ------------------------------------------------------------------
    def background(self, demand_shape: np.ndarray, pv_shape: np.ndarray):
        """
        Background (non-EV) bus power for a day.

        P_bg[t, b] = κ·PD_b·D_t − π·κ·PD_b·S_t     (PV at unity PF)
        Q_bg[t, b] = κ·QD_b·D_t

        D_t: demand shape (regional demand / its 99th percentile, so κ·PD_b
        is the bus's peak load); S_t: PV shape (0..1). π is the PV peak as a
        fraction of the bus's peak load. Returns arrays of shape (T, nb).
        """
        k = self.cfg.kappa_load
        d = np.asarray(demand_shape, dtype=float)[:, None]
        s = np.asarray(pv_shape, dtype=float)[:, None]
        load_p = k * self.pd_nom[None, :] * d
        pv = self.cfg.pv_penetration * k * self.pd_nom[None, :] * s
        return load_p - pv, k * self.qd_nom[None, :] * d

    # ------------------------------------------------------------------
    # Power flow
    # ------------------------------------------------------------------
    def voltages(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        |V| (pu) for scenarios P, Q of shape (S, nb) in kW/kvar.

        The solver fixes the slack at 1.0 pu; with constant-power loads the
        solution for a slack at Vs equals Vs × the solution for powers / Vs²,
        which is how the substation set point is applied.
        """
        P = np.atleast_2d(P); Q = np.atleast_2d(Q)
        vs = self.cfg.v_slack
        v, converged = self.pf.solve(P / vs**2, Q / vs**2)
        if not converged:
            raise RuntimeError("power flow did not converge")
        return vs * np.abs(v).reshape(P.shape)

    def with_hubs(self, P_bg, hub_p):
        """Add hub net flows (…, H) to background bus powers (…, nb)."""
        P = np.array(P_bg, dtype=float, copy=True)
        P[..., self.hub_bus] += hub_p          # hub buses are distinct
        return P

    # ------------------------------------------------------------------
    # DOE computation
    # ------------------------------------------------------------------
    def _max_feasible(self, P_win, Q_win, inj, direction):
        """
        Largest feasible scaling per scenario family.

        P_win, Q_win : (W, nb) background over the DOE window (forecast)
        inj          : (F, L, nb) bus power added by each candidate level
                       (F families × L increasing levels)
        Returns index of the largest level whose whole prefix is feasible,
        per family (F,), or -1 if even level 0 is infeasible.
        """
        F, L, nb = inj.shape
        W = P_win.shape[0]
        P = P_win[None, None, :, :] + inj[:, :, None, :]           # (F, L, W, nb)
        Q = np.broadcast_to(Q_win[None, None, :, :], P.shape)
        V = self.voltages(P.reshape(-1, nb), Q.reshape(-1, nb)).reshape(F, L, W, nb)
        flow = self.section_flows(P)                                # (F, L, W, sections)
        # A level is infeasible only if it pushes a limit beyond where the
        # background alone already is (pre-existing stress is not the hub's).
        base_flow = self.section_flows(P_win)[None, None]           # (1, 1, W, sections)
        rating = self.rating[None, None, None, :]
        if direction == "export":
            v_ok = (V <= self.cfg.v_max + 1e-9).all(axis=(2, 3))
            t_ok = ((flow >= -rating - 1e-9) | (flow >= base_flow)).all(axis=(2, 3))
        else:
            v_ok = (V >= self.cfg.v_min - 1e-9).all(axis=(2, 3))
            t_ok = ((flow <= rating + 1e-9) | (flow <= base_flow)).all(axis=(2, 3))
        ok = v_ok & t_ok
        # largest k such that levels 0..k are all feasible
        prefix = np.cumprod(ok, axis=1)
        return prefix.sum(axis=1) - 1

    def _hosting(self, P_win, Q_win, direction, levels=17, refine=9):
        """Individual hosting capacity per hub (others at baseline), kW."""
        sign = -1.0 if direction == "export" else 1.0
        lo = np.zeros(self.H); hi = self.hub_cap.copy()
        for n_lv in (levels, refine):
            grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, n_lv)[None, :]   # (H, L)
            inj = np.zeros((self.H, n_lv, self.nb))
            inj[np.arange(self.H), :, self.hub_bus] = sign * grid
            k = self._max_feasible(P_win, Q_win, inj, direction)
            best = np.where(k >= 0, grid[np.arange(self.H), np.maximum(k, 0)], 0.0)
            nxt = np.where(k + 1 < n_lv, grid[np.arange(self.H), np.minimum(k + 1, n_lv - 1)], best)
            lo, hi = best, np.maximum(nxt, best)
        return lo

    def _joint_scale(self, P_win, Q_win, h, direction, levels=21, refine=11):
        """Largest κ ≤ 1 with all hubs at κ·h_i jointly feasible."""
        sign = -1.0 if direction == "export" else 1.0
        lo, hi = 0.0, 1.0
        for n_lv in (levels, refine):
            ks = np.linspace(lo, hi, n_lv)
            inj = np.zeros((1, n_lv, self.nb))
            inj[0][:, self.hub_bus] = sign * ks[:, None] * h[None, :]
            k = int(self._max_feasible(P_win, Q_win, inj, direction)[0])
            if k < 0:
                return 0.0
            lo, hi = ks[k], ks[min(k + 1, n_lv - 1)]
        return lo

    def compute_doe(self, P_win, Q_win):
        """
        DOEs (kW) for one publication window.

        Returns (import_kw (H,), export_kw (H,)), each capped by hub capacity.
        """
        out = {}
        for direction in ("import", "export"):
            h = self._hosting(P_win, Q_win, direction)
            kappa = self._joint_scale(P_win, Q_win, h, direction)
            out[direction] = np.minimum(self.hub_cap, kappa * h)
        return out["import"], out["export"]

    def doe_day(self, P_bg, Q_bg):
        """
        Step-held DOEs for a whole day from the forecast background.

        P_bg, Q_bg : (T, nb). Returns (doe_imp (T, H), doe_exp (T, H)).
        """
        T = P_bg.shape[0]
        B = self.cfg.doe_block
        imp = np.zeros((T, self.H)); exp = np.zeros((T, self.H))
        for s in range(0, T, B):
            i, e = self.compute_doe(P_bg[s:s + B], Q_bg[s:s + B])
            imp[s:s + B] = i; exp[s:s + B] = e
        return imp, exp

    def hosting_day(self, P_bg, Q_bg):
        """
        Individual hosting capacities for a day (spec E2, network-aware
        mode): each hub's limit with all other hubs at baseline, *without*
        the joint scaling. Their sum can exceed what the feeder can take
        jointly, so hubs must coordinate when the feeder is stressed.

        Returns (host_imp (T, H), host_exp (T, H)), capped by hub capacity.
        """
        T = P_bg.shape[0]
        B = self.cfg.doe_block
        imp = np.zeros((T, self.H)); exp = np.zeros((T, self.H))
        for s in range(0, T, B):
            Pw, Qw = P_bg[s:s + B], Q_bg[s:s + B]
            imp[s:s + B] = np.minimum(self.hub_cap, self._hosting(Pw, Qw, "import"))
            exp[s:s + B] = np.minimum(self.hub_cap, self._hosting(Pw, Qw, "export"))
        return imp, exp

    # ------------------------------------------------------------------
    # Realised network outcome
    # ------------------------------------------------------------------
    def realised(self, P_bg_real, Q_bg_real, hub_p):
        """
        Physical outcome of one step: power flow with the *realised*
        background and the hubs' realised net site flows.

        Returns dict:
          v_min, v_max      : extreme bus voltages (pu)
          v_viol_pu         : Σ_b max(0, V_b − v_max, v_min − V_b)
          overload_kw       : Σ_sections max(0, |flow| − rating)
          hub_v             : voltage at each hub's bus (pu)
        """
        P = self.with_hubs(P_bg_real, hub_p)[None, :]
        V = self.voltages(P, np.atleast_2d(Q_bg_real))[0]
        flow = self.section_flows(P[0])
        return {
            "v_min": float(V.min()), "v_max": float(V.max()),
            "v_viol_pu": float(np.sum(np.maximum(0.0, V - self.cfg.v_max)
                                      + np.maximum(0.0, self.cfg.v_min - V))),
            "overload_kw": float(np.sum(np.maximum(0.0, np.abs(flow) - self.rating))),
            "hub_v": V[self.hub_bus],
        }
