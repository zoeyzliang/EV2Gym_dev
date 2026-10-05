"""
baselines/mpc/oracle_mpc.py
============================
MPC oracle baseline (Table 3, upper bound).

Has oracle access to the true participation model ρ(c,d,s) — information
the SAC agent must discover through exploration. Uses one-step lookahead
to optimise signed dispatch (kW) and incentive price ($/kWh) at each interval.

Serves as an approximate upper bound on achievable performance under the
DOE/arbitrage design (master summary §4). The gap between OracleMPC and
SAC-GNN quantifies the cost of having to discover ρ(·) from experience.

Design aligned with master summary:
- Signed dispatch: negative = charge, positive = discharge
- Price in $/kWh (not $/MWh)
- No WDR/conformance — pure arbitrage: r = RRP × net_MWh - incentive × |kWh|
- DOE clipping applied (oracle respects DNSP constraints)
"""

import numpy as np
from nem_env.nem_doe_env import NEMDOEEnv
from nem_env.participation_model import ParticipationModel


def expected_delivered_kw(pool, probs, limit_kw, per_ev_kw):
    """
    Exact E[min(limit_kw, N · per_ev_kw)] per hub, N ~ Binomial(pool, probs).

    All arguments are (H,) arrays. Sums over the Binomial pmf (pool is at
    most a few chargers per hub, so this is cheap).
    """
    from scipy.stats import binom
    pool = np.asarray(pool, dtype=int)
    k = np.arange(int(pool.max(initial=0)) + 1)[:, None]          # (K+1, 1)
    pmf = binom.pmf(k, pool[None, :], np.asarray(probs)[None, :])  # (K+1, H)
    return (pmf * np.minimum(np.asarray(limit_kw)[None, :],
                             k * np.asarray(per_ev_kw)[None, :])).sum(0)


class OracleMPCBaseline:
    """
    One-step lookahead MPC with oracle access to participation model.

    At each step maximises expected reward:
        r = RRP × net_MWh − incentive_price × |participated_kWh| / 1000

    Uses grid search over incentive price, then analytical dispatch decision.

    Parameters
    ----------
    n_hubs : int
    participation_model : ParticipationModel
        True participation model (hidden from SAC agent).
    hub_distances : list[float]
        Road-network distance from centroid per hub (km).
    n_price_grid : int
        Price grid points for optimisation.
    equipment_cap_kw : float
        Default equipment cap for dispatch clipping.
    """

    def __init__(
        self,
        n_hubs: int,
        participation_model: ParticipationModel,
        hub_distances: list,
        n_price_grid: int = 20,
        equipment_cap_kw: float = 100.0,
    ):
        self.n_hubs = n_hubs
        self.model = participation_model
        self.hub_distances = np.array(hub_distances)
        self.n_price_grid = n_price_grid
        self.equipment_cap_kw = equipment_cap_kw
        self.name = "OracleMPC"
        self.price_min = 0.0
        self.price_max = 0.50   # $/kWh

    def select_action(self, obs: np.ndarray, env: NEMDOEEnv) -> np.ndarray:
        """
        Select action maximising expected one-step reward.

        Dispatches on env.cfg.energy_model so Oracle always evaluates the
        same energy model the environment actually uses.
        """
        if env.cfg.energy_model == "legacy":
            return self._select_action_legacy(obs, env)
        return self._select_action_coupled(obs, env)

    def _select_action_coupled(self, obs: np.ndarray, env: NEMDOEEnv) -> np.ndarray:
        """
        One-step expected-reward maximisation under the coupled energy model
        (see EnvConfig "Energy model" in nem_doe_env.py).

        For each incentive price c on the grid, and each hub i:
          pool_i  = min(n_connected_i, n_chargers_i)      (same pool as env)
          N_i     ~ Binomial(pool_i, ρ_i(c))
          E_dir   = Δt · E[min(L_dir, N_i · p_dir,i)]       (exact, via pmf)
        where L_dir is the DOE/cap-clipped limit for that direction and
        p_dir,i the per-EV power (charger rating, capped by SoC headroom /
        Δt exactly as in the env). Each hub
        then takes whichever of discharge / charge / idle has the highest
        expected value, (±RRP/1000 − c) · E_dir, and the price with the
        highest network total is chosen. Still myopic (one step), so this is
        an informed baseline, not an upper bound on the intertemporal problem.
        """
        node_feats = env.obs_to_node_features(obs)  # (H, 9)
        doe_import_kw = node_feats[:, 0] * env.cfg.doe_normalise_by_w / 1000.0
        doe_export_kw = node_feats[:, 1] * env.cfg.doe_normalise_by_w / 1000.0
        n_connected   = node_feats[:, 2].astype(int)
        mean_socs     = node_feats[:, 3]
        equipment_caps = node_feats[:, 5]
        rrp = float(node_feats[0, 6]) * env.cfg.rrp_clip_high   # $/MWh

        eff_discharge_kw = np.minimum(doe_export_kw, equipment_caps)
        eff_charge_kw    = np.minimum(doe_import_kw, equipment_caps)

        n_chargers = np.array([hc.n_chargers for hc in env.hub_configs], dtype=int)
        charger_kw = np.array([hc.charger_max_kw for hc in env.hub_configs], dtype=float)
        pool = np.minimum(n_connected, n_chargers)                # (H,)
        dt_hr = 5.0 / 60.0

        # Per-EV power per direction, including the env's SoC headroom bound
        batt = env.cfg.ev_battery_kwh
        per_ev_dis_kw = np.minimum(
            charger_kw, batt * np.maximum(mean_socs - env.cfg.soc_min, 0.0) / dt_hr)
        per_ev_chg_kw = np.minimum(
            charger_kw, batt * np.maximum(env.cfg.soc_max - mean_socs, 0.0) / dt_hr)

        best_action = None
        best_expected_reward = -np.inf
        price_grid = np.linspace(self.price_min, self.price_max, self.n_price_grid)

        for c in price_grid:
            probs = self.model.participation_prob_vector(
                c_t=c * 1000.0,   # $/kWh → $/MWh for participation model
                distances_km=self.hub_distances,
                mean_socs=mean_socs,
                doe_export_ws=node_feats[:, 1] * env.cfg.doe_normalise_by_w,  # W
                n_connecteds=pool,
            )
            e_dis_kwh = dt_hr * expected_delivered_kw(pool, probs, eff_discharge_kw, per_ev_dis_kw)
            e_chg_kwh = dt_hr * expected_delivered_kw(pool, probs, eff_charge_kw, per_ev_chg_kw)

            v_dis = (rrp / 1000.0 - c) * e_dis_kwh
            v_chg = (-rrp / 1000.0 - c) * e_chg_kwh
            dispatch_kw = np.where(
                (v_dis >= v_chg) & (v_dis > 0), eff_discharge_kw,
                np.where(v_chg > 0, -eff_charge_kw, 0.0),
            )
            expected_r = float(np.maximum(np.maximum(v_dis, v_chg), 0.0).sum())

            if expected_r > best_expected_reward:
                best_expected_reward = expected_r
                best_action = np.append(dispatch_kw, c).astype(np.float32)

        return best_action

    def _select_action_legacy(self, obs: np.ndarray, env: NEMDOEEnv) -> np.ndarray:
        """
        Draft-v1 Oracle, kept unchanged for energy_model="legacy" so v1
        results reproduce exactly. Known issues (fixed in the coupled path):
        uses n_connected as the Binomial pool while the legacy env draws
        from n_enrolled, and uses a fixed discharge/charge rule rather than
        a per-hub expected-value comparison.
        """
        node_feats = env.obs_to_node_features(obs)  # (H, 9)

        # Extract from node features (master summary §6 layout)
        doe_import_norm  = node_feats[:, 0]   # normalised
        doe_export_norm  = node_feats[:, 1]
        mean_socs        = node_feats[:, 3]
        equipment_caps   = node_feats[:, 5]   # kW

        # RRP: feature [6], normalised by rrp_clip_high=20300
        rrp_norm = float(node_feats[0, 6])
        rrp = rrp_norm * env.cfg.rrp_clip_high   # $/MWh

        # DOE limits in kW
        doe_import_kw = doe_import_norm * env.cfg.doe_normalise_by_w / 1000.0
        doe_export_kw = doe_export_norm * env.cfg.doe_normalise_by_w / 1000.0

        # Effective bounds per hub: min(DOE, equipment_cap)
        eff_discharge_kw = np.minimum(doe_export_kw, equipment_caps)
        eff_charge_kw    = np.minimum(doe_import_kw, equipment_caps)

        # Approximate enrolled EV count from occupancy (feature [2])
        n_enrolled = np.maximum(node_feats[:, 2].astype(int), 1)

        best_action = None
        best_expected_reward = -np.inf

        price_grid = np.linspace(self.price_min, self.price_max, self.n_price_grid)

        for c in price_grid:
            # Expected participation per hub
            #
            # doe_export_ws/n_connecteds fix
            # --------------------------------
            # These two arguments were previously omitted entirely,
            # silently defaulting to "no battery degradation cost"
            # (see participation_model.py's _anticipated_discharge_kwh_vector:
            # doe_export_ws=None -> returns zeros -> degradation term is 0).
            # Since the real environment (nem_doe_env.py) DOES pass these
            # through and therefore DOES include the degradation term in
            # every actual EV owner's response, Oracle was silently
            # optimising against a STALE, incomplete version of "the true
            # participation model" -- undermining its own core premise of
            # having privileged, exact knowledge of ρ(·). Fixed by passing
            # the same information Oracle already reads/computes for its
            # own DOE-clipping logic below, just forwarded to the
            # participation model too. Units: doe_export_ws must be WATTS
            # (per the function's docstring and its internal /1000.0
            # conversion) -- NOT the kW variable (doe_export_kw) used
            # elsewhere in this function for equipment-cap clipping.
            probs = self.model.participation_prob_vector(
                c_t=c * 1000.0,   # $/kWh → $/MWh for participation model
                distances_km=self.hub_distances,
                mean_socs=mean_socs,
                doe_export_ws=doe_export_norm * env.cfg.doe_normalise_by_w,  # W
                n_connecteds=n_enrolled,
            )
            expected_n = n_enrolled * probs   # (H,)

            # Dispatch decision: discharge if RRP > incentive cost, charge if RRP < 0
            kwh_per_ev = env.cfg.mean_discharge_kwh_per_ev

            if rrp > c * 1000.0:
                # Profitable to discharge: positive dispatch
                dispatch_kw = eff_discharge_kw.copy()
            elif rrp < 0:
                # Negative RRP: profitable to charge
                dispatch_kw = -eff_charge_kw.copy()
            else:
                dispatch_kw = np.zeros(self.n_hubs)

            # Expected energy delivered
            direction = np.sign(dispatch_kw)
            participated_kwh = direction * expected_n * kwh_per_ev
            participated_mwh = participated_kwh / 1000.0

            r_wholesale  = rrp * participated_mwh.sum()
            r_incentive  = c * np.abs(participated_kwh).sum()
            expected_r   = r_wholesale - r_incentive

            if expected_r > best_expected_reward:
                best_expected_reward = expected_r
                best_action = np.append(dispatch_kw, c).astype(np.float32)

        return best_action if best_action is not None else np.zeros(self.n_hubs + 1)

    def reset(self):
        pass
