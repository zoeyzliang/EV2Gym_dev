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

        Uses obs_to_node_features() — no zone features (RRP in node [6]).
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
            probs = self.model.participation_prob_vector(
                c_t=c * 1000.0,   # $/kWh → $/MWh for participation model
                distances_km=self.hub_distances,
                mean_socs=mean_socs,
            )
            expected_n = n_enrolled * probs   # (H,)

            # Dispatch decision: discharge if RRP > incentive cost, charge if RRP < 0
            kwh_per_ev = env.cfg.mean_discharge_kwh_per_ev
            dt_hr = 5.0 / 60.0

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
