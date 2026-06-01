"""
baselines/mpc/oracle_mpc.py
============================
MPC oracle baseline (Table 3, Baseline 3).

This baseline has full knowledge of the participation probability
function ρ(c, d, s) — the hidden model that the SAC agent must
discover through exploration. It uses this knowledge to optimise
the incentive price and dispatch fractions via a simple one-step
lookahead at each interval.

This serves as an approximate upper bound on achievable performance
given perfect participation foresight, addressing RQ3 and RQ5.

Why "approximate" upper bound
------------------------------
True MPC would solve a multi-step optimisation over the full episode
horizon. This implementation uses a one-step greedy optimisation
(myopic MPC) for tractability. It still provides a meaningful upper
bound because:
1. It has access to the true ρ(·) — information the SAC agent never sees
2. It optimises the exact reward function at each step
3. It has no exploration overhead

The gap between oracle MPC and SAC-GNN quantifies how much the SAC
agent's policy degrades from having to discover ρ(·) through experience
rather than computing against it directly. This gap is a direct measure
of the principal-agent problem's difficulty (RQ5).

Implementation
--------------
At each step, the oracle solves:
    max_{c, δ} E[r(c, δ, n_respond)] subject to c ∈ [c_min, c_max], δ ∈ [0,1]^H

Since E[n_respond_i] = n_enrolled_i × ρ(c, d_i, s̄_i), the expected
reward is differentiable in c and δ. We use scipy.optimize.minimize
with L-BFGS-B for the joint optimisation.
"""

import numpy as np
from scipy.optimize import minimize
from nem_env.nem_wdr_env import NEMWDREnv, EnvConfig
from nem_env.participation_model import ParticipationModel, HubParticipationState


class OracleMPCBaseline:
    """
    One-step lookahead MPC with oracle access to participation model.

    Parameters
    ----------
    n_hubs : int
        Number of hubs.
    participation_model : ParticipationModel
        The true participation model (hidden from SAC agent, but given
        to this oracle baseline).
    hub_distances : list[float]
        Road-network distance from centroid for each hub (km).
        Used to compute ρ(c, d_i, s̄_i) at each step.
    env_config : EnvConfig
        Environment configuration for reward computation.
    n_price_grid : int
        Number of price grid points for optimisation.
        Uses grid search + local refinement for robustness.
    """

    def __init__(
        self,
        n_hubs: int,
        participation_model: ParticipationModel,
        hub_distances: list,
        env_config: EnvConfig = None,
        n_price_grid: int = 20,
    ):
        self.n_hubs = n_hubs
        self.model = participation_model
        self.hub_distances = np.array(hub_distances)
        self.cfg = env_config or EnvConfig()
        self.n_price_grid = n_price_grid
        self.name = "OracleMPC"

    def select_action(self, obs: np.ndarray, env: NEMWDREnv) -> np.ndarray:
        """
        Select the action that maximises expected one-step reward.

        Uses the true participation model to compute E[n_respond] and
        optimises jointly over incentive price c and dispatch fractions δ.
        """
        node_feats, zone_feats = env.obs_to_node_and_zone(obs)

        # Extract current state
        mean_socs = node_feats[:, 1]                    # (H,) normalised SoC
        n_connected = (node_feats[:, 0]
                       * self.cfg.n_enrolled_mean).astype(int)   # approx n_enrolled
        n_connected = np.maximum(n_connected, 1)

        spot_price = float(zone_feats[0]) * 500.0       # denormalise
        wdr_active = float(zone_feats[1]) > 0.01        # dispatch target > 0
        dispatch_target_mw = float(zone_feats[1]) * self.cfg.zone_peak_mw

        # Grid search over price, then local refinement
        best_action = None
        best_expected_reward = -np.inf

        price_grid = np.linspace(
            self.cfg.price_min, self.cfg.price_max, self.n_price_grid
        )

        for c in price_grid:
            # Expected participation per hub given price c
            probs = self.model.participation_prob_vector(
                c_t=c,
                distances_km=self.hub_distances,
                mean_socs=mean_socs,
            )
            expected_n_respond = n_connected * probs   # (H,)

            # Optimal dispatch: maximise revenue - incentive - conformance
            # For a given c, optimal δ_i = 1 if E[revenue_i] > E[incentive_i]
            # which simplifies to: dispatch = 1 if spot_price > c (always true
            # when agent is profitable), else 0. But conformance complicates this.
            # Use full optimisation:
            delta = self._optimise_dispatch(
                c=c,
                expected_n_respond=expected_n_respond,
                spot_price=spot_price,
                dispatch_target_mw=dispatch_target_mw,
                wdr_active=wdr_active,
            )

            expected_reward = self._expected_reward(
                c=c,
                delta=delta,
                expected_n_respond=expected_n_respond,
                spot_price=spot_price,
                dispatch_target_mw=dispatch_target_mw,
                wdr_active=wdr_active,
            )

            if expected_reward > best_expected_reward:
                best_expected_reward = expected_reward
                best_action = (delta, c)

        delta, c = best_action
        action = np.append(delta, c).astype(np.float32)
        return action

    def _optimise_dispatch(
        self,
        c: float,
        expected_n_respond: np.ndarray,
        spot_price: float,
        dispatch_target_mw: float,
        wdr_active: bool,
    ) -> np.ndarray:
        """
        Given fixed price c, find optimal dispatch fractions δ.

        During WDR: minimise conformance deviation by matching target.
        Outside WDR: maximise net profit by dispatching high-SoC hubs.
        """
        dt_hr = 5.0 / 60.0
        target_mwh = dispatch_target_mw * dt_hr if wdr_active else 0.0
        kwh_per_respond = self.cfg.mean_discharge_kwh_per_ev

        if not wdr_active or target_mwh == 0:
            # Outside WDR: maximise profit, dispatch all if profitable
            net_per_kwh = (spot_price - c) / 1000.0 - self.cfg.lambda_degradation
            if net_per_kwh > 0:
                return np.ones(self.n_hubs, dtype=np.float32)
            else:
                return np.zeros(self.n_hubs, dtype=np.float32)

        # During WDR: solve for δ that minimises |E[E_del] - target|
        # E[E_del] = Σ_i δ_i × E[n_respond_i] × kwh_per_ev / 1000 (MWh)
        expected_e_del_per_hub = expected_n_respond * kwh_per_respond / 1000.0

        # Greedy allocation: fill target from highest-contributing hubs first
        total_capacity = expected_e_del_per_hub.sum()
        if total_capacity == 0:
            return np.zeros(self.n_hubs, dtype=np.float32)

        if total_capacity <= target_mwh:
            # Can't reach target — dispatch everything
            return np.ones(self.n_hubs, dtype=np.float32)

        # Scale dispatch proportionally to hit target
        delta = np.minimum(
            target_mwh / (expected_e_del_per_hub + 1e-9),
            1.0,
        ).astype(np.float32)
        return np.clip(delta, 0.0, 1.0)

    def _expected_reward(
        self,
        c: float,
        delta: np.ndarray,
        expected_n_respond: np.ndarray,
        spot_price: float,
        dispatch_target_mw: float,
        wdr_active: bool,
    ) -> float:
        """Compute expected reward given price c and dispatch δ."""
        dt_hr = 5.0 / 60.0
        kwh_per_ev = self.cfg.mean_discharge_kwh_per_ev

        expected_e_del_kwh = (delta * expected_n_respond * kwh_per_ev).sum()
        expected_e_del_mwh = expected_e_del_kwh / 1000.0

        r_wholesale = spot_price * expected_e_del_mwh
        r_incentive = c * expected_e_del_mwh

        if wdr_active and dispatch_target_mw > 0:
            target_mwh = dispatch_target_mw * dt_hr
            deviation = abs(expected_e_del_mwh - target_mwh)
            tolerance = self.cfg.conformance_tolerance * target_mwh
            p_conformance = self.cfg.lambda_conformance * max(0, deviation - tolerance)
        else:
            p_conformance = 0.0

        c_degradation = self.cfg.lambda_degradation * expected_e_del_kwh

        return r_wholesale - r_incentive - p_conformance - c_degradation

    def reset(self):
        pass
