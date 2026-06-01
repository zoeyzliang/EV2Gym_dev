"""
nem_wdr_env.py
==============
Gymnasium environment for NEM VSRP public V2G hub dispatch under stochastic
EV owner participation.

This is the central MDP environment. It wraps the NEM price loader and
participation model into a Gym-compatible interface consumable by SAC.

MDP structure
-------------
State (observation) at step t:
  Per-hub node features (assembled into graph node matrix by spatial_graph.py):
    - n_connected_i_t  : number of EVs currently connected at hub i
    - mean_soc_i_t     : mean SoC of connected EVs at hub i (normalised)
    - P_max_i_t        : max available discharge capacity at hub i (kW)
    - loc_i            : hub location relative to VSR zone centroid (2D, normalised)

  Zone-level features (appended to graph embedding after GAT):
    - p_t              : NEM spot price (normalised by market cap)
    - D_star_t         : AEMO dispatch target (normalised by zone peak MW)
    - delta_conf_t     : cumulative conformance deviation this episode (normalised)
    - sin/cos(h/24π)   : cyclical hour-of-day encoding (2 features)
    - sin/cos(d/7)     : cyclical day-of-week encoding (2 features)
    - c_{t-1}          : previous incentive price offered (normalised)
    - rho_hat_{t-1}    : empirical lagged participation rate (n_respond / n_enrolled)
                         THIS IS OBSERVABLE: computed from last step's outcome,
                         not from the hidden ρ(·) model. See design note below.

Action at step t (continuous, from SAC actor):
  - delta_i ∈ [0, 1]  : dispatch fraction for hub i (H values)
  - c_t ∈ [c_min, c_max] : incentive price scalar ($/MWh), zone-wide

Reward at step t:
  r_t = R_wholesale - R_incentive - P_conformance - C_degradation

  R_wholesale     = p_t × E_del_t / 1000          ($/step, wholesale revenue)
  R_incentive     = c_t × E_del_t / 1000          ($/step, cost to EV owners)
  P_conformance   = λ_conf × max(0, |E_del_t - D*_t × dt| - ε_tol)
                    only when wdr_active=True; 0 otherwise
  C_degradation   = λ_deg × E_del_t               (battery cycle proxy)

  where dt = 5/60 hr (one interval in hours), ε_tol = 10% of target.

Design notes
------------
1. ρ̂_{t-1} is an *empirical* lagged observation, not the model parameter.
   At step t, the agent observes the fraction of enrolled owners who responded
   in step t-1: ρ̂_{t-1} = n_respond_{t-1} / n_enrolled_{t-1}.
   This is computationally observable by the real VSRP (they count the EVs
   that arrived) but does NOT reveal the latent ρ(·) function parameters.
   The agent must still explore to learn the response surface.

2. The participation model is called inside step() and its output is NOT
   added to the observation. The only feedback the agent gets about ρ(·)
   is through (a) the reward and (b) the lagged ρ̂_{t-1} empirical rate.

3. EV SoC dynamics are simplified: each responding EV discharges by a fixed
   mean_discharge_kwh until the episode ends. Degradation is modelled as a
   linear cost per kWh discharged. A future extension could use EV2Gym's
   physics model for exact SoC simulation.

4. Hub SoC (mean_soc_i_t) evolves stochastically: at each step, EVs arrive
   and depart based on a simplified Poisson process. This creates realistic
   variation in n_connected and mean_soc without requiring the full EV2Gym
   mobility model. The full EV2Gym integration is reserved for Phase 5.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Any

from .aemo_price_loader import PriceLoader
from .participation_model import ParticipationModel, HubParticipationState
from .spatial_graph import HubConfig


# ---------------------------------------------------------------------------
# Hub and zone configuration
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """
    Hyperparameters for the NEM WDR dispatch environment.

    All values here should appear in your thesis config YAML / sensitivity
    analysis (§4.3.4). Do not hardcode in step() logic.
    """
    # VSR zone
    zone_peak_mw: float = 2.0          # MW — scale to real hub network later
    n_enrolled_mean: int = 20          # mean enrolled EV owners per hub
    n_enrolled_std: int = 5            # std of enrolled pool size across hubs

    # EV discharge model (simplified pending full EV2Gym integration)
    mean_discharge_kwh_per_ev: float = 8.0    # kWh per responding EV per event
    mean_soc_init: float = 0.65               # mean initial SoC (normalised)
    mean_soc_std: float = 0.15

    # Reward weights (tune these during training; candidates for sweep)
    lambda_conformance: float = 50.0   # penalty per kWh deviation from target
    lambda_degradation: float = 0.02   # cost per kWh discharged
    conformance_tolerance: float = 0.10  # fraction of target within which no penalty

    # Incentive price bounds ($/MWh) — define the agent's pricing action range
    price_min: float = 0.0
    price_max: float = 500.0           # ~2.5% of market price cap; tune

    # SoC dynamics: simple AR(1) model for between-step EV arrivals/departures
    soc_ar1_phi: float = 0.90          # persistence coefficient
    soc_ar1_noise: float = 0.05        # Gaussian noise std

    # Observation normalisation bounds
    price_normalise_by: float = 500.0  # $/MWh — normalise spot price to ~[0,1]


# ---------------------------------------------------------------------------
# Dynamic hub state (updated each step)
# ---------------------------------------------------------------------------

@dataclass
class HubState:
    """Mutable per-hub state, updated each step."""
    n_connected: int       # EVs physically present and plugged in
    n_enrolled: int        # total registered pool
    mean_soc: float        # normalised [0, 1]
    p_max_kw: float        # max discharge capacity this step
    # Lagged participation: updated after each step for observation
    rho_hat_prev: float = 0.0   # ρ̂_{t-1}


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class NEMWDREnv(gym.Env):
    """
    NEM VSRP public V2G hub dispatch environment.

    Implements the gymnasium.Env interface for compatibility with stable-
    baselines3 and the SAC implementation adapted from EV-GNN.

    Parameters
    ----------
    hub_configs : list of HubConfig
        Static config for each hub. Length H defines the number of hubs.
    price_loader : PriceLoader
        Loaded price data source. Call price_loader.load_synthetic() or
        price_loader.fetch_and_cache() before passing here.
    participation_model : ParticipationModel
        Stochastic participation model (hidden from agent).
    env_config : EnvConfig, optional
        Environment hyperparameters.
    force_wdr : bool
        Whether to guarantee a WDR event per episode (curriculum mode).
    seed : int, optional
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": []}

    # Node feature dimension per hub (Table 4, per-hub rows)
    #   n_connected (1) + mean_soc (1) + p_max_kw (1) + loc_x (1) + loc_y (1)
    NODE_FEATURE_DIM = 5

    # Zone-level feature dimension (Table 4, zone-level rows)
    #   spot_price (1) + dispatch_target (1) + cumulative_deviation (1)
    #   + sin_hour (1) + cos_hour (1) + sin_dow (1) + cos_dow (1)
    #   + prev_price (1) + rho_hat_prev (1)
    ZONE_FEATURE_DIM = 9

    def __init__(
        self,
        hub_configs: list[HubConfig],
        price_loader: PriceLoader,
        participation_model: ParticipationModel,
        env_config: Optional[EnvConfig] = None,
        force_wdr: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.hub_configs = hub_configs
        self.H = len(hub_configs)           # number of hubs
        self.price_loader = price_loader
        self.participation_model = participation_model
        self.cfg = env_config or EnvConfig()
        self.force_wdr = force_wdr
        self._rng = np.random.default_rng(seed)

        # --- Action space ---
        # H dispatch fractions δ_i ∈ [0, 1] + 1 price scalar c_t ∈ [c_min, c_max]
        action_low = np.concatenate([
            np.zeros(self.H),
            [self.cfg.price_min],
        ]).astype(np.float32)
        action_high = np.concatenate([
            np.ones(self.H),
            [self.cfg.price_max],
        ]).astype(np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # --- Observation space ---
        # Flat vector: H × NODE_FEATURE_DIM per-hub features + ZONE_FEATURE_DIM
        # The GAT agent will reshape this internally into node feature matrix + zone vec.
        # We use a flat Box here for compatibility with standard Gym wrappers.
        obs_dim = self.H * self.NODE_FEATURE_DIM + self.ZONE_FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state (populated on reset)
        self._episode_df = None      # 288-row DataFrame from price_loader
        self._step_idx = 0
        self._hub_states: list[HubState] = []
        self._cumulative_conformance_dev = 0.0
        self._prev_incentive_price = 0.0
        self._prev_rho_hat = 0.0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Initialise a new episode.

        Samples a fresh price/WDR episode from the price loader and
        re-initialises all hub states from their enrollment distributions.
        """
        super().reset(seed=seed)

        # Sample episode (288 steps of prices + WDR events)
        self._episode_df = self.price_loader.sample_episode(
            force_wdr=self.force_wdr
        )
        self._step_idx = 0
        self._cumulative_conformance_dev = 0.0
        self._prev_incentive_price = 0.0
        self._prev_rho_hat = 0.0

        # Initialise hub states
        self._hub_states = self._init_hub_states()

        obs = self._build_observation()
        info = {"step": 0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one 5-minute dispatch interval.

        Parameters
        ----------
        action : np.ndarray, shape (H + 1,)
            Concatenation of [δ_0, ..., δ_{H-1}, c_t].
            Produced by SAC actor; already clipped to action_space bounds.

        Returns
        -------
        obs : np.ndarray
            Next state observation.
        reward : float
            Immediate reward r_t (see reward structure §4.2.5).
        terminated : bool
            True at end of 288-step episode.
        truncated : bool
            Always False (no early termination in this env).
        info : dict
            Diagnostic quantities for logging (not passed to agent policy).
        """
        assert self._episode_df is not None, "Call reset() before step()"

        # --- Unpack action ---
        dispatch_fractions = np.clip(action[:self.H], 0.0, 1.0)   # δ_i
        incentive_price = float(np.clip(
            action[self.H], self.cfg.price_min, self.cfg.price_max
        ))  # c_t ($/MWh)

        # --- Current interval market data ---
        interval = self._episode_df.iloc[self._step_idx]
        spot_price = float(interval["spot_price"])         # $/MWh
        wdr_active = bool(interval["wdr_active"])
        dispatch_target_mw = float(interval["dispatch_target_mw"])  # MW

        # --- Participation draw (hidden from agent) ---
        participation_states = self._get_participation_states()
        n_respond, true_probs = (
            self.participation_model.sample_responses_with_probs(
                c_t=incentive_price,
                hub_states=participation_states,
            )
        )
        # Empirical participation rate (observable by VSRP — they count arrivals)
        n_enrolled_total = sum(h.n_enrolled for h in self._hub_states)
        rho_hat = float(n_respond.sum()) / max(n_enrolled_total, 1)

        # --- Energy delivered ---
        # Each responding EV discharges mean_discharge_kwh_per_ev, gated by
        # the dispatch fraction the agent allocated to this hub.
        e_del_hub_kwh = (
            dispatch_fractions
            * n_respond
            * self.cfg.mean_discharge_kwh_per_ev
        )  # shape (H,), kWh per hub this interval

        e_del_total_kwh = float(e_del_hub_kwh.sum())
        # Convert to MWh for market calculations
        e_del_total_mwh = e_del_total_kwh / 1000.0

        # --- Reward computation ---
        reward, reward_components = self._compute_reward(
            spot_price=spot_price,
            incentive_price=incentive_price,
            e_del_total_mwh=e_del_total_mwh,
            dispatch_target_mw=dispatch_target_mw,
            wdr_active=wdr_active,
        )

        # --- Conformance tracking (cumulative) ---
        if wdr_active:
            dt_hr = 5.0 / 60.0   # one interval in hours
            target_mwh = dispatch_target_mw * dt_hr
            deviation_mwh = abs(e_del_total_mwh - target_mwh)
            self._cumulative_conformance_dev += deviation_mwh

        # --- Update hub states ---
        self._update_hub_states(e_del_hub_kwh)

        # Save lagged observables for next step's observation
        self._prev_incentive_price = incentive_price
        self._prev_rho_hat = rho_hat

        # --- Advance step counter ---
        self._step_idx += 1
        terminated = self._step_idx >= PriceLoader.STEPS_PER_DAY

        # --- Build observation for next step ---
        obs = self._build_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        # Diagnostic info (logged by trainer, never seen by policy)
        info = {
            "step": self._step_idx,
            "spot_price": spot_price,
            "incentive_price": incentive_price,
            "wdr_active": wdr_active,
            "dispatch_target_mw": dispatch_target_mw,
            "e_del_total_kwh": e_del_total_kwh,
            "n_respond": n_respond.tolist(),
            "true_probs": true_probs.tolist(),   # true ρ values (not given to agent)
            "rho_hat": rho_hat,
            "cumulative_conformance_dev_mwh": self._cumulative_conformance_dev,
            **reward_components,
        }

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        Assemble the flat observation vector from current state.

        Layout: [hub_0_features | hub_1_features | ... | hub_{H-1}_features | zone_features]

        The GAT agent reshapes this into:
          node_features: (H, NODE_FEATURE_DIM)
          zone_features: (ZONE_FEATURE_DIM,)

        See spatial_graph.py for how the GAT wraps this into a PyG Data object.

        Note on normalisation:
        All features are normalised to approximately [-1, 1] or [0, 1] to
        ease neural network training. The normalisation constants are documented
        below and should match those used in the GAT agent's observation parser.
        """
        interval = self._episode_df.iloc[self._step_idx]
        spot_price = float(interval["spot_price"])
        dispatch_target_mw = float(interval["dispatch_target_mw"])
        wdr_active = bool(interval["wdr_active"])

        # Per-hub node features
        hub_features = []
        for hub_state in self._hub_states:
            hub_cfg = self.hub_configs[hub_state.p_max_kw.__class__.__name__ == "float" and 0 or 0]
            # Careful: find the matching config by hub position in list
            node_feats = np.array([
                hub_state.n_connected / max(self.cfg.n_enrolled_mean, 1),  # normalised
                hub_state.mean_soc,                                         # already [0,1]
                hub_state.p_max_kw / 1000.0,                               # kW → MW
                # loc_x, loc_y come from hub_configs — use hub_id as index
                self.hub_configs[
                    next(i for i, h in enumerate(self.hub_configs)
                         if True)   # placeholder; fixed below
                ].loc_x,
                0.0,  # loc_y placeholder — fixed below
            ], dtype=np.float32)
            hub_features.append(node_feats)

        # Fix: proper per-hub config lookup
        hub_features = []
        for i, (hub_state, hub_cfg) in enumerate(zip(self._hub_states, self.hub_configs)):
            node_feats = np.array([
                hub_state.n_connected / max(self.cfg.n_enrolled_mean, 1),
                hub_state.mean_soc,
                hub_state.p_max_kw / 1000.0,   # normalise to MW
                hub_cfg.loc_x,                  # fixed spatial feature
                hub_cfg.loc_y,
            ], dtype=np.float32)
            hub_features.append(node_feats)

        hub_feature_vec = np.concatenate(hub_features)   # (H × NODE_FEATURE_DIM,)

        # Cyclical time encoding (prevents discontinuity at midnight/week boundary)
        step = self._step_idx
        hour_of_day = (step * 5 / 60) % 24        # fractional hour
        day_of_week = 0                             # placeholder without date

        time_features = np.array([
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
        ], dtype=np.float32)

        # Zone-level features
        zone_features = np.array([
            spot_price / self.cfg.price_normalise_by,          # normalise
            (dispatch_target_mw / self.cfg.zone_peak_mw
             if self.cfg.zone_peak_mw > 0 else 0.0),           # 0 when no WDR
            self._cumulative_conformance_dev / 1.0,            # MWh; scale later
            *time_features,
            self._prev_incentive_price / self.cfg.price_max,
            self._prev_rho_hat,                                # already [0,1]
        ], dtype=np.float32)

        obs = np.concatenate([hub_feature_vec, zone_features]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        spot_price: float,
        incentive_price: float,
        e_del_total_mwh: float,
        dispatch_target_mw: float,
        wdr_active: bool,
    ) -> tuple[float, dict]:
        """
        Compute the four-term reward from §4.2.5.

        Returns
        -------
        reward : float
            Scalar reward r_t.
        components : dict
            Breakdown of each term for logging/diagnosis.
        """
        dt_hr = 5.0 / 60.0   # one 5-minute interval in hours

        # Term 1: wholesale revenue — energy dispatched at spot price
        r_wholesale = spot_price * e_del_total_mwh

        # Term 2: incentive cost — paid to EV owners per MWh delivered
        # Note: if no energy delivered, no cost even if price is high
        r_incentive = incentive_price * e_del_total_mwh

        # Term 3: conformance penalty — only during WDR activation
        # Applied when delivered energy deviates >ε_tol from target
        if wdr_active and dispatch_target_mw > 0:
            target_mwh = dispatch_target_mw * dt_hr
            deviation_mwh = abs(e_del_total_mwh - target_mwh)
            tolerance_mwh = self.cfg.conformance_tolerance * target_mwh
            excess_deviation = max(0.0, deviation_mwh - tolerance_mwh)
            p_conformance = self.cfg.lambda_conformance * excess_deviation
        else:
            p_conformance = 0.0
            deviation_mwh = 0.0
            target_mwh = 0.0

        # Term 4: battery degradation proxy — linear cost per kWh discharged
        c_degradation = self.cfg.lambda_degradation * (e_del_total_mwh * 1000)

        # Net reward: revenue - costs
        reward = r_wholesale - r_incentive - p_conformance - c_degradation

        components = {
            "r_wholesale": r_wholesale,
            "r_incentive": r_incentive,
            "p_conformance": p_conformance,
            "c_degradation": c_degradation,
            "reward": reward,
        }

        return float(reward), components

    # ------------------------------------------------------------------
    # Hub state management
    # ------------------------------------------------------------------

    def _init_hub_states(self) -> list[HubState]:
        """
        Initialise dynamic hub states at episode start.

        n_enrolled is drawn from N(mean, std) per hub, capturing realistic
        variation in enrolled pool size across the VSR zone.
        n_connected starts at a fraction of n_enrolled.
        """
        states = []
        for hub_cfg in self.hub_configs:
            n_enrolled = max(1, int(self._rng.normal(
                self.cfg.n_enrolled_mean, self.cfg.n_enrolled_std
            )))
            # Initially, ~50% of enrolled owners are connected
            n_connected = int(self._rng.binomial(n_enrolled, 0.5))
            mean_soc = float(np.clip(
                self._rng.normal(self.cfg.mean_soc_init, self.cfg.mean_soc_std),
                0.1, 0.95,
            ))
            states.append(HubState(
                n_connected=n_connected,
                n_enrolled=n_enrolled,
                mean_soc=mean_soc,
                p_max_kw=hub_cfg.p_max_kw,
            ))
        return states

    def _get_participation_states(self) -> list[HubParticipationState]:
        """Convert current HubState list to ParticipationState list."""
        return [
            HubParticipationState(
                hub_id=i,
                n_enrolled=hub_state.n_enrolled,
                distance_km=hub_cfg.distance_km,
                mean_soc=hub_state.mean_soc,
            )
            for i, (hub_state, hub_cfg) in enumerate(
                zip(self._hub_states, self.hub_configs)
            )
        ]

    def _update_hub_states(self, e_del_hub_kwh: np.ndarray) -> None:
        """
        Update hub states after a dispatch step.

        SoC dynamics: AR(1) process capturing EV arrivals/departures.
        Each discharging EV reduces SoC proportionally.
        New arrivals bring higher SoC; departures remove lower-SoC vehicles.

        This is a simplified placeholder pending full EV2Gym SoC simulation
        (planned for Phase 4 integration with zoeyzliang/EV2Gym_dev).
        """
        for i, (hub_state, hub_cfg) in enumerate(zip(self._hub_states, self.hub_configs)):
            # SoC decay from discharge
            if hub_state.n_connected > 0 and e_del_hub_kwh[i] > 0:
                kwh_per_ev = e_del_hub_kwh[i] / max(hub_state.n_connected, 1)
                # Assume 60 kWh battery; SoC decreases proportionally
                soc_drop = kwh_per_ev / 60.0
                hub_state.mean_soc = max(0.0, hub_state.mean_soc - soc_drop)

            # AR(1) SoC noise: new arrivals with fresh SoC, departures remove EVs
            noise = float(self._rng.normal(0.0, self.cfg.soc_ar1_noise))
            hub_state.mean_soc = float(np.clip(
                self.cfg.soc_ar1_phi * hub_state.mean_soc
                + (1 - self.cfg.soc_ar1_phi) * self.cfg.mean_soc_init
                + noise,
                0.05, 0.95,
            ))

            # n_connected: Poisson arrivals/departures
            arrivals = int(self._rng.poisson(1.5))
            departures = int(self._rng.poisson(1.5))
            hub_state.n_connected = max(
                0,
                min(hub_state.n_enrolled, hub_state.n_connected + arrivals - departures),
            )

    # ------------------------------------------------------------------
    # Observation/action space helpers (for agent code)
    # ------------------------------------------------------------------

    def obs_to_node_and_zone(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split flat observation vector into node feature matrix and zone features.

        Used by the GAT agent to reshape the flat obs into graph inputs.

        Returns
        -------
        node_features : np.ndarray, shape (H, NODE_FEATURE_DIM)
        zone_features : np.ndarray, shape (ZONE_FEATURE_DIM,)
        """
        n_hub_feats = self.H * self.NODE_FEATURE_DIM
        node_features = obs[:n_hub_feats].reshape(self.H, self.NODE_FEATURE_DIM)
        zone_features = obs[n_hub_feats:]
        return node_features, zone_features

    @property
    def n_hubs(self) -> int:
        return self.H

    @property
    def node_feature_dim(self) -> int:
        return self.NODE_FEATURE_DIM

    @property
    def zone_feature_dim(self) -> int:
        return self.ZONE_FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return self.H + 1   # H dispatch fractions + 1 price scalar
