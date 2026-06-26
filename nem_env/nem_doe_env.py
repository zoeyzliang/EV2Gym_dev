"""
nem_doe_env.py
==============
Gymnasium MDP environment for NEM public V2G hub bidirectional dispatch
under DNSP-issued Dynamic Operating Envelope (DOE) constraints and
stochastic EV owner participation.

This replaces nem_wdr_env.py. All WDR/VSRP/conformance-penalty logic
has been removed. The agent is now a price-taker retailer/aggregator
that observes AEMO RRP and independently sets signed dispatch targets
(negative = charge, positive = discharge) for each hub.

MDP structure (aligned with master problem setting summary, June 2026)
----------------------------------------------------------------------

State (per 5-minute step):
  Per-hub node features — NODE_FEATURE_DIM = 9 per hub:
    [0] DOE import limit  (W, constrains charging magnitude)
    [1] DOE export limit  (W, constrains discharging magnitude)
    [2] occupancy         (number of connected EVs, integer)
    [3] mean SoC          (mean state-of-charge of connected EVs, normalised 0-1)
    [4] queue length      (EVs waiting but not yet connected)
    [5] equipment cap     (kW, static symmetric hardware limit from CPO config)
    [6] RRP               ($/MWh, broadcast identically to all hub nodes)
    [7] hour of day       (fractional, 0-24)
    [8] day of week       (integer 0=Mon, 6=Sun)

  The flat observation vector is:
    [hub_0_features | hub_1_features | ... | hub_{H-1}_features]
  Shape: (H × 9,) — no separate zone feature block.
  The GAT agent reshapes this to (H, 9) as node_features for the graph.

Action (continuous, per-hub independent):
  - N × signed dispatch target (kW): raw output from tanh-squashed actor,
    rescaled to [-equipment_cap_i, +equipment_cap_i].
    Negative = charge/import. Positive = discharge/export.
  - 1 × incentive price ($/kWh, zone-wide): bounded [price_min, price_max].

Action clipping (two sequential clips inside step()):
  Step 1 — DOE clip (time-varying, DNSP-issued, asymmetric):
      action_i = clip(raw_action_i, -DOE_import_i, +DOE_export_i)
  Step 2 — Equipment cap clip (static, symmetric, CPO hardware):
      action_i = clip(action_i, -equipment_cap_i, +equipment_cap_i)
  Effective bound per direction = min(|DOE_limit|, equipment_cap).

Reward:
  r_t = Σ_i [RRP_t × net_dispatched_kWh_{i,t}]
        − Σ_i [incentive_price_t × participated_kWh_{i,t}]
        − λ_conf × Σ_i [max(0, |dispatch_{i,t}| − DOE_{i,t})]

  net_dispatched_kWh is signed:
    positive (export) → positive revenue when RRP > 0
    negative (import) → positive revenue when RRP < 0 (buying cheap)

  Agent is never penalised for direction of dispatch — only for DOE
  violation and incentive cost.

DOE dynamics:
  DOE limits are updated every ~30 minutes (6 steps) and held constant
  between DNSP updates. Synthetic DOEs are derived from VIC1 demand
  profiles + thermal headroom model. Real CSIP-AUS feeds are not
  publicly available.

Participation model:
  ρ(c, d, s) = σ(β₀ + β₁c + β₂d + β₃s)  [hidden from agent]
  n_respond_{i,t} ~ Binomial(n_enrolled_{i,t}, ρ)
  Agent infers participation elasticity from dispatch outcomes only.

Episode:
  288 steps × 5 min = one simulated NEM trading day.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional

from .aemo_price_loader import PriceLoader
from .participation_model import ParticipationModel, HubParticipationState
from .spatial_graph import HubConfig


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """
    Hyperparameters for the NEM DOE dispatch environment.
    All values appear in thesis sensitivity analysis (§4.3.4).

    DOE dynamics
    ------------
    doe_update_every : int
        Number of 5-min steps between DNSP DOE updates (~30 min = 6 steps).
    doe_import_mean_w, doe_export_mean_w : float
        Mean DOE limits in Watts for synthetic generation.
        Real values come from CSIP-AUS (not publicly available).
    doe_noise_std_w : float
        Gaussian noise std on DOE values between updates (W).

    EV dynamics (simplified pending full EV2Gym integration)
    ---------------------------------------------------------
    n_enrolled_mean, n_enrolled_std : int
        Hub enrolled EV owner pool size distribution.
    mean_soc_init, mean_soc_std : float
        Initial mean SoC distribution across hubs.
    mean_discharge_kwh_per_ev : float
        Mean energy per responding EV per interval (kWh).
    soc_ar1_phi, soc_ar1_noise : float
        AR(1) SoC dynamics between steps.

    Reward weights
    --------------
    lambda_conformance : float
        Penalty weight per kW of DOE violation (λ_conf = 200 per master summary).
    price_min, price_max : float
        Incentive price action bounds ($/kWh).
    """
    # DOE dynamics
    doe_update_every: int = 6              # steps between DNSP updates (~30 min)
    doe_import_mean_w: float = 50_000.0   # 50 kW mean import limit
    doe_export_mean_w: float = 40_000.0   # 40 kW mean export limit
    doe_noise_std_w: float = 5_000.0      # variation between updates

    # EV pool dynamics
    n_enrolled_mean: int = 20
    n_enrolled_std: int = 5
    mean_soc_init: float = 0.65
    mean_soc_std: float = 0.15
    mean_discharge_kwh_per_ev: float = 8.0

    # SoC AR(1) model
    soc_ar1_phi: float = 0.90
    soc_ar1_noise: float = 0.05

    # Reward
    lambda_conformance: float = 200.0     # per kW DOE violation (master summary §4)

    # Incentive price action space ($/kWh)
    price_min: float = 0.0
    price_max: float = 0.50               # $/kWh; ~$500/MWh upper bound

    # Observation normalisation
    rrp_clip_low: float = -1_000.0        # NEM floor $/MWh
    rrp_clip_high: float = 20_300.0       # NEM price cap $/MWh 2025-26
    doe_normalise_by_w: float = 100_000.0 # 100 kW reference for normalisation


# ---------------------------------------------------------------------------
# Per-hub dynamic state
# ---------------------------------------------------------------------------

@dataclass
class HubState:
    """Mutable per-hub state, updated each step."""
    n_connected: int        # EVs physically present (occupancy)
    n_enrolled: int         # total registered pool
    queue_length: int       # EVs waiting to connect
    mean_soc: float         # normalised [0, 1]
    equipment_cap_kw: float # static CPO hardware limit (kW), symmetric
    doe_import_w: float     # current DOE import limit (W), from DNSP
    doe_export_w: float     # current DOE export limit (W), from DNSP


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class NEMDOEEnv(gym.Env):
    """
    NEM public V2G hub bidirectional dispatch under DOE constraints.

    Implements gymnasium.Env for compatibility with SAC training loop.

    Parameters
    ----------
    hub_configs : list of HubConfig
        Static config for each hub (location, charger specs, equipment cap).
        Length H determines the number of hubs.
    price_loader : PriceLoader
        Loaded NEM price data. Call price_loader.fetch_and_cache() or
        price_loader.load_synthetic() before passing here.
    participation_model : ParticipationModel
        Stochastic participation model (hidden from agent).
    env_config : EnvConfig, optional
        Environment hyperparameters.
    seed : int, optional
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": []}

    # Node feature vector per hub — must match master summary §6
    # Layout: [doe_import_w, doe_export_w, occupancy, mean_soc,
    #          queue_length, equipment_cap_kw, rrp, hour, day_of_week]
    NODE_FEATURE_DIM = 9

    def __init__(
        self,
        hub_configs: list,          # list[HubConfig]
        price_loader: "PriceLoader",
        participation_model: "ParticipationModel",
        env_config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.hub_configs = hub_configs
        self.H = len(hub_configs)
        self.price_loader = price_loader
        self.participation_model = participation_model
        self.cfg = env_config or EnvConfig()
        self._rng = np.random.default_rng(seed)

        # --- Action space ---
        # H signed dispatch targets (kW) + 1 incentive price ($/kWh)
        # The actor uses tanh rescaling internally; the env clips to DOE +
        # equipment cap in step(). We declare a wide box here so the agent
        # can output any signed value and the clips inside step() handle
        # feasibility enforcement.
        max_cap = max(hc.p_max_kw for hc in hub_configs)
        action_low = np.concatenate([
            np.full(self.H, -max_cap, dtype=np.float32),
            [self.cfg.price_min],
        ])
        action_high = np.concatenate([
            np.full(self.H, +max_cap, dtype=np.float32),
            [self.cfg.price_max],
        ])
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # --- Observation space ---
        # Flat vector: H × NODE_FEATURE_DIM
        # GAT agent reshapes to (H, NODE_FEATURE_DIM) internally.
        obs_dim = self.H * self.NODE_FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal episode state (populated by reset())
        self._episode_df = None          # 288-row DataFrame from price_loader
        self._step_idx = 0
        self._hub_states: list = []      # list[HubState]
        self._doe_step = 0               # tracks when next DOE update fires

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple:
        """
        Initialise a new episode.

        Samples a fresh price day from the price loader, initialises hub
        states and DOE limits, returns first observation.
        """
        super().reset(seed=seed)

        self._episode_df = self.price_loader.sample_episode(force_wdr=False)
        self._step_idx = 0
        self._doe_step = 0
        self._hub_states = self._init_hub_states()

        obs = self._build_observation()
        return obs, {"step": 0}

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one 5-minute dispatch interval.

        Parameters
        ----------
        action : np.ndarray, shape (H + 1,)
            [dispatch_0_kw, ..., dispatch_{H-1}_kw, incentive_price_per_kwh]
            Signed dispatch: negative = charge (import), positive = discharge (export).
            Produced by SAC actor via tanh rescaling; further clipped here.

        Returns
        -------
        obs : np.ndarray, shape (H × NODE_FEATURE_DIM,)
        reward : float
        terminated : bool   — True at step 288 (end of NEM trading day)
        truncated : bool    — always False
        info : dict         — diagnostic quantities (not seen by policy)
        """
        assert self._episode_df is not None, "Call reset() before step()."

        # --- Unpack action ---
        raw_dispatch_kw = action[:self.H].astype(np.float64)   # signed kW per hub
        incentive_price = float(np.clip(
            action[self.H], self.cfg.price_min, self.cfg.price_max
        ))  # $/kWh

        # --- Current interval RRP ---
        interval = self._episode_df.iloc[self._step_idx]
        rrp = float(interval["spot_price"])   # $/MWh

        # --- Two-stage action clipping (master summary §4) ---
        # Stage 1: DOE clip — asymmetric per direction, time-varying
        clipped_dispatch_kw = np.array([
            np.clip(
                raw_dispatch_kw[i],
                -self._hub_states[i].doe_import_w / 1000.0,   # W → kW
                +self._hub_states[i].doe_export_w / 1000.0,
            )
            for i in range(self.H)
        ], dtype=np.float64)

        # Stage 2: Equipment cap clip — static, symmetric
        clipped_dispatch_kw = np.array([
            np.clip(
                clipped_dispatch_kw[i],
                -self._hub_states[i].equipment_cap_kw,
                +self._hub_states[i].equipment_cap_kw,
            )
            for i in range(self.H)
        ], dtype=np.float64)

        # DOE violation: how much of raw action exceeded DOE limits
        # Used only for penalty; agent observes DOE in state so can avoid this.
        doe_violations_kw = np.maximum(
            0.0,
            np.abs(raw_dispatch_kw) - np.array([
                min(
                    self._hub_states[i].doe_import_w / 1000.0,
                    self._hub_states[i].doe_export_w / 1000.0,
                )
                for i in range(self.H)
            ])
        )

        # --- Participation draw (hidden from agent) ---
        participation_states = self._get_participation_states()
        n_respond, true_probs = self.participation_model.sample_responses_with_probs(
            c_t=incentive_price * 1000.0,   # convert $/kWh → $/MWh for participation model
            hub_states=participation_states,
        )

        # --- Energy actually delivered (kWh per hub, signed) ---
        # Each responding EV delivers mean_discharge_kwh_per_ev, signed by
        # direction of clipped dispatch. Charge = negative kWh, discharge = positive.
        direction = np.sign(clipped_dispatch_kw)   # -1, 0, or +1
        participated_kwh = (
            direction
            * n_respond
            * self.cfg.mean_discharge_kwh_per_ev
        )   # shape (H,), signed kWh

        # Empirical participation rate (aggregate, for logging)
        n_enrolled_total = sum(h.n_enrolled for h in self._hub_states)
        rho_hat = float(n_respond.sum()) / max(n_enrolled_total, 1)

        # --- Reward (master summary §4) ---
        reward, reward_components = self._compute_reward(
            rrp=rrp,
            participated_kwh=participated_kwh,
            incentive_price=incentive_price,
            doe_violations_kw=doe_violations_kw,
        )

        # --- Update DOE (every doe_update_every steps) ---
        self._doe_step += 1
        if self._doe_step >= self.cfg.doe_update_every:
            self._update_doe()
            self._doe_step = 0

        # --- Update hub occupancy / SoC ---
        self._update_hub_states(participated_kwh)

        # --- Advance step ---
        self._step_idx += 1
        terminated = self._step_idx >= PriceLoader.STEPS_PER_DAY

        obs = (
            self._build_observation()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "step": self._step_idx,
            "rrp": rrp,
            "incentive_price_per_kwh": incentive_price,
            "clipped_dispatch_kw": clipped_dispatch_kw.tolist(),
            "raw_dispatch_kw": raw_dispatch_kw.tolist(),
            "doe_violations_kw": doe_violations_kw.tolist(),
            "n_respond": n_respond.tolist(),
            "true_probs": true_probs.tolist(),
            "rho_hat": rho_hat,
            "participated_kwh": participated_kwh.tolist(),
            **reward_components,
        }

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        Assemble the flat observation vector from current hub states and RRP.

        Layout per hub (NODE_FEATURE_DIM = 9):
          [0] doe_import_w       normalised by doe_normalise_by_w
          [1] doe_export_w       normalised by doe_normalise_by_w
          [2] occupancy          raw integer (connected EVs)
          [3] mean_soc           already [0, 1]
          [4] queue_length       raw integer
          [5] equipment_cap_kw   raw kW (static, same every step)
          [6] rrp                $/MWh, clipped to [rrp_clip_low, rrp_clip_high]
                                 then normalised to [-1, 1] via price cap
          [7] hour_of_day        fractional hour 0.0–23.99
          [8] day_of_week        integer 0 (Mon) – 6 (Sun)

        RRP is broadcast identically to every hub node — the GAT can
        treat it as a shared global signal and its attention weights will
        determine how price information propagates across hub embeddings.
        """
        interval = self._episode_df.iloc[self._step_idx]
        rrp = float(np.clip(
            interval["spot_price"],
            self.cfg.rrp_clip_low,
            self.cfg.rrp_clip_high,
        ))
        # Normalise RRP to [-1, 1] using price cap as reference
        rrp_norm = rrp / self.cfg.rrp_clip_high

        # Time features from sim date embedded in the price loader index
        try:
            ts = self._episode_df.index[self._step_idx]
            hour = ts.hour + ts.minute / 60.0
            dow = float(ts.dayofweek)
        except Exception:
            # Fallback: derive from step index
            hour = (self._step_idx * 5 / 60.0) % 24.0
            dow = 0.0

        hub_features = []
        for hub_state in self._hub_states:
            node = np.array([
                hub_state.doe_import_w / self.cfg.doe_normalise_by_w,
                hub_state.doe_export_w / self.cfg.doe_normalise_by_w,
                float(hub_state.n_connected),
                hub_state.mean_soc,
                float(hub_state.queue_length),
                hub_state.equipment_cap_kw,
                rrp_norm,
                hour,
                dow,
            ], dtype=np.float32)
            hub_features.append(node)

        obs = np.concatenate(hub_features)   # (H × NODE_FEATURE_DIM,)
        return obs

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        rrp: float,
        participated_kwh: np.ndarray,
        incentive_price: float,
        doe_violations_kw: np.ndarray,
    ) -> tuple:
        """
        Compute reward aligned with master summary §4:

        r_t = Σ_i [RRP_t × net_dispatched_kWh_{i,t} / 1000]   (MWh basis)
              − Σ_i [incentive_price_t × |participated_kWh_{i,t}|]
              − λ_conf × Σ_i [max(0, |dispatch_{i,t}| − DOE_{i,t})]

        RRP in $/MWh, participated_kwh in kWh → convert to MWh for revenue.
        incentive_price in $/kWh, applied to absolute kWh delivered.
        DOE violation penalty on raw kW excess (before clipping).

        Returns
        -------
        reward : float
        components : dict  (for logging; never passed to agent)
        """
        dt_hr = 5.0 / 60.0   # one NEM interval in hours

        # Term 1: wholesale revenue/cost
        # participated_kwh is signed: positive = export (revenue), negative = import (cost)
        # When RRP is negative, importing (negative kwh) yields positive revenue:
        #   rrp (negative) × kwh (negative) / 1000 = positive $/step ✓
        participated_mwh = participated_kwh / 1000.0
        r_wholesale = float(rrp * participated_mwh.sum())

        # Term 2: incentive cost — paid per kWh of V2G service delivered
        # Only applies to energy actually delivered, regardless of direction
        r_incentive = float(incentive_price * np.abs(participated_kwh).sum())

        # Term 3: DOE violation penalty
        p_conformance = float(
            self.cfg.lambda_conformance * doe_violations_kw.sum()
        )

        reward = r_wholesale - r_incentive - p_conformance

        components = {
            "r_wholesale": r_wholesale,
            "r_incentive": r_incentive,
            "p_conformance": p_conformance,
            "reward": reward,
            "net_mwh": float(participated_mwh.sum()),
        }
        return reward, components

    # ------------------------------------------------------------------
    # DOE dynamics
    # ------------------------------------------------------------------

    def _update_doe(self) -> None:
        """
        Update DOE limits for all hubs (DNSP update every ~30 min).

        Synthetic DOE generation: Gaussian noise around episode-stable mean
        values. Correlated across hubs sharing upstream infrastructure
        (captured implicitly by the GAT attention weights — the environment
        does not need to model the correlation explicitly here, it emerges
        from co-movement in the observation).

        Real CSIP-AUS DOE feeds are unavailable; this synthetic model
        produces the time-varying, per-NMI structure the agent needs to
        learn from. See limitations §10 of master summary.
        """
        for hub_state in self._hub_states:
            # Import limit: draw from N(mean, noise), clip to [0, equipment_cap]
            new_import = float(np.clip(
                self._rng.normal(
                    self.cfg.doe_import_mean_w,
                    self.cfg.doe_noise_std_w,
                ),
                0.0,
                hub_state.equipment_cap_kw * 1000.0,   # kW → W
            ))
            # Export limit: same structure, independent per direction
            new_export = float(np.clip(
                self._rng.normal(
                    self.cfg.doe_export_mean_w,
                    self.cfg.doe_noise_std_w,
                ),
                0.0,
                hub_state.equipment_cap_kw * 1000.0,
            ))
            hub_state.doe_import_w = new_import
            hub_state.doe_export_w = new_export

    # ------------------------------------------------------------------
    # Hub state management
    # ------------------------------------------------------------------

    def _init_hub_states(self) -> list:
        """
        Initialise dynamic hub states at episode start.

        Equipment cap is taken from HubConfig.p_max_kw (static, from CPO
        transformer/charger config). DOE limits are drawn from the synthetic
        generator. EV occupancy and SoC are drawn from their prior distributions.
        """
        states = []
        for hub_cfg in self.hub_configs:
            n_enrolled = max(1, int(self._rng.normal(
                self.cfg.n_enrolled_mean, self.cfg.n_enrolled_std
            )))
            n_connected = int(self._rng.binomial(n_enrolled, 0.5))
            queue = int(self._rng.poisson(1.0))
            mean_soc = float(np.clip(
                self._rng.normal(self.cfg.mean_soc_init, self.cfg.mean_soc_std),
                0.05, 0.95,
            ))
            doe_import = float(np.clip(
                self._rng.normal(self.cfg.doe_import_mean_w, self.cfg.doe_noise_std_w),
                0.0, hub_cfg.p_max_kw * 1000.0,
            ))
            doe_export = float(np.clip(
                self._rng.normal(self.cfg.doe_export_mean_w, self.cfg.doe_noise_std_w),
                0.0, hub_cfg.p_max_kw * 1000.0,
            ))
            states.append(HubState(
                n_connected=n_connected,
                n_enrolled=n_enrolled,
                queue_length=queue,
                mean_soc=mean_soc,
                equipment_cap_kw=hub_cfg.p_max_kw,   # static from CPO config
                doe_import_w=doe_import,
                doe_export_w=doe_export,
            ))
        return states

    def _get_participation_states(self) -> list:
        """Convert HubState list to HubParticipationState list."""
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

    def _update_hub_states(self, participated_kwh: np.ndarray) -> None:
        """
        Update per-hub occupancy and SoC after a dispatch step.

        SoC dynamics: AR(1) process with mean reversion toward mean_soc_init.
        Discharge reduces SoC proportionally; charging increases it.
        Arrivals/departures modelled as Poisson process.
        """
        for i, hub_state in enumerate(self._hub_states):
            # SoC update from energy delivered
            if hub_state.n_connected > 0:
                kwh_per_ev = participated_kwh[i] / max(hub_state.n_connected, 1)
                # Assume 60 kWh battery; positive kwh (discharge) reduces SoC,
                # negative kwh (charge) increases SoC.
                soc_delta = -kwh_per_ev / 60.0
                hub_state.mean_soc = float(np.clip(
                    hub_state.mean_soc + soc_delta, 0.0, 1.0
                ))

            # AR(1) SoC noise: new arrivals bring fresh SoC
            noise = float(self._rng.normal(0.0, self.cfg.soc_ar1_noise))
            hub_state.mean_soc = float(np.clip(
                self.cfg.soc_ar1_phi * hub_state.mean_soc
                + (1 - self.cfg.soc_ar1_phi) * self.cfg.mean_soc_init
                + noise,
                0.05, 0.95,
            ))

            # Occupancy: Poisson arrivals and departures
            arrivals = int(self._rng.poisson(1.5))
            departures = int(self._rng.poisson(1.5))
            hub_state.n_connected = int(np.clip(
                hub_state.n_connected + arrivals - departures,
                0, hub_state.n_enrolled,
            ))

            # Queue: replenish from enrolled pool not yet connected
            hub_state.queue_length = max(
                0, hub_state.n_enrolled - hub_state.n_connected
            )

    # ------------------------------------------------------------------
    # Observation splitting helper (used by SAC agent and baselines)
    # ------------------------------------------------------------------

    def obs_to_node_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Reshape flat observation to node feature matrix.

        Parameters
        ----------
        obs : np.ndarray, shape (H × NODE_FEATURE_DIM,)

        Returns
        -------
        node_features : np.ndarray, shape (H, NODE_FEATURE_DIM)
        """
        return obs.reshape(self.H, self.NODE_FEATURE_DIM)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_hubs(self) -> int:
        return self.H

    @property
    def node_feature_dim(self) -> int:
        return self.NODE_FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return self.H + 1   # H dispatch targets (kW) + 1 incentive price
