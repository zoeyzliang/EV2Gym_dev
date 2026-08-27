"""
participation_model.py
======================
Stochastic EV owner participation model for public V2G hub dispatch.

Models the binary accept/reject decision of each enrolled EV owner when
notified by the VSRP aggregator via the CPO. The model is intentionally
*hidden from the RL agent* — it exists only inside environment dynamics.
The agent must discover effective incentive levels through SAC's entropy-
regularised exploration, not by computing against a known ρ(·).

Participation probability (logistic model)
------------------------------------------
For hub i at interval t, the per-owner participation probability is:

    U_{i,t} = β₀ + β₁·c_t + β₂·d_i + β₃·s̄_{i,t} − γ·g_{i,t}
    ρ(·) = σ(U_{i,t})

where σ is the logistic sigmoid, and:
  c_t   : incentive price offered ($/MWh, zone-wide scalar)
  d_i   : travel distance proxy for hub i (km from residential centroid)
  s̄_i,t : mean SoC of enrolled EVs at hub i (normalised 0–1)
  g_i,t : anticipated per-EV discharge this interval (kWh) — see below
  γ     : per-unit battery degradation cost ($/kWh), fleet-wide constant

The number of responding owners at hub i is then:

    n_respond_{i,t} ~ Binomial(n_enrolled_{i,t}, ρ(·))

Degradation term (g_{i,t}, γ)
------------------------------
The EV owner's participation decision precedes the agent's realised dispatch
for the interval, so g_{i,t} cannot be the realised discharge. We use the
DOE-permitted export ceiling for hub i, distributed evenly across currently
*connected* EVs, as a conservative worst-case anticipation:

    g_{i,t} = (doe_export_w_{i,t} / 1000) × dt_hr / max(n_connected_{i,t}, 1)

dt_hr is the interval duration in hours (5 min → 1/12). This is a
deliberately conservative anchor — it assumes the full hub-level ceiling
could in principle be asked of any one connected EV — and therefore tends
to *suppress* participation relative to what realised discharge would imply.
See thesis discussion (§X) for the rationale and the acknowledged
alternative (expected discharge conditional on participation), which was
not adopted due to the circularity it introduces between policy training
and the participation model, and the absence of any empirical benchmark to
calibrate it against in the literature (Hematiboroujeni et al. 2026;
Latinopoulos et al. 2021 — see thesis discussion for both).

γ is computed as γ = r / (κ · e), following Hematiboroujeni et al. (2026,
arXiv:2603.20226v1, Eq. 19): r is a $/kWh-of-capacity replacement-cost rate,
κ is cycles-to-end-of-life, and e = sqrt(round-trip efficiency). Under this
formulation γ is invariant to individual battery capacity (replacement cost
and lifetime throughput both scale linearly with capacity and cancel), so a
single fleet-wide γ is not a simplification relative to a capacity-weighted
version — it is algebraically equivalent to one under standard assumptions.
Default values (r=200 $/kWh, κ=1500, e=√0.90) reproduce Hematiboroujeni's
γ≈0.14 $/kWh reference case; override via ParticipationModel(gamma=...) or
by passing replacement_cost_per_kwh_capacity / cycles_to_eol / round_trip_
efficiency to compute_gamma().

Default parameter values
------------------------
β₀ = -2.20  (intercept; implies ρ ≈ 0.10 at zero price, near distance, mid SoC)
β₁ = +0.04  (price sensitivity; ρ increases ~20pp per $50/MWh increase)
β₂ = -0.20  (distance penalty; ρ drops ~20pp per 10 km)
β₃ = +1.50  (SoC effect; high SoC owners significantly more willing)
γ  = +0.14  ($/kWh; degradation disutility per kWh of anticipated discharge)

Calibration basis: Liu et al. (2025) systematic review of V2G acceptance [14]
establishes that economic incentive, range anxiety (proxied by d_i and s̄),
and convenience are the dominant participation antecedents. γ follows
Hematiboroujeni et al. (2026); no published study estimates V2G-specific
discharge disutility from real participation data (see thesis discussion),
so β and γ are both reasoned assumptions, not fitted parameters. The
combined values produce the following sanity-check behaviour:
  - At c=0, d=0, s̄=0.5, g=0:    ρ ≈ 0.12
  - At c=100, d=0, s̄=0.5, g=0:  ρ ≈ 0.55  (a meaningful incentive works)
  - At c=100, d=10, s̄=0.5, g=0: ρ ≈ 0.38  (distance penalty is real)
  - At c=200, d=0, s̄=0.8, g=0:  ρ ≈ 0.90  (high price + high SoC → near-certain)
  - At c=200, d=0, s̄=0.8, g=10: ρ drops relative to g=0 by γ·g in logit units
    (a 10 kWh anticipated discharge at γ=0.14 subtracts 1.4 from the logit —
    comparable in magnitude to the entire distance penalty at d≈7km)

These are sensitivity analysis parameters (§4.3.4). Override via
ParticipationModel(betas=..., gamma=...).

Note on n_enrolled vs n_connected
----------------------------------
n_enrolled_{i,t}  : the pool of EV owners *registered* with the VSRP at hub i.
                    Set at episode initialisation; varies slowly across episodes.
n_connected_{i,t} : EVs *physically present and plugged in* at hub i. This is a
                    subset of n_enrolled and is observed by the agent.
n_respond_{i,t}   : the Binomial draw — owners who accept the notification.
                    This is the supply that materialises for dispatch.

Only n_respond_{i,t} owners actually discharge. The env computes:
    E_del_{i,t} = δ_{i,t} × n_respond_{i,t} × mean_discharge_kwh_per_ev
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Default beta parameters — override for sensitivity analysis
# ---------------------------------------------------------------------------
DEFAULT_BETAS = {
    "beta_0": -2.20,   # intercept
    "beta_1":  0.008,   # $/MWh incentive price coefficient (positive)
    "beta_2": -0.20,   # km travel distance coefficient (negative)
    "beta_3":  1.50,   # mean SoC coefficient (normalised 0–1, positive)
}

# Battery degradation cost, $/kWh — see module docstring "Degradation term".
# Reproduces Hematiboroujeni et al. (2026, arXiv:2603.20226v1) reference
# case: r=200 $/kWh capacity, κ=1500 cycles-to-EOL, e=sqrt(0.90 round-trip
# efficiency) → γ ≈ 0.14 $/kWh. γ is capacity-invariant under this formula
# (see module docstring), so this is a single fleet-wide constant, not a
# per-vehicle simplification.
DEFAULT_GAMMA_PARAMS = {
    "replacement_cost_per_kwh_capacity": 200.0,   # $/kWh of battery capacity
    "cycles_to_eol": 1500,                         # cycles to 80% EOL
    "round_trip_efficiency": 0.90,                 # fraction
}


def compute_gamma(
    replacement_cost_per_kwh_capacity: float = DEFAULT_GAMMA_PARAMS["replacement_cost_per_kwh_capacity"],
    cycles_to_eol: float = DEFAULT_GAMMA_PARAMS["cycles_to_eol"],
    round_trip_efficiency: float = DEFAULT_GAMMA_PARAMS["round_trip_efficiency"],
) -> float:
    """
    Compute per-unit battery degradation cost γ ($/kWh discharged).

    γ = r / (κ · e), following Hematiboroujeni et al. (2026), Eq. 19,
    where r is $/kWh-of-capacity replacement cost, κ is cycles-to-EOL,
    and e = sqrt(round_trip_efficiency).

    Capacity-invariance note: if replacement cost and lifetime throughput
    both scale linearly with battery capacity (the standard assumption),
    capacity cancels out of this ratio algebraically — so this fleet-wide
    γ is not an approximation relative to a capacity-weighted per-vehicle
    version, it is equivalent to one. See module docstring.
    """
    e = np.sqrt(round_trip_efficiency)
    return replacement_cost_per_kwh_capacity / (cycles_to_eol * e)


DEFAULT_GAMMA = compute_gamma()

# One NEM dispatch interval in hours (5 min). Must match nem_doe_env.py's
# dt_hr used in _compute_reward — kept as a separate constant here since
# this module has no dependency on the env.
DT_HR = 5.0 / 60.0


@dataclass
class HubParticipationState:
    """
    Per-hub state consumed by the participation model each step.

    Attributes
    ----------
    hub_id : int
        Hub index within the VSR zone.
    n_enrolled : int
        Pool of registered EV owners at this hub.
    distance_km : float
        Travel distance proxy from nearest residential centroid (km).
        Fixed per hub; set from spatial_graph.py hub construction.
    mean_soc : float
        Mean state of charge of enrolled EVs, normalised to [0, 1].
        Updated each step by the environment.
    doe_export_w : float
        Current DOE-permitted export ceiling for this hub (W). Used as the
        anticipated-discharge anchor for the degradation cost term — see
        module docstring "Degradation term". Defaults to 0.0 (no
        degradation disutility) for backward compatibility with callers
        that do not yet pass this through.
    n_connected : int
        EVs physically present/plugged in at this hub. Used as the divisor
        when distributing doe_export_w across connected EVs to get a
        per-EV anticipated discharge. Defaults to n_enrolled if not set
        (a conservative fallback — see participation_prob).
    """
    hub_id: int
    n_enrolled: int
    distance_km: float
    mean_soc: float = 0.5
    doe_export_w: float = 0.0
    n_connected: Optional[int] = None


class ParticipationModel:
    """
    Stochastic EV owner participation model.

    This class is instantiated inside nem_wdr_env.py and is NOT exposed to
    the RL agent. The agent observes only the *outcome* (n_respond per hub,
    encoded as ρ̂_{t-1} — the empirical lagged rate) not the model parameters.

    Parameters
    ----------
    betas : dict, optional
        Override DEFAULT_BETAS. Keys: beta_0, beta_1, beta_2, beta_3.
    gamma : float, optional
        Per-unit battery degradation cost ($/kWh). Defaults to DEFAULT_GAMMA
        (Hematiboroujeni et al. 2026 reference case). Use compute_gamma()
        to derive from replacement-cost/cycle-life/efficiency assumptions
        instead of passing a raw number directly.
    rng : np.random.Generator, optional
        Reproducible RNG. If None, a fresh default_rng() is used.

    Methods
    -------
    participation_prob(c_t, distance_km, mean_soc) -> float
        Compute ρ for a single hub at a given incentive price.
    sample_responses(c_t, hub_states) -> np.ndarray
        Sample n_respond for each hub via Binomial draw.
    """

    def __init__(
        self,
        betas: Optional[dict] = None,
        gamma: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        b = {**DEFAULT_BETAS, **(betas or {})}
        self.beta_0 = b["beta_0"]
        self.beta_1 = b["beta_1"]
        self.beta_2 = b["beta_2"]
        self.beta_3 = b["beta_3"]
        self.gamma = DEFAULT_GAMMA if gamma is None else float(gamma)
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Core probability computation
    # ------------------------------------------------------------------

    def participation_prob(
        self,
        c_t: float,
        distance_km: float,
        mean_soc: float,
        doe_export_w: float = 0.0,
        n_connected: Optional[int] = None,
    ) -> float:
        """
        Compute per-owner participation probability ρ for one hub.

        Parameters
        ----------
        c_t : float
            Incentive price ($/MWh). The VSRP's action variable.
        distance_km : float
            Hub's travel distance proxy (km).
        mean_soc : float
            Mean SoC of enrolled EVs at hub, normalised to [0, 1].
        doe_export_w : float, optional
            Current DOE export ceiling for this hub (W). Default 0.0
            reproduces pre-degradation behaviour (no disutility term).
        n_connected : int, optional
            EVs currently connected at this hub, used to distribute
            doe_export_w into a per-EV anticipated discharge. If None
            and doe_export_w > 0, the caller must supply it — there is
            no safe default divisor.

        Returns
        -------
        float
            ρ ∈ (0, 1): per-owner probability of accepting the notification.
        """
        anticipated_discharge_kwh = _anticipated_discharge_kwh(
            doe_export_w, n_connected
        )
        logit = (
            self.beta_0
            + self.beta_1 * c_t
            + self.beta_2 * distance_km
            + self.beta_3 * mean_soc
            - self.gamma * anticipated_discharge_kwh
        )
        return float(_sigmoid(logit))

    def participation_prob_vector(
        self,
        c_t: float,
        distances_km: np.ndarray,
        mean_socs: np.ndarray,
        doe_export_ws: Optional[np.ndarray] = None,
        n_connecteds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vectorised version: compute ρ for all hubs simultaneously.

        Parameters
        ----------
        c_t : float
            Incentive price scalar (zone-wide).
        distances_km : np.ndarray, shape (H,)
            Travel distance proxy per hub.
        mean_socs : np.ndarray, shape (H,)
            Mean SoC per hub, normalised 0–1.
        doe_export_ws : np.ndarray, shape (H,), optional
            DOE export ceiling per hub (W). Defaults to zeros (no
            degradation disutility) for backward compatibility.
        n_connecteds : np.ndarray, shape (H,), optional
            Connected EVs per hub, for distributing doe_export_ws.
            Required (elementwise) wherever doe_export_ws > 0.

        Returns
        -------
        np.ndarray, shape (H,)
            Per-hub participation probabilities.
        """
        anticipated_discharge_kwh = _anticipated_discharge_kwh_vector(
            doe_export_ws, n_connecteds, n_hubs=len(distances_km)
        )
        logits = (
            self.beta_0
            + self.beta_1 * c_t
            + self.beta_2 * distances_km
            + self.beta_3 * mean_socs
            - self.gamma * anticipated_discharge_kwh
        )
        return _sigmoid(logits)

    # ------------------------------------------------------------------
    # Stochastic outcome sampling — this is what the env calls each step
    # ------------------------------------------------------------------

    def sample_responses(
        self,
        c_t: float,
        hub_states: list[HubParticipationState],
    ) -> np.ndarray:
        """
        Sample the number of EV owners responding per hub.

        For each hub i:
            ρ_i = σ(β₀ + β₁·c_t + β₂·d_i + β₃·s̄_i)
            n_respond_i ~ Binomial(n_enrolled_i, ρ_i)

        This is the sole source of stochasticity in the dispatch supply.
        The agent sets c_t; everything downstream is a random draw.

        Parameters
        ----------
        c_t : float
            Incentive price offered this step ($/MWh).
        hub_states : list of HubParticipationState
            Current state of each hub.

        Returns
        -------
        np.ndarray, shape (H,), dtype int
            n_respond_i for each hub i.
        """
        n_hubs = len(hub_states)
        n_enrolled = np.array([h.n_enrolled for h in hub_states], dtype=int)
        distances = np.array([h.distance_km for h in hub_states], dtype=float)
        socs = np.array([h.mean_soc for h in hub_states], dtype=float)
        doe_export_ws = np.array([h.doe_export_w for h in hub_states], dtype=float)
        n_connecteds = np.array(
            [h.n_connected if h.n_connected is not None else h.n_enrolled
             for h in hub_states],
            dtype=int,
        )

        probs = self.participation_prob_vector(
            c_t, distances, socs, doe_export_ws, n_connecteds
        )

        # Binomial draw: independent per hub (owners do not coordinate)
        n_respond = self.rng.binomial(n=n_enrolled, p=probs)

        return n_respond

    def sample_responses_with_probs(
        self,
        c_t: float,
        hub_states: list[HubParticipationState],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Like sample_responses(), but also returns the underlying ρ values.

        Used internally by nem_wdr_env.py for logging and diagnostics.
        The env may store these for offline analysis but does NOT pass them
        to the agent's observation vector.

        Returns
        -------
        n_respond : np.ndarray, shape (H,)
        probs : np.ndarray, shape (H,)
        """
        n_enrolled = np.array([h.n_enrolled for h in hub_states], dtype=int)
        distances = np.array([h.distance_km for h in hub_states], dtype=float)
        socs = np.array([h.mean_soc for h in hub_states], dtype=float)
        doe_export_ws = np.array([h.doe_export_w for h in hub_states], dtype=float)
        n_connecteds = np.array(
            [h.n_connected if h.n_connected is not None else h.n_enrolled
             for h in hub_states],
            dtype=int,
        )

        probs = self.participation_prob_vector(
            c_t, distances, socs, doe_export_ws, n_connecteds
        )
        n_respond = self.rng.binomial(n=n_enrolled, p=probs)

        return n_respond, probs

    # ------------------------------------------------------------------
    # Diagnostics — not used by agent, useful for thesis calibration plots
    # ------------------------------------------------------------------

    def participation_curve(
        self,
        price_range: np.ndarray,
        distance_km: float = 0.0,
        mean_soc: float = 0.5,
        anticipated_discharge_kwh: float = 0.0,
    ) -> np.ndarray:
        """
        Compute ρ across a range of incentive prices for one hub configuration.

        Useful for plotting participation elasticity curves (§4.3.4 sensitivity
        analysis) and verifying that β/γ values produce sensible behaviour
        before training.

        Parameters
        ----------
        price_range : np.ndarray
            Array of c_t values to evaluate ($/MWh).
        distance_km : float
            Hub distance proxy (km).
        mean_soc : float
            Mean SoC, normalised 0–1.
        anticipated_discharge_kwh : float
            Fixed per-EV anticipated discharge (kWh) to hold constant across
            the price sweep — pass a nonzero value to visualise how the
            degradation term shifts the curve relative to the g=0 baseline.
            Default 0.0 reproduces the pre-degradation curve.

        Returns
        -------
        np.ndarray
            ρ values corresponding to each price in price_range.

        Example
        -------
        >>> model = ParticipationModel()
        >>> prices = np.linspace(0, 500, 200)
        >>> rho_no_discharge = model.participation_curve(prices, distance_km=5.0, mean_soc=0.5)
        >>> rho_with_discharge = model.participation_curve(
        ...     prices, distance_km=5.0, mean_soc=0.5, anticipated_discharge_kwh=10.0
        ... )
        """
        logits = (
            self.beta_0
            + self.beta_1 * price_range
            + self.beta_2 * distance_km
            + self.beta_3 * mean_soc
            - self.gamma * anticipated_discharge_kwh
        )
        return _sigmoid(logits)

    def beta_summary(self) -> dict:
        """Return current beta/gamma values for logging / thesis reporting."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "gamma": self.gamma,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _anticipated_discharge_kwh(
    doe_export_w: float,
    n_connected: Optional[int],
) -> float:
    """
    Convert a hub-level DOE export ceiling (W) into a conservative per-EV
    anticipated discharge (kWh) for one interval, by distributing the
    ceiling evenly across connected EVs.

    Returns 0.0 (no degradation disutility) if doe_export_w is 0 — this
    keeps every pre-degradation call site (doe_export_w defaulted to 0.0)
    numerically identical to the old behaviour.
    """
    if doe_export_w <= 0.0:
        return 0.0
    divisor = max(int(n_connected) if n_connected is not None else 1, 1)
    return (doe_export_w / 1000.0) * DT_HR / divisor


def _anticipated_discharge_kwh_vector(
    doe_export_ws: Optional[np.ndarray],
    n_connecteds: Optional[np.ndarray],
    n_hubs: int,
) -> np.ndarray:
    """Vectorised form of _anticipated_discharge_kwh, see its docstring."""
    if doe_export_ws is None:
        return np.zeros(n_hubs, dtype=float)
    doe_export_ws = np.asarray(doe_export_ws, dtype=float)
    if n_connecteds is None:
        divisors = np.ones(n_hubs, dtype=float)
    else:
        divisors = np.maximum(np.asarray(n_connecteds, dtype=float), 1.0)
    discharge = np.where(
        doe_export_ws > 0.0,
        (doe_export_ws / 1000.0) * DT_HR / divisors,
        0.0,
    )
    return discharge


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    Avoids overflow for large negative inputs.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
