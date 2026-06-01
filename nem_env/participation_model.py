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

    ρ(c_t, d_i, s̄_{i,t}) = σ(β₀ + β₁·c_t + β₂·d_i + β₃·s̄_{i,t})

where σ is the logistic sigmoid, and:
  c_t   : incentive price offered ($/MWh, zone-wide scalar)
  d_i   : travel distance proxy for hub i (km from residential centroid)
  s̄_i,t : mean SoC of enrolled EVs at hub i (normalised 0–1)

The number of responding owners at hub i is then:

    n_respond_{i,t} ~ Binomial(n_enrolled_{i,t}, ρ(c_t, d_i, s̄_{i,t}))

Default parameter values
------------------------
β₀ = -2.20  (intercept; implies ρ ≈ 0.10 at zero price, near distance, mid SoC)
β₁ = +0.04  (price sensitivity; ρ increases ~20pp per $50/MWh increase)
β₂ = -0.20  (distance penalty; ρ drops ~20pp per 10 km)
β₃ = +1.50  (SoC effect; high SoC owners significantly more willing)

Calibration basis: Liu et al. (2025) systematic review of V2G acceptance [14]
establishes that economic incentive, range anxiety (proxied by d_i and s̄),
and convenience are the dominant participation antecedents. The β values
produce the following sanity-check behaviour:
  - At c=0, d=0, s̄=0.5:  ρ ≈ 0.12
  - At c=100, d=0, s̄=0.5: ρ ≈ 0.55  (a meaningful incentive works)
  - At c=100, d=10, s̄=0.5: ρ ≈ 0.38  (distance penalty is real)
  - At c=200, d=0, s̄=0.8: ρ ≈ 0.90  (high price + high SoC → near-certain)

These are sensitivity analysis parameters (§4.3.4). Override via ParticipationModel(betas=...).

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
    """
    hub_id: int
    n_enrolled: int
    distance_km: float
    mean_soc: float = 0.5


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
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        b = {**DEFAULT_BETAS, **(betas or {})}
        self.beta_0 = b["beta_0"]
        self.beta_1 = b["beta_1"]
        self.beta_2 = b["beta_2"]
        self.beta_3 = b["beta_3"]
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

        Returns
        -------
        float
            ρ ∈ (0, 1): per-owner probability of accepting the notification.
        """
        logit = (
            self.beta_0
            + self.beta_1 * c_t
            + self.beta_2 * distance_km
            + self.beta_3 * mean_soc
        )
        return float(_sigmoid(logit))

    def participation_prob_vector(
        self,
        c_t: float,
        distances_km: np.ndarray,
        mean_socs: np.ndarray,
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

        Returns
        -------
        np.ndarray, shape (H,)
            Per-hub participation probabilities.
        """
        logits = (
            self.beta_0
            + self.beta_1 * c_t
            + self.beta_2 * distances_km
            + self.beta_3 * mean_socs
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

        probs = self.participation_prob_vector(c_t, distances, socs)

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

        probs = self.participation_prob_vector(c_t, distances, socs)
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
    ) -> np.ndarray:
        """
        Compute ρ across a range of incentive prices for one hub configuration.

        Useful for plotting participation elasticity curves (§4.3.4 sensitivity
        analysis) and verifying that β values produce sensible behaviour before
        training.

        Parameters
        ----------
        price_range : np.ndarray
            Array of c_t values to evaluate ($/MWh).
        distance_km : float
            Hub distance proxy (km).
        mean_soc : float
            Mean SoC, normalised 0–1.

        Returns
        -------
        np.ndarray
            ρ values corresponding to each price in price_range.

        Example
        -------
        >>> model = ParticipationModel()
        >>> prices = np.linspace(0, 500, 200)
        >>> rho = model.participation_curve(prices, distance_km=5.0, mean_soc=0.5)
        """
        logits = (
            self.beta_0
            + self.beta_1 * price_range
            + self.beta_2 * distance_km
            + self.beta_3 * mean_soc
        )
        return _sigmoid(logits)

    def beta_summary(self) -> dict:
        """Return current beta values for logging / thesis reporting."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

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
