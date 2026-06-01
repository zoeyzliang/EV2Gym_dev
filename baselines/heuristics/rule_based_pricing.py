"""
baselines/heuristics/rule_based_pricing.py
==========================================
Rule-based pricing baseline (Table 3, Baseline 2).

Policy: Offer a fixed percentage of the current spot price as incentive,
with uniform dispatch fractions across all hubs.

Addresses RQ2 and RQ5: evaluates the value of jointly optimised pricing
over a static price rule. The agent must learn that spot-price-linked
pricing is suboptimal because participation probability depends on the
absolute incentive level, not its relationship to the spot price.

Rationale for this baseline: a naive VSRP might index their incentive
to the spot price (offer EV owners a share of what they earn from AEMO).
This baseline tests whether SAC's learned pricing significantly
outperforms this simple rule.
"""

import numpy as np
from nem_env.nem_wdr_env import NEMWDREnv


class RuleBasedPricingBaseline:
    """
    Fixed spot-price-fraction incentive with uniform hub dispatch.

    Parameters
    ----------
    n_hubs : int
        Number of hubs.
    price_fraction : float
        Fraction of spot price to offer as incentive.
        Default 0.5 — offer EV owners 50% of wholesale revenue.
    dispatch_fraction : float
        Uniform dispatch fraction applied to all hubs.
        Default 1.0 — always request full discharge.
    price_min : float
        Minimum incentive price floor ($/MWh).
    price_max : float
        Maximum incentive price cap ($/MWh).
    """

    def __init__(
        self,
        n_hubs: int,
        price_fraction: float = 0.5,
        dispatch_fraction: float = 1.0,
        price_min: float = 0.0,
        price_max: float = 500.0,
    ):
        self.n_hubs = n_hubs
        self.price_fraction = price_fraction
        self.dispatch_fraction = dispatch_fraction
        self.price_min = price_min
        self.price_max = price_max
        self.name = "RuleBasedPricing"

    def select_action(self, obs: np.ndarray, env: NEMWDREnv) -> np.ndarray:
        """
        Select action from observation.

        Price: price_fraction × current spot price, clipped to [min, max].
        Dispatch: uniform dispatch_fraction across all hubs.

        The spot price is extracted from zone-level features (index 0,
        normalised by price_normalise_by=500 in nem_wdr_env.py).
        """
        _, zone_feats = env.obs_to_node_and_zone(obs)

        # Zone feature index 0 = spot_price / price_normalise_by
        spot_price_norm = float(zone_feats[0])
        spot_price = spot_price_norm * 500.0   # denormalise

        # Incentive price: fraction of spot price
        incentive_price = float(np.clip(
            self.price_fraction * max(spot_price, 0.0),
            self.price_min,
            self.price_max,
        ))

        dispatch = np.full(self.n_hubs, self.dispatch_fraction, dtype=np.float32)
        action = np.append(dispatch, incentive_price).astype(np.float32)
        return action

    def reset(self):
        pass
