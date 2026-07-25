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
from nem_env.nem_doe_env import NEMDOEEnv


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
        price_max: float = 0.50,   # $/kWh
    ):
        self.n_hubs = n_hubs
        self.price_fraction = price_fraction
        self.dispatch_fraction = dispatch_fraction
        self.price_min = price_min
        self.price_max = price_max
        self.name = "RuleBasedPricing"

    def select_action(self, obs: np.ndarray, env: NEMDOEEnv) -> np.ndarray:
        """
        Select action from observation.

        Price: price_fraction × current spot price, clipped to [min, max].
        Dispatch: uniform dispatch_fraction across all hubs.

        RRP is node feature [6], normalised by rrp_clip_high=20300 $/MWh.
        No zone features — RRP is broadcast to all hub nodes.
        """
        node_feats = env.obs_to_node_features(obs)  # (H, 9)

        # RRP: feature [6], denormalise
        spot_price_norm = float(node_feats[0, 6])
        spot_price = spot_price_norm * env.cfg.rrp_clip_high   # $/MWh

        # Incentive price: fraction of spot price
        incentive_price = float(np.clip(
            self.price_fraction * max(spot_price, 0.0) / 1000.0,  # $/MWh → $/kWh
            self.price_min,
            self.price_max,
        ))

        # Signed dispatch: discharge at full cap when RRP > incentive price
        if spot_price > incentive_price * 1000.0:  # compare $/MWh
            dispatch = np.full(self.n_hubs, self.dispatch_fraction * 100.0, dtype=np.float32)  # kW
        else:
            dispatch = np.full(self.n_hubs, -self.dispatch_fraction * 100.0, dtype=np.float32)  # charge
        action = np.append(dispatch, incentive_price).astype(np.float32)
        return action

    def reset(self):
        pass
