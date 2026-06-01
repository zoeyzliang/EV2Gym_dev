"""
baselines/heuristics/greedy_dispatch.py
========================================
Greedy dispatch baseline (Table 3, Baseline 1).

Policy: At each step, allocate full dispatch (δ=1.0) to hubs ranked by
descending mean SoC, and offer a fixed incentive price equal to the
mean spot price observed so far in the episode.

Addresses RQ1 and RQ2: evaluates the value of learned dispatch
allocation and pricing over a simple heuristic that requires no
training.

This is intentionally simple — it represents what a naive VSRP operator
might do without any machine learning.
"""

import numpy as np
from nem_env.nem_wdr_env import NEMWDREnv


class GreedyDispatchBaseline:
    """
    Greedy dispatch with fixed mean-price incentive.

    Parameters
    ----------
    n_hubs : int
        Number of hubs.
    fixed_price : float
        Fixed incentive price offered every step ($/MWh).
        Default 80.0 — approximate historical NEM mean price.
    """

    def __init__(self, n_hubs: int, fixed_price: float = 80.0):
        self.n_hubs = n_hubs
        self.fixed_price = fixed_price
        self.name = "GreedyDispatch"

    def select_action(self, obs: np.ndarray, env: NEMWDREnv) -> np.ndarray:
        """
        Select action from observation.

        Dispatch fractions: rank hubs by mean_soc (node feature index 1),
        assign δ=1.0 to top half, δ=0.0 to bottom half.
        Price: fixed at self.fixed_price.
        """
        # Extract per-hub node features from flat obs
        node_feats, _ = env.obs_to_node_and_zone(obs)
        mean_socs = node_feats[:, 1]  # column 1 = mean_soc

        # Rank hubs by SoC descending
        ranked = np.argsort(-mean_socs)
        dispatch = np.zeros(self.n_hubs, dtype=np.float32)
        # Dispatch top half fully
        top_k = max(1, self.n_hubs // 2)
        dispatch[ranked[:top_k]] = 1.0

        action = np.append(dispatch, self.fixed_price).astype(np.float32)
        return action

    def reset(self):
        """No state to reset for this baseline."""
        pass
