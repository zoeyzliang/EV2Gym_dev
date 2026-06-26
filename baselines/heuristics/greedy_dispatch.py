"""
baselines/heuristics/greedy_dispatch.py
========================================
Greedy dispatch baseline (Table 3, Baseline 1).

Policy: discharge at full equipment cap when RRP > threshold, charge at
full cap when RRP <= threshold. Offers a fixed incentive price each step.

This is a price-taker heuristic requiring no training — it represents
what a naive operator might do using only the current spot price signal.
Addresses RQ1 and RQ2: establishes the value of learned SAC-GNN policy
over a simple rule-based alternative.
"""

import numpy as np
from nem_env.nem_doe_env import NEMDOEEnv


class GreedyDispatchBaseline:
    """
    Greedy dispatch with fixed mean-price incentive.

    Parameters
    ----------
    n_hubs : int
        Number of hubs.
    discharge_threshold : float
        RRP threshold ($/MWh) above which the agent discharges.
        Below this, agent charges. Default 100.0.
    fixed_price : float
        Fixed incentive price offered every step ($/kWh).
        Default 0.10 — approximate mid-range incentive.
    """

    def __init__(self, n_hubs: int, discharge_threshold: float = 100.0,
                 fixed_price: float = 0.10):
        self.n_hubs = n_hubs
        self.discharge_threshold = discharge_threshold
        self.fixed_price = fixed_price
        self.name = "GreedyDispatch"

    def select_action(self, obs: np.ndarray, env: NEMDOEEnv) -> np.ndarray:
        """
        Select action from observation.

        Dispatch: if RRP > discharge_threshold → discharge all hubs at
        equipment cap (positive kW); else charge all hubs at full cap
        (negative kW). Equipment cap is node feature [5].
        Price: fixed at self.fixed_price ($/kWh).
        """
        # Reshape flat obs to (H, 9) node features — no zone block
        node_feats = env.obs_to_node_features(obs)  # (H, 9)

        # RRP is broadcast to all nodes as feature [6]; read from hub 0
        rrp_norm = float(node_feats[0, 6])  # normalised
        rrp = rrp_norm * env.cfg.rrp_clip_high   # denormalise to $/MWh

        # Equipment cap per hub: node feature [5] in kW
        equipment_caps = node_feats[:, 5]  # (H,)

        if rrp > self.discharge_threshold:
            # High price: discharge (positive kW)
            dispatch = equipment_caps.copy()
        else:
            # Low/negative price: charge (negative kW)
            dispatch = -equipment_caps.copy()

        action = np.append(dispatch, self.fixed_price).astype(np.float32)
        return action

    def reset(self):
        """No state to reset for this baseline."""
        pass