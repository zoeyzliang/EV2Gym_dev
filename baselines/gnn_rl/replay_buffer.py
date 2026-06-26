"""
replay_buffer.py
================
Off-policy experience replay buffer for SAC-GNN.

Why off-policy replay matters for this problem
----------------------------------------------
SAC's replay buffer retains all transitions and resamples them uniformly,
providing stable gradient estimates across the full distribution of price
conditions (low/negative RRP, high RRP spikes, normal trading). An
on-policy algorithm like PPO would discard experience after each update,
wasting transitions from rare high-RRP spike intervals that carry the
strongest arbitrage learning signal. This is the primary sample-efficiency
argument for SAC over PPO given in §2.2.2 and RQ5.

Buffer design
-------------
Stores flat numpy arrays for observations and actions (no graph structure
in the buffer — the graph is static and reconstructed from the observation
at sample time). Each transition is:
    (obs, action, reward, next_obs, done)

where obs is the flat (H × NODE_FEATURE_DIM,) = (H × 9,) vector from
NEMDOEEnv. No zone feature block — RRP is in node feature [6].

The graph (edge_index, edge_attr) is stored once separately and attached
to every sampled batch — it doesn't change between steps within an episode
or across episodes (same hub network throughout training).

Prioritised replay (future extension)
--------------------------------------
A uniform buffer is used here. Prioritised Experience Replay (PER) would
give higher sampling weight to high-RRP spike transitions, further improving
sample efficiency on rare price events. This is noted as a future extension
in §5. The buffer interface is designed to support PER without changing the
agent code.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Batch:
    """
    A batch of transitions sampled from the replay buffer.

    Attributes
    ----------
    obs : np.ndarray, shape (B, obs_dim)
        Flat observation: (H × 9,) from NEMDOEEnv.
    actions : np.ndarray, shape (B, action_dim)
        [dispatch_0_kw, ..., dispatch_{H-1}_kw, incentive_price_per_kwh]
    rewards : np.ndarray, shape (B, 1)
    next_obs : np.ndarray, shape (B, obs_dim)
    dones : np.ndarray, shape (B, 1)
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """
    Circular replay buffer for off-policy SAC training.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension. = H × NODE_FEATURE_DIM = H × 9.
        No zone feature block — RRP is broadcast into each hub's node features.
    action_dim : int
        Action dimension. = H + 1 (H signed dispatch kW + 1 price $/kWh).
    capacity : int
        Maximum number of transitions to store. When full, oldest
        transitions are overwritten (circular buffer).
        Default 1,000,000 — standard for continuous control SAC.
    seed : int, optional
        RNG seed for reproducible sampling.

    Notes on capacity
    -----------------
    At 288 steps/episode, 1M capacity ≈ 3,472 episodes ≈ ~9.5 years of
    simulated NEM trading days. In practice SAC converges well before this.
    For memory-constrained machines (e.g. MacBook dev runs), 100,000 is
    sufficient for 500-episode local runs.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        seed: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        # Pre-allocate arrays (avoids Python list append overhead)
        self._obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),          dtype=np.float32)

        self._ptr = 0       # write pointer
        self._size = 0      # current number of stored transitions

        # Diagnostic counter
        self._total_added = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one transition to the buffer.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
        action : np.ndarray, shape (action_dim,)
        reward : float
        next_obs : np.ndarray, shape (obs_dim,)
        done : bool
            True if this is the last step of an episode.
        """
        self._obs[self._ptr]      = obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr]    = float(done)

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        self._total_added += 1

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a random minibatch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample. Typically 256 for SAC.

        Returns
        -------
        Batch
            Sampled transitions as numpy arrays.

        Raises
        ------
        RuntimeError
            If the buffer contains fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._size} transitions, "
                f"need at least {batch_size} to sample. "
                f"Continue collecting experience before training."
            )

        indices = self._rng.integers(0, self._size, size=batch_size)

        return Batch(
            obs=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            dones=self._dones[indices],
        )

    def can_sample(self, batch_size: int) -> bool:
        """Return True if the buffer has enough transitions to sample."""
        return self._size >= batch_size

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def summary(self) -> dict:
        """Diagnostic summary for logging."""
        return {
            "buffer_size": self._size,
            "buffer_capacity": self.capacity,
            "buffer_fill_pct": 100 * self._size / self.capacity,
            "total_transitions_added": self._total_added,
        }