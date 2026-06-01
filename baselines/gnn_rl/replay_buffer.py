"""
replay_buffer.py
================
Off-policy experience replay buffer for SAC-GNN.

Why off-policy replay matters for this problem
----------------------------------------------
WDR activation events are sparse in reality (~3 activations/year in VIC).
Even with curriculum training (force_wdr=True), the conformance penalty
term in the reward only fires during WDR intervals — roughly 1–4% of
training steps. An on-policy algorithm like PPO discards experience after
each update, wasting the rare WDR transitions. SAC's replay buffer retains
all transitions and resamples them uniformly, so WDR transitions accumulate
over time and are replayed many times. This is the primary sample-efficiency
argument for SAC over PPO given in §2.2.2 and RQ5.

Buffer design
-------------
Stores flat numpy arrays for observations and actions (no graph structure
in the buffer — the graph is static and reconstructed from the observation
at sample time). Each transition is:
    (obs, action, reward, next_obs, done)

where obs is the flat (H×NODE_DIM + ZONE_DIM,) vector from NEMWDREnv.

The graph (edge_index, edge_attr) is stored once separately and attached
to every sampled batch — it doesn't change between steps within an episode
or across episodes (same hub network throughout training).

Prioritised replay (future extension)
--------------------------------------
A uniform buffer is used here. Prioritised Experience Replay (PER) would
give higher sampling weight to WDR transitions, further improving sample
efficiency. This is noted as a future extension in §5. The buffer interface
is designed to support PER addition without changing the agent code.
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
    actions : np.ndarray, shape (B, action_dim)
    rewards : np.ndarray, shape (B, 1)
    next_obs : np.ndarray, shape (B, obs_dim)
    dones : np.ndarray, shape (B, 1)
    wdr_active : np.ndarray, shape (B, 1), dtype bool
        Whether a WDR event was active at each transition.
        Stored separately for diagnostic logging (not used in loss computation).
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    wdr_active: np.ndarray


class ReplayBuffer:
    """
    Circular replay buffer for off-policy SAC training.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension. = H × NODE_FEATURE_DIM + ZONE_FEATURE_DIM.
    action_dim : int
        Action dimension. = H + 1 (H dispatch fractions + 1 price scalar).
    capacity : int
        Maximum number of transitions to store. When full, oldest
        transitions are overwritten (circular buffer).
        Default 1,000,000 — standard for continuous control SAC.
    seed : int, optional
        RNG seed for reproducible sampling.

    Notes on capacity
    -----------------
    At 288 steps/episode, 1M capacity ≈ 3,472 episodes ≈ ~9.5 years of
    simulated NEM days. In practice training converges well before this.
    For memory-constrained machines, 100,000 is sufficient.
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
        self._wdr      = np.zeros((capacity, 1),          dtype=bool)

        self._ptr = 0       # write pointer
        self._size = 0      # current number of stored transitions

        # Diagnostic counters — logged by trainer for thesis reporting
        self._total_added = 0
        self._wdr_added = 0

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
        wdr_active: bool = False,
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
        wdr_active : bool
            Whether a WDR dispatch event was active this step.
            Logged for diagnostic purposes.
        """
        self._obs[self._ptr]      = obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr]    = float(done)
        self._wdr[self._ptr]      = wdr_active

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        self._total_added += 1
        self._wdr_added += int(wdr_active)

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
            wdr_active=self._wdr[indices],
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

    @property
    def wdr_fraction(self) -> float:
        """Fraction of stored transitions that had WDR active."""
        if self._total_added == 0:
            return 0.0
        return self._wdr_added / self._total_added

    def summary(self) -> dict:
        """Diagnostic summary for logging."""
        return {
            "buffer_size": self._size,
            "buffer_capacity": self.capacity,
            "buffer_fill_pct": 100 * self._size / self.capacity,
            "total_transitions_added": self._total_added,
            "wdr_transitions_added": self._wdr_added,
            "wdr_fraction_in_buffer": round(self.wdr_fraction, 4),
        }
