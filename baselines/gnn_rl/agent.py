"""
agent.py
========
Soft Actor-Critic (SAC) agent with GAT spatial encoder.

This is the primary contribution of Phase 4. The agent wraps the
networks and replay buffer into a complete training loop.

SAC algorithm recap (Haarnoja et al., 2018)
--------------------------------------------
SAC maximises a trade-off between expected return and entropy:

    J(π) = Σ_t E[r_t + α H(π(·|s_t))]

where α is the entropy temperature (automatically tuned) and H is
the policy entropy. The entropy term discourages premature convergence
to a fixed action — in this problem, it keeps the agent exploring the
incentive price space until the participation response surface has been
adequately characterised. This is the mechanism that addresses RQ5.

Three network pairs are maintained:
  - Actor (policy): π_θ(a|s) — outputs action distribution
  - Critic 1+2 (Q-functions): Q_φ1, Q_φ2 — clipped double-Q trick
  - Target critics: Q_φ1', Q_φ2' — soft-updated for stability

Loss functions
--------------
Critic loss (MSE against Bellman target):
    y = r + γ(1-d)[min(Q_φ1'(s',a'), Q_φ2'(s',a')) - α log π(a'|s')]
    L_Q = (Q_φ(s,a) - y)²

Actor loss (maximise Q - entropy):
    L_π = α log π(a|s) - min(Q_φ1(s,a_π), Q_φ2(s,a_π))

Temperature loss (auto-tune α to target entropy):
    L_α = -α(log π(a|s) + H_target)

Observation handling
--------------------
The flat observation vector from NEMDOEEnv is (H × 9,) — no separate
zone feature block. It is reshaped to (H, 9) node features inside each
forward pass using obs_to_node_features(). RRP is already broadcast
into node feature [6] of every hub node, so the GAT encoder has full
price information without a separate zone vector.
The static graph (edge_index, edge_attr) is stored in the agent and
attached to every forward pass — it doesn't change during training.

Numpy fallback mode
-------------------
When torch_geometric is not available, the agent runs in validation
mode: it can store transitions and sample batches but cannot compute
gradients. All forward passes use the NumpyActor/NumpyCritic from
networks.py. This mode is used for shape validation in the test suite.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional

from .networks import (
    NetworkConfig,
    NumpyActor,
    NumpyCritic,
    build_torch_networks,
    _try_import_torch,
)
from .replay_buffer import ReplayBuffer, Batch

logger = logging.getLogger(__name__)


class SACGNNAgent:
    """
    SAC agent with GAT spatial encoder for NEM V2G hub dispatch.

    Parameters
    ----------
    n_hubs : int
        Number of hubs (H). Determines action and observation dimensions.
    graph_data : GraphData
        Static hub graph from spatial_graph.HubGraphBuilder.
        Contains edge_index and edge_attr used in every forward pass.
    obs_dim : int
        Flat observation dimension from NEMDOEEnv.observation_space.shape[0].
        = H × NEMDOEEnv.NODE_FEATURE_DIM = H × 9.
    net_cfg : NetworkConfig, optional
        Neural network hyperparameters.
    gamma : float
        Discount factor. Default 0.99.
    tau : float
        Soft update coefficient for target networks. Default 0.005.
    lr_actor : float
        Actor learning rate. Default 3e-4.
    lr_critic : float
        Critic learning rate. Default 3e-4.
    lr_alpha : float
        Entropy temperature learning rate. Default 3e-4.
    target_entropy : float, optional
        Target entropy for automatic alpha tuning.
        Default: -action_dim (standard SAC heuristic).
    batch_size : int
        Minibatch size for gradient updates. Default 256.
    buffer_capacity : int
        Replay buffer size. Default 1,000,000.
    learning_starts : int
        Number of transitions to collect before first gradient update.
        Default 1000 (≈ 3.5 episodes).
    update_every : int
        Number of environment steps between gradient updates. Default 1
        (update after every step, standard for SAC).
    seed : int, optional
        RNG seed.
    """

    def __init__(
        self,
        n_hubs: int,
        graph_data,                     # GraphData from spatial_graph.py
        obs_dim: int,
        net_cfg: Optional[NetworkConfig] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        target_entropy: Optional[float] = None,
        batch_size: int = 256,
        buffer_capacity: int = 1_000_000,
        learning_starts: int = 1000,
        update_every: int = 1,
        seed: Optional[int] = None,
    ):
        self.n_hubs = n_hubs
        self.graph_data = graph_data
        self.obs_dim = obs_dim
        self.action_dim = n_hubs + 1    # H signed dispatch targets (kW) + 1 price ($/kWh)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_every = update_every
        self._rng = np.random.default_rng(seed)

        self.net_cfg = net_cfg or NetworkConfig()

        # Target entropy: standard SAC heuristic = -action_dim
        self.target_entropy = (
            target_entropy
            if target_entropy is not None
            else -float(self.action_dim)
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_capacity,
            seed=seed,
        )

        # Initialise networks (PyTorch or numpy fallback)
        # Set env var FORCE_NUMPY_AGENT=1 to skip torch probe (e.g. in CI)
        import os
        force_numpy = os.environ.get("FORCE_NUMPY_AGENT", "0") == "1"
        self._use_torch = (not force_numpy) and _try_import_torch()
        if self._use_torch:
            self._init_torch_networks(lr_actor, lr_critic, lr_alpha)
            logger.info("SACGNNAgent: using PyTorch + GAT networks")
        else:
            self._init_numpy_networks()
            logger.info(
                "SACGNNAgent: torch_geometric unavailable — "
                "running numpy fallback (no training, shape validation only)"
            )

        # Training step counter
        self._total_steps = 0
        self._total_updates = 0

        # Running loss tracking
        self._loss_history = {
            "critic_loss": [],
            "actor_loss": [],
            "alpha_loss": [],
            "alpha": [],
        }

    # ------------------------------------------------------------------
    # Network initialisation
    # ------------------------------------------------------------------

    def _init_torch_networks(self, lr_actor, lr_critic, lr_alpha):
        import torch
        import torch.optim as optim

        actor, critic1, critic2 = build_torch_networks(self.net_cfg, self.n_hubs)
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

        # Target critics (soft-updated, not directly trained)
        _, target_critic1, target_critic2 = build_torch_networks(
            self.net_cfg, self.n_hubs
        )
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2

        # Initialise target networks with same weights as online critics
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)

        # Freeze target critics (updated only via soft update)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Entropy temperature α (learnable, log-parameterised for stability)
        self.log_alpha = torch.tensor(
            np.log(0.1), dtype=torch.float32, requires_grad=True
        )

        # Optimisers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Convert static graph to tensors (stored once, reused every step)
        self._edge_index_t = torch.tensor(
            self.graph_data.edge_index, dtype=torch.long
        )
        self._edge_attr_t = torch.tensor(
            self.graph_data.edge_attr, dtype=torch.float32
        )

    def _init_numpy_networks(self):
        """Fallback: numpy networks for shape validation without PyG."""
        self.actor = NumpyActor(self.net_cfg, self.n_hubs)
        self.critic1 = NumpyCritic(self.net_cfg, self.n_hubs)
        self.critic2 = NumpyCritic(self.net_cfg, self.n_hubs)
        self.log_alpha = np.log(0.1)
        self._edge_index_t = self.graph_data.edge_index
        self._edge_attr_t = self.graph_data.edge_attr

    # ------------------------------------------------------------------
    # Core agent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action given a flat observation.

        During training: samples stochastically from the policy distribution.
        During evaluation: returns the deterministic mean action.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
            Flat observation from NEMWDREnv.step().
        deterministic : bool
            Use deterministic policy (evaluation) or stochastic (training).

        Returns
        -------
        action : np.ndarray, shape (action_dim,)
            [δ_0, ..., δ_{H-1}, c_t] clipped to valid bounds.
        """
        # Reshape flat obs to node feature matrix (H, 9)
        node_feats = self._split_obs(obs)

        if self._use_torch:
            import torch
            with torch.no_grad():
                node_t = torch.tensor(node_feats, dtype=torch.float32)
                action, _ = self.actor(
                    node_t, self._edge_index_t,
                    deterministic=deterministic,
                )
            action = action.numpy()
        else:
            # Numpy fallback: always deterministic (no reparameterisation)
            action = self.actor.forward(node_feats, self._edge_index_t)

        return np.clip(action, self._action_low(), self._action_high())

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add one environment transition to the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self._total_steps += 1

    def update(self) -> Optional[dict]:
        """
        Perform one SAC gradient update if conditions are met.

        Conditions:
          1. Buffer has at least learning_starts transitions.
          2. Called on an update_every step boundary.

        Returns
        -------
        dict of loss values, or None if update was skipped.
        """
        if not self.buffer.can_sample(self.batch_size):
            return None
        if self._total_steps < self.learning_starts:
            return None
        if self._total_steps % self.update_every != 0:
            return None
        if not self._use_torch:
            return None  # no gradient updates in numpy fallback

        batch = self.buffer.sample(self.batch_size)
        losses = self._gradient_update(batch)
        self._total_updates += 1

        # Track loss history
        for k, v in losses.items():
            self._loss_history[k].append(v)

        return losses

    # ------------------------------------------------------------------
    # SAC gradient update
    # ------------------------------------------------------------------

    def _gradient_update(self, batch: Batch) -> dict:
        """
        Compute and apply SAC gradient updates for all networks.

        Update order (standard SAC):
          1. Critic update (Bellman regression)
          2. Actor update (maximise Q - entropy)
          3. Alpha update (auto-tune entropy temperature)
          4. Target critic soft update
        """
        import torch
        import torch.nn.functional as F

        alpha = self.log_alpha.exp().detach()

        # --- Convert batch to tensors ---
        obs_t      = torch.tensor(batch.obs,      dtype=torch.float32)
        act_t      = torch.tensor(batch.actions,  dtype=torch.float32)
        rew_t      = torch.tensor(batch.rewards,  dtype=torch.float32)
        next_obs_t = torch.tensor(batch.next_obs, dtype=torch.float32)
        done_t     = torch.tensor(batch.dones,    dtype=torch.float32)

        # --- Step 1: Critic update ---
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self._batch_actor_forward(next_obs_t)

            # Clipped double-Q target (takes min of two critics)
            next_node = self._batch_split_obs(next_obs_t)
            q1_next = self._batch_critic_forward(
                self.target_critic1, next_node, next_actions
            )
            q2_next = self._batch_critic_forward(
                self.target_critic2, next_node, next_actions
            )
            q_next = torch.min(q1_next, q2_next)

            # Bellman target: r + γ(1-d)[min_Q' - α log π]
            y = rew_t + self.gamma * (1 - done_t) * (
                q_next - alpha * next_log_probs.unsqueeze(1)
            )

        # Current node features
        curr_node = self._batch_split_obs(obs_t)

        # Critic 1 loss
        q1 = self._batch_critic_forward(self.critic1, curr_node, act_t)
        critic1_loss = F.mse_loss(q1, y)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()

        # Critic 2 loss
        q2 = self._batch_critic_forward(self.critic2, curr_node, act_t)
        critic2_loss = F.mse_loss(q2, y)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()

        # --- Step 2: Actor update ---
        new_actions, log_probs = self._batch_actor_forward(obs_t)
        q1_pi = self._batch_critic_forward(
            self.critic1, curr_node, new_actions
        )
        q2_pi = self._batch_critic_forward(
            self.critic2, curr_node, new_actions
        )
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss: minimise (α log π - Q)
        actor_loss = (alpha * log_probs - q_pi.squeeze()).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # --- Step 3: Alpha (entropy temperature) update ---
        # Auto-tune: adjust α so that H(π) ≈ target_entropy
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Step 4: Soft update target critics ---
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        return {
            "critic_loss": float((critic1_loss + critic2_loss) / 2),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.log_alpha.exp()),
        }

    # ------------------------------------------------------------------
    # Batched forward passes (process full minibatch efficiently)
    # ------------------------------------------------------------------

    def _batch_actor_forward(self, obs_batch):
        """
        Run actor forward pass on a batch of observations.

        Processes each observation individually (the graph is the same
        for all, but node features differ per observation). Returns
        batched actions and log probs.

        Note: A more efficient implementation would batch the GAT forward
        pass using PyG's batch object. This sequential version is used
        for clarity and correctness. Batch-level GAT is a performance
        optimisation for Phase 5 if training is slow.
        """
        import torch
        B = obs_batch.shape[0]
        actions_list = []
        log_probs_list = []

        for i in range(B):
            node_t = self._batch_split_obs_single(obs_batch[i])
            action, log_prob = self.actor(
                node_t, self._edge_index_t, deterministic=False
            )
            actions_list.append(action)
            log_probs_list.append(log_prob)

        return torch.stack(actions_list), torch.stack(log_probs_list)

    def _batch_critic_forward(self, critic, node_batch, action_batch):
        """Run critic on a batch, return (B, 1) Q-values."""
        import torch
        B = node_batch.shape[0]
        q_list = []
        for i in range(B):
            q = critic(
                node_batch[i],
                self._edge_index_t,
                action_batch[i],
            )
            q_list.append(q)
        return torch.stack(q_list)

    def _batch_split_obs(self, obs_batch):
        """Reshape (B, obs_dim) into (B, H, node_feature_dim=9).
        No zone feature block — RRP is in node feature [6]."""
        B = obs_batch.shape[0]
        return obs_batch.reshape(B, self.n_hubs, self.net_cfg.node_feature_dim)

    def _batch_split_obs_single(self, obs):
        """Reshape single (obs_dim,) obs to node tensor (H, node_feature_dim=9).
        No zone feature block — RRP is in node feature [6]."""
        return obs.reshape(self.n_hubs, self.net_cfg.node_feature_dim)

    # ------------------------------------------------------------------
    # Observation splitting (numpy, for select_action)
    # ------------------------------------------------------------------

    def _split_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reshape flat obs (H×9,) to node feature matrix (H, 9).
        No zone feature block — RRP is broadcast into node feature [6]."""
        return obs.reshape(self.n_hubs, self.net_cfg.node_feature_dim)

    # ------------------------------------------------------------------
    # Action space bounds
    # ------------------------------------------------------------------

    def _action_low(self) -> np.ndarray:
        # Dispatch is signed kW: [-equipment_cap, +equipment_cap]
        # Price is $/kWh: [price_min, price_max]
        # The env clips more tightly (DOE + equipment cap), so these are
        # loose outer bounds for safety clipping only.
        low = np.full(self.action_dim, -self.net_cfg.equipment_cap_kw, dtype=np.float32)
        low[-1] = self.net_cfg.price_min
        return low

    def _action_high(self) -> np.ndarray:
        high = np.full(self.action_dim, +self.net_cfg.equipment_cap_kw, dtype=np.float32)
        high[-1] = self.net_cfg.price_max
        return high

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def _soft_update(self, target, online):
        """Polyak averaging: θ_target ← τ θ_online + (1-τ) θ_target"""
        import torch
        with torch.no_grad():
            for t_param, o_param in zip(
                target.parameters(), online.parameters()
            ):
                t_param.data.copy_(
                    self.tau * o_param.data + (1 - self.tau) * t_param.data
                )

    def _hard_update(self, target, online):
        """Copy weights exactly: θ_target ← θ_online"""
        import torch
        with torch.no_grad():
            for t_param, o_param in zip(
                target.parameters(), online.parameters()
            ):
                t_param.data.copy_(o_param.data)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all network weights and training state to disk."""
        if not self._use_torch:
            logger.warning("Cannot save numpy fallback agent (no weights to save)")
            return
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha,
            "total_steps": self._total_steps,
            "total_updates": self._total_updates,
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load network weights from a checkpoint."""
        if not self._use_torch:
            logger.warning("Cannot load into numpy fallback agent")
            return
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.log_alpha = ckpt["log_alpha"]
        self._total_steps = ckpt["total_steps"]
        self._total_updates = ckpt["total_updates"]
        logger.info(f"Agent loaded from {path} (step {self._total_steps})")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "total_updates": self._total_updates,
            "buffer_size": self.buffer.size,
            "using_torch": self._use_torch,
            "alpha": float(np.exp(self.log_alpha))
            if not self._use_torch
            else float(self.log_alpha.exp()),
            "n_hubs": self.n_hubs,
            "action_dim": self.action_dim,
        }