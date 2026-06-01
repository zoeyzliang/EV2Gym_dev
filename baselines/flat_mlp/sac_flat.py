"""
baselines/flat_mlp/sac_flat.py
================================
SAC with flat MLP policy baseline (Table 3, Baseline 4).

Identical SAC training loop but the actor and critics use a flat
MLP that receives the full observation as a concatenated vector,
with no graph structure encoding.

This isolates the empirical contribution of the GNN spatial encoder
over a non-spatial RL baseline (RQ4). The comparison SAC-GNN vs
SAC-Flat answers: does encoding hub relational structure improve
dispatch performance beyond what a flat policy can learn from the
raw feature vector?

Why this matters for the thesis argument
-----------------------------------------
The thesis claims the GNN provides an inductive bias that captures
inter-hub dependencies (competing catchment areas) that a flat MLP
cannot represent without rediscovering spatial relationships from
the reward signal alone. SAC-Flat tests this claim directly:
if SAC-Flat achieves comparable performance, the GNN contribution
is marginal. If SAC-GNN significantly outperforms, the spatial
inductive bias is empirically justified.

The flat MLP receives the identical obs vector as SAC-GNN — the
difference is purely architectural: no graph structure, no message
passing, no attention.
"""

import os
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_HAS_TORCH = False

def _try_torch():
    global _HAS_TORCH
    if _HAS_TORCH:
        return True
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return False
        import torch  # noqa
        _HAS_TORCH = True
    except Exception:
        pass
    return _HAS_TORCH


def build_flat_networks(obs_dim: int, action_dim: int,
                        hidden_dim: int = 256, price_max: float = 500.0):
    """Build flat MLP actor and twin critics."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FlatActor(nn.Module):
        """
        Flat MLP actor. Receives concatenated obs vector directly.
        Two heads: dispatch (H outputs, sigmoid) and price (1 output, tanh-scaled).
        """

        def __init__(self):
            super().__init__()
            n_hubs = action_dim - 1
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            # Dispatch head: hidden → H × 2 (mean, log_std)
            self.dispatch_head = nn.Linear(hidden_dim, n_hubs * 2)
            # Price head: hidden → 2 (mean, log_std)
            self.price_head = nn.Linear(hidden_dim, 2)
            self.log_std_min = -5.0
            self.log_std_max = 2.0
            self.price_max = price_max
            self.n_hubs = n_hubs

        def forward(self, obs, deterministic=False):
            import torch
            h = self.shared(obs)

            dispatch_out = self.dispatch_head(h).reshape(-1, self.n_hubs, 2)
            dispatch_mean = dispatch_out[..., 0].squeeze(0)
            dispatch_log_std = dispatch_out[..., 1].squeeze(0).clamp(
                self.log_std_min, self.log_std_max
            )

            price_out = self.price_head(h).squeeze(0)
            price_mean = price_out[0]
            price_log_std = price_out[1].clamp(self.log_std_min, self.log_std_max)

            if deterministic:
                dispatch = torch.sigmoid(dispatch_mean)
                price = (self.price_max / 2) + (self.price_max / 2) * torch.tanh(price_mean)
                return torch.cat([dispatch, price.unsqueeze(0)]), torch.tensor(0.0)

            dispatch_eps = torch.randn_like(dispatch_mean)
            dispatch_pre = dispatch_mean + dispatch_log_std.exp() * dispatch_eps
            dispatch = torch.sigmoid(dispatch_pre)

            price_eps = torch.randn_like(price_mean)
            price_pre = price_mean + price_log_std.exp() * price_eps
            price_mid = self.price_max / 2
            price = price_mid + price_mid * torch.tanh(price_pre)

            action = torch.cat([dispatch, price.unsqueeze(0)])

            dispatch_log_prob = (
                -0.5 * dispatch_eps ** 2 - dispatch_log_std
                - 0.5 * np.log(2 * np.pi)
                - torch.log(dispatch * (1 - dispatch) + 1e-6)
            ).sum()

            tanh_price = torch.tanh(price_pre)
            price_log_prob = (
                -0.5 * price_eps ** 2 - price_log_std
                - 0.5 * np.log(2 * np.pi)
                - torch.log(1 - tanh_price ** 2 + 1e-6)
                - np.log(price_mid)
            )

            return action, dispatch_log_prob + price_log_prob

    class FlatCritic(nn.Module):
        """Flat MLP critic. Input: concatenated obs + action."""

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, obs, action):
            return self.mlp(torch.cat([obs, action], dim=-1))

    return FlatActor(), FlatCritic(), FlatCritic()


class SACFlatAgent:
    """
    SAC agent with flat MLP policy (no graph encoder).

    Provides the same interface as SACGNNAgent so it can be
    dropped into the same training and evaluation scripts.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension.
    action_dim : int
        Action dimension (H + 1).
    hidden_dim : int
        MLP hidden layer width.
    gamma, tau, lr_* : float
        Standard SAC hyperparameters.
    batch_size, buffer_capacity, learning_starts : int
        Replay buffer parameters.
    seed : int, optional
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        batch_size: int = 256,
        buffer_capacity: int = 500_000,
        learning_starts: int = 1000,
        update_every: int = 1,
        price_max: float = 500.0,
        seed: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_hubs = action_dim - 1
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_every = update_every
        self.price_max = price_max
        self.name = "SAC-Flat"

        from .replay_buffer import ReplayBuffer
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=buffer_capacity,
            seed=seed,
        )

        self._use_torch = (
            os.environ.get("FORCE_NUMPY_AGENT", "0") != "1"
            and _try_torch()
        )
        self._total_steps = 0
        self._total_updates = 0

        if self._use_torch:
            self._init_torch(lr_actor, lr_critic, lr_alpha, hidden_dim, price_max)
        else:
            logger.info("SACFlatAgent: numpy fallback (no training)")

    def _init_torch(self, lr_actor, lr_critic, lr_alpha, hidden_dim, price_max):
        import torch
        import torch.optim as optim

        self.actor, self.critic1, self.critic2 = build_flat_networks(
            self.obs_dim, self.action_dim, hidden_dim, price_max
        )
        _, self.target_critic1, self.target_critic2 = build_flat_networks(
            self.obs_dim, self.action_dim, hidden_dim, price_max
        )

        # Hard init targets
        for t, o in zip(self.target_critic1.parameters(),
                        self.critic1.parameters()):
            t.data.copy_(o.data)
        for t, o in zip(self.target_critic2.parameters(),
                        self.critic2.parameters()):
            t.data.copy_(o.data)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32,
                                       requires_grad=True)
        self.target_entropy = -float(self.action_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        if not self._use_torch:
            return np.random.rand(self.action_dim).astype(np.float32)

        import torch
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _ = self.actor(obs_t, deterministic=deterministic)
        action = action.numpy()
        action[-1] = np.clip(action[-1], 0.0, self.price_max)
        return action.astype(np.float32)

    def store_transition(self, obs, action, reward, next_obs,
                         done, wdr_active=False):
        self.buffer.add(obs, action, reward, next_obs, done, wdr_active)
        self._total_steps += 1

    def update(self):
        if not self._use_torch:
            return None
        if not self.buffer.can_sample(self.batch_size):
            return None
        if self._total_steps < self.learning_starts:
            return None
        if self._total_steps % self.update_every != 0:
            return None

        import torch
        import torch.nn.functional as F

        batch = self.buffer.sample(self.batch_size)
        alpha = self.log_alpha.exp().detach()

        obs_t = torch.tensor(batch.obs, dtype=torch.float32)
        act_t = torch.tensor(batch.actions, dtype=torch.float32)
        rew_t = torch.tensor(batch.rewards, dtype=torch.float32)
        next_t = torch.tensor(batch.next_obs, dtype=torch.float32)
        done_t = torch.tensor(batch.dones, dtype=torch.float32)

        with torch.no_grad():
            next_act, next_log_prob = self.actor(next_t)
            q1n = self.target_critic1(next_t, next_act.unsqueeze(0)
                                       if next_act.dim() == 1 else next_act)
            q2n = self.target_critic2(next_t, next_act.unsqueeze(0)
                                       if next_act.dim() == 1 else next_act)
            # Batch forward
            next_actions_list, next_lp_list = [], []
            for i in range(len(obs_t)):
                a, lp = self.actor(next_t[i:i+1])
                next_actions_list.append(a)
                next_lp_list.append(lp)
            next_actions = torch.stack(next_actions_list).squeeze(1)
            next_lps = torch.stack(next_lp_list)

            q1n = self.target_critic1(next_t, next_actions)
            q2n = self.target_critic2(next_t, next_actions)
            y = rew_t + self.gamma * (1 - done_t) * (
                torch.min(q1n, q2n) - alpha * next_lps.unsqueeze(1)
            )

        q1 = self.critic1(obs_t, act_t)
        q2 = self.critic2(obs_t, act_t)
        c1_loss = F.mse_loss(q1, y)
        c2_loss = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad(); c1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(); c2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()

        actions_list, lp_list = [], []
        for i in range(len(obs_t)):
            a, lp = self.actor(obs_t[i:i+1])
            actions_list.append(a)
            lp_list.append(lp)
        new_actions = torch.stack(actions_list).squeeze(1)
        log_probs = torch.stack(lp_list)

        q1p = self.critic1(obs_t, new_actions)
        q2p = self.critic2(obs_t, new_actions)
        actor_loss = (alpha * log_probs - torch.min(q1p, q2p).squeeze()).mean()

        self.actor_opt.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update targets
        for t, o in zip(self.target_critic1.parameters(),
                        self.critic1.parameters()):
            t.data.copy_(self.tau * o.data + (1 - self.tau) * t.data)
        for t, o in zip(self.target_critic2.parameters(),
                        self.critic2.parameters()):
            t.data.copy_(self.tau * o.data + (1 - self.tau) * t.data)

        self._total_updates += 1
        return {
            "critic_loss": float((c1_loss + c2_loss) / 2),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.log_alpha.exp()),
        }

    def save(self, path: str):
        if not self._use_torch:
            return
        import torch
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "log_alpha": self.log_alpha,
            "total_steps": self._total_steps,
        }, path)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_steps": self._total_steps,
            "buffer_size": self.buffer.size,
            "using_torch": self._use_torch,
        }
