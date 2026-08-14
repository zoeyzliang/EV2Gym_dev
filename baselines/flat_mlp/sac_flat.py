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
                        hidden_dim: int = 256, price_max: float = 0.50):  # $/kWh
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
            """
            Forward pass handling both single obs (H*9,) and batched (B, H*9).
            Always returns action (H+1,) for single, (B, H+1) for batch.

            Equipment cap fix
            ------------------
            Hub equipment caps are genuinely heterogeneous (derived from
            real OpenChargeMap charger counts per station — see
            spatial_graph.py HubConfig.p_max_kw), NOT a uniform 100 kW.
            SAC-GNN and SAC-GCN both read the true per-hub cap from node
            feature [5] via the graph encoder. This flat MLP previously
            hardcoded 100.0 kW for every hub, which is a genuine
            confound for the graph-vs-no-graph ablation (RQ3): SAC-Flat
            would waste tanh output range on values later clipped away
            by the environment's per-hub cap, for reasons unrelated to
            the presence/absence of graph structure. Fixed by reading
            the same feature [5] directly from the flat obs vector,
            which is trivially available without any graph structure —
            using it does not reintroduce graph information, it only
            removes an unrelated implementation bug.
            """
            import torch, math
            # Ensure 2D: (B, obs_dim)
            single = obs.dim() == 1
            if single:
                obs = obs.unsqueeze(0)   # (1, obs_dim)
            B = obs.shape[0]

            # Extract true per-hub equipment cap (kW) from node feature [5]
            # of the flat obs. Layout: [hub0(9 feats), hub1(9 feats), ...]
            node_feats = obs.view(B, self.n_hubs, 9)
            equipment_caps = node_feats[:, :, 5]   # (B, n_hubs), kW, unnormalised

            h = self.shared(obs)   # (B, hidden_dim)

            # Dispatch: (B, n_hubs*2) -> (B, n_hubs, 2)
            dispatch_out = self.dispatch_head(h).view(B, self.n_hubs, 2)
            dispatch_mean    = dispatch_out[:, :, 0]   # (B, n_hubs)
            dispatch_log_std = dispatch_out[:, :, 1].clamp(
                self.log_std_min, self.log_std_max
            )

            # Price: (B, 2)
            price_out    = self.price_head(h)           # (B, 2)
            price_mean   = price_out[:, 0]              # (B,)
            price_log_std = price_out[:, 1].clamp(self.log_std_min, self.log_std_max)

            price_mid = self.price_max / 2

            if deterministic:
                dispatch_kw = equipment_caps * torch.tanh(dispatch_mean)   # (B, n_hubs)
                price = price_mid + price_mid * torch.tanh(price_mean)  # (B,)
                action = torch.cat([dispatch_kw, price.unsqueeze(1)], dim=1)  # (B, H+1)
                if single:
                    action = action.squeeze(0)
                return action, torch.tensor(0.0)

            # Reparameterisation
            dispatch_eps = torch.randn_like(dispatch_mean)
            dispatch_pre = dispatch_mean + dispatch_log_std.exp() * dispatch_eps
            tanh_d       = torch.tanh(dispatch_pre)
            dispatch_kw  = equipment_caps * tanh_d   # (B, n_hubs)

            price_eps  = torch.randn_like(price_mean)
            price_pre  = price_mean + price_log_std.exp() * price_eps
            tanh_p     = torch.tanh(price_pre)
            price      = price_mid + price_mid * tanh_p   # (B,)

            action = torch.cat([dispatch_kw, price.unsqueeze(1)], dim=1)  # (B, H+1)

            # Log prob with tanh correction. The Jacobian term for a
            # per-hub scale factor is log(equipment_caps[hub]) rather
            # than a single constant log(100.0) — each hub now has its
            # own scale.
            dispatch_log_prob = (
                -0.5 * dispatch_eps ** 2 - dispatch_log_std
                - 0.5 * math.log(2 * math.pi)
                - torch.log(1 - tanh_d ** 2 + 1e-6)
                - torch.log(equipment_caps.clamp(min=1e-3))
            ).sum(dim=-1)   # (B,)

            price_log_prob = (
                -0.5 * price_eps ** 2 - price_log_std
                - 0.5 * math.log(2 * math.pi)
                - torch.log(1 - tanh_p ** 2 + 1e-6)
                - math.log(price_mid)
            )   # (B,)

            log_prob = dispatch_log_prob + price_log_prob   # (B,)

            if single:
                action   = action.squeeze(0)
                log_prob = log_prob.squeeze(0)

            return action, log_prob

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
        price_max: float = 0.50,    # $/kWh
        seed: Optional[int] = None,
        device: Optional[str] = None,
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

        from baselines.gnn_rl.replay_buffer import ReplayBuffer
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

        # Device selection: explicit device arg > CUDA if available > CPU.
        # Same fix as SACGNNAgent — without this every tensor defaults to
        # CPU even with a GPU allocated, wasting the SLURM GPU allocation.
        if self._use_torch:
            import torch
            if device is not None:
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = None

        if self._use_torch:
            self._init_torch(lr_actor, lr_critic, lr_alpha, hidden_dim, price_max)
            logger.info(f"SACFlatAgent: using PyTorch on device={self.device}")
        else:
            logger.info("SACFlatAgent: numpy fallback (no training)")

    def _init_torch(self, lr_actor, lr_critic, lr_alpha, hidden_dim, price_max):
        import torch
        import torch.optim as optim

        actor, critic1, critic2 = build_flat_networks(
            self.obs_dim, self.action_dim, hidden_dim, price_max
        )
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)

        _, target_critic1, target_critic2 = build_flat_networks(
            self.obs_dim, self.action_dim, hidden_dim, price_max
        )
        self.target_critic1 = target_critic1.to(self.device)
        self.target_critic2 = target_critic2.to(self.device)

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

        self.log_alpha = torch.tensor(
            np.log(1.0), dtype=torch.float32,
            requires_grad=True, device=self.device
        )
        self.target_entropy = -float(self.action_dim) * 0.5  # = -11

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
            # Pass 1D obs directly — actor.forward() handles unsqueeze internally
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            )   # (obs_dim,)
            action, _ = self.actor(obs_t, deterministic=deterministic)
        action = action.cpu().numpy()   # (action_dim,) — single obs returns 1D
        action[-1] = np.clip(action[-1], 0.0, self.price_max)
        return action.astype(np.float32)

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
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
        # Alpha floor raised from 0.01 to 0.05 — see agent.py store_transition
        # docstring for rationale (more recovery capacity after bad updates)
        # Alpha ceiling added (max=2.0) after observing this exact agent's
        # alpha explode to 13.09 by episode 100, with DOE violations never
        # reaching zero even at episode 1500 (238,126 kW) — see agent.py
        # store_transition docstring / _gradient_update comment for the
        # full root-cause explanation (no graph structure -> no spatial
        # credit assignment -> some action dims collapse -> extremely
        # negative log_probs -> alpha auto-tuner runs away trying to
        # compensate, with no natural ceiling to stop it).
        alpha = self.log_alpha.exp().detach().clamp(min=0.05, max=2.0)

        obs_t  = torch.tensor(batch.obs,      dtype=torch.float32, device=self.device)
        act_t  = torch.tensor(batch.actions,  dtype=torch.float32, device=self.device)
        rew_t  = torch.tensor(batch.rewards,  dtype=torch.float32, device=self.device)
        next_t = torch.tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(batch.dones,    dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Batched forward pass — actor now handles (B, obs_dim) directly
            next_actions, next_lps = self.actor(next_t, deterministic=False)
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

        # Batched actor forward
        new_actions, log_probs = self.actor(obs_t, deterministic=False)

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
            "alpha": float(self.log_alpha.detach().exp()),
        }

    def save(self, path: str):
        """
        Save all network weights and training state to disk.
        Includes target critics and total_updates for full parity with
        SACGNNAgent.save() — needed so a resumed SAC-Flat run has correctly
        initialised target networks rather than re-copying from the online
        critics (which would discard the soft-update history).
        """
        if not self._use_torch:
            logger.warning("Cannot save numpy fallback agent (no weights to save)")
            return
        import torch
        from pathlib import Path
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
        """
        Load network weights from a checkpoint.

        This method was previously missing entirely — evaluate.py calls
        agent.load(checkpoint) on every agent type including SAC-Flat, so
        without this, evaluating a trained SAC-Flat checkpoint would crash
        with AttributeError. map_location=self.device makes checkpoints
        portable across devices (e.g. train on GPU, evaluate on CPU laptop).
        """
        if not self._use_torch:
            logger.warning("Cannot load into numpy fallback agent")
            return
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        if "target_critic1" in ckpt:
            self.target_critic1.load_state_dict(ckpt["target_critic1"])
            self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self._total_steps = ckpt["total_steps"]
        self._total_updates = ckpt.get("total_updates", 0)
        logger.info(
            f"Agent loaded from {path} onto device={self.device} "
            f"(step {self._total_steps})"
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_steps": self._total_steps,
            "total_updates": self._total_updates,
            "buffer_size": self.buffer.size,
            "using_torch": self._use_torch,
            "device": str(self.device) if self.device is not None else "n/a",
        }
