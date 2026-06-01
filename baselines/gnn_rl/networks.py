"""
networks.py
===========
Neural network components for the SAC-GNN agent.

Architecture overview
---------------------
The policy network has three stages:

  Stage 1 — Node embedding MLP
    Each hub's raw node features (static spatial + dynamic operational)
    are projected to a uniform embedding dimension via a shared MLP.
    This is the same pattern as EV-GNN's node-type-specific MLPs, but
    simplified to one type (hubs only, no EV/charger/transformer hierarchy).

  Stage 2 — GAT encoder
    A 2-layer Graph Attention Network processes the hub embeddings,
    propagating information between connected hubs weighted by learned
    attention coefficients. This is the architectural divergence from
    EV-GNN, which uses GCN (uniform aggregation). GAT is chosen because
    the relevance of one hub's state for another varies with VSR zone
    load, spot price, and time of day — exactly the regime where learned
    attention outperforms fixed structural aggregation [42].

  Stage 3a — Actor head (two outputs, jointly optimised)
    Per-hub dispatch fractions δ_i: one sigmoid output per hub node,
    produced from each hub's GAT embedding independently.
    Incentive price scalar c_t: produced from the mean-pooled graph
    embedding (zone-level summary), bounded to [c_min, c_max].

  Stage 3b — Twin critic heads (SAC uses two Q-functions)
    Each critic takes (graph_embedding, full_action) and outputs a
    scalar Q-value. Zone-level features and the full action vector
    (H dispatch fractions + price scalar) are concatenated with the
    mean-pooled graph embedding before the MLP head.

Why GAT over GCN
----------------
GCN aggregates neighbour features with fixed weights determined by
graph structure alone (normalised adjacency). GAT learns per-edge
attention weights α_{ij} as a function of both hub i and hub j's
current features. In the public hub dispatch setting, the catchment
area overlap between two hubs depends on the current incentive price,
the time of day, and the zone load — none of which are captured by
the static graph topology. GAT's learned attention mechanism adapts
to these operational conditions; GCN cannot.

Why mean-pooling for the price head
------------------------------------
The incentive price c_t is a single scalar broadcast to all enrolled
EV owners across all hubs simultaneously. It is a zone-level decision,
not a per-hub decision. Mean-pooling the GAT embeddings produces a
zone-level summary vector that appropriately aggregates information
from all hubs before the price MLP head makes this zone-wide decision.

PyG / numpy compatibility
--------------------------
When torch_geometric is available, GATConv layers are used directly.
When it is not (e.g., during CI or on incompatible hardware), a
numpy-backed MessagePassing fallback is used. The fallback implements
the same GAT attention formula but without autograd — suitable for
validating shapes and forward pass logic without training.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy PyG import — same pattern as spatial_graph.py
# ---------------------------------------------------------------------------
_HAS_TORCH = False

def _try_import_torch():
    global _HAS_TORCH
    if _HAS_TORCH:
        return True
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return False
        import torch  # noqa
        import torch.nn as nn  # noqa
        from torch_geometric.nn import GATConv  # noqa
        _HAS_TORCH = True
    except Exception:
        _HAS_TORCH = False
    return _HAS_TORCH


# ---------------------------------------------------------------------------
# Network hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """
    Hyperparameters for all network components.
    All values are thesis sensitivity parameters — sweep in §4.3.4.

    Attributes
    ----------
    node_feature_dim : int
        Input features per hub node. Must match NEMWDREnv.NODE_FEATURE_DIM = 5.
        Layout: [n_connected_norm, mean_soc, loc_x, loc_y, distance_norm]
    zone_feature_dim : int
        Zone-level features appended after GAT. Must match
        NEMWDREnv.ZONE_FEATURE_DIM = 9.
    embed_dim : int
        Hidden embedding dimension for node MLP and GAT layers.
    gat_heads : int
        Number of attention heads in each GAT layer. Output dim per layer
        = embed_dim × gat_heads (concatenated), then projected back to
        embed_dim for the second layer.
    gat_layers : int
        Number of GAT message-passing layers. 2 is standard; more risks
        over-smoothing on small graphs (H=10–30).
    actor_hidden : int
        Hidden units in actor MLP heads.
    critic_hidden : int
        Hidden units in critic MLP heads.
    price_min, price_max : float
        Incentive price action bounds ($/MWh). Must match EnvConfig.
    dropout : float
        Dropout rate applied after each GAT layer during training.
    """
    node_feature_dim: int = 5
    zone_feature_dim: int = 9
    embed_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    actor_hidden: int = 128
    critic_hidden: int = 256
    price_min: float = 0.0
    price_max: float = 500.0
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Numpy fallback: manual GAT forward pass (no autograd, for validation only)
# ---------------------------------------------------------------------------

class NumpyGATLayer:
    """
    Single GAT layer implemented in numpy.

    Computes attention-weighted neighbourhood aggregation:
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        α_ij = softmax_j(e_ij)
        h'_i = σ(Σ_j α_ij W h_j)

    Used only when torch_geometric is not available.
    Parameters are random (not trained) — this is for shape validation only.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_heads = n_heads
        self.out_dim = out_dim
        # Weight matrix W: (in_dim, out_dim * n_heads)
        self.W = rng.normal(0, 0.01, (in_dim, out_dim * n_heads)).astype(np.float32)
        # Attention vector a: (2 * out_dim,) per head
        self.a = rng.normal(0, 0.01, (n_heads, 2 * out_dim)).astype(np.float32)

    def forward(
        self,
        x: np.ndarray,          # (H, in_dim)
        edge_index: np.ndarray, # (2, E)
    ) -> np.ndarray:             # (H, out_dim * n_heads)
        H = x.shape[0]
        Wh = x @ self.W          # (H, out_dim * n_heads)

        # Reshape for multi-head: (H, n_heads, out_dim)
        Wh_heads = Wh.reshape(H, self.n_heads, self.out_dim)

        # Output accumulator
        out = np.zeros((H, self.n_heads, self.out_dim), dtype=np.float32)
        attn_sum = np.zeros((H, self.n_heads), dtype=np.float32)

        src, dst = edge_index[0], edge_index[1]

        # Compute unnormalised attention scores e_ij per head
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        for head in range(self.n_heads):
            a_h = self.a[head]  # (2 * out_dim,)
            Wh_h = Wh_heads[:, head, :]  # (H, out_dim)

            # Concatenate source and destination embeddings for each edge
            src_feat = Wh_h[src]  # (E, out_dim)
            dst_feat = Wh_h[dst]  # (E, out_dim)
            concat = np.concatenate([src_feat, dst_feat], axis=1)  # (E, 2*out_dim)

            # Attention energy: LeakyReLU
            e = concat @ a_h  # (E,)
            e = np.where(e >= 0, e, 0.01 * e)  # LeakyReLU α=0.01

            # Softmax per destination node
            # Group edges by destination node
            alpha = np.zeros_like(e)
            for i in range(H):
                mask = dst == i
                if mask.any():
                    e_i = e[mask]
                    e_i_exp = np.exp(e_i - e_i.max())
                    alpha[mask] = e_i_exp / e_i_exp.sum()

            # Aggregate: h'_i = Σ_j α_ij Wh_j
            for k in range(len(src)):
                d = dst[k]
                out[d, head] += alpha[k] * Wh_h[src[k]]
            attn_sum[:, head] += 1  # placeholder

        # Concatenate heads: (H, n_heads * out_dim)
        result = out.reshape(H, self.n_heads * self.out_dim)
        # ELU activation
        result = np.where(result >= 0, result, np.exp(result) - 1)
        return result.astype(np.float32)


class NumpyGATEncoder:
    """
    2-layer numpy GAT encoder for shape validation without PyG.
    Not used for training — only for testing network dimensions.
    """

    def __init__(self, cfg: NetworkConfig):
        self.cfg = cfg
        head_dim = cfg.embed_dim // cfg.gat_heads
        # Layer 1 output: head_dim × gat_heads = embed_dim
        self.layer1 = NumpyGATLayer(
            cfg.node_feature_dim,
            head_dim,
            cfg.gat_heads,
            seed=0,
        )
        # Layer 2 input = embed_dim (layer1 concatenated output)
        self.layer2 = NumpyGATLayer(
            cfg.embed_dim,          # = head_dim * gat_heads
            cfg.embed_dim,          # output per head
            1,
            seed=1,
        )

    def forward(
        self,
        x: np.ndarray,           # (H, node_feature_dim)
        edge_index: np.ndarray,  # (2, E)
    ) -> np.ndarray:              # (H, embed_dim)
        h = self.layer1.forward(x, edge_index)   # (H, embed_dim)
        h = self.layer2.forward(h, edge_index)   # (H, embed_dim)
        return h


class NumpyActor:
    """
    Numpy actor for shape validation.
    Produces deterministic actions (no reparameterisation trick).
    For training, use the PyTorch version.
    """

    def __init__(self, cfg: NetworkConfig, n_hubs: int):
        self.cfg = cfg
        self.n_hubs = n_hubs
        self.encoder = NumpyGATEncoder(cfg)
        rng = np.random.default_rng(3)

        # Per-hub dispatch head: embed_dim → 1 (sigmoid → [0,1])
        self.dispatch_W = rng.normal(
            0, 0.01, (cfg.embed_dim, 1)
        ).astype(np.float32)

        # Price head: (embed_dim + zone_feature_dim) → 1
        self.price_W = rng.normal(
            0, 0.01, (cfg.embed_dim + cfg.zone_feature_dim, 1)
        ).astype(np.float32)

    def forward(
        self,
        node_features: np.ndarray,   # (H, node_feature_dim)
        edge_index: np.ndarray,      # (2, E)
        zone_features: np.ndarray,   # (zone_feature_dim,)
    ) -> np.ndarray:                  # (H + 1,): [δ_0,...,δ_{H-1}, c_t]
        # Stage 1+2: GAT encoding
        h = self.encoder.forward(node_features, edge_index)  # (H, embed_dim)

        # Stage 3a-i: per-hub dispatch fractions
        dispatch_logits = h @ self.dispatch_W   # (H, 1)
        dispatch = _sigmoid(dispatch_logits).flatten()  # (H,) ∈ (0,1)

        # Stage 3a-ii: zone-level price
        h_mean = h.mean(axis=0)  # (embed_dim,) — mean pool across hubs
        price_input = np.concatenate([h_mean, zone_features])  # (embed_dim + zone_dim,)
        price_raw = float((price_input @ self.price_W).item())
        # Scale tanh output to [price_min, price_max]
        price_mid = (self.cfg.price_max + self.cfg.price_min) / 2
        price_range = (self.cfg.price_max - self.cfg.price_min) / 2
        price = price_mid + price_range * np.tanh(price_raw)

        return np.concatenate([dispatch, [price]]).astype(np.float32)


class NumpyCritic:
    """
    Numpy critic (Q-function) for shape validation.
    Takes full observation + action, returns scalar Q-value.
    """

    def __init__(self, cfg: NetworkConfig, n_hubs: int):
        self.cfg = cfg
        self.n_hubs = n_hubs
        self.encoder = NumpyGATEncoder(cfg)
        rng = np.random.default_rng(4)

        # Input: mean-pooled graph embedding + zone features + full action
        # action dim = H dispatch fractions + 1 price scalar
        action_dim = n_hubs + 1
        critic_input_dim = cfg.embed_dim + cfg.zone_feature_dim + action_dim

        self.W1 = rng.normal(
            0, 0.01, (critic_input_dim, cfg.critic_hidden)
        ).astype(np.float32)
        self.W2 = rng.normal(
            0, 0.01, (cfg.critic_hidden, 1)
        ).astype(np.float32)

    def forward(
        self,
        node_features: np.ndarray,   # (H, node_feature_dim)
        edge_index: np.ndarray,      # (2, E)
        zone_features: np.ndarray,   # (zone_feature_dim,)
        action: np.ndarray,          # (H + 1,)
    ) -> float:
        h = self.encoder.forward(node_features, edge_index)   # (H, embed_dim)
        h_mean = h.mean(axis=0)                               # (embed_dim,)

        # Concatenate: graph summary + zone state + action
        critic_input = np.concatenate([h_mean, zone_features, action])
        # Two-layer MLP
        hidden = np.maximum(0, critic_input @ self.W1)   # ReLU
        q_value = float((hidden @ self.W2).item())
        return q_value


# ---------------------------------------------------------------------------
# PyTorch networks (used when torch_geometric is available)
# ---------------------------------------------------------------------------

def build_torch_networks(cfg: NetworkConfig, n_hubs: int):
    """
    Build the full PyTorch actor and twin critics using GATConv.

    Returns
    -------
    actor : GATActor
    critic1, critic2 : GATCritic  (twin critics for SAC's clipped double-Q)

    Called by SACGNNAgent.__init__() when PyG is available.
    Only imported/executed at runtime to avoid bus errors at module load.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv

    class GATEncoder(nn.Module):
        """
        2-layer GAT encoder.

        Layer 1: node_feature_dim → embed_dim, gat_heads heads (concatenated)
        Layer 2: embed_dim → embed_dim, 1 head (averaged for stable embedding)

        The two-layer design allows each hub to aggregate information from
        its direct neighbours (layer 1) and from its neighbours' neighbours
        (layer 2), capturing second-order spatial dependencies in the hub
        network. For H=10–30 hubs, 2 layers is sufficient and avoids the
        over-smoothing that occurs with deeper GNNs on small graphs.
        """

        def __init__(self, cfg: NetworkConfig):
            super().__init__()
            self.cfg = cfg
            head_dim = cfg.embed_dim // cfg.gat_heads

            # Layer 1: multi-head attention (heads concatenated)
            self.gat1 = GATConv(
                in_channels=cfg.node_feature_dim,
                out_channels=head_dim,
                heads=cfg.gat_heads,
                concat=True,        # output: embed_dim = head_dim × gat_heads
                dropout=cfg.dropout,
                add_self_loops=True,
            )
            # Layer 2: single-head (averaged), maps back to embed_dim
            self.gat2 = GATConv(
                in_channels=cfg.embed_dim,  # = head_dim × gat_heads
                out_channels=cfg.embed_dim,
                heads=1,
                concat=False,       # output: embed_dim (averaged)
                dropout=cfg.dropout,
                add_self_loops=True,
            )
            self.norm1 = nn.LayerNorm(cfg.embed_dim)
            self.norm2 = nn.LayerNorm(cfg.embed_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, edge_index, edge_attr=None):
            # Layer 1
            h = self.gat1(x, edge_index)           # (H, embed_dim)
            h = F.elu(h)
            h = self.norm1(h)
            h = self.dropout(h)

            # Residual connection from layer 1 to layer 2
            h_res = h
            h = self.gat2(h, edge_index)           # (H, embed_dim)
            h = F.elu(h + h_res)                   # residual
            h = self.norm2(h)

            return h                                # (H, embed_dim)

    class GATActor(nn.Module):
        """
        SAC actor: outputs mean and log_std for a squashed Gaussian policy.

        Two output heads:
        1. Dispatch head: per-hub, from each hub's GAT embedding.
           Output: δ_i ∈ (0, 1) via sigmoid (deterministic) or
                   sampled from N(μ_i, σ_i) → sigmoid (stochastic).
        2. Price head: zone-level, from mean-pooled GAT embedding + zone features.
           Output: c_t ∈ [price_min, price_max] via tanh rescaling.

        Both heads output (mean, log_std) for the reparameterisation trick.
        During evaluation, the deterministic mean action is used.
        """

        def __init__(self, cfg: NetworkConfig, n_hubs: int):
            super().__init__()
            self.cfg = cfg
            self.n_hubs = n_hubs
            self.encoder = GATEncoder(cfg)

            # Dispatch head: embed_dim → hidden → 2 (mean, log_std) per hub
            self.dispatch_head = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.actor_hidden),
                nn.ReLU(),
                nn.Linear(cfg.actor_hidden, 2),  # mean + log_std
            )

            # Price head: (embed_dim + zone_feature_dim) → hidden → 2
            self.price_head = nn.Sequential(
                nn.Linear(cfg.embed_dim + cfg.zone_feature_dim, cfg.actor_hidden),
                nn.ReLU(),
                nn.Linear(cfg.actor_hidden, 2),  # mean + log_std
            )

            # Log std clamping bounds (SAC standard)
            self.log_std_min = -5.0
            self.log_std_max = 2.0

        def forward(self, node_features, edge_index, zone_features,
                    deterministic=False):
            """
            Parameters
            ----------
            node_features : Tensor (H, node_feature_dim)
            edge_index : Tensor (2, E)
            zone_features : Tensor (zone_feature_dim,) or (B, zone_feature_dim)
            deterministic : bool
                If True, return mean action (evaluation mode).
                If False, sample via reparameterisation (training mode).

            Returns
            -------
            action : Tensor (H + 1,)
                [δ_0, ..., δ_{H-1}, c_t]
            log_prob : Tensor (scalar)
                Log probability of the sampled action (for SAC entropy term).
                Returns 0 if deterministic=True.
            """
            import torch
            import torch.nn.functional as F

            # GAT encoding
            h = self.encoder(node_features, edge_index)  # (H, embed_dim)

            # --- Dispatch head ---
            dispatch_out = self.dispatch_head(h)          # (H, 2)
            dispatch_mean = dispatch_out[:, 0]            # (H,)
            dispatch_log_std = dispatch_out[:, 1].clamp(
                self.log_std_min, self.log_std_max
            )

            # --- Price head ---
            h_mean = h.mean(dim=0)                        # (embed_dim,)
            price_input = torch.cat([h_mean, zone_features], dim=-1)
            price_out = self.price_head(price_input)      # (2,)
            price_mean = price_out[0]
            price_log_std = price_out[1].clamp(self.log_std_min, self.log_std_max)

            if deterministic:
                # Deterministic action for evaluation
                dispatch = torch.sigmoid(dispatch_mean)
                price_mid = (self.cfg.price_max + self.cfg.price_min) / 2
                price_range = (self.cfg.price_max - self.cfg.price_min) / 2
                price = price_mid + price_range * torch.tanh(price_mean)
                action = torch.cat([dispatch, price.unsqueeze(0)])
                return action, torch.tensor(0.0)

            # Reparameterisation trick: sample from N(mean, std)
            dispatch_std = dispatch_log_std.exp()
            dispatch_eps = torch.randn_like(dispatch_mean)
            dispatch_pre_squash = dispatch_mean + dispatch_std * dispatch_eps

            price_std = price_log_std.exp()
            price_eps = torch.randn_like(price_mean)
            price_pre_squash = price_mean + price_std * price_eps

            # Squash dispatch to (0,1) via sigmoid
            dispatch = torch.sigmoid(dispatch_pre_squash)

            # Squash price to [price_min, price_max] via tanh rescaling
            price_mid = (self.cfg.price_max + self.cfg.price_min) / 2
            price_range = (self.cfg.price_max - self.cfg.price_min) / 2
            price = price_mid + price_range * torch.tanh(price_pre_squash)

            action = torch.cat([dispatch, price.unsqueeze(0)])

            # Log probability (with change-of-variables correction for squashing)
            # For dispatch (sigmoid squashing):
            # log π(a) = log N(pre_squash) - Σ log |dσ/dx|
            #          = log N(pre_squash) - Σ log(σ(x)(1-σ(x)))
            dispatch_log_prob = (
                -0.5 * dispatch_eps ** 2
                - dispatch_log_std
                - 0.5 * np.log(2 * np.pi)
                - torch.log(dispatch * (1 - dispatch) + 1e-6)
            ).sum()

            # For price (tanh squashing):
            # log π(a) = log N(pre_squash) - Σ log(1 - tanh²(x)) - log(price_range)
            tanh_price = torch.tanh(price_pre_squash)
            price_log_prob = (
                -0.5 * price_eps ** 2
                - price_log_std
                - 0.5 * np.log(2 * np.pi)
                - torch.log(1 - tanh_price ** 2 + 1e-6)
                - np.log(price_range)
            )

            log_prob = dispatch_log_prob + price_log_prob

            return action, log_prob

    class GATCritic(nn.Module):
        """
        SAC Q-function (one of two twins).

        Input: full observation (node features + zone features) + full action
        Output: scalar Q-value

        The critic uses the same GAT encoder as the actor (separate weights)
        to process the hub graph, then concatenates the mean-pooled embedding
        with zone features and the full action vector before a 2-layer MLP.

        The full action vector includes both dispatch fractions (H values) and
        the incentive price (1 value). This is critical: the Q-function must
        evaluate the joint quality of the dispatch allocation AND the price,
        since these interact through the participation model.
        """

        def __init__(self, cfg: NetworkConfig, n_hubs: int):
            super().__init__()
            self.cfg = cfg
            self.n_hubs = n_hubs
            self.encoder = GATEncoder(cfg)

            action_dim = n_hubs + 1  # H dispatch fractions + 1 price
            critic_input_dim = (
                cfg.embed_dim        # mean-pooled graph embedding
                + cfg.zone_feature_dim  # zone-level state
                + action_dim         # full action
            )

            self.mlp = nn.Sequential(
                nn.Linear(critic_input_dim, cfg.critic_hidden),
                nn.ReLU(),
                nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
                nn.ReLU(),
                nn.Linear(cfg.critic_hidden, 1),
            )

        def forward(self, node_features, edge_index, zone_features, action):
            """
            Parameters
            ----------
            node_features : Tensor (H, node_feature_dim)
            edge_index : Tensor (2, E)
            zone_features : Tensor (zone_feature_dim,)
            action : Tensor (H + 1,)

            Returns
            -------
            q_value : Tensor (1,)
            """
            import torch
            h = self.encoder(node_features, edge_index)  # (H, embed_dim)
            h_mean = h.mean(dim=0)                       # (embed_dim,)
            critic_input = torch.cat([h_mean, zone_features, action])
            return self.mlp(critic_input)                # (1,)

    actor = GATActor(cfg, n_hubs)
    critic1 = GATCritic(cfg, n_hubs)
    critic2 = GATCritic(cfg, n_hubs)

    return actor, critic1, critic2


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
