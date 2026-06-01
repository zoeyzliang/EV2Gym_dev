"""
baselines/gnn_rl/sac_gcn.py
============================
SAC with GCN encoder ablation baseline (Table 3, Baseline 5).

Identical to SAC-GNN except GATConv is replaced with GCNConv —
uniform neighbourhood aggregation with fixed structural weights
instead of learned attention weights.

This isolates the contribution of the attention mechanism (RQ4).
The comparison SAC-GNN vs SAC-GCN answers: does learning context-
dependent edge weights outperform fixed structural aggregation for
the hub dispatch problem?

Architecture difference from SAC-GNN:
    SAC-GNN : GATConv (learned attention α_ij per edge, per head)
    SAC-GCN : GCNConv (fixed D^{-1/2} A D^{-1/2} aggregation)

Everything else — actor head, critic head, SAC loop, replay buffer —
is identical. This ensures any performance difference is attributable
solely to the attention mechanism.
"""

import os
import numpy as np
import logging
from typing import Optional

from .networks import NetworkConfig, NumpyActor, NumpyCritic, _try_import_torch
from .replay_buffer import ReplayBuffer
from .agent import SACGNNAgent

logger = logging.getLogger(__name__)


def build_gcn_networks(cfg: NetworkConfig, n_hubs: int):
    """
    Build actor and twin critics using GCNConv instead of GATConv.
    Called by SACGCNAgent when PyG is available.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCNEncoder(nn.Module):
        """
        2-layer GCN encoder.
        Identical structure to GATEncoder but uses GCNConv.
        GCN aggregates with fixed D^{-1/2}AD^{-1/2} weights —
        no attention, no context-dependence.
        """

        def __init__(self, cfg: NetworkConfig):
            super().__init__()
            self.cfg = cfg

            self.gcn1 = GCNConv(
                in_channels=cfg.node_feature_dim,
                out_channels=cfg.embed_dim,
            )
            self.gcn2 = GCNConv(
                in_channels=cfg.embed_dim,
                out_channels=cfg.embed_dim,
            )
            self.norm1 = nn.LayerNorm(cfg.embed_dim)
            self.norm2 = nn.LayerNorm(cfg.embed_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, edge_index, edge_attr=None):
            h = F.elu(self.gcn1(x, edge_index))
            h = self.norm1(h)
            h = self.dropout(h)
            h_res = h
            h = F.elu(self.gcn2(h, edge_index) + h_res)
            h = self.norm2(h)
            return h

    # Reuse GATActor/GATCritic heads but swap encoder
    # Build full networks via GAT factory then replace encoder
    from .networks import build_torch_networks
    actor, critic1, critic2 = build_torch_networks(cfg, n_hubs)

    # Replace GAT encoders with GCN encoders
    gcn_enc_1 = GCNEncoder(cfg)
    gcn_enc_2 = GCNEncoder(cfg)
    gcn_enc_3 = GCNEncoder(cfg)

    actor.encoder = gcn_enc_1
    critic1.encoder = gcn_enc_2
    critic2.encoder = gcn_enc_3

    return actor, critic1, critic2


class SACGCNAgent(SACGNNAgent):
    """
    SAC agent with GCN encoder (ablation of SAC-GNN).

    Inherits entire training loop from SACGNNAgent.
    Only overrides network initialisation to use GCNConv.
    """

    def __init__(self, *args, **kwargs):
        # Call parent init but intercept network building
        super().__init__(*args, **kwargs)
        self.name = "SAC-GCN"

    def _init_torch_networks(self, lr_actor, lr_critic, lr_alpha):
        """Override: use GCN encoder instead of GAT."""
        import torch
        import torch.optim as optim

        actor, critic1, critic2 = build_gcn_networks(self.net_cfg, self.n_hubs)
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

        _, target_critic1, target_critic2 = build_gcn_networks(
            self.net_cfg, self.n_hubs
        )
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2

        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)

        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(
            np.log(0.1), dtype=torch.float32, requires_grad=True
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )
        self.critic1_opt = torch.optim.Adam(
            self.critic1.parameters(), lr=lr_critic
        )
        self.critic2_opt = torch.optim.Adam(
            self.critic2.parameters(), lr=lr_critic
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self._edge_index_t = torch.tensor(
            self.graph_data.edge_index, dtype=torch.long
        )
        self._edge_attr_t = torch.tensor(
            self.graph_data.edge_attr, dtype=torch.float32
        )

        logger.info("SACGCNAgent: GCN encoder initialised (ablation baseline)")
