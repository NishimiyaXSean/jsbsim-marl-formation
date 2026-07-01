"""Self-Attention Formation Actor for CTDE coordination.

Core innovation: instead of a flat MLP that treats all 33 observation dims equally,
we decompose the observation into three semantic token groups and apply multi-head
self-attention so the Actor can dynamically decide how much attention to allocate
to mate features vs. target features at each timestep.

Observation segmentation (33 dims → 3 tokens):
  ┌──────────┬──────┬────────────────────────────────────────────┐
  │ Token    │ Dims │ Content                                    │
  ├──────────┼──────┼────────────────────────────────────────────┤
  │ Self     │  13  │ own_vel(3), attitude(3), ang_vel(3),       │
  │          │      │ height(1), alpha(1), airspeed(1), pad(1)   │
  │ Target   │  14  │ target_rel_pos(3), target_vel(3),          │
  │          │      │ tac_geo(3), los_rate(1), bearing_err(1),   │
  │          │      │ placeholder(3)                             │
  │ Mate     │   6  │ mate_rel_pos(3), mate_rel_vel(3)           │
  └──────────┴──────┴────────────────────────────────────────────┘

Architecture:
  obs[33] → segment → [Self(13), Target(14), Mate(6)]
    → Linear projection to d_model each
    → Stack → [batch, 3, d_model]
    → MultiHeadSelfAttention(qkv over 3 tokens)
    → Flatten → [batch, 3*d_model]
    → MLP head → action_mean(2) + log_std(2)

Key properties:
  - Attention weights are interpretable: we can inspect which token groups
    the Actor attends to at each decision step.
  - The Mate token can be zeroed out for 1v1 training, then scaled up
    for 2v1 without changing the architecture.
  - From-scratch training forces the network to learn mate-attention
    patterns organically rather than relying on tiled SB3 weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Observation segmentation indices (0-based into 33-dim obs vector)
# ═══════════════════════════════════════════════════════════════════════════

# Self-state features: what the pursuer knows about itself
SELF_INDICES = [
    3, 4, 5,      # own velocity in body frame (3)
    6, 7, 8,      # own attitude rpy (3)
    9, 10, 11,    # own angular velocity (3)
    12,           # own height / MAX_HEIGHT
    22,           # alpha / MAX_AOA
    23,           # airspeed / MAX_VEL
    24,           # placeholder (was Ps)
]
SELF_DIM = len(SELF_INDICES)  # 13

# Target-related features: what the pursuer perceives about the target
TARGET_INDICES = [
    0, 1, 2,      # target relative position in body frame (3)
    13, 14, 15,   # target velocity in body frame (3)
    16, 17, 18,   # placeholder (was target ang_vel) — kept for structure
    19, 20, 21,   # tactical geometry: cos(ATA), cos(AA), cos(HCA) (3)
    25,           # LOS rate / MAX_LOS_RATE
    26,           # bearing error / 180
]
TARGET_DIM = len(TARGET_INDICES)  # 14

# Mate-related features: what the pursuer perceives about its wingman
MATE_INDICES = [
    27, 28, 29,   # mate relative position in body frame (3)
    30, 31, 32,   # mate relative velocity in body frame (3)
]
MATE_DIM = len(MATE_INDICES)  # 6

# Verify: 13 + 14 + 6 = 33 ✓
assert SELF_DIM + TARGET_DIM + MATE_DIM == 33, \
    f"Segment dims don't sum to 33: {SELF_DIM}+{TARGET_DIM}+{MATE_DIM}={SELF_DIM+TARGET_DIM+MATE_DIM}"


def segment_obs(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split flat 33-dim observation into [self, target, mate] token groups.

    Args:
        obs: [batch, 33] or [33] tensor

    Returns:
        (self_feat, target_feat, mate_feat): each [batch, group_dim]
    """
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    return (
        obs[:, SELF_INDICES],
        obs[:, TARGET_INDICES],
        obs[:, MATE_INDICES],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Network
# ═══════════════════════════════════════════════════════════════════════════

class AttentionFormationActor(nn.Module):
    """Self-Attention Actor for formation coordination.

    Decomposes the 33-dim observation into Self/Target/Mate token groups,
    applies multi-head self-attention across the 3 tokens, then produces
    action distribution parameters.

    Args:
        obs_dim: Total observation dimension (default 33)
        act_dim: Action dimension (default 2: turn_rate, speed)
        d_model: Embedding dimension for each token
        n_heads: Number of attention heads (must divide d_model)
        mlp_hidden: Hidden dimension for the final MLP head
        dropout: Attention dropout rate
        log_std_init: Initial log-standard-deviation for action distribution
        mate_scale: Multiplier for mate token embedding (0.0 = ignore mate,
                    1.0 = full mate attention). Useful for curriculum: train
                    1v1 with mate_scale=0, then 2v1 with mate_scale=1.
    """

    def __init__(
        self,
        obs_dim: int = 33,
        act_dim: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
        log_std_init: float = 0.0,
        mate_scale: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) must be divisible by n_heads({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.mate_scale = mate_scale

        # ── Token projection layers (different input dims → same d_model) ──
        self.self_proj = nn.Linear(SELF_DIM, d_model)
        self.target_proj = nn.Linear(TARGET_DIM, d_model)
        self.mate_proj = nn.Linear(MATE_DIM, d_model)

        # ── Learnable token-type embeddings (so attention can distinguish token roles) ──
        self.token_type_embed = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.normal_(self.token_type_embed, std=0.02)

        # ── Multi-head self-attention over 3 tokens ────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Attention pooling: learned weighted sum over 3 tokens ──────────
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Final MLP head (3*d_model → mlp_hidden → mlp_hidden → act_dim) ─
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),  # pooled attention → hidden
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(mlp_hidden, act_dim)
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * log_std_init)

        # ── Weight initialization ──────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for MLP layers, Xavier for attention."""
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "mlp_head" in name or "proj" in name:
                    nn.init.orthogonal_(mod.weight, gain=np.sqrt(2))
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
        # Small init for output layer (encourages near-zero actions initially)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            obs: [batch, 33] or [33] observation tensor (or numpy array)
            return_attention: If True, also return attention weights for analysis

        Returns:
            If return_attention=False: (loc[batch,2], scale[batch,2])
            If return_attention=True:  ((loc, scale), {"attn_weights": [...], "tokens": [...]})
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]

        # 1. Segment and project to token embeddings
        self_feat, target_feat, mate_feat = segment_obs(obs)

        token_self = self.self_proj(self_feat)       # [B, d_model]
        token_target = self.target_proj(target_feat)  # [B, d_model]
        token_mate = self.mate_proj(mate_feat) * self.mate_scale  # [B, d_model]

        # Stack into sequence: [Self, Target, Mate]
        tokens = torch.stack([token_self, token_target, token_mate], dim=1)  # [B, 3, d_model]

        # Add learnable token-type embeddings
        tokens = tokens + self.token_type_embed

        # 2. Multi-head self-attention
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        # attn_out: [B, 3, d_model]
        # attn_weights: [B, 3, 3] — attention from each token to each token

        # Residual connection
        tokens_out = tokens + attn_out

        # 3. Learned attention pooling: weighted average over 3 tokens
        # pool_query: [1, 1, d_model]
        pool_scores = torch.matmul(
            self.attn_pool_query, tokens_out.transpose(1, 2)
        )  # [B, 1, 3]
        pool_weights = F.softmax(pool_scores / np.sqrt(self.d_model), dim=-1)  # [B, 1, 3]
        pooled = torch.matmul(pool_weights, tokens_out).squeeze(1)  # [B, d_model]

        # 4. MLP head → action distribution
        feat = self.mlp_head(pooled)
        loc = torch.tanh(self.mean(feat))  # [-1, 1] squashed
        scale = torch.exp(self.log_std).expand(batch, -1)

        if return_attention:
            return (loc, scale), {
                "attn_weights": attn_weights,       # [B, 3, 3]
                "pool_weights": pool_weights,       # [B, 1, 3]
                "tokens_raw": torch.stack([token_self, token_target, token_mate], dim=1),
                "tokens_attended": tokens_out,
            }
        return loc, scale

    def set_mate_scale(self, scale: float):
        """Dynamically adjust mate token contribution (for curriculum training)."""
        self.mate_scale = scale


class AttentionCritic(nn.Module):
    """Standard centralized critic — unchanged from baseline.

    Global state (21-dim) → MLP → scalar value.
    The critic doesn't need attention because it already sees the full global state.
    """

    def __init__(self, global_dim: int = 21, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight, gain=np.sqrt(2))
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
        # Output layer: gain=1.0 for value
        nn.init.orthogonal_(self.net[4].weight, gain=1.0)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Test / sanity check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Self-Attention FormationActor — Sanity Check ===\n")

    actor = AttentionFormationActor()
    critic = AttentionCritic()

    # Count parameters
    n_params_actor = sum(p.numel() for p in actor.parameters())
    n_params_critic = sum(p.numel() for p in critic.parameters())
    print(f"Actor parameters:  {n_params_actor:,}")
    print(f"Critic parameters: {n_params_critic:,}")
    print(f"Total:             {n_params_actor + n_params_critic:,}")

    # Baseline MLP comparison
    base_actor_params = 33*256 + 256 + 256*256 + 256 + 256*2 + 2  # ~76K with bias
    print(f"Baseline MLP Actor: ~{base_actor_params:,} (33→256→256→2)")

    # Forward pass test
    batch = torch.randn(4, 33)
    loc, scale = actor(batch)
    print(f"\nForward pass [4,33]: loc={loc.shape}, scale={scale.shape}")
    print(f"  loc range: [{loc.min().item():.3f}, {loc.max().item():.3f}]")
    print(f"  scale:      {scale[0].detach().numpy()}")

    # With attention weights
    (loc, scale), attn_info = actor(batch, return_attention=True)
    print(f"\nAttention weights [4,3,3]:\n{attn_info['attn_weights'].detach()}")
    print(f"Pool weights [4,1,3]:\n{attn_info['pool_weights'].detach()}")

    # Critic test
    gs = torch.randn(4, 21)
    val = critic(gs)
    print(f"\nCritic [4,21]: val={val.shape}, range=[{val.min():.3f}, {val.max():.3f}]")

    # 1v1 mode (mate_scale=0)
    actor_1v1 = AttentionFormationActor(mate_scale=0.0)
    loc_1v1, _ = actor_1v1(batch)
    print(f"\n1v1 mode (mate_scale=0): loc range=[{loc_1v1.min():.3f}, {loc_1v1.max():.3f}]")

    print("\n[OK] All checks passed!")
