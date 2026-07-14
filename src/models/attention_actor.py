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
    27, 28,       # agent one-hot ID: [1,0]=P0, [0,1]=P1 — breaks symmetry
]
SELF_DIM = len(SELF_INDICES)  # 15 (was 13)

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
    29, 30, 31,   # mate relative position in body frame (3)
    32, 33, 34,   # mate relative velocity in body frame (3)
    35, 36, 37, 38,  # mate broadcast: cmd_turn, cmd_speed, cos(hdg), sin(hdg) (4)
]
MATE_DIM = len(MATE_INDICES)  # 10

# Verify: 15 + 14 + 10 = 39 ✓
assert SELF_DIM + TARGET_DIM + MATE_DIM == 39, \
    f"Segment dims don't sum to 39: {SELF_DIM}+{TARGET_DIM}+{MATE_DIM}={SELF_DIM+TARGET_DIM+MATE_DIM}"


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
        # Scale 0.1: for d_model=128, pre-softmax scores have std ≈ sqrt(128)*0.1*1 ≈ 1.13,
        # giving meaningful initial differentiation across 3 tokens (prevents collapse).
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

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

    def forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim intermediate features before the final output heads.

        Used by discrete-action models that replace mean/scale with categorical heads.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]

        self_feat, target_feat, mate_feat = segment_obs(obs)
        token_self = self.self_proj(self_feat)
        token_target = self.target_proj(target_feat)
        token_mate = self.mate_proj(mate_feat) * self.mate_scale
        tokens = torch.stack([token_self, token_target, token_mate], dim=1)
        tokens = tokens + self.token_type_embed
        attn_out, _attn_w = self.attention(tokens, tokens, tokens)
        tokens_out = tokens + attn_out

        pool_scores = torch.matmul(
            self.attn_pool_query, tokens_out.transpose(1, 2))
        pool_weights = F.softmax(
            pool_scores / np.sqrt(self.d_model), dim=-1)
        pooled = torch.matmul(pool_weights, tokens_out).squeeze(1)

        feat = self.mlp_head(pooled)  # [B, 256]
        return feat

    def set_mate_scale(self, scale: float):
        """Dynamically adjust mate token contribution (for curriculum training)."""
        self.mate_scale = scale

    def attention_entropy(self, obs: torch.Tensor) -> dict:
        """Compute per-head attention entropy for collapse detection.

        Returns dict with:
          - mha_entropy: mean entropy across all MHA heads (bits, max=log(3)≈1.10)
          - pool_entropy: entropy of learned pooling weights (bits, max=log(3)≈1.10)
          - mate_attention: fraction of MHA attention allocated to mate token (row avg)

        Low entropy (< 0.3) signals attention collapse (one token dominates).
        High entropy (> 1.0) signals uniform attention (no differentiation).
        Healthy range: 0.4–0.9 (meaningful differentiation without collapse).
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            (_, _), info = self.forward(obs, return_attention=True)
            attn = info['attn_weights']  # [B, 3, 3]

            # MHA entropy per batch item, averaged
            eps = 1e-8
            mha_entropy = -(attn * (attn + eps).log()).sum(-1).mean().item()  # avg over B×3 rows

            # Pool entropy
            pool = info['pool_weights'].squeeze(1)  # [B, 3]
            pool_entropy = -(pool * (pool + eps).log()).sum(-1).mean().item()

            # Mate attention fraction (column 2 of attention matrix = attention TO mate)
            mate_attn = attn[:, :, 2].mean().item()  # avg over B×3 query positions

        return {
            'mha_entropy': float(mha_entropy),
            'pool_entropy': float(pool_entropy),
            'mate_attention': float(mate_attn),
        }


class AttentionCritic(nn.Module):
    """Tokenized Attention Critic — Self-Attention over entity sequence.

    Global state is reshaped from flat 21-dim to sequence of 3 entities
    (Self, Mate, Target), each with 7 features (pos xyz, vel xyz, heading).

    Architecture:
      (B, 3, 7) → Linear(7, d_model) → + TypeEmbed(3, d_model)
      → MultiHeadSelfAttention → LayerNorm + Residual
      → Learned Pooling Query → (B, d_model)
      → Value Head [256, 1] → scalar value

    The attention mechanism lets the Critic dynamically focus on different
    entities — attending to Mate when proximity risks collision, attending
    to Target when estimating pursuit progress.
    """

    def __init__(self, token_dim: int = 7, d_model: int = 128, n_heads: int = 4,
                 mlp_hidden: int = 256):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads

        # 1. Token projection: 7-dim entity state → d_model
        self.input_proj = nn.Linear(token_dim, d_model)

        # 2. Type embeddings: tell the network which entity is Self/Mate/Target
        self.type_embeddings = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)

        # 3. Multi-head self-attention over 3 entities
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # 4. Learned pooling query — aggregates 3 tokens into one
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # 5. Value head
        self.value_net = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for value head
        for m in self.value_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Output layer: gain=1.0 for stable initial value predictions
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)

    def forward(self, global_state, return_attention: bool = False):
        """
        Args:
            global_state: (B, 3, 7) tensor — [Self, Mate, Target] tokens
            return_attention: if True, return (value, attn_weights_dict)

        Returns:
            value: (B,) tensor of scalar state-values
        """
        if isinstance(global_state, np.ndarray):
            global_state = torch.as_tensor(global_state, dtype=torch.float32)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0)

        batch_size = global_state.size(0)

        # 1. Project to d_model
        x = self.input_proj(global_state)  # (B, 3, d_model)

        # 2. Inject type embeddings: token 0=Self, 1=Mate, 2=Target
        device = global_state.device
        types = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, 3)
        x = x + self.type_embeddings(types)

        # 3. Self-Attention with residual + LayerNorm
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)  # (B, 3, d_model)

        # 4. Learned pooling: query attends to all 3 token outputs
        q = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, d_model)
        pooled_out, pool_weights = self.attention(q, x, x)  # (B, 1, d_model)
        pooled = pooled_out.squeeze(1)  # (B, d_model)

        # 5. Value head
        value = self.value_net(pooled).squeeze(-1)  # (B,)

        if return_attention:
            return value, {
                'mha_weights': attn_weights,      # (B, 3, 3)
                'pool_weights': pool_weights,      # (B, 1, 3)
            }
        return value

    def critic_attention_entropy(self, global_state) -> dict:
        """Monitor attention entropy for collapse detection."""
        if isinstance(global_state, np.ndarray):
            global_state = torch.as_tensor(global_state, dtype=torch.float32)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0)

        with torch.no_grad():
            _, attn = self.forward(global_state, return_attention=True)
            mha = attn['mha_weights']   # (B, 3, 3)
            pool = attn['pool_weights']  # (B, 1, 3)

            eps = 1e-8
            mha_entropy = -(mha * (mha + eps).log()).sum(-1).mean().item()
            pool_entropy = -(pool * (pool + eps).log()).sum(-1).mean().item()

            # Attention to mate token (column 1 in MHA, column 1 in pool)
            mha_mate = mha[:, :, 1].mean().item()
            pool_mate = pool[:, 0, 1].mean().item()

        return {
            'mha_entropy': float(mha_entropy),
            'pool_entropy': float(pool_entropy),
            'mha_mate_attn': float(mha_mate),
            'pool_mate_attn': float(pool_mate),
        }


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
