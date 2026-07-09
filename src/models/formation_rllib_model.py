"""RLlib TorchModelV2 — Self-Attention CTDE Model for Formation MAPPO.

Wraps AttentionFormationActor (33-dim → Self-Attention → action) and
AttentionCritic (21-dim → Tokenized Attention → value) inside RLlib's
TorchModelV2 interface.

Key design decisions:
  - forward() computes BOTH action logits AND critic value (required by RLlib PPO)
  - logits = [action_mean(2), action_log_std(2)] = 4-dim for Box(2) action space
  - Global state reshaped from flat (B,21) to token sequence (B,3,7) for Critic
  - Orthogonal init applied as double insurance on top of model's built-in init
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

from src.models.attention_actor import (
    AttentionFormationActor,
    AttentionCritic,
    SELF_DIM,
    TARGET_DIM,
    MATE_DIM,
)


class RLlibAttentionActor(TorchModelV2, nn.Module):
    """CTDE Self-Attention model for RLlib MAPPO.

    Actor:  33-dim local obs → 3-token Self-Attention → Box(2) action
    Critic: 21-dim global state → 3-entity Tokenized Attention → scalar value

    RLlib expects:
      - forward() returns (logits, state) where logits = mean||log_std concatenated
      - value_function() returns the pre-computed critic value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 d_model: int = 128, n_heads: int = 4, mlp_hidden: int = 256,
                 mate_scale: float = 1.0):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Derive dimensions from observation space
        orig = getattr(obs_space, "original_space", obs_space)
        self._local_dim = orig["obs"].shape[0]       # 33
        self._global_dim = orig["global_state"].shape[0]  # 21

        # Discrete action space: MultiDiscrete([n_turn, n_speed])
        if hasattr(action_space, "nvec"):
            self._n_turn = int(action_space.nvec[0])
            self._n_speed = int(action_space.nvec[1])
        else:
            self._n_turn = 5
            self._n_speed = 3

        # num_outputs = n_turn + n_speed (categorical logits concatenated)
        assert num_outputs == self._n_turn + self._n_speed, \
            f"Expected num_outputs={self._n_turn + self._n_speed}, got {num_outputs}"
        self._n_actions_total = num_outputs

        # ── Embed full AttentionFormationActor (feature extractor) ──────
        # The actor's MLP head outputs [B, mlp_hidden] features.
        # We replace its final mean layer with categorical heads.
        self.actor = AttentionFormationActor(
            obs_dim=self._local_dim,
            act_dim=2,  # dummy — we replace output heads
            d_model=d_model,
            n_heads=n_heads,
            mlp_hidden=mlp_hidden,
            mate_scale=mate_scale,
        )

        # ── Simple MLP fallback (bypasses Self-Attention for NaN isolation) ─
        self._fallback_mlp = nn.Sequential(
            nn.Linear(self._local_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh(),
        )

        # ── Categorical output heads ─────────────────────────────────────
        self.turn_head = nn.Linear(mlp_hidden, self._n_turn)
        self.speed_head = nn.Linear(mlp_hidden, self._n_speed)
        nn.init.orthogonal_(self.turn_head.weight, gain=0.01)
        nn.init.constant_(self.turn_head.bias, 0.0)
        nn.init.orthogonal_(self.speed_head.weight, gain=0.01)
        nn.init.constant_(self.speed_head.bias, 0.0)

        # ── Embed full AttentionCritic ─────────────────────────────────
        # Critic expects (B, 3, 7) token sequence: [Self, Mate, Target]
        self.critic = AttentionCritic(
            token_dim=7,
            d_model=d_model,
            n_heads=n_heads,
            mlp_hidden=mlp_hidden,
        )

        # Cache for value_function()
        self._last_value = None

        # Track attention health
        self._last_attn_info = {}

        # ── Remove double-init: AttentionFormationActor has its own
        #     careful init; re-initializing all Linear layers would
        #     corrupt MHA's in_proj_weight and cause NaN on GPU

    def forward(self, input_dict, state, seq_lens):
        """Compute action logits and critic value."""
        obs = input_dict["obs"]

        if isinstance(obs, dict):
            local = obs["obs"].float()
            global_flat = obs["global_state"].float()
            action_mask = obs.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.float()
        else:
            local = obs[:, :self._local_dim].float()
            global_flat = obs[:, self._local_dim:].float()
            action_mask = None

        batch_size = local.shape[0]

        # ── Critic ──────────────────────────────────────────────────
        if global_flat.shape[-1] == self._global_dim:
            global_tokens = global_flat.view(batch_size, 3, 7)
        else:
            g_dim = global_flat.shape[-1]
            if g_dim < self._global_dim:
                pad = torch.zeros(batch_size, self._global_dim - g_dim, device=global_flat.device)
                global_flat = torch.cat([global_flat, pad], dim=-1)
            else:
                global_flat = global_flat[:, :self._global_dim]
            global_tokens = global_flat.view(batch_size, 3, 7)
        self._last_value = self.critic(global_tokens)

        # ── Actor: Self-Attention token pipeline → categorical heads ──
        feat = self.actor.forward_features(local)  # [B, 256]
        turn_logits = self.turn_head(feat)
        speed_logits = self.speed_head(feat)
        logits = torch.cat([turn_logits, speed_logits], dim=1)

        # ── Action masking ───────────────────────────────────────────
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e9)

        return logits, state

    def value_function(self):
        """Return the critic value computed in the last forward pass."""
        assert self._last_value is not None, (
            "value_function() called before forward(). "
            "RLlib should always call forward() first in the same batch.")
        return self._last_value

    def get_attention_stats(self, obs_tensor: torch.Tensor) -> dict:
        """Monitor attention health for debugging (not used in training loop)."""
        if isinstance(obs_tensor, np.ndarray):
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            local = obs_tensor[:, :self._local_dim]
            return self.actor.attention_entropy(local)

    def get_critic_attention_stats(self, global_tensor: torch.Tensor) -> dict:
        """Monitor Critic attention health."""
        if isinstance(global_tensor, np.ndarray):
            global_tensor = torch.as_tensor(global_tensor, dtype=torch.float32)
        if global_tensor.dim() == 2:
            global_tensor = global_tensor.unsqueeze(0)

        batch = global_tensor.shape[0]
        global_tokens = global_tensor.view(batch, 3, 7)

        with torch.no_grad():
            return self.critic.critic_attention_entropy(global_tokens)
