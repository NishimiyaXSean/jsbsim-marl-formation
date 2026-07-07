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

import numpy as np
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

        # Derive dimensions from observation space (robust, not hardcoded)
        # RLlib wraps the original space → use original_space if available
        orig = getattr(obs_space, "original_space", obs_space)
        self._local_dim = orig["obs"].shape[0]       # 33
        self._global_dim = orig["global_state"].shape[0]  # 21
        self._act_dim = action_space.shape[0]  # 2

        # num_outputs should be 2 * act_dim (mean + log_std for each action)
        assert num_outputs == 2 * self._act_dim, \
            f"Expected num_outputs={2*self._act_dim}, got {num_outputs}"

        # ── Embed full AttentionFormationActor ──────────────────────────
        self.actor = AttentionFormationActor(
            obs_dim=self._local_dim,
            act_dim=self._act_dim,
            d_model=d_model,
            n_heads=n_heads,
            mlp_hidden=mlp_hidden,
            mate_scale=mate_scale,
        )

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

        # ── Double-insurance orthogonal init ────────────────────────────
        self._init_weights(self.actor)
        self._init_weights(self.critic)

    @staticmethod
    def _init_weights(module):
        """SB3-validated orthogonal initialization for all Linear layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, input_dict, state, seq_lens):
        """Compute action logits and critic value.

        Args:
            input_dict: RLlib-provided dict with "obs" key
            state: RNN state (unused)
            seq_lens: Sequence lengths (unused)

        Returns:
            (logits, state) where logits = [mean(2), log_std(2)] = 4-dim
        """
        obs = input_dict["obs"]

        # ── Extract local observation and global state ──────────────────
        # RLlib may present obs as nested dict or flat tensor
        if isinstance(obs, dict):
            local = obs["obs"].float()                  # [B, 33]
            global_flat = obs["global_state"].float()   # [B, 21]
        else:
            # Fallback: split flat tensor
            local = obs[:, :self._local_dim].float()
            global_flat = obs[:, self._local_dim:].float()

        batch_size = local.shape[0]

        # ── 1. Critic: global_state → value ─────────────────────────────
        #    Reshape flat [B,21] → token sequence [B,3,7]
        #    Order: [Self(0), Mate(1), Target(2)]
        if global_flat.shape[-1] == self._global_dim:
            global_tokens = global_flat.view(batch_size, 3, 7)
        else:
            # Safety: if dimensions don't match, pad or truncate
            g_dim = global_flat.shape[-1]
            if g_dim < self._global_dim:
                pad = torch.zeros(batch_size, self._global_dim - g_dim,
                                 device=global_flat.device)
                global_flat = torch.cat([global_flat, pad], dim=-1)
            else:
                global_flat = global_flat[:, :self._global_dim]
            global_tokens = global_flat.view(batch_size, 3, 7)

        self._last_value = self.critic(global_tokens)  # [B]

        # ── 2. Actor: local obs → action distribution params ────────────
        loc, scale = self.actor(local)  # loc=[B,2], scale=[B,2]

        # Clamp scale to prevent NaN/Inf in log
        scale = torch.clamp(scale, min=1e-6, max=1e6)

        # ── 3. Build logits = mean || log_std ────────────────────────────
        log_std = torch.log(scale)
        logits = torch.cat([loc, log_std], dim=1)  # [B, 4]

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
