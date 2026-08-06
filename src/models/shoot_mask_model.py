"""ShootMaskModel — TorchModelV2 with action masking for 1v1 missile task.

Reads a flat Box observation (36-dim legacy or 38-dim with closure features)
split into:
  - obs[:, :obs_dim]     — aircraft + target + missile state
  - obs[:, obs_dim:]     — action mask (N_ACTIONS=11 dims)

obs_dim is derived at construction from the observation space, so legacy and
extended observations both work without code changes.

Applies mask to MultiDiscrete logits: invalid actions → -1e9 (≈ probability 0).
Reference: formation_rllib_model.py mask logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ShootMaskModel(TorchModelV2, nn.Module):
    """MLP policy with proper action mask support for MultiDiscrete actions.

    Action space: MultiDiscrete([3 speed, 5 heading, 1 altitude, 2 fire])
    Observation: Box(39/41) — first (dim-11) obs, last 11 mask.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._action_dims = [3, 5, 1, 2]  # speed, heading, altitude, fire
        self._total_actions = sum(self._action_dims)  # 11

        # Derive obs dim from the action-mask tail: flat = obs + mask.
        obs_shape = getattr(obs_space, "shape", None)
        self._obs_dim = int(obs_shape[0]) - self._total_actions if obs_shape is not None else 25

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(self._obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )

        # Per-dimension action heads
        self.action_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in self._action_dims
        ])

        # Value branch
        self.value_net = nn.Sequential(
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        flat_obs = input_dict["obs"]  # [B, 36] or [36]

        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        # Split: first obs_dim = real obs, last 11 = action mask
        obs = flat_obs[:, :self._obs_dim]
        mask = flat_obs[:, self._obs_dim:self._obs_dim + self._total_actions]

        feat = self.encoder(obs)  # [B, 128]
        self._features = feat

        # Action logits per head → concat
        logits = torch.cat([head(feat) for head in self.action_heads], dim=1)  # [B, 11]

        # ── Action masking ──────────────────────────────────────────────
        # mask=1.0 means VALID, mask=0.0 means INVALID
        # Convert: invalid → logit = -1e9 (≈ probability 0 after softmax)
        logits = logits + (1.0 - mask) * (-1e9)

        return logits, state

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_net(self._features).squeeze(1)


class ShootFireOnlyModel(TorchModelV2, nn.Module):
    """Frozen-BC actor + INDEPENDENT critic (PPO P1/P2).

    Actor path mirrors ShootMaskModel (encoder + action heads [3,5,1,2]) so
    the archived BC state-dict loads directly into 'encoder'/'action_heads'.
    The critic uses its OWN encoder + value head ('critic_encoder' /
    'critic_value') -- no gradient ever flows from the critic into the actor.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self._action_dims = [3, 5, 1, 2]
        self._total_actions = sum(self._action_dims)
        obs_shape = getattr(obs_space, "shape", None)
        self._obs_dim = int(obs_shape[0]) - self._total_actions \
            if obs_shape is not None else 30

        # frozen BC actor
        self.encoder = nn.Sequential(
            nn.Linear(self._obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in self._action_dims
        ])

        # independent critic
        self.critic_encoder = nn.Sequential(
            nn.Linear(self._obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.critic_value = nn.Sequential(
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._features = None
        self._obs_cache = None

    def forward(self, input_dict, state, seq_lens):
        flat_obs = input_dict["obs"]
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)
        obs = flat_obs[:, :self._obs_dim]
        mask = flat_obs[:, self._obs_dim:self._obs_dim + self._total_actions]
        self._obs_cache = obs.detach()
        feat = self.encoder(obs)
        self._features = feat
        logits = torch.cat([head(feat) for head in self.action_heads], dim=1)
        logits = logits + (1.0 - mask) * (-1e9)
        return logits, state

    def value_function(self):
        assert self._obs_cache is not None, "must call forward() first"
        return self.critic_value(self.critic_encoder(self._obs_cache)).squeeze(1)

