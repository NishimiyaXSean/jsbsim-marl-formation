"""CTDE TorchModelV2 for formation MAPPO.

Actor:  33-dim local obs → 256 → Tanh → 256 → Tanh → Box(2)
Critic: global_state (N*7 + M*7) → 256 → Tanh → 256 → Tanh → scalar
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from gymnasium.spaces import Box, Dict


class FormationMAPPOModel(TorchModelV2, nn.Module):
    """CTDE model: separate actor (local obs) and critic (global state)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        local_obs_dim = 33
        global_dim = 21  # 2 pursuers + 1 target × 7

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(local_obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.action_mean = nn.Linear(256, 2)
        self.action_log_std = nn.Parameter(torch.zeros(1, 2))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(global_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, dict):
            local = obs["obs"].float()
            global_state = obs["global_state"].float()
        else:
            # Flat obs: first 33 are local, rest is global
            local = obs[:, :33].float()
            global_state = obs[:, 33:].float()

        # Actor
        feat = self.actor(local)
        mean = self.action_mean(feat)
        log_std = self.action_log_std.expand_as(mean)

        # Critic
        self._last_value = self.critic(global_state).squeeze(-1)

        return torch.cat([mean, log_std], dim=1), state

    def value_function(self):
        return self._last_value
