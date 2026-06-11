"""MAPPO Actor-Critic model with centralized critic (CTDE).

Actor:  local observation (19-dim) → action (4-dim continuous: throttle, elevator, aileron, rudder)
Critic: global state (26-dim) → scalar value

For continuous actions, the actor outputs mean + log_std for a Gaussian policy.
"""

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class MAPPOModel(TorchModelV2, nn.Module):
    """MAPPO CTDE: decentralized actor + centralized critic."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        local_obs_dim = obs_space.original_space["obs"].shape[0]    # 19
        global_state_dim = obs_space.original_space["global_state"].shape[0]  # 26

        self._action_dim = action_space.shape[0]  # 4

        # Actor: 19 → 256 → 256 → 8 (mean + log_std for 4 actions)
        self.actor = nn.Sequential(
            nn.Linear(local_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, self._action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self._action_dim))

        # Critic: 26 → 256 → 256 → 1
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        local_obs = input_dict["obs"]["obs"]
        global_state = input_dict["obs"]["global_state"]

        # Critic
        self._last_value = self.critic(global_state).squeeze(1)

        # Actor: output mean + log_std concatenated
        features = self.actor(local_obs)
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_output = torch.cat([action_mean, action_log_std], dim=1)

        return action_output, state

    def value_function(self):
        return self._last_value
