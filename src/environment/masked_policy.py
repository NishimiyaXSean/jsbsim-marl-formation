"""Maskable Actor-Critic Policy for SB3 with Discrete action space.

Reads an "action_mask" key from a Dict observation and applies invalid
action masking to the policy logits before sampling.

Usage with PPO:
    from src.environment.masked_policy import MaskableActorCriticPolicy
    model = PPO(MaskableActorCriticPolicy, env, ...)
"""

from __future__ import annotations

from typing import Optional, Callable

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class MaskedFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observation with "obs" + "action_mask".

    Only the "obs" key is fed to the MLP; action_mask is passed through
    and applied downstream in the policy's forward().
    """

    def __init__(self, observation_space: gym.spaces.Dict, net_arch=None):
        obs_space = observation_space["obs"]
        features_dim = int(get_flattened_obs_dim(obs_space))
        super().__init__(observation_space, features_dim)

        if net_arch is None:
            net_arch = [128, 128]

        layers = []
        input_dim = features_dim
        for units in net_arch:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        self._net = nn.Sequential(*layers)
        self._features_dim = input_dim

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._net(observations["obs"])


class MaskableActorCriticPolicy(ActorCriticPolicy):
    """Policy that applies action masking to discrete action logits.

    Expects a Dict observation space with keys:
        "obs":           Box(obs_dim,)     — agent observation
        "action_mask":   Box(num_actions,) — 1.0 = valid, 0.0 = masked

    Invalid action logits are set to -1e9, making their softmax
    probability effectively zero.
    """

    def __init__(self, *args, **kwargs):
        # Override features_extractor_class to our masked one
        kwargs["features_extractor_class"] = MaskedFeatureExtractor
        super().__init__(*args, **kwargs)

    def forward(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override to apply action_mask before distribution."""
        action_mask = obs["action_mask"]  # [batch, num_actions]

        # Standard feature extraction + policy/value heads
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        # Compute logits and apply mask
        distribution = self._get_action_dist_from_latent(latent_pi)
        logits = distribution.distribution.logits

        # Mask: set invalid actions to -1e9
        inf_mask = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        masked_logits = torch.where(action_mask > 0.5, logits, inf_mask)

        # Create a new distribution with masked logits
        if hasattr(distribution.distribution, "logits"):
            distribution.distribution.logits = masked_logits
        elif hasattr(distribution.distribution, "probs"):
            distribution.distribution.probs = torch.softmax(masked_logits, dim=-1)

        action = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(action)

        return action, values, log_prob

    def _predict(self, obs: dict[str, torch.Tensor], deterministic: bool = True):
        """Override to apply masking during prediction."""
        action_mask = obs["action_mask"]
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        logits = distribution.distribution.logits

        inf_mask = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        masked_logits = torch.where(action_mask > 0.5, logits, inf_mask)

        if hasattr(distribution.distribution, "logits"):
            distribution.distribution.logits = masked_logits
        elif hasattr(distribution.distribution, "probs"):
            distribution.distribution.probs = torch.softmax(masked_logits, dim=-1)

        return distribution.get_actions(deterministic=deterministic)
