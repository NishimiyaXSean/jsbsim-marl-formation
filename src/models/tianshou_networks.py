"""Actor/Critic networks for Tianshou MAPPO (pure PyTorch).

Actor:  33-dim local obs → 256→Tanh→256→Tanh → action_mean(2) + log_std
Critic: 21-dim global state → 256→Tanh→256→Tanh → scalar value

Both are standard nn.Module — no framework inheritance needed.
Tianshou integration: Actor reads obs["obs"], Critic reads obs["global_state"].
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32)
    return x


class FormationActor(nn.Module):
    """Decentralized actor: local observation → Box(2) action.

    Input:  Dict {"obs": (33,), "global_state": (21,)} or plain (33,) array
    Output: (loc(2), scale(2)) for Independent(Normal) distribution.
    """

    def __init__(self, obs_dim: int = 33, act_dim: int = 2,
                 hidden: int = 256, log_std_init: float = -0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * log_std_init)

    def forward(
        self, obs: Any, state: Any = None, info: Any = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Return (loc, scale) tuple for Independent(Normal)."""
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        obs = _to_tensor(obs)
        feat = self.net(obs)
        loc = torch.tanh(self.mean(feat))
        scale = torch.exp(self.log_std).expand_as(loc)
        return (loc, scale), state


class FormationCritic(nn.Module):
    """Centralized critic: global state → scalar state-value.

    Input:  Dict {"obs": (33,), "global_state": (21,)} or plain (21,) array
    Output: scalar value (or [batch] values).
    """

    def __init__(self, global_dim: int = 21, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: Any, state: Any = None, info: Any = None) -> torch.Tensor:
        """Return scalar value estimate."""
        if isinstance(obs, dict):
            obs = obs.get("global_state", obs)
        obs = _to_tensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)
