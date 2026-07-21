"""Abstract BaseTask — decouples task business logic from physics simulation.

Inspired by LAG's BaseTask design, adapted for RLlib MultiAgentEnv compatibility.

A BaseTask defines everything that is specific to a particular air-combat scenario:
  - Observation / action spaces
  - Observation assembly from aircraft state
  - Reward calculation
  - Termination conditions
  - Action masking (optional)
  - Action interpretation → PID flight control targets

BaseEnv only handles JSBSim lifecycle and the micro-step physics loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class BaseTask(ABC):
    """Abstract interface for a formation / combat task.

    Subclass this to implement specific scenarios (formation pursuit,
    missile dodge, shoot, etc.) without touching the physics engine.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._agent_ids: List[str] = []

    # ══════════════════════════════════════════════════════════════════════════
    #  Spaces (must be set by subclass __init__)
    # ══════════════════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        """Per-agent observation space as a Dict of agent_id → space."""

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Dict:
        """Per-agent action space as a Dict of agent_id → space."""

    @property
    def agent_ids(self) -> List[str]:
        """RL-controlled agent IDs for this task."""
        return self._agent_ids

    # ══════════════════════════════════════════════════════════════════════════
    #  Lifecycle hooks (called by BaseEnv)
    # ══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def reset(self, env) -> None:
        """Reset task-specific state at episode start.

        Called AFTER BaseEnv has reset all aircraft positions.
        `env` is the BaseEnv instance — use env.pursuers, env.targets, etc.
        """

    @abstractmethod
    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Interpret RL actions → set PID flight-control targets on each aircraft.

        Called BEFORE each 12-step physics loop in BaseEnv.step().

        Args:
            env: BaseEnv instance (with .pursuers[] and .targets[]).
            action_dict: {"p0": np.array([turn, speed]), "p1": ...}
                         or arbitrary shape defined by self.action_space.
        """

    @abstractmethod
    def step(self, env) -> None:
        """Task-level per-decision-step logic.

        Called AFTER the 12 micro-step physics loop. Handle:
          - Missile simulation updates (if any)
          - Internal state-machine transitions
          - Curriculum stage advancement

        Args:
            env: BaseEnv instance.
        """

    # ══════════════════════════════════════════════════════════════════════════
    #  Observation / Reward / Termination (called by BaseEnv)
    # ══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def get_obs(self, env) -> Dict[str, dict]:
        """Build per-agent observation dicts from current aircraft state.

        Returns:
            {"p0": {"obs": ..., "global_state": ..., "action_mask": ...}, "p1": ...}
        """

    @abstractmethod
    def get_reward(self, env) -> Dict[str, float]:
        """Compute per-agent rewards for the current decision step.

        Returns:
            {"p0": reward_p0, "p1": reward_p1}
        """

    @abstractmethod
    def get_termination(
        self, env
    ) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Check termination / truncation conditions.

        Returns:
            (terminateds, truncateds, infos) — each with "__all__" key.
        """

    # ══════════════════════════════════════════════════════════════════════════
    #  Optional hooks
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_mask(self, env, agent_id: str) -> Optional[np.ndarray]:
        """Return per-agent action mask, or None if no masking is needed."""
        return None

    def get_global_state(self, env) -> np.ndarray:
        """Return the centralized critic's global state (flat vector).

        Default: concatenation of all agent-local observations.
        Override for task-specific global features.
        """
        return np.zeros(0)
