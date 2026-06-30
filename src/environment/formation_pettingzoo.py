"""PettingZoo ParallelEnv wrapper for FormationEnv 2v1 MAPPO training.

Converts the single-Gym-Env FormationEnv (Box(66) obs, Box(4) act)
into PettingZoo's dict-based ParallelEnv API required by Tianshou.

Per-agent:
  obs: Box(33) — local observation
  act: Box(2)  — [turn_rate_factor, speed_factor]
  global_state: Box(21) — in infos dict for centralized critic
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from src.environment.formation_env import FormationEnv


class FormationPettingZooEnv(ParallelEnv):
    """PettingZoo wrapper: N pursuer agents + 1 scripted target.

    Agents: "pursuer_0", "pursuer_1"
    Global state injected into infos["global_state"] for CTDE critic.
    """

    metadata = {"name": "formation_pettingzoo_v0"}

    def __init__(self, num_pursuers: int = 2, difficulty_level: float = 0.0,
                 lock_altitude: bool = True, record_tacview: bool = False):
        super().__init__()

        self.N = num_pursuers
        self.possible_agents = [f"pursuer_{i}" for i in range(self.N)]
        self.agents = list(self.possible_agents)

        # Underlying FormationEnv
        self._env = FormationEnv(
            num_pursuers=self.N, num_targets=1,
            difficulty_level=difficulty_level, lock_altitude=lock_altitude,
            record_tacview=record_tacview)

        self._obs_per_agent = self._env._obs_per_pursuer
        self._global_dim = (self.N + 1) * 7  # N pursuers + 1 target × 7

        # PettingZoo dict spaces
        self.observation_spaces = {
            aid: gym.spaces.Box(-1, 1, (self._obs_per_agent,), dtype=np.float32)
            for aid in self.possible_agents
        }
        self.action_spaces = {
            aid: gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            for aid in self.possible_agents
        }

        self._record = record_tacview

    def reset(self, seed=None, options=None):
        obs_concat, _ = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)

        # Split concatenated obs into per-agent dict
        observations = {}
        for i, aid in enumerate(self.agents):
            start = i * self._obs_per_agent
            observations[aid] = obs_concat[start:start + self._obs_per_agent].copy()

        self._last_obs = observations
        self._last_rew = {aid: 0.0 for aid in self.agents}
        self._last_term = {aid: False for aid in self.agents}
        self._last_trunc = {aid: False for aid in self.agents}
        infos = {aid: {"global_state": self._build_global_state()} for aid in self.agents}
        self._last_info = infos
        return observations, infos

    def step(self, actions: dict):
        # Concatenate per-agent actions → Box(2N)
        act_list = []
        for aid in self.possible_agents:
            a = np.asarray(actions.get(aid, [0.0, 0.0]), dtype=np.float32)
            act_list.append(np.clip(a, -1, 1))
        concat_action = np.concatenate(act_list)

        obs_concat, reward_total, terminated, truncated, info = self._env.step(concat_action)

        # Split obs, share reward
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos_out = {}
        global_state = self._build_global_state()

        base_reward = float(reward_total) / self.N  # split evenly

        for i, aid in enumerate(self.possible_agents):
            start = i * self._obs_per_agent
            observations[aid] = obs_concat[start:start + self._obs_per_agent].copy()
            rewards[aid] = base_reward
            terminations[aid] = terminated
            truncations[aid] = truncated
            infos_out[aid] = {"global_state": global_state,
                              "reason": info.get("reason", "unknown")}

        self._last_obs = observations
        self._last_rew = rewards
        self._last_term = terminations
        self._last_trunc = truncations
        self._last_info = infos_out

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos_out

    def render(self):
        pass

    def close(self):
        pass

    # ── AEC API compatibility (for Tianshou 0.5.x PettingZooEnv) ─────────

    def last(self, agent=None):
        """Return (obs, rew, term, trunc, info) for AEC-compatible API."""
        # After step(), return the last observation for each agent
        if not hasattr(self, '_last_obs'):
            return None, 0.0, True, True, {}
        aid = agent if agent else self.possible_agents[0]
        return (
            self._last_obs.get(aid, np.zeros(self._obs_per_agent, dtype=np.float32)),
            self._last_rew.get(aid, 0.0),
            self._last_term.get(aid, False),
            self._last_trunc.get(aid, False),
            self._last_info.get(aid, {}),
        )

    # ── Global state (centralized critic input) ──────────────────────────

    def _build_global_state(self) -> np.ndarray:
        """21-dim: 2 pursuers + 1 target, each with pos(3)+vel(3)+heading(1)."""
        MAX_DIST = 10000.0; MAX_HEIGHT = 5000.0; MAX_VEL = 400.0
        vec = []
        for ps in self._env.pursuers:
            p = ps.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ps.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ps.aircraft.state["yaw_deg"]) / 180.0])
            vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        for ts in self._env.targets:
            p = ts.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ts.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ts.aircraft.state["yaw_deg"]) / 180.0])
            vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        return np.array(vec, dtype=np.float32)

    # ── Delegation ──────────────────────────────────────────────────────

    def export_tacview(self, path):
        self._env.export_tacview(path)

    @property
    def difficulty_level(self):
        return self._env.difficulty_level

    def set_ata_penalty_weight(self, w):
        self._env.set_ata_penalty_weight(w)

    def set_formation_weight(self, w):
        self._env.set_formation_weight(w)
