"""HeadingTrackingTask — simple 1-agent heading hold task.

The agent controls a single F-16 and must maintain a target heading,
altitude, and speed. This is the simplest task for validating the
Task-Based architecture end-to-end.

Action: Discrete(3) — [left(-5°/s), hold(0°/s), right(+5°/s)]
Observation: Box(6) — [heading_err, roll_sin, roll_cos, pitch, alt_err, speed_err]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .task_base import BaseTask


class HeadingTrackingTask(BaseTask):
    """Single-agent heading hold with fixed target.

    Agent ID: "p0"
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._agent_ids = ["p0"]
        self.N = 1  # single pursuer
        self.M = 0  # no target

        # Target values
        self.target_heading_deg = config.get("target_heading", 90.0) if config else 90.0
        self.target_altitude_m = config.get("target_altitude", 5000.0) if config else 5000.0
        self.target_speed_mps = config.get("target_speed", 250.0) if config else 250.0

        # Action: 3 discrete heading deltas
        self._delta_headings = [-10.0, 0.0, 10.0]  # deg/s — applied at 5Hz → ±2° per decision step

        # Spaces
        single_obs = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        single_act = gym.spaces.Discrete(3)

        self._observation_space = gym.spaces.Dict({"p0": single_obs})
        self._action_space = gym.spaces.Dict({"p0": single_act})

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self, env) -> None:
        """Nothing to reset for a fixed-target task."""
        pass

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Map discrete heading action → FlightTarget on pursuer."""
        for aid in self._agent_ids:
            a = int(action_dict.get(aid, 1))
            a = np.clip(a, 0, 2)
            delta_hdg = self._delta_headings[a]

        for ps in env.pursuers:
            ps.ref_hdg = (ps.ref_hdg + delta_hdg * 0.2) % 360.0  # 0.2s decision interval
            ps.ref_alt_m = self.target_altitude_m
            ps._cmd_speed = self.target_speed_mps

    def step(self, env) -> None:
        """No additional task-level logic needed."""
        pass

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        """Build heading-tracking observation for the single agent."""
        ps = env.pursuers[0]
        s = ps.aircraft.state

        hdg = float(s["yaw_deg"])
        roll_rad = math.radians(float(s["roll_deg"]))
        pitch_rad = math.radians(float(s["pitch_deg"]))
        alt_m = float(s["alt_m"])
        spd_mps = float(s["airspeed_mps"])

        heading_err = (self.target_heading_deg - hdg + 180.0) % 360.0 - 180.0

        obs = np.array([
            heading_err / 180.0,          # [-1, 1]
            math.sin(roll_rad),
            math.cos(roll_rad),
            pitch_rad / (math.pi / 2),   # [-1, 1]
            (alt_m - self.target_altitude_m) / 1000.0,
            (spd_mps - self.target_speed_mps) / 50.0,
        ], dtype=np.float32)

        return {"p0": np.clip(obs, -1, 1)}

    # ── Reward ──────────────────────────────────────────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        """Dense reward: negative absolute heading error."""
        ps = env.pursuers[0]
        hdg = float(ps.aircraft.state["yaw_deg"])
        heading_err = abs((self.target_heading_deg - hdg + 180.0) % 360.0 - 180.0)
        reward = -heading_err / 180.0  # [-1, 0]
        return {"p0": reward}

    # ── Termination ─────────────────────────────────────────────────────────

    def get_termination(self, env) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Terminate on crash or timeout."""
        ps = env.pursuers[0]
        alt_m = float(ps.aircraft.state["alt_m"])

        terminateds = {"p0": False, "__all__": False}
        truncateds = {"p0": False, "__all__": False}
        infos: Dict[str, Any] = {"p0": {}}

        if alt_m < 500.0:
            terminateds = {"p0": True, "__all__": True}
            infos["p0"]["termination_reason"] = "low_altitude"

        if env._step_counter >= 500:
            truncateds = {"p0": True, "__all__": True}
            infos["p0"]["termination_reason"] = "timeout"

        return terminateds, truncateds, infos
