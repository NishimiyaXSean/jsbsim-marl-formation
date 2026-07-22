"""HeadingTrackingTask — simple 1-agent heading hold task.

The agent controls a single F-16 and must maintain a target heading,
altitude, and speed. This is the simplest task for validating the
Task-Based architecture end-to-end.

Action: Discrete(3) — [left(-10°/s), hold(0°/s), right(+10°/s)]
Observation: Box(8) — [heading_err, roll_sin, roll_cos, pitch, alt_err, spd_err, vs, reserved]
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

        # Target values (altitude/speed set at reset to avoid aggressive PID transients)
        self.target_heading_deg = config.get("target_heading", 90.0) if config else 90.0
        self.target_altitude_m = 3000.0   # placeholder — set to actual initial alt at reset
        self.target_speed_mps = 250.0

        # Action: 3 discrete heading deltas
        self._delta_headings = [-10.0, 0.0, 10.0]  # deg/s → ±2° per decision step at 5Hz

        # Observation: 8 dims (heading_err, roll_sin/cos, pitch, alt_err, spd_err, vs)
        single_obs = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
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
        """Sync PID references to current aircraft state (prevents wild transients)."""
        self._last_actions: Dict[str, int] = {}
        for i, aid in enumerate(self._agent_ids):
            ps = env.pursuers[i]
            s = ps.aircraft.state
            self.target_altitude_m = float(s["alt_m"])
            ps.ref_hdg = float(s["yaw_deg"])          # ← critical: sync PID heading reference
            ps.ref_alt_m = self.target_altitude_m
            ps._cmd_speed = self.target_speed_mps

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Map discrete heading action → FlightTarget on each pursuer.

        Uses index-aligned iteration: agent_ids[i] ↔ pursuers[i].
        Each agent controls exactly its own aircraft.
        """
        for i, aid in enumerate(self._agent_ids):
            a = int(action_dict.get(aid, 1))
            a = np.clip(a, 0, 2)
            delta_hdg = self._delta_headings[a]

            ps = env.pursuers[i]
            ps.ref_hdg = (ps.ref_hdg + delta_hdg * 0.2) % 360.0
            ps.ref_alt_m = self.target_altitude_m
            ps._cmd_speed = self.target_speed_mps
            self._last_actions[aid] = a  # store for action penalty in reward

    def step(self, env) -> None:
        """No additional task-level logic needed."""
        pass

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        """Build heading-tracking observation (8 dims)."""
        ps = env.pursuers[0]
        s = ps.aircraft.state

        hdg = float(s["yaw_deg"])
        roll_rad = math.radians(float(s["roll_deg"]))
        pitch_rad = math.radians(float(s["pitch_deg"]))
        alt_m = float(s["alt_m"])
        spd_mps = float(s["airspeed_mps"])

        heading_err = (self.target_heading_deg - hdg + 180.0) % 360.0 - 180.0
        alt_err = alt_m - self.target_altitude_m

        obs = np.array([
            heading_err / 180.0,            # [-1, 1]
            math.sin(roll_rad),
            math.cos(roll_rad),
            pitch_rad / (math.pi / 2),      # [-1, 1]
            alt_err / 500.0,                # clamp large deviations
            (spd_mps - self.target_speed_mps) / 50.0,
            float(s.get("h_dot_fps", 0)) * 0.3048 / 50.0,  # vertical speed m/s → normalized
            0.0,  # reserved
        ], dtype=np.float32)

        return {"p0": np.clip(obs, -1, 1)}

    # ── Reward ──────────────────────────────────────────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        """Dense reward: heading error + altitude deviation penalty."""
        ps = env.pursuers[0]
        s = ps.aircraft.state
        hdg = float(s["yaw_deg"])
        alt_m = float(s["alt_m"])
        pitch_deg = float(s["pitch_deg"])

        heading_err = abs((self.target_heading_deg - hdg + 180.0) % 360.0 - 180.0)
        alt_err = abs(alt_m - self.target_altitude_m)

        reward = (
            -heading_err / 180.0           # [-1, 0] heading
            - alt_err / 1000.0             # altitude penalty
            - abs(pitch_deg) / 90.0 * 0.5  # pitch penalty (prevent vertical flight)
        )

        # Action penalty: discourage unnecessary turns → eliminates oscillation
        # 0.05 is ~1/4 of a unit heading error, enough to matter vs -abs(err)/180
        last_action = self._last_actions.get("p0", 1)
        if last_action != 1:
            reward -= 0.05

        return {"p0": float(np.clip(reward, -5, 5))}

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
