"""SinglePursuitTask — 1v1 pursuit with hierarchical action space.

Stage 2 of Task-Based architecture validation. The RL agent (p0) controls
one F-16 and must intercept a scripted target (t0) moving at constant speed
with sinusoidal evasion at higher difficulties.

Action:    MultiDiscrete([3 speed, 5 heading, 3 altitude]) = 45 tactical deltas
Observation: Box(25) — body-frame relative state + tactical geometry
Reward:     Progress + ATA + Proximity + Capture − GroundWarning − LowSpeedTurn
Termination: Envelope + Timeout + CaptureSuccess

Key difference from FormationTask: single-agent, M=1 target, no cooperation logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .task_base import BaseTask
from .reward_functions import (
    ProgressReward, ATAAlignmentReward, ProximityTierReward,
    StepPenaltyReward, GroundWarningReward, LowSpeedTurnPenalty,
    CaptureSuccessReward,
)
from .termination_conditions import (
    FlightEnvelopeTermination, TimeoutTermination, CaptureSuccessTermination,
)
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_LOS_RATE = 0.5


class SinglePursuitTask(BaseTask):
    """1v1 pursuit — hierarchical tactical deltas → FlightTarget → PID/Neural control.

    Agent IDs: "p0"
    Target:    "t0" (scripted, straight-line or sinusoidal evasion)
    """

    # ── Hierarchical action deltas ──────────────────────────────────────
    DELTA_SPEEDS    = [-20.0,   0.0,  20.0]       # m/s
    DELTA_HEADINGS  = [-30.0, -15.0, 0.0, 15.0, 30.0]  # degrees
    DELTA_ALTITUDES = [-100.0,   0.0, 100.0]      # meters

    N_SPD = len(DELTA_SPEEDS)    # 3
    N_HDG = len(DELTA_HEADINGS)  # 5
    N_ALT = len(DELTA_ALTITUDES) # 3
    N_MASK = N_SPD + N_HDG + N_ALT  # 11

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._agent_ids = ["p0"]
        self.N = 1  # pursuers
        self.M = 1  # targets

        # ── Spaces ──────────────────────────────────────────────────────
        single_obs = gym.spaces.Box(-1.0, 1.0, (25,), dtype=np.float32)
        single_act = gym.spaces.MultiDiscrete([self.N_SPD, self.N_HDG, self.N_ALT])

        self._observation_space = gym.spaces.Dict({"p0": single_obs})
        self._action_space = gym.spaces.Dict({"p0": single_act})

        # ── Target config ───────────────────────────────────────────────
        self._difficulty = float(np.clip(
            self.config.get("difficulty_level", 0.0) if self.config else 0.0, 0.0, 1.0))
        self._target_speed_kts = 310  # target cruise speed

        # ── Reward modules ──────────────────────────────────────────────
        self.reward_functions = [
            ProgressReward(self.config),          # closing distance
            ATAAlignmentReward(self.config),       # nose-on-target
            ProximityTierReward(self.config),      # distance milestones
            StepPenaltyReward(self.config),        # time pressure
            GroundWarningReward(self.config),      # anti-CFIT
            LowSpeedTurnPenalty(self.config),      # stall risk
            CaptureSuccessReward(self.config),     # capture bonus
        ]

        # ── Termination modules ─────────────────────────────────────────
        self.termination_conditions = [
            FlightEnvelopeTermination(self.config),
            TimeoutTermination(self.config),
            CaptureSuccessTermination(self.config),
        ]

        # ── Per-episode state ───────────────────────────────────────────
        self._last_actions: Dict[str, dict] = {}
        self._step_counter: int = 0
        self._target_base_hdg: float = 0.0  # set at reset

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self, env) -> None:
        """Sync PID refs + set target initial trajectory."""
        self._last_actions = {"p0": {}}
        self._step_counter = 0

        for ps in env.pursuers:
            s = ps.aircraft.state
            ps.ref_hdg = float(s["yaw_deg"])
            ps.ref_alt_m = float(s["alt_m"])
            ps._cmd_speed = float(s["airspeed_mps"])
            ps._capture_awarded = False
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - env.targets[0].aircraft.position_ned))

        # Target base heading (random, used for sinusoidal evasion)
        self._target_base_hdg = float(env.targets[0].aircraft.state["yaw_deg"])

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Map tactical delta indices → FlightTarget for pursuer."""
        for i, aid in enumerate(self._agent_ids):
            a = action_dict.get(aid, np.array([1, 2, 1], dtype=np.int64))
            a = np.asarray(a, dtype=np.int64)
            spd_idx = int(np.clip(a[0], 0, self.N_SPD - 1))
            hdg_idx = int(np.clip(a[1], 0, self.N_HDG - 1))
            alt_idx = int(np.clip(a[2], 0, self.N_ALT - 1))
            self._last_actions[aid] = {
                'delta_speed': self.DELTA_SPEEDS[spd_idx],
                'delta_heading': self.DELTA_HEADINGS[hdg_idx],
                'delta_altitude': self.DELTA_ALTITUDES[alt_idx],
            }

            ps = env.pursuers[i]
            s = ps.aircraft.state
            current_hdg = float(s["yaw_deg"])
            current_alt = float(s["alt_m"])
            current_spd = float(s["airspeed_mps"])

            ps.ref_hdg = (current_hdg + self._last_actions[aid]['delta_heading']) % 360.0
            ps.ref_alt_m = np.clip(current_alt + self._last_actions[aid]['delta_altitude'],
                                   100.0, 5000.0)
            ps._cmd_speed = np.clip(current_spd + self._last_actions[aid]['delta_speed'],
                                    100.0, 380.0)

    def step(self, env) -> None:
        """Update target trajectory (sinusoidal evasion based on difficulty)."""
        self._step_counter += 1

        if env.M > 0:
            ts = env.targets[0]
            d = self._difficulty
            t = self._step_counter * 0.2  # 5 Hz decision rate

            # Sinusoidal heading variation (amplitude grows with difficulty)
            hdg_var = d * 30.0 * math.sin(t * 0.3)
            ts.ref_hdg = (self._target_base_hdg + hdg_var) % 360.0

            # Altitude oscillation at high difficulty
            ts.ref_alt_m = 3000.0 + d * 200.0 * math.sin(t * 0.15)

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        """Build 25-dim body-frame relative observation."""
        ps = env.pursuers[0]
        ts = env.targets[0]
        s = ps.aircraft.state

        a_pos = ps.aircraft.position_ned
        a_rpy = ps.aircraft.rpy_rad
        a_vel = ps.aircraft.velocity_ned
        t_pos = ts.aircraft.position_ned
        t_vel = ts.aircraft.velocity_ned

        # Body-frame transforms
        rel_w = t_pos - a_pos
        ch, sh = np.cos(a_rpy[2]), np.sin(a_rpy[2])
        rel_body = np.array([
            rel_w[0] * ch + rel_w[1] * sh,
            -rel_w[0] * sh + rel_w[1] * ch,
            -rel_w[2],
        ])
        vel_body = np.array([
            a_vel[0] * ch + a_vel[1] * sh,
            -a_vel[0] * sh + a_vel[1] * ch,
            a_vel[2],
        ])
        t_vel_body = np.array([
            t_vel[0] * ch + t_vel[1] * sh,
            -t_vel[0] * sh + t_vel[1] * ch,
            t_vel[2],
        ])

        # Angular velocity (finite-diff)
        ang_vel = self._ang_vel(ps.aircraft.rpy_rad, ps.prev_rpy)
        ps.prev_rpy = ps.aircraft.rpy_rad.copy()

        # Tactical geometry
        a_fwd = compute_forward_vector(a_rpy)
        t_fwd = compute_forward_vector(ts.aircraft.rpy_rad)
        _, los_dir, _ = compute_los(a_pos, t_pos)
        geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)

        # LOS rate
        r_h = t_pos[:2] - a_pos[:2]
        dh = float(np.linalg.norm(r_h))
        if dh > 1.0:
            v_rel_h = t_vel[:2] - a_vel[:2]
            lambda_dot = float(np.cross(r_h, v_rel_h)) / (dh * dh)
            lambda_dot_norm = float(np.clip(lambda_dot / MAX_LOS_RATE, -1, 1))
        else:
            lambda_dot_norm = 0.0

        alpha = float(s["alpha_deg"])
        spd = float(s["airspeed_mps"])

        obs = np.array([
            rel_body[0] / MAX_DIST, rel_body[1] / MAX_DIST, rel_body[2] / MAX_DIST,
            vel_body[0] / MAX_VEL, vel_body[1] / MAX_VEL, vel_body[2] / MAX_VEL,
            a_rpy[0] / np.pi, a_rpy[1] / (np.pi / 2), a_rpy[2] / np.pi,
            ang_vel[0] / np.pi, ang_vel[1] / np.pi, ang_vel[2] / np.pi,
            a_pos[2] / MAX_HEIGHT,
            t_vel_body[0] / MAX_VEL, t_vel_body[1] / MAX_VEL, t_vel_body[2] / MAX_VEL,
            0.0, 0.0, 0.0,  # target ang_vel placeholder
            geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
            alpha / 30.0, spd / MAX_VEL, 0.0,  # Ps placeholder
            lambda_dot_norm,
        ], dtype=np.float32)

        return {"p0": np.clip(obs, -1, 1)}

    def _ang_vel(self, cur, prev):
        d = cur - prev
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return d / 0.016667  # 1/60 Hz

    # ── Action mask ─────────────────────────────────────────────────────────

    def _build_action_mask(self, ps) -> np.ndarray:
        """11-dim high-level safety mask."""
        mask = np.ones(self.N_MASK, dtype=np.float32)
        airspeed = float(ps.aircraft.state["airspeed_mps"])
        alt_m = float(ps.aircraft.state["alt_m"])

        # Speed mask [0..2]
        if airspeed < 130.0:
            mask[0] = 0.0  # forbid decelerate
        if airspeed > MAX_VEL * 0.95:
            mask[2] = 0.0  # forbid accelerate

        # Heading mask [3..7] — always allowed

        # Altitude mask [8..10]
        if alt_m < 300.0:
            mask[8] = 0.0  # forbid descend

        return mask

    # ── Reward + Termination (delegated to modules) ─────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        rewards = {aid: 0.0 for aid in self._agent_ids}
        for fn in self.reward_functions:
            sub = fn(self, env)
            for aid in self._agent_ids:
                rewards[aid] += sub.get(aid, 0.0)
        return rewards

    def get_termination(self, env) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        terminateds = {aid: False for aid in self._agent_ids}
        terminateds["__all__"] = False
        truncateds = {aid: False for aid in self._agent_ids}
        truncateds["__all__"] = False
        infos: Dict[str, Any] = {"p0": {}}

        for cond in self.termination_conditions:
            reason = cond(self, env)
            if reason is None:
                continue
            if reason == "timeout":
                for aid in self._agent_ids:
                    truncateds[aid] = True
                    infos[aid]["termination_reason"] = "timeout"
                truncateds["__all__"] = True
            else:
                for aid in self._agent_ids:
                    terminateds[aid] = True
                    infos[aid]["termination_reason"] = reason
                terminateds["__all__"] = True
            break

        return terminateds, truncateds, infos
