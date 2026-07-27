"""SingleCombatShootTask — 1v1 missile combat for BaseEnv + RLlib MAPPO.

The RL agent (p0) controls:
  - Flight:   MultiDiscrete([speed_delta(3), heading_delta(5), altitude_delta(3)])
  - Fire:     fire/no-fire as an extra action dimension

The target (t0) is rule-based — flies straight-and-level with gentle heading
changes for now, providing a stationary-like target for initial training.

KEY DESIGN DECISIONS (per user feedback):
  1. No new Env class — instantiate as BaseEnv(task=SingleCombatShootTask(...))
  2. ValidLaunchReward (+50): immediate credit for firing within valid parameters
     (1–5 km, ATA < 20°), solving the +200 hit reward's long credit-assignment delay
  3. Action mask: fire action is masked when remaining_missiles == 0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.dynamics.flight_controller import FlightControlTargets
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles
from src.utils.units import kts_to_mps

from .task_base import BaseTask
from .missile_simulator import MissileSimulator
from .reward_functions import (
    ProgressReward, ATAAlignmentReward, AltitudeDeviationPenalty,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

DECISION_DT = 0.2
DECISION_STEPS = 12

MAX_DIST = 15000.0         # 15 km — beyond this, target is lost
MAX_ALT = 10000.0           # 10 km ceiling
MAX_VEL = 500.0             # ~Mach 1.5

# ── Agent configuration ───────────────────────────────────────────────────────
N_PURSUERS = 1
N_TARGETS = 1
AGENT_IDS = ["p0"]

# ── Observation dimensions ───────────────────────────────────────────────────
# Self: alt, roll_sin, roll_cos, pitch_sin, pitch_cos, v_body_xyz (3), vc, AoA, heading_sin, heading_cos
SELF_DIM = 12
# Target: delta_altitude, delta_heading, delta_speed, AO, TA, distance, side_flag
TARGET_DIM = 7
# Missile threat (incoming): delta_v, delta_alt, AO, TA, distance, side_flag
MISSILE_DIM = 6
OBS_DIM = SELF_DIM + TARGET_DIM + MISSILE_DIM  # 25

# ── Global state (for centralized critic) ────────────────────────────────────
GLOBAL_PER_AIRCRAFT = 8    # alt, roll_sin, roll_cos, pitch_sin, pitch_cos, vc, heading_sin, heading_cos
GLOBAL_DIM = (N_PURSUERS + N_TARGETS) * GLOBAL_PER_AIRCRAFT  # 16

# ── Action space ─────────────────────────────────────────────────────────────
N_SPEED_DELTA = 3
N_HEADING_DELTA = 5
N_ALT_DELTA = 3
N_FIRE = 2
N_ACTIONS = N_SPEED_DELTA + N_HEADING_DELTA + N_ALT_DELTA + N_FIRE  # 13 for mask

DELTA_SPEEDS    = [-20.0,   0.0,  20.0]       # m/s
DELTA_HEADINGS  = [-30.0, -15.0, 0.0, 15.0, 30.0]  # degrees
DELTA_ALTITUDES = [-100.0,   0.0, 100.0]      # meters

# ── Missile launch parameters ────────────────────────────────────────────────
MAX_ATTACK_ANGLE = 45.0        # degrees — max ATA for valid launch
MAX_ATTACK_DISTANCE = 14000.0   # meters — max range
MIN_ATTACK_DISTANCE = 1000.0    # meters — min range (too close = danger)
MIN_ATTACK_INTERVAL = 125       # decision steps — cooldown between launches
NUM_MISSILES = 2                # per aircraft

# ── Reward weights ───────────────────────────────────────────────────────────
REWARD_VALID_LAUNCH = 50.0      # immediate credit for firing in valid envelope
REWARD_HIT = 200.0              # missile hit on enemy
REWARD_SHOTDOWN = -200.0        # hit by enemy missile
REWARD_CRASH = -200.0           # low altitude / overstress
REWARD_SHOOT_PENALTY = -10.0    # cost per missile fired (anti-spam)


# ═══════════════════════════════════════════════════════════════════════════════
#  SingleCombatShootTask
# ═══════════════════════════════════════════════════════════════════════════════

class SingleCombatShootTask(BaseTask):
    """1v1 missile combat — RL pursuer vs rule-based target.

    Architecture:
      BaseEnv(task=SingleCombatShootTask(config))  — no new Env needed.

    Agent "p0" controls:
      - Flight targets (speed, heading, altitude deltas) via HRL
      - Fire button (Discrete(2))
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._agent_ids = AGENT_IDS
        self.N = N_PURSUERS          # for BaseEnv to build aircraft
        self.M = N_TARGETS

        # ── Spaces ──────────────────────────────────────────────────────────
        single_obs = gym.spaces.Dict({
            "obs": gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32),
            "global_state": gym.spaces.Box(-1.0, 1.0, (GLOBAL_DIM,), dtype=np.float32),
            "action_mask": gym.spaces.Box(0.0, 1.0, (N_ACTIONS,), dtype=np.float32),
        })
        single_act = gym.spaces.MultiDiscrete(
            [N_SPEED_DELTA, N_HEADING_DELTA, N_ALT_DELTA, N_FIRE])

        self._observation_space = gym.spaces.Dict({aid: single_obs for aid in AGENT_IDS})
        self._action_space = gym.spaces.Dict({aid: single_act for aid in AGENT_IDS})

        # ── Reward modules (reuse existing) ─────────────────────────────────
        self._progress = ProgressReward(config)
        self._ata = ATAAlignmentReward(config)
        self._alt_penalty = AltitudeDeviationPenalty()

        # ── Task state ──────────────────────────────────────────────────────
        self._difficulty = float(np.clip(config.get("difficulty_level", 0.0), 0.0, 1.0))
        self._last_termination_reason: str = "none"

    # ══════════════════════════════════════════════════════════════════════════
    #  Space properties
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    # ══════════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, env) -> None:
        """Initialize missile state, target evasion state, and reward tracking."""
        self._last_termination_reason = "none"

        # ── Missile state ───────────────────────────────────────────────────
        self.remaining_missiles = {aid: NUM_MISSILES for aid in AGENT_IDS}
        self._last_shoot_step = {aid: -MIN_ATTACK_INTERVAL for aid in AGENT_IDS}

        # ── Reward tracking ─────────────────────────────────────────────────
        self._prev_alive = {aid: True for aid in AGENT_IDS}
        self._prev_missile_count = {aid: NUM_MISSILES for aid in AGENT_IDS}
        self._has_launched_this_step: Dict[str, bool] = {aid: False for aid in AGENT_IDS}

        # ── Reward breakdown for diagnostics ────────────────────────────────
        self._reward_breakdown: Dict[str, Dict[str, float]] = {}

        # ── Reset reward modules ────────────────────────────────────────────
        for ps in env.pursuers:
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - env.targets[0].aircraft.position_ned))

        # ── Target evasion state ────────────────────────────────────────────
        self._target_base_hdg = float(env.targets[0].aircraft.state["yaw_deg"])
        self._target_evasion_step = 0

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Interpret RL action → flight targets + missile launch."""
        step = env._step_counter

        for i, (ps, aid) in enumerate(zip(env.pursuers, AGENT_IDS)):
            action = action_dict.get(aid)
            if action is None:
                continue

            if not ps.is_alive:
                continue

            speed_idx, hdg_idx, alt_idx, fire = (
                int(action[0]), int(action[1]), int(action[2]), int(action[3]))

            # ── Flight: incremental deltas on current reference ─────────────
            ps.ref_hdg = float(
                (ps.ref_hdg + DELTA_HEADINGS[hdg_idx]) % 360.0)
            ps.ref_alt_m = float(np.clip(
                ps.ref_alt_m + DELTA_ALTITUDES[alt_idx], 500.0, 8000.0))
            ps._cmd_speed = float(np.clip(
                getattr(ps, '_cmd_speed', 200.0) + DELTA_SPEEDS[speed_idx],
                120.0, 400.0))

            # ── Fire logic ──────────────────────────────────────────────────
            self._has_launched_this_step[aid] = False
            if fire == 1 and ps.is_alive and self.remaining_missiles[aid] > 0:
                self._try_launch_missile(env, ps, aid, step)

        # ── Target: rule-based evasion ──────────────────────────────────────
        for ts in env.targets:
            if ts.is_alive:
                self._update_target_evasion(env, ts)

    def step(self, env) -> None:
        """Task-level per-step: apply hit effects, check self-launched missiles."""
        # Check if any pursuer-launched missile hit
        for ps in env.pursuers:
            for m in list(ps.launch_missiles):
                if m.is_success:
                    # Mark reward — handled in get_reward
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Observation
    # ══════════════════════════════════════════════════════════════════════════

    def get_obs(self, env) -> Dict[str, dict]:
        """Build per-agent observation dicts."""
        obs_dict = {}
        target = env.targets[0]

        for ps, aid in zip(env.pursuers, AGENT_IDS):
            s = ps.aircraft.state
            t_pos = target.aircraft.position_ned
            p_pos = ps.aircraft.position_ned

            # ── Self features ──────────────────────────────────────────────
            alt_m = float(s["alt_m"])
            roll_r = math.radians(float(s["roll_deg"]))
            pitch_r = math.radians(float(s["pitch_deg"]))
            yaw_r = math.radians(float(s["yaw_deg"]))

            self_feat = np.array([
                np.clip(alt_m / MAX_ALT, 0.0, 1.0),          # 0. altitude
                math.sin(roll_r),                              # 1. roll_sin
                math.cos(roll_r),                              # 2. roll_cos
                math.sin(pitch_r),                             # 3. pitch_sin
                math.cos(pitch_r),                             # 4. pitch_cos
                float(s["u_fps"]) * 0.3048 / MAX_VEL,         # 5. v_body_x
                float(s["v_fps"]) * 0.3048 / MAX_VEL,         # 6. v_body_y
                float(s["w_fps"]) * 0.3048 / MAX_VEL,         # 7. v_body_z
                float(s["airspeed_mps"]) / MAX_VEL,           # 8. vc
                float(s["alpha_deg"]) / 30.0,                 # 9. alpha
                math.sin(yaw_r),                               # 10. heading_sin
                math.cos(yaw_r),                               # 11. heading_cos
            ], dtype=np.float32)

            # ── Target relative features ───────────────────────────────────
            los_vec = t_pos - p_pos
            dist = float(np.linalg.norm(los_vec))
            los_dir = los_vec / max(dist, 1e-6)

            p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            t_fwd = compute_forward_vector(target.aircraft.rpy_rad)
            geo = compute_tactical_angles(p_fwd, t_fwd, los_dir)

            cos_ata = geo["cos_ata"]
            ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))

            side_flag = float(np.sign(np.cross(
                p_fwd[:2], los_dir[:2])))

            target_feat = np.array([
                np.clip((float(target.aircraft.state["alt_m"]) - alt_m) / 5000.0, -1.0, 1.0),  # 0. delta_alt
                self._delta_heading_rad(ps, target),                               # 1. delta_hdg
                np.clip((float(target.aircraft.state["airspeed_mps"]) - float(s["airspeed_mps"]))
                        / 200.0, -1.0, 1.0),                                       # 2. delta_speed
                ata_deg / 180.0,                                                    # 3. ATA
                self._compute_aa_deg(p_fwd, t_fwd, los_dir) / 180.0,              # 4. AA
                np.clip(dist / MAX_DIST, 0.0, 1.0),                               # 5. distance
                side_flag,                                                          # 6. side
            ], dtype=np.float32)

            # ── Missile threat features (incoming) ──────────────────────────
            missile_feat = np.zeros(MISSILE_DIM, dtype=np.float32)
            if len(ps.under_missiles) > 0:
                m = ps.under_missiles[0]
                if m.is_alive:
                    m_pos = m.get_absolute_position()
                    m_vel = m.get_velocity()
                    m_dist = float(np.linalg.norm(m_pos - p_pos))
                    m_los = (m_pos - p_pos) / max(m_dist, 1e-6)
                    m_ata = math.degrees(math.acos(max(-1.0, min(1.0,
                        float(np.dot(p_fwd, m_los))))))
                    m_side = float(np.sign(np.cross(p_fwd[:2], m_los[:2])))

                    missile_feat = np.array([
                        np.clip((np.linalg.norm(m_vel) - float(s["airspeed_mps"]))
                                / 200.0, -1.0, 1.0),                               # 0. delta_v
                        np.clip((m_pos[2] - p_pos[2]) / 5000.0, -1.0, 1.0),       # 1. delta_alt (note: NED z=down)
                        m_ata / 180.0,                                              # 2. ATA
                        0.0,                                                        # 3. AA (placeholder)
                        np.clip(m_dist / MAX_DIST, 0.0, 1.0),                     # 4. distance
                        m_side,                                                     # 5. side
                    ], dtype=np.float32)

            obs = np.concatenate([self_feat, target_feat, missile_feat]).astype(np.float32)

            # ── Global state (both aircraft) ────────────────────────────────
            global_state = self._build_global_state(env)

            # ── Action mask ─────────────────────────────────────────────────
            mask = self.get_action_mask(env, aid)

            obs_dict[aid] = {
                "obs": obs,
                "global_state": global_state,
                "action_mask": mask,
            }

        return obs_dict

    # ══════════════════════════════════════════════════════════════════════════
    #  Reward
    # ══════════════════════════════════════════════════════════════════════════

    def get_reward(self, env) -> Dict[str, float]:
        rewards = {}
        target = env.targets[0]

        for ps, aid in zip(env.pursuers, AGENT_IDS):
            r = 0.0

            # ── Continuous shaping rewards ─────────────────────────────────
            r_progress = self._progress_reward(ps, target)
            r_ata = self._ata_reward(ps, target)
            r_alt = self._alt_reward(ps, target)

            r += r_progress + r_ata + r_alt

            # ── Valid launch reward ─────────────────────────────────────────
            r_launch = 0.0
            if self._has_launched_this_step.get(aid, False):
                # Check if launch was within valid envelope
                if self._is_valid_launch_envelope(ps, target):
                    r_launch = REWARD_VALID_LAUNCH
                else:
                    r_launch = REWARD_SHOOT_PENALTY  # bad launch
                r += r_launch

            # ── Event-driven rewards ────────────────────────────────────────
            r_event = 0.0

            # Hit: check launched missiles for success
            for m in ps.launch_missiles:
                if m.is_success:
                    r_event += REWARD_HIT

            # Shot down by enemy
            if self._prev_alive.get(aid, True) and not ps.is_alive:
                r_event += REWARD_SHOTDOWN

            r += r_event

            # ── Store breakdown for diagnostics ────────────────────────────
            self._reward_breakdown = {
                "ProgressReward": {"p0": r_progress},
                "ATAAlignmentReward": {"p0": r_ata},
                "AltitudeDeviationPenalty": {"p0": r_alt},
                "ValidLaunchReward": {"p0": r_launch},
                "EventReward": {"p0": r_event},
            }

            # Update tracking
            self._prev_alive[aid] = ps.is_alive

            rewards[aid] = r

        return rewards

    # ══════════════════════════════════════════════════════════════════════════
    #  Termination
    # ══════════════════════════════════════════════════════════════════════════

    def get_termination(self, env) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        terminateds = {}
        truncateds = {}
        infos = {}

        max_steps = 500
        low_alt = 2500.0  # meters

        for ps, aid in zip(env.pursuers, AGENT_IDS):
            s = ps.aircraft.state
            alt_m = float(s["alt_m"])

            # Purseuer shotdown
            if not ps.is_alive:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "pursuer_shotdown"}
                self._last_termination_reason = "pursuer_shotdown"
            # Low altitude
            elif alt_m < low_alt:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "low_altitude"}
                self._last_termination_reason = "low_altitude"
            # Target killed
            elif not env.targets[0].is_alive:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "target_killed"}
                self._last_termination_reason = "target_killed"
            # Lost target (too far)
            elif ps.is_alive and env.targets[0].is_alive:
                dist = float(np.linalg.norm(
                    ps.aircraft.position_ned - env.targets[0].aircraft.position_ned))
                if dist > MAX_DIST:
                    terminateds[aid] = True
                    infos[aid] = {"termination_reason": "lost_target"}
                    self._last_termination_reason = "lost_target"
                else:
                    terminateds[aid] = False
                    infos[aid] = {"termination_reason": "none"}
            else:
                terminateds[aid] = False
                infos[aid] = {"termination_reason": "none"}

            # Truncation: timeout
            truncated = env._step_counter >= max_steps
            truncateds[aid] = truncated
            if truncated:
                infos[aid]["termination_reason"] = "timeout"

        terminateds["__all__"] = any(terminateds.get(aid, False) for aid in AGENT_IDS)
        truncateds["__all__"] = all(truncateds.get(aid, False) for aid in AGENT_IDS)

        return terminateds, truncateds, infos

    # ══════════════════════════════════════════════════════════════════════════
    #  Action masking
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_mask(self, env, agent_id: str) -> np.ndarray:
        """Mask fire action when no missiles remain or target not alive."""
        mask = np.ones(N_ACTIONS, dtype=np.float32)

        # Fire action index = 10 (after speed(3) + heading(5) + altitude(3) = 11, but
        # we encode fire as the 4th MultiDiscrete dimension)
        # The mask is flat: [speed_delta(3), heading_delta(5), altitude_delta(3), fire(2)]
        # Fire is at mask indices 11 and 12 (flat index for MultiDiscrete)
        fire_start = N_SPEED_DELTA + N_HEADING_DELTA + N_ALT_DELTA  # 11

        if self.remaining_missiles.get(agent_id, 0) <= 0:
            mask[fire_start:fire_start + N_FIRE] = 0.0

        # Also mask fire if enemy not alive (no point shooting a dead target)
        # but keep this soft since scoring still matters
        return mask

    # ══════════════════════════════════════════════════════════════════════════
    #  Global state (for centralized critic)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_global_state(self, env) -> np.ndarray:
        """Concatenate all aircraft states into a flat global feature vector."""
        features = []
        for ac in list(env.pursuers) + list(env.targets):
            s = ac.aircraft.state
            yaw_r = np.radians(float(s["yaw_deg"]))
            features.extend([
                float(s["alt_m"]) / MAX_ALT,
                np.sin(np.radians(float(s["roll_deg"]))),
                np.cos(np.radians(float(s["roll_deg"]))),
                np.sin(np.radians(float(s["pitch_deg"]))),
                np.cos(np.radians(float(s["pitch_deg"]))),
                float(s["airspeed_mps"]) / MAX_VEL,
                np.sin(yaw_r),
                np.cos(yaw_r),
            ])
        return np.array(features, dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal: missile launch
    # ══════════════════════════════════════════════════════════════════════════

    def _try_launch_missile(self, env, ps, aid: str, step: int) -> None:
        """Attempt missile launch — check cooldown and create MissileSimulator."""
        # Cooldown check
        if step - self._last_shoot_step[aid] < MIN_ATTACK_INTERVAL:
            return

        # Count check
        if self.remaining_missiles[aid] <= 0:
            return

        # Find target
        if env.M == 0 or not env.targets[0].is_alive:
            return

        target = env.targets[0]
        # Match LAG's missile ID pattern: parent ID + sequential number (no underscore)
        uid = f"M{env._missile_uid_counter:03d}"
        env._missile_uid_counter += 1

        # Create and launch
        m = MissileSimulator.create(
            parent=ps, target=target, uid=uid, dt=1.0 / 60.0)
        env.add_temp_simulator(m)

        # Update state
        self.remaining_missiles[aid] -= 1
        self._last_shoot_step[aid] = step
        self._has_launched_this_step[aid] = True

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal: target evasion
    # ══════════════════════════════════════════════════════════════════════════

    def _update_target_evasion(self, env, ts) -> None:
        """Rule-based target control — gentle S-turns.

        The target simply maintains altitude and speed while varying heading
        sinusoidally. This gives the pursuer a moving target that is easy to
        track, suitable for initial training.
        """
        self._target_evasion_step += 1
        t = self._target_evasion_step * DECISION_DT

        # Gentle S-turn: ±30° oscillation, period ~20s
        d = self._difficulty
        hdg_var = d * 30.0 * math.sin(t * 0.3)

        ts.ref_hdg = float((self._target_base_hdg + hdg_var) % 360.0)
        ts.ref_alt_m = 3000.0

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal: reward helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _progress_reward(self, ps, target) -> float:
        """2D progress toward target (positive = closing, negative = separating)."""
        cur_dist_2d = float(np.linalg.norm(
            ps.aircraft.position_ned[:2] - target.aircraft.position_ned[:2]))
        prev_dist_2d = getattr(ps, 'prev_dist', cur_dist_2d)
        delta = prev_dist_2d - cur_dist_2d  # positive = closing
        dist_factor = 1.0 + max(0.0, (500.0 - cur_dist_2d) / 250.0)
        reward = delta * 0.5 * DECISION_STEPS * dist_factor
        ps.prev_dist = cur_dist_2d
        return float(reward)

    def _ata_reward(self, ps, target) -> float:
        """Nose-on-target reward."""
        p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
        los_vec = target.aircraft.position_ned - ps.aircraft.position_ned
        dist = float(np.linalg.norm(los_vec))
        los_dir = los_vec / max(dist, 1e-6)
        cos_ata = float(np.dot(p_fwd, los_dir))
        dist_factor = np.clip(1.0 - dist / MAX_DIST, 0.1, 1.0)
        return float(8.0 * cos_ata * dist_factor * DECISION_STEPS)

    def _alt_reward(self, ps, target) -> float:
        """Penalty for extreme altitude deviation from target."""
        t_alt = float(target.aircraft.state["alt_m"])
        alt_diff = abs(float(ps.aircraft.state["alt_m"]) - t_alt)
        if alt_diff > 300.0:
            return float(-0.1 * (alt_diff - 300.0) / 1000.0 * DECISION_STEPS)
        return 0.0

    def _is_valid_launch_envelope(self, ps, target) -> bool:
        """Check if current state is within valid missile launch parameters."""
        if not target.is_alive:
            return False

        p_pos = ps.aircraft.position_ned
        t_pos = target.aircraft.position_ned
        dist = float(np.linalg.norm(p_pos - t_pos))

        if dist < MIN_ATTACK_DISTANCE or dist > MAX_ATTACK_DISTANCE:
            return False

        # ATA check
        p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
        los_dir = (t_pos - p_pos) / max(dist, 1e-6)
        cos_ata = float(np.dot(p_fwd, los_dir))
        ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))

        if ata_deg > MAX_ATTACK_ANGLE:
            return False

        return True

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal: geometry
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _delta_heading_rad(ps, target) -> float:
        """Normalized heading difference between pursuer and target."""
        p_yaw = np.radians(float(ps.aircraft.state["yaw_deg"]))
        t_yaw = np.radians(float(target.aircraft.state["yaw_deg"]))
        diff = (t_yaw - p_yaw + np.pi) % (2 * np.pi) - np.pi
        return float(diff / np.pi)

    @staticmethod
    def _compute_aa_deg(p_fwd, t_fwd, los_dir) -> float:
        """Aspect Angle: target nose vs LOS in degrees."""
        cos_aa = float(np.dot(t_fwd, los_dir))
        return float(np.degrees(np.arccos(np.clip(cos_aa, -1.0, 1.0))))
