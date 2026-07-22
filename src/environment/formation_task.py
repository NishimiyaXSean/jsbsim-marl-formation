"""FormationTask — 2v1 cooperative formation pursuit for CTDE MAPPO.

Extracted from FormationRLlibEnv. Contains all task-specific logic:
  - Observation assembly (Self/Target/Mate token split + global state)
  - Reward computation (progress, pincer, ATA, OOC, loiter, collision, etc.)
  - Termination conditions (timeout, envelope, cooperative success)
  - Action masking (stall, GPWS, overspeed)
  - Curriculum state machine (Stage 1/2/3, AND-gate annealing)
  - Cooperative phases (OR-gate → AND-gate)

Compatible with BaseEnv — receives env.pursuers / env.targets for state access.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.dynamics.flight_controller import FlightControlTargets
from src.utils.units import kts_to_mps
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles

from .task_base import BaseTask


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.2
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 12

MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0

# Reward weights
REWARD_PROGRESS = 1.0
REWARD_ATA = 8.0
REWARD_SUCCESS = 5000.0
REWARD_CRASH = -3000.0
REWARD_LOST_TARGET = -3000.0
REWARD_OOB = -3000.0
REWARD_TIMEOUT = -500.0
REWARD_OR_FALLBACK = 1000.0
STEP_PENALTY = 0.25
ANTI_STALL_WINDOW = 35
ANTI_STALL_MIN_VC = 15.0
ANTI_STALL_MIN_DIST = 200.0
ANTI_STALL_PENALTY = 200.0
ANTI_STALL_SPEED_WARN = 130.0
ANTI_STALL_SPEED_WARN_WEIGHT = 1.0
REWARD_CLOSURE_RATE = 6.0
CLOSURE_RATE_NORM = 30.0
PROXIMITY_TIERS = [(800.0, 25.0), (500.0, 50.0), (300.0, 100.0)]

ATA_DEGRADATION_THRESH = 20.0
ATA_DEGRADATION_WEIGHT = 1.0
TERMINAL_PULL_MAX = 50.0

FORMATION_COLLISION_DIST = 100.0
FORMATION_COLLISION_PENALTY = -3000.0
COLLISION_SHAPING_WEIGHT = 10.0

PINCER_SHAPING_COEFF = 35.0
PINCER_DIST_MAX = 2000.0

COOP_PHASE_OR = 0
COOP_PHASE_AND = 1
COOP_PHASE1_OR_DIST = 200.0
COOP_PHASE2_AND_DIST = 800.0
COOP_PHASE2_AND_DIST_INIT = 2000.0
COOP_PHASE2_AND_ANGLE = 30.0
COOP_SUSTAIN_STEPS = 6

LOST_PURSUER_DIST = 6000.0
LOST_PURSUER_STEPS = 30

OOC_MARGIN = 400.0
OOC_PENALTY_STEPS = 30
OOC_PENALTY_PER_STEP = 2.0

ENHANCED_LOITER_PENALTY = 10.0

STRIKER_TRACKING_BONUS = 1.5
INTERCEPTOR_PINCER_BONUS = 2.0

ASYMMETRIC_RESET_PROB = 0.7
ASYMMETRIC_DIST_FAR = 1500.0
ASYMMETRIC_HEADING_OFF = 120.0

DIST_ASYMMETRY_THRESH = 800.0
DIST_ASYMMETRY_WEIGHT = 0.3
DIST_ASYMMETRY_NORM = 1000.0

SYNC_PACING_STRIKER_DIST = 1200.0
SYNC_PACING_INTERCEPTOR_DIST = 1500.0
SYNC_PACING_WEIGHT = 0.5

GLOBAL_DIM_PER_AIRCRAFT = 7
OBS_PER_PURSUER = 39

SPEEDS = [180.0, 250.0, 320.0]
N_SPEED = 3

_TURN_SCALE = {0: 1.33, 1: 1.0, 2: 0.8}


def _get_turn_rates(speed_idx: int) -> list[float]:
    base = [-15.0, -5.0, 0.0, 5.0, 15.0]
    scale = _TURN_SCALE.get(speed_idx, 1.0)
    return [r * scale for r in base]


TURN_RATES = _get_turn_rates(1)
N_TURN = 5
N_ACTIONS = N_TURN + N_SPEED  # 8

# ═══════════════════════════════════════════════════════════════════════════════
#  FormationTask
# ═══════════════════════════════════════════════════════════════════════════════


class FormationTask(BaseTask):
    """2v1 cooperative formation pursuit — token-based CTDE observation.

    Agent IDs: "p0", "p1"
    Observation: Dict(obs=Box(39), global_state=Box(21), action_mask=Box(11))
    Action: MultiDiscrete([3 speed_delta, 5 heading_delta, 3 altitude_delta]) = 45
      → interpreted as incremental FlightTarget applied to current aircraft state
    """

    # ── Hierarchical action space: tactical deltas (LAG-compatible ranges) ──
    DELTA_SPEEDS     = [-20.0,   0.0,  20.0]      # m/s: decelerate / hold / accelerate
    DELTA_HEADINGS   = [-30.0, -15.0, 0.0, 15.0, 30.0]  # degrees
    DELTA_ALTITUDES  = [-100.0,   0.0, 100.0]     # meters: descend / hold / climb

    N_SPEED_DELTA  = len(DELTA_SPEEDS)     # 3
    N_HEADING_DELTA = len(DELTA_HEADINGS)  # 5
    N_ALT_DELTA    = len(DELTA_ALTITUDES)  # 3
    N_HIGH_ACTIONS = N_SPEED_DELTA + N_HEADING_DELTA + N_ALT_DELTA  # 11-dim mask

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._agent_ids = ["p0", "p1"]
        self.N = 2
        self.M = 1

        # ── Spaces ──────────────────────────────────────────────────────
        self._global_dim = (self.N + self.M) * GLOBAL_DIM_PER_AIRCRAFT  # 21

        single_obs = gym.spaces.Dict({
            "obs": gym.spaces.Box(-1.0, 1.0, (OBS_PER_PURSUER,), dtype=np.float32),
            "global_state": gym.spaces.Box(-1.0, 1.0, (self._global_dim,), dtype=np.float32),
            "action_mask": gym.spaces.Box(0.0, 1.0, (self.N_HIGH_ACTIONS,), dtype=np.float32),
        })
        # Hierarchical: [speed_delta(3), heading_delta(5), altitude_delta(3)]
        single_act = gym.spaces.MultiDiscrete(
            [self.N_SPEED_DELTA, self.N_HEADING_DELTA, self.N_ALT_DELTA])

        self._observation_space = gym.spaces.Dict({
            aid: single_obs for aid in self._agent_ids
        })
        self._action_space = gym.spaces.Dict({
            aid: single_act for aid in self._agent_ids
        })

        # ── Task state (moved from env) ─────────────────────────────────
        self._difficulty = float(np.clip(self.config.get("difficulty_level", 0.0), 0.0, 1.0))
        self.cooperative_mode = self.config.get("cooperative_mode", True)
        self._striker_idx: int = 0
        self._coop_sustain_counter: int = 0
        self._sustain_required: int = COOP_SUSTAIN_STEPS
        self._coop_phase: int = COOP_PHASE_OR
        self._and_dist: float = COOP_PHASE2_AND_DIST_INIT
        self._and_angle: float = COOP_PHASE2_AND_ANGLE
        self._init_bearing_range: tuple = (-180.0, 180.0)
        self._target_dist_range: tuple = (900.0, 1300.0)
        self._curriculum_stage: int = 0
        self._last_termination_reason: str = "none"
        self._reward_breakdown: dict = {}
        self._lost_pursuer_steps: int = 0
        self._ooc_counters: list = [0, 0]
        self._or_triggered: list = [False, False]
        self._last_actions: dict = {}

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self, env) -> None:
        """Reset cooperative state, OOC counters, and other per-episode state."""
        self._step_counter = 0
        self._coop_sustain_counter = 0
        self._ooc_counters = [0, 0]
        self._or_triggered = [False, False]
        self._last_termination_reason = "none"
        self._lost_pursuer_steps = 0
        self._last_asymmetric = getattr(env, '_last_asymmetric', False)
        self._last_disadvantaged = getattr(env, '_last_disadvantaged', 0)

        # Post-warmup: init prev_dist + sync PID references for each pursuer
        for ps in env.pursuers:
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - env.targets[0].aircraft.position_ned))
            ps.episode_start_dist = ps.prev_dist
            # Sync PID refs to actual state → prevents step-0 control transients
            s = ps.aircraft.state
            ps.ref_hdg = float(s["yaw_deg"])
            ps.ref_alt_m = float(s["alt_m"])
            ps._cmd_speed = float(s["airspeed_mps"])

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Map high-level tactical actions → FlightTarget for each pursuer.

        Action format: [speed_delta_idx, heading_delta_idx, altitude_delta_idx]
        The FlightTarget is computed as: current_state + delta, then clamped.
        The actual control surface deflections are produced by the aircraft's
        controller (PID or Neural) inside the physics loop.

        Stores _last_actions as FlightTarget-compatible dicts for mate broadcast.
        """
        actions = {}
        for aid in self._agent_ids:
            a = action_dict.get(aid, np.array([1, 2, 1], dtype=np.int64))
            a = np.asarray(a, dtype=np.int64)
            speed_idx = int(np.clip(a[0], 0, self.N_SPEED_DELTA - 1))
            heading_idx = int(np.clip(a[1], 0, self.N_HEADING_DELTA - 1))
            alt_idx = int(np.clip(a[2], 0, self.N_ALT_DELTA - 1))
            actions[aid] = {
                'delta_speed': self.DELTA_SPEEDS[speed_idx],
                'delta_heading': self.DELTA_HEADINGS[heading_idx],
                'delta_altitude': self.DELTA_ALTITUDES[alt_idx],
                'speed_idx': speed_idx,
                'heading_idx': heading_idx,
                'alt_idx': alt_idx,
            }
        self._last_actions = actions

        for ps, aid in zip(env.pursuers, self._agent_ids):
            ac = actions[aid]
            s = ps.aircraft.state
            current_hdg = float(s["yaw_deg"])
            current_alt = float(s["alt_m"])
            current_spd = float(s["airspeed_mps"])

            # Compute incremental target
            target_hdg = (current_hdg + ac['delta_heading']) % 360.0
            target_alt = np.clip(current_alt + ac['delta_altitude'], 100.0, 5000.0)
            target_spd = np.clip(current_spd + ac['delta_speed'], 100.0, 380.0)

            # Store on pursuer for physics loop + broadcast
            ps.ref_hdg = target_hdg
            ps.ref_alt_m = target_alt
            ps._cmd_speed = target_spd

    def step(self, env) -> None:
        """Task-level per-decision-step logic (nothing to do for formation task)."""
        self._step_counter += 1

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        """Build per-agent Dict observation with 39-dim tokens + 21-dim global + mask.

        Returns format strictly aligned with AttentionFormationActor's expected input:
          - "obs":          Box(39) — Self/Target/Mate token features
          - "global_state": Box(21) — ego-centric [Self, Mate, Target] for critic
          - "action_mask":  Box(11) — high-level safety mask (speed_delta + heading_delta + alt_delta)
        """
        target_pos = env.targets[0].aircraft.position_ned
        target_vel = env.targets[0].aircraft.velocity_ned

        obs = {}
        for i, (ps, aid) in enumerate(zip(env.pursuers, self._agent_ids)):
            local = self._build_local_obs(i, ps, env, target_pos, target_vel)
            mask = self._build_high_level_action_mask(ps)

            # Ego-centric global state: [Self, Mate, Target]
            mate_idx = 1 - i
            ego_pursuers = [env.pursuers[i], env.pursuers[mate_idx]]
            global_parts = []
            for p_agent in ego_pursuers:
                p = p_agent.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
                v = p_agent.aircraft.velocity_ned / MAX_VEL
                h = np.array([float(p_agent.aircraft.state["yaw_deg"]) / 180.0])
                global_parts.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
            for ts in env.targets:
                p = ts.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
                v = ts.aircraft.velocity_ned / MAX_VEL
                h = np.array([float(ts.aircraft.state["yaw_deg"]) / 180.0])
                global_parts.extend(np.clip(np.concatenate([p, v, h]), -1, 1))

            obs[aid] = {
                "obs": local.astype(np.float32),
                "global_state": np.array(global_parts, dtype=np.float32),
                "action_mask": mask.astype(np.float32),
            }
        return obs

    def _build_high_level_action_mask(self, ps) -> np.ndarray:
        """Build high-level action mask [11] for tactical deltas.

        Layout: [speed_0, speed_1, speed_2, heading_0, ..., heading_4,
                 altitude_0, altitude_1, altitude_2]
        1=allowed, 0=forbidden.
        """
        mask = np.ones(self.N_HIGH_ACTIONS, dtype=np.float32)
        airspeed = float(ps.aircraft.state["airspeed_mps"])
        alt_m = float(ps.aircraft.state["alt_m"])

        # Speed delta mask: indices 0..2
        if airspeed < ANTI_STALL_SPEED_WARN:
            mask[0] = 0.0  # forbid decelerate (would stall)
        if airspeed > MAX_VEL * 0.95:
            mask[2] = 0.0  # forbid accelerate (would overspeed)

        # Altitude delta mask: indices 8..10
        if alt_m < 200.0:
            mask[8] = 0.0  # forbid descend

        # Heading deltas (indices 3..7) are always allowed — turning is safe

        return mask

    def _build_local_obs(self, idx: int, ps, env, target_pos, target_vel) -> np.ndarray:
        """39-dim per-pursuer local observation (matches FormationRLlibEnv exactly).

        Layout: base(27) + agent_onehot(2) + mate(10) = 39
        """
        a_pos = ps.aircraft.position_ned
        a_rpy = ps.aircraft.rpy_rad
        a_vel = ps.aircraft.velocity_ned

        # Body-frame transforms
        rel_w = target_pos - a_pos
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
            target_vel[0] * ch + target_vel[1] * sh,
            -target_vel[0] * sh + target_vel[1] * ch,
            target_vel[2],
        ])

        ang_vel = self._ang_vel(ps.aircraft.rpy_rad, ps.prev_rpy)
        ps.prev_rpy = ps.aircraft.rpy_rad.copy()

        a_fwd = compute_forward_vector(a_rpy)
        t_fwd = compute_forward_vector(env.targets[0].aircraft.rpy_rad)
        _, los_dir, _ = compute_los(a_pos, target_pos)
        geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)

        spd = float(ps.aircraft.state["airspeed_mps"])
        alpha = float(ps.aircraft.state["alpha_deg"])

        # LOS rate
        r_h = target_pos[:2] - a_pos[:2]
        dh = float(np.linalg.norm(r_h))
        if dh > 1.0:
            v_rel_h = target_vel[:2] - a_vel[:2]
            lambda_dot = float(np.cross(r_h, v_rel_h)) / (dh * dh)
            lambda_dot_norm = float(np.clip(lambda_dot / 0.5, -1, 1))
        else:
            lambda_dot_norm = 0.0

        bearing = float(np.degrees(np.arctan2(r_h[1], r_h[0]))) % 360.0
        hdg = float(ps.aircraft.state["yaw_deg"]) % 360.0
        berr = (bearing - hdg + 180) % 360 - 180
        berr_norm = float(np.clip(berr / 180.0, -1, 1))

        # Base observation (indices 0-26, same as FormationEnv)
        base = np.array([
            rel_body[0] / MAX_DIST, rel_body[1] / MAX_DIST,
            rel_body[2] / MAX_DIST,
            vel_body[0] / MAX_VEL, vel_body[1] / MAX_VEL, vel_body[2] / MAX_VEL,
            a_rpy[0] / np.pi, a_rpy[1] / (np.pi / 2), a_rpy[2] / np.pi,
            ang_vel[0] / np.pi, ang_vel[1] / np.pi,
            ang_vel[2] / np.pi,
            a_pos[2] / MAX_HEIGHT,
            t_vel_body[0] / MAX_VEL, t_vel_body[1] / MAX_VEL,
            t_vel_body[2] / MAX_VEL,
            0.0, 0.0, 0.0,  # target ang_vel placeholder
            geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
            alpha / 30.0, spd / MAX_VEL, 0.0,  # Ps placeholder
            lambda_dot_norm, berr_norm,
        ], dtype=np.float32)

        # Agent one-hot ID
        agent_onehot = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float32)
        base = np.concatenate([base, agent_onehot])  # 27→29

        # Mate observation (indices 27-38, total 10 dims)
        if self.N >= 2:
            mate_idx = 1 - idx
            mp = env.pursuers[mate_idx].aircraft.position_ned
            mv = env.pursuers[mate_idx].aircraft.velocity_ned
            mrw = mp - a_pos
            mrv = mv - a_vel
            mate_body_pos = np.array([
                mrw[0] * ch + mrw[1] * sh,
                -mrw[0] * sh + mrw[1] * ch,
                -mrw[2],
            ])
            mate_body_vel = np.array([
                mrv[0] * ch + mrv[1] * sh,
                -mrv[0] * sh + mrv[1] * ch,
                mrv[2],
            ])
            mate_obs = np.array([
                mate_body_pos[0] / MAX_DIST, mate_body_pos[1] / MAX_DIST,
                mate_body_pos[2] / MAX_DIST,
                mate_body_vel[0] / MAX_VEL, mate_body_vel[1] / MAX_VEL,
                mate_body_vel[2] / MAX_VEL,
            ], dtype=np.float32)

            # Broadcast: mate's hierarchical tactical intent (4 dims)
            # _agent_ids may be a list or a set (RLlib compatibility)
            if hasattr(env, '_agent_ids'):
                agent_id_list = list(env._agent_ids)
                mate_aid = agent_id_list[mate_idx]
            else:
                mate_aid = self._agent_ids[mate_idx]
            mate_act = self._last_actions.get(mate_aid, {})
            mate_delta_hdg = mate_act.get('delta_heading', 0.0) / 30.0   # [-1, 1]
            mate_delta_spd = mate_act.get('delta_speed', 0.0) / 20.0      # [-1, 1]
            mate_ref_hdg = np.deg2rad(env.pursuers[mate_idx].ref_hdg)
            mate_broadcast = np.array([
                mate_delta_hdg,
                mate_delta_spd,
                np.cos(mate_ref_hdg),
                np.sin(mate_ref_hdg),
            ], dtype=np.float32)
            mate = np.concatenate([mate_obs, mate_broadcast])
        else:
            mate = np.zeros(10, dtype=np.float32)

        return np.clip(np.concatenate([base, mate]), -1, 1)

    def _ang_vel(self, cur, prev):
        d = cur - prev
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return d / PHYSICS_DT

    # ── Reward ──────────────────────────────────────────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        """Compute per-agent rewards for the current decision step.

        Note: The original FormationRLlibEnv accumulates rewards inside the
        12-step physics loop. Since BaseEnv delegates physics to itself,
        the per-step rewards here are multiplied by DECISION_STEPS (12)
        to produce approximately equivalent total rewards.
        """
        dt = PHYSICS_DT
        rewards = {aid: 0.0 for aid in self._agent_ids}

        for i, (ps, aid) in enumerate(zip(env.pursuers, self._agent_ids)):
            t_pos = env.targets[0].aircraft.position_ned
            a_pos = ps.aircraft.position_ned
            cur_dist = float(np.linalg.norm(a_pos - t_pos))
            delta = ps.prev_dist - cur_dist

            # Progress (scaled by DECISION_STEPS to match per-micro-step accumulation)
            rewards[aid] += REWARD_PROGRESS * delta * 0.5 * DECISION_STEPS
            if cur_dist < 500.0:
                rewards[aid] += REWARD_PROGRESS * delta * 5.0 * DECISION_STEPS

            # ATA
            a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            t_fwd = compute_forward_vector(env.targets[0].aircraft.rpy_rad)
            _, los_dir, _ = compute_los(a_pos, t_pos)
            geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
            cos_ata = geo.get('cos_ata', 0.0)
            dist_factor = np.clip(1.0 - cur_dist / MAX_DIST, 0.1, 1.0)
            rewards[aid] += REWARD_ATA * cos_ata * dist_factor * DECISION_STEPS

            # Proximity tiers (only once per tier per episode)
            for tier_dist, bonus in PROXIMITY_TIERS:
                if cur_dist < tier_dist and tier_dist not in ps.proximity_awarded:
                    rewards[aid] += bonus
                    ps.proximity_awarded.add(tier_dist)

            # Step penalty
            rewards[aid] -= STEP_PENALTY * DECISION_STEPS

            ps.prev_dist = cur_dist

        # ── Pincer shaping ───────────────────────────────────────────────
        p0_pos = env.pursuers[0].aircraft.position_ned
        p1_pos = env.pursuers[1].aircraft.position_ned
        t_pos = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(p0_pos - t_pos))
        d1 = float(np.linalg.norm(p1_pos - t_pos))

        if self._coop_phase == COOP_PHASE_AND:
            and_dist = self._and_dist
            los0 = t_pos - p0_pos
            los1 = t_pos - p1_pos
            cos_pincer = np.dot(los0, los1) / max(d0 * d1, 1e-6)
            pincer_angle = float(np.degrees(np.arccos(np.clip(cos_pincer, -1, 1))))
            if d0 < and_dist and d1 < and_dist:
                pincer_reward = PINCER_SHAPING_COEFF * min(pincer_angle, self._and_angle) * dt * DECISION_STEPS
                rewards["p0"] += pincer_reward
                rewards["p1"] += pincer_reward

        # ── Distance asymmetry penalty ───────────────────────────────────
        dist_diff = abs(d0 - d1)
        if dist_diff > DIST_ASYMMETRY_THRESH:
            penalty = DIST_ASYMMETRY_WEIGHT * (dist_diff - DIST_ASYMMETRY_THRESH) / DIST_ASYMMETRY_NORM * dt * DECISION_STEPS
            rewards["p0"] -= penalty
            rewards["p1"] -= penalty

        # Stored for termination check
        self._last_pincer = pincer_angle if self._coop_phase == COOP_PHASE_AND else 0.0
        self._last_d0 = d0
        self._last_d1 = d1

        return rewards

    # ── Termination ─────────────────────────────────────────────────────────

    def get_termination(self, env) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Check termination conditions.

        Returns:
            (terminateds, truncateds, infos) — each with "__all__" key.
        """
        terminateds = {aid: False for aid in self._agent_ids}
        terminateds["__all__"] = False
        truncateds = {aid: False for aid in self._agent_ids}
        truncateds["__all__"] = False
        infos: Dict[str, Any] = {"p0": {}, "p1": {}}
        reason = "timeout"
        done_all = False

        # ── Flight envelope violations ───────────────────────────────────
        for ps, aid in zip(env.pursuers, self._agent_ids):
            alt_m = float(ps.aircraft.state["alt_m"])
            airspeed = float(ps.aircraft.state["airspeed_mps"])
            nz_g = float(ps.aircraft.state.get("n_z_g", 1.0))

            if alt_m < 100.0:
                reason = f"{aid}_low_altitude"
                done_all = True
            elif alt_m > 5000.0:
                reason = f"{aid}_high_altitude"
                done_all = True
            elif airspeed < 100.0:
                reason = f"{aid}_stall"
                done_all = True
            elif abs(nz_g) > 9.0:
                reason = f"{aid}_overload"
                done_all = True

        if done_all:
            for aid in self._agent_ids:
                terminateds[aid] = True
                terminateds["__all__"] = True
                infos[aid]["termination_reason"] = reason
            return terminateds, truncateds, infos
        # ── Timeout ──────────────────────────────────────────────────────
        if self._step_counter >= 1000:
            for aid in self._agent_ids:
                truncateds[aid] = True
                infos[aid]["termination_reason"] = "timeout"
            truncateds["__all__"] = True
            return terminateds, truncateds, infos

        # ── Cooperative success checks ───────────────────────────────────
        d0 = self._last_d0 if hasattr(self, '_last_d0') else 9999.0
        d1 = self._last_d1 if hasattr(self, '_last_d1') else 9999.0
        pincer = self._last_pincer if hasattr(self, '_last_pincer') else 0.0

        if self._coop_phase == COOP_PHASE_OR:
            if d0 < COOP_PHASE1_OR_DIST or d1 < COOP_PHASE1_OR_DIST:
                reason = "cooperative_success_or"
                done_all = True
        elif self._coop_phase == COOP_PHASE_AND:
            if d0 < self._and_dist and d1 < self._and_dist and pincer > self._and_angle:
                self._coop_sustain_counter += 1
                if self._coop_sustain_counter >= self._sustain_required:
                    reason = "cooperative_success_and"
                    done_all = True
            else:
                self._coop_sustain_counter = 0

        if done_all:
            for aid in self._agent_ids:
                terminateds[aid] = True
                infos[aid]["termination_reason"] = reason
            terminateds["__all__"] = True

        return terminateds, truncateds, infos
