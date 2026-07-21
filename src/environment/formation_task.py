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
    Observation: Dict(obs=Box(39), global_state=Box(21), action_mask=Box(8))
    Action: MultiDiscrete([5 turn, 3 speed])
    """

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
            "action_mask": gym.spaces.Box(0.0, 1.0, (N_ACTIONS,), dtype=np.float32),
        })
        single_act = gym.spaces.MultiDiscrete([N_TURN, N_SPEED])

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

        # Post-warmup: init prev_dist for each pursuer
        for ps in env.pursuers:
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - env.targets[0].aircraft.position_ned))
            ps.episode_start_dist = ps.prev_dist

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Parse discrete tactical primitives → set PID targets on each pursuer."""
        dt = PHYSICS_DT
        actions = {}
        for i, aid in enumerate(self._agent_ids):
            a = action_dict.get(aid, np.array([2, 1], dtype=np.int64))
            a = np.asarray(a, dtype=np.int64)
            turn_idx = int(np.clip(a[0], 0, N_TURN - 1))
            speed_idx = int(np.clip(a[1], 0, N_SPEED - 1))
            turn_rates = _get_turn_rates(speed_idx)
            actions[aid] = {
                'turn_idx': turn_idx,
                'speed_idx': speed_idx,
                'cmd_turn_rate': turn_rates[turn_idx],
                'cmd_speed': SPEEDS[speed_idx],
            }
        self._last_actions = actions

        for i, (ps, aid) in enumerate(zip(env.pursuers, self._agent_ids)):
            ac = actions[aid]
            ps.ref_hdg = (ps.ref_hdg + ac['cmd_turn_rate'] * dt) % 360.0
            ps._cmd_speed = ac['cmd_speed']  # stored for physics loop

    def step(self, env) -> None:
        """Task-level per-decision-step logic (nothing to do for formation task)."""
        self._step_counter += 1

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        """Build per-agent Dict observation with local obs, global_state, action_mask."""
        target_pos = env.targets[0].aircraft.position_ned
        target_vel = env.targets[0].aircraft.velocity_ned

        obs = {}
        for i, (ps, aid) in enumerate(zip(env.pursuers, self._agent_ids)):
            local = self._build_local_obs(i, ps, env, target_pos, target_vel)
            mask = self._build_action_mask(ps)

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

    def _build_local_obs(self, i: int, ps, env, target_pos, target_vel) -> np.ndarray:
        """Build 39-dim local observation for one pursuer.

        Segments: Self(15) + Target(14) + Mate(10) = 39
        """
        p_pos = ps.aircraft.position_ned
        p_vel = ps.aircraft.velocity_ned
        p_yaw = float(ps.aircraft.state["yaw_deg"])
        p_airspeed = float(ps.aircraft.state["airspeed_mps"])
        p_alt_m = float(ps.aircraft.state["alt_m"])
        p_roll = float(ps.aircraft.state["roll_deg"])
        p_pitch = float(ps.aircraft.state["pitch_deg"])
        p_alpha = float(ps.aircraft.state["alpha_deg"])

        mate_idx = 1 - i
        mate = env.pursuers[mate_idx]
        m_pos = mate.aircraft.position_ned
        m_vel = mate.aircraft.velocity_ned
        m_yaw = float(mate.aircraft.state["yaw_deg"])
        m_airspeed = float(mate.aircraft.state["airspeed_mps"])
        m_alt_m = float(mate.aircraft.state["alt_m"])

        # Target relative (body-frame)
        delta_pos = target_pos - p_pos
        cos_y, sin_y = np.cos(np.radians(p_yaw)), np.sin(np.radians(p_yaw))
        rel_x = cos_y * delta_pos[0] + sin_y * delta_pos[1]
        rel_y = -sin_y * delta_pos[0] + cos_y * delta_pos[1]
        rel_z_body = target_pos[2] - p_pos[2]

        # Own velocity (body-frame)
        vn, ve, vd = p_vel[0], p_vel[1], p_vel[2]
        vx_body = cos_y * vn + sin_y * ve
        vy_body = -sin_y * vn + cos_y * ve

        # Target velocity (body-frame)
        tvn, tve, tvd = target_vel[0], target_vel[1], target_vel[2]
        tvx_body = cos_y * tvn + sin_y * tve
        tvy_body = -sin_y * tvn + cos_y * tve

        # Tactical geometry
        a_pos = p_pos
        a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
        t_fwd = compute_forward_vector(env.targets[0].aircraft.rpy_rad)
        _, los_dir, _ = compute_los(a_pos, target_pos)
        geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
        cos_ata = geo.get('cos_ATA', 0.0)
        cos_aa = geo.get('cos_AA', 0.0)
        cos_hca = geo.get('cos_HCA', 0.0)
        closing = geo.get('closing_speed_mps', 0.0)
        los_rate = geo.get('LOS_rate', 0.0)

        # Bearing error
        target_bearing = float(np.degrees(np.arctan2(delta_pos[1], delta_pos[0]))) % 360.0
        bearing_err = (target_bearing - p_yaw + 180.0) % 360.0 - 180.0

        # Mate relative (body-frame)
        m_rel = m_pos - p_pos
        m_rel_x = cos_y * m_rel[0] + sin_y * m_rel[1]
        m_rel_y = -sin_y * m_rel[0] + cos_y * m_rel[1]
        m_rel_z = m_pos[2] - p_pos[2]
        m_vx = cos_y * (m_vel[0] - vn) + sin_y * (m_vel[1] - ve)
        m_vy = -sin_y * (m_vel[0] - vn) + cos_y * (m_vel[1] - ve)
        m_vz = m_vel[2] - vd

        # Mate broadcast: last action of mate
        mate_action = self._last_actions.get(env._agent_ids[mate_idx] if hasattr(env, '_agent_ids') else self._agent_ids[mate_idx],
                                              {'turn_idx': 2, 'speed_idx': 1})
        mate_turn_onehot = np.zeros(N_TURN)
        mate_turn_onehot[mate_action['turn_idx']] = 1.0
        mate_speed_onehot = np.zeros(N_SPEED)
        mate_speed_onehot[mate_action['speed_idx']] = 1.0

        # Agent identity one-hot
        agent_onehot = np.zeros(2)
        agent_onehot[i] = 1.0

        obs = np.array([
            # Self (15)
            rel_x / MAX_DIST, rel_y / MAX_DIST, rel_z_body / MAX_HEIGHT,
            vx_body / MAX_VEL, vy_body / MAX_VEL, vd / MAX_VEL,
            np.sin(np.radians(p_roll)), np.cos(np.radians(p_roll)),
            np.sin(np.radians(p_pitch)), np.cos(np.radians(p_pitch)),
            p_alt_m / MAX_HEIGHT, p_airspeed / MAX_VEL, p_alpha / 30.0,
            bearing_err / 180.0, los_rate / 0.5,
            # Target (14)
            tvx_body / MAX_VEL, tvy_body / MAX_VEL, tvd / MAX_VEL,
            cos_ata, cos_aa, cos_hca,
            delta_pos[2] / MAX_HEIGHT, closing / MAX_VEL,
            np.linalg.norm(delta_pos) / MAX_DIST,
            target_vel[0] / MAX_VEL, target_vel[1] / MAX_VEL, target_vel[2] / MAX_VEL,
            self._difficulty, 0.0,  # pad to 14
            # Mate (10)
            m_rel_x / MAX_DIST, m_rel_y / MAX_DIST, m_rel_z / MAX_HEIGHT,
            m_vx / MAX_VEL, m_vy / MAX_VEL, m_vz / MAX_VEL,
            *mate_turn_onehot[:5], *mate_speed_onehot[:3],
            # Agent ID one-hot (2)
            *agent_onehot,
        ], dtype=np.float32)

        # Clip to [-1, 1]
        return np.clip(obs, -1, 1)

    def _build_action_mask(self, ps) -> np.ndarray:
        """Build action mask [8] based on flight safety."""
        mask = np.ones(N_ACTIONS, dtype=np.float32)
        airspeed = float(ps.aircraft.state["airspeed_mps"])
        alt_m = float(ps.aircraft.state["alt_m"])

        if airspeed < ANTI_STALL_SPEED_WARN:
            mask[5] = 0.0
            mask[0] = 0.0
            mask[4] = 0.0
        if alt_m < 200.0:
            mask[0] = 0.0
            mask[4] = 0.0
        if airspeed > 0.95 * 320.0:
            mask[7] = 0.0

        return mask

    # ── Reward ──────────────────────────────────────────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        """Compute per-agent rewards for the current decision step.

        Returns a dict of {agent_id: scalar_reward}.
        The actual reward accumulation happens per micro-step inside BaseEnv,
        but the cooperative shaping is computed per decision step here.
        """
        rewards = {aid: 0.0 for aid in self._agent_ids}

        for i, (ps, aid) in enumerate(zip(env.pursuers, self._agent_ids)):
            t_pos = env.targets[0].aircraft.position_ned
            a_pos = ps.aircraft.position_ned
            cur_dist = float(np.linalg.norm(a_pos - t_pos))
            delta = ps.prev_dist - cur_dist

            # Progress
            rewards[aid] += REWARD_PROGRESS * delta * 0.5
            if cur_dist < 500.0:
                rewards[aid] += REWARD_PROGRESS * delta * 5.0

            # ATA
            a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            t_fwd = compute_forward_vector(env.targets[0].aircraft.rpy_rad)
            _, los_dir, _ = compute_los(a_pos, t_pos)
            geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
            cos_ata = geo.get('cos_ATA', 0.0)
            dist_factor = np.clip(1.0 - cur_dist / MAX_DIST, 0.1, 1.0)
            rewards[aid] += REWARD_ATA * cos_ata * dist_factor

            # Proximity tiers
            for tier_dist, bonus in PROXIMITY_TIERS:
                if cur_dist < tier_dist and tier_dist not in ps.proximity_awarded:
                    rewards[aid] += bonus
                    ps.proximity_awarded.add(tier_dist)

            # Step penalty
            rewards[aid] -= STEP_PENALTY

            ps.prev_dist = cur_dist

        # ── Pincer shaping (per decision step) ──────────────────────────
        p0_pos = env.pursuers[0].aircraft.position_ned
        p1_pos = env.pursuers[1].aircraft.position_ned
        t_pos = env.targets[0].aircraft.position_ned
        los0 = t_pos - p0_pos
        los1 = t_pos - p1_pos
        d0 = float(np.linalg.norm(los0))
        d1 = float(np.linalg.norm(los1))
        cos_pincer = np.dot(los0, los1) / max(d0 * d1, 1e-6)
        pincer_angle = float(np.degrees(np.arccos(np.clip(cos_pincer, -1, 1))))

        if self._coop_phase == COOP_PHASE_AND:
            and_dist = self._and_dist
            if d0 < and_dist and d1 < and_dist:
                pincer_reward = PINCER_SHAPING_COEFF * min(pincer_angle, self._and_angle) * DECISION_DT
                rewards["p0"] += pincer_reward
                rewards["p1"] += pincer_reward

        # ── Distance asymmetry penalty ───────────────────────────────────
        dist_diff = abs(d0 - d1)
        if dist_diff > DIST_ASYMMETRY_THRESH:
            penalty = DIST_ASYMMETRY_WEIGHT * (dist_diff - DIST_ASYMMETRY_THRESH) / DIST_ASYMMETRY_NORM * DECISION_DT
            rewards["p0"] -= penalty
            rewards["p1"] -= penalty

        # Stored for termination check
        self._last_pincer = pincer_angle
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
        truncateds = {aid: False for aid in self._agent_ids}
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
