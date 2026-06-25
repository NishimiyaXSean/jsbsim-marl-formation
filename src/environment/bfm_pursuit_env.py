"""Single-agent pursuit with discrete BFM action space.

The RL agent selects from 9 basic fighter maneuvers (PURSUIT_ACTIONS).
Each action is a tuple ``(n_x, n_n, mu)`` fed through FlightEnvelope →
BFMAutopilot → JSBSim FCS → control surfaces.

This replaces the continuous [d_heading, d_alt, d_speed] interface with
a tactical-level command set — the agent thinks in terms of "turn left"
or "climb" rather than raw flight control targets.

Observation
-----------
25-dim, identical to SinglePursuitEnv for compatibility with existing
reward wrappers and training infrastructure.

Action
------
Discrete(9): 0-8 index into PURSUIT_ACTIONS (see src/dynamics/bfm_actions.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import AltitudeStabilizer
from src.dynamics.bfm_actions import PURSUIT_ACTIONS, describe_pursuit_action, NUM_PURSUIT_ACTIONS
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration (shared with SinglePursuitEnv for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.5            # 2 Hz — discrete BFM decisions match human pilot cadence
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 30 micro-steps per decision

MAX_EPISODE_TIME = 180.0

# Observation normalisation constants
MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi
MAX_AOA = 30.0
MAX_PS = 300.0
LOW_SPEED_THRESHOLD = 100.0

# Reward weights
REWARD_PROGRESS = 0.5
REWARD_ATA = 5.0
REWARD_GROUND_WARNING = 2.0
REWARD_SUCCESS = 2000.0
REWARD_CRASH = -200.0
REWARD_LOST_TARGET = -200.0
REWARD_LOW_SPEED_TURN = 5.0
STEP_PENALTY = 1.0
LOW_ENERGY_PENALTY = 5.0
ANTI_STALL_WINDOW = 30
ANTI_STALL_MIN_VC = 15.0
ANTI_STALL_MIN_DIST = 300.0
ANTI_STALL_PENALTY = 200.0

ZONE_DEATH_DIST_LO = 300.0
ZONE_DEATH_DIST_HI = 800.0
ZONE_DEATH_DIST_HI_SCALE = 400.0
ZONE_DEATH_MIN_VC = 15.0
ZONE_DEATH_WINDOW = 20
ZONE_DEATH_PENALTY = 50.0
VELOCITY_SHAPING_WEIGHT = 3.0
VELOCITY_SHAPING_ATA_THRESH = 0.95

PROXIMITY_TIERS = [
    (800.0, 25.0),
    (500.0, 50.0),
    (300.0, 100.0),
]


@dataclass
class TargetProfile:
    alt_m: float = 3000.0
    speed_mps: float = 180.0
    heading_deg: float = 90.0
    heading_rate_dps: float = 0.0
    alt_rate_mps: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════════

class BFMPursuitEnv(gym.Env):
    """Single-agent pursuit with Discrete(9) BFM action space.

    Action: 0-8  →  PURSUIT_ACTIONS[i] = (n_x, n_n, mu)
            →  FlightEnvelope  →  BFMAutopilot  →  JSBSim FCS
    """

    metadata = {"render_modes": ["human", "tacview"], "name": "bfm_pursuit_v0"}

    _STAGE_MAP = [1.0, 1.5, 2.0, 2.5, 3.0]

    def __init__(
        self,
        curriculum_stage: Optional[float] = None,
        difficulty_level: float = 0.0,
        jsbsim_data_dir: Optional[str] = None,
        record_tacview: bool = False,
    ):
        super().__init__()

        if curriculum_stage is not None:
            idx = next((i for i, s in enumerate(self._STAGE_MAP)
                       if np.isclose(curriculum_stage, s)), 0)
            self._difficulty = idx / (len(self._STAGE_MAP) - 1)
        else:
            self._difficulty = float(np.clip(difficulty_level, 0.0, 1.0))

        self.record_tacview = record_tacview
        self._tacview_frames: List[dict] = []
        self._ref_lla = (30.0, 120.0, 3000.0)

        # Aircraft
        self.pursuer = Aircraft(jsbsim_data_dir)
        self.target_ac = Aircraft(jsbsim_data_dir)

        # Phase 3.5 autopilot pipeline: BFM commands → surfaces
        self._autopilot = BFMAutopilot(
            BFMAutopilotConfig(),
            trim=TrimSchedule(),
            scheduler=GainScheduler(),
        )
        self._envelope = FlightEnvelope(EnvelopeConfig())

        # Target altitude stabiliser
        self._target_alt_stab = AltitudeStabilizer()

        # Observation / action spaces
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(25,), dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(NUM_PURSUIT_ACTIONS,), dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Discrete(NUM_PURSUIT_ACTIONS)

        # Episode state
        self._step_counter = 0
        self._prev_dist = 0.0
        self._proximity_awarded: set = set()
        self._prev_rpy = np.zeros(3, dtype=np.float64)
        self._prev_target_rpy = np.zeros(3, dtype=np.float64)
        self._prev_airspeed = 180.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._closure_rates: deque = deque(maxlen=ANTI_STALL_WINDOW)
        self._zone_death_counter: int = 0
        self._target_profile: Optional[TargetProfile] = None

    # ── Difficulty property ─────────────────────────────────────────────────

    @property
    def difficulty_level(self) -> float:
        return self._difficulty

    @difficulty_level.setter
    def difficulty_level(self, value: float) -> None:
        self._difficulty = float(np.clip(value, 0.0, 1.0))

    @property
    def curriculum_stage(self) -> float:
        n = len(self._STAGE_MAP) - 1
        idx = int(round(self._difficulty * n))
        return self._STAGE_MAP[min(idx, n)]

    @curriculum_stage.setter
    def curriculum_stage(self, value: float) -> None:
        idx = next((i for i, s in enumerate(self._STAGE_MAP)
                   if np.isclose(value, s)), 0)
        self._difficulty = idx / (len(self._STAGE_MAP) - 1)

    # ── Action mask ───────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return valid-action mask for current flight state.

        True = action allowed, False = action forbidden by safety rules.
        This prevents the agent from selecting self-destructive actions.
        """
        s = self.pursuer.state
        alt = s["alt_m"]
        speed = s["airspeed_mps"]
        alpha = s["alpha_deg"]

        mask = np.ones(NUM_PURSUIT_ACTIONS, dtype=bool)

        # 1. Hard deck: no pure dive below 1500 m
        if alt < 1500.0:
            mask[6] = False   # Descend

        # 2. Stall prevention: at low speed, block energy-losing actions
        if speed < 160.0:
            mask[2] = False   # Decelerate
            mask[5] = False   # Climb (bleeds speed rapidly)

        # 3. Overspeed protection: block acceleration above 350 m/s
        if speed > 350.0:
            mask[1] = False   # Accelerate
            mask[7] = False   # Accel + turn right
            mask[8] = False   # Accel + turn left

        # 4. High-alpha protection: block aggressive turns when near stall
        if alpha > 20.0:
            mask[3] = False   # Turn right
            mask[4] = False   # Turn left
            mask[5] = False   # Climb
            mask[7] = False   # Accel + turn right
            mask[8] = False   # Accel + turn left

        # 5. Safety net: guarantee at least level flight is available
        if not mask.any():
            mask[0] = True

        return mask

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)

        pursuer_hdg = rng.uniform(0.0, 360.0)
        self.pursuer.reset(
            lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
            heading_deg=pursuer_hdg, speed_kts=400, trim=False,
        )
        pursuer_ned = np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), 3000.0])
        self.pursuer.position_ned = pursuer_ned

        d = self._difficulty
        target_dist = rng.uniform(900 + d * 1100, 1300 + d * 1700)
        bearing_max = d * 45.0
        bearing_offset = rng.uniform(-bearing_max, bearing_max)
        alt_offset_max = 50.0 + d * 250.0
        target_alt_offset = rng.uniform(-alt_offset_max, alt_offset_max)
        heading_diff_max = d * 30.0
        heading_diff = rng.uniform(-heading_diff_max, heading_diff_max)

        target_bearing = (pursuer_hdg + bearing_offset) % 360.0
        target_bearing_rad = np.deg2rad(target_bearing)
        target_ned = np.array([
            pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
            pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
            pursuer_ned[2] + target_alt_offset,
        ])
        target_hdg = (pursuer_hdg + heading_diff) % 360.0

        self.target_ac.reset(
            lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
            heading_deg=target_hdg, speed_kts=310, trim=False,
        )
        self.target_ac.position_ned = target_ned

        self._autopilot.reset(initial_speed_mps=200.0)
        self._envelope.reset()
        self._target_alt_stab.reset()

        self._step_counter = 0

        # Warmup: 3s at trim
        warmup_steps = int(3.0 * CTRL_FREQ)
        for _ in range(warmup_steps):
            s = self.pursuer.state
            thr, elev, ail, rud = self._autopilot.step(
                0.0, 1.0, 0.0, PHYSICS_DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"], q_rps=s["q_rps"],
            )
            self.pursuer.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            self.target_ac.set_controls(throttle=0.80, elevator=-0.05, aileron=0.0, rudder=0.0)
            self.pursuer.run()
            self.target_ac.run()
            self.pursuer.position_ned[0:2] += self.pursuer.velocity_ned[0:2] * PHYSICS_DT
            self.target_ac.position_ned[0:2] += self.target_ac.velocity_ned[0:2] * PHYSICS_DT
            self.pursuer.position_ned[2] = self.pursuer.state["alt_m"]
            self.target_ac.position_ned[2] = self.target_ac.state["alt_m"]

        self._prev_dist = float(np.linalg.norm(
            self.pursuer.position_ned - self.target_ac.position_ned))
        self._proximity_awarded.clear()
        self._tacview_frames = []
        self._prev_rpy = self.pursuer.rpy_rad.copy()
        self._prev_target_rpy = self.target_ac.rpy_rad.copy()
        self._prev_airspeed = float(self.pursuer.state["airspeed_mps"])
        self._closure_rates.clear()
        self._zone_death_counter = 0
        self._target_profile = self._generate_target_profile(rng, target_hdg,
                                                            spawn_alt_m=float(target_ned[2]))

        if self.record_tacview:
            self._record_tacview_frame(0.0)

        return self._get_obs(), {}

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        dt = PHYSICS_DT

        # Map discrete action → BFM command
        n_x_raw, n_n_raw, mu_raw = PURSUIT_ACTIONS.get(int(action), PURSUIT_ACTIONS[0])

        _airspeed = float(self.pursuer.state["airspeed_mps"])
        _target_spd = float(self.target_ac.state.get("airspeed_mps", 180.0))
        _energy_ok = _airspeed >= _target_spd
        self._last_action = np.array([n_x_raw, n_n_raw, mu_raw], dtype=np.float32)

        total_reward = 0.0
        _r_progress = 0.0
        _r_terminal_boost = 0.0
        _r_ata = 0.0
        _r_time_pressure = 0.0
        _r_ground_warning = 0.0
        _r_proximity = 0.0
        _r_low_speed_penalty = 0.0
        _r_step_penalty = 0.0

        total_reward -= STEP_PENALTY
        _r_step_penalty -= STEP_PENALTY

        terminated = False
        truncated = False
        reason = "timeout"
        min_dist = self._prev_dist
        start_dist = self._prev_dist

        for _ in range(DECISION_STEPS):
            # FlightEnvelope → BFMAutopilot → control surfaces
            s = self.pursuer.state
            n_x_env, n_n_env, mu_env = self._envelope.step(
                n_x_raw, n_n_raw, mu_raw,
                speed_mps=s["airspeed_mps"], alt_m=s["alt_m"],
                vz_mps=s["h_dot_fps"] * 0.3048,
                current_roll_rad=np.deg2rad(s["roll_deg"]), dt=dt,
            )
            thr, elev, ail, rud = self._autopilot.step(
                n_x_env, n_n_env, mu_env, dt,
                n_z_g=s["n_z_g"],
                roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"],
                beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"],
            )
            self.pursuer.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)

            # Move target
            self._move_target(dt)

            self.pursuer.run()
            self.target_ac.run()
            self._step_counter += 1

            self.pursuer.position_ned[0:2] += self.pursuer.velocity_ned[0:2] * dt
            self.pursuer.position_ned[2] = self.pursuer.state["alt_m"]
            self.target_ac.position_ned[0:2] += self.target_ac.velocity_ned[0:2] * dt
            self.target_ac.position_ned[2] = self.target_ac.state["alt_m"]

            # NaN guard
            if any(not np.isfinite(float(self.pursuer.state[k]))
                   for k in ["n_z_g", "airspeed_mps", "alt_m"]):
                total_reward += REWARD_CRASH
                terminated = True
                reason = "jsbsim_nan"
                break

            a_pos = self.pursuer.position_ned
            t_pos = self.target_ac.position_ned

            current_dist = float(np.linalg.norm(a_pos - t_pos))
            if current_dist < min_dist:
                min_dist = current_dist

            a_forward = compute_forward_vector(self.pursuer.rpy_rad)
            t_forward = compute_forward_vector(self.target_ac.rpy_rad)
            _, los_dir, _ = compute_los(a_pos, t_pos)
            geo = compute_tactical_angles(a_forward, t_forward, los_dir)

            delta_dist = self._prev_dist - current_dist

            prog = REWARD_PROGRESS * delta_dist * 0.5
            total_reward += prog
            _r_progress += prog
            if current_dist < 500.0:
                boost = REWARD_PROGRESS * delta_dist * 5.0
                total_reward += boost
                _r_terminal_boost += boost

            if not _energy_ok:
                le_penalty = LOW_ENERGY_PENALTY * dt
                total_reward -= le_penalty
                _r_low_speed_penalty -= le_penalty

            ata_r = REWARD_ATA * max(geo["cos_ata"], -0.2) * dt
            total_reward += ata_r
            _r_ata += ata_r

            if geo["cos_ata"] > VELOCITY_SHAPING_ATA_THRESH:
                aspd = float(self.pursuer.state["airspeed_mps"])
                vel_bonus = (aspd / MAX_VEL) * VELOCITY_SHAPING_WEIGHT * dt
                total_reward += vel_bonus
                _r_ata += vel_bonus

            time_ratio = self._step_counter / (CTRL_FREQ * MAX_EPISODE_TIME)
            tp = -0.5 * time_ratio * dt
            total_reward += tp
            _r_time_pressure += tp

            if a_pos[2] < 800.0:
                gw = -REWARD_GROUND_WARNING * dt
                total_reward += gw
                _r_ground_warning += gw

            for threshold, bonus in PROXIMITY_TIERS:
                if current_dist < threshold and threshold not in self._proximity_awarded:
                    total_reward += bonus
                    _r_proximity += bonus
                    self._proximity_awarded.add(threshold)

            self._prev_dist = current_dist

            if current_dist < 200.0:
                total_reward += REWARD_SUCCESS
                terminated = True
                reason = "success"
                break
            if current_dist > 10000.0:
                total_reward += REWARD_LOST_TARGET
                terminated = True
                reason = "lost_target"
                break
            if a_pos[2] < 10.0:
                total_reward += REWARD_CRASH
                terminated = True
                reason = "ground_crash"
                break
            if a_pos[2] > 12000.0:
                terminated = True
                reason = "out_of_bounds"
                break

        # Anti-stall + zone-of-death
        _zone_death_hi = ZONE_DEATH_DIST_HI + self.difficulty_level * ZONE_DEATH_DIST_HI_SCALE
        if not terminated:
            end_dist = self._prev_dist
            closure_rate = (start_dist - end_dist) / DECISION_DT
            self._closure_rates.append(closure_rate)
            if (len(self._closure_rates) >= ANTI_STALL_WINDOW
                    and end_dist > ANTI_STALL_MIN_DIST
                    and all(v < ANTI_STALL_MIN_VC for v in self._closure_rates)):
                truncated = True
                reason = "stall"
                total_reward -= ANTI_STALL_PENALTY

            in_zone = (ZONE_DEATH_DIST_LO <= end_dist <= _zone_death_hi)
            vc_low = closure_rate < ZONE_DEATH_MIN_VC
            if in_zone and vc_low:
                self._zone_death_counter += 1
            else:
                self._zone_death_counter = 0
            if self._zone_death_counter > ZONE_DEATH_WINDOW:
                total_reward -= ZONE_DEATH_PENALTY
                _r_low_speed_penalty -= ZONE_DEATH_PENALTY

        current_time = self._step_counter / CTRL_FREQ
        if not terminated and not truncated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"

        if self.record_tacview:
            self._record_tacview_frame(current_time)

        end_dist = self._prev_dist
        _closure_rate = (start_dist - end_dist) / DECISION_DT

        info = {
            "reason": reason,
            "min_dist": min_dist,
            "r_progress": _r_progress,
            "r_terminal_boost": _r_terminal_boost,
            "r_ata": _r_ata,
            "r_time_pressure": _r_time_pressure,
            "r_ground_warning": _r_ground_warning,
            "r_proximity": _r_proximity,
            "r_low_speed_penalty": _r_low_speed_penalty,
            "r_step_penalty": _r_step_penalty,
            "energy_ok": _energy_ok,
            "last_action": self._last_action,
            "closure_rate": _closure_rate,
            "end_dist": end_dist,
            "zone_death_dist_hi": _zone_death_hi,
            "bfm_action": int(action),
            "bfm_action_name": describe_pursuit_action(int(action)),
        }
        return self._get_obs(), total_reward, terminated, truncated, info

    # ── Observation (identical to SinglePursuitEnv) ────────────────────────

    def _get_obs(self) -> dict:
        a_pos = self.pursuer.position_ned
        t_pos = self.target_ac.position_ned
        a_rpy = self.pursuer.rpy_rad
        a_vel = self.pursuer.velocity_ned
        t_vel = self.target_ac.velocity_ned

        rel_pos_world = t_pos - a_pos
        cos_a_hdg = np.cos(a_rpy[2])
        sin_a_hdg = np.sin(a_rpy[2])
        rel_pos_body = np.array([
            rel_pos_world[0] * cos_a_hdg + rel_pos_world[1] * sin_a_hdg,
            -rel_pos_world[0] * sin_a_hdg + rel_pos_world[1] * cos_a_hdg,
            -rel_pos_world[2],
        ])

        vel_body = np.array([
            a_vel[0] * cos_a_hdg + a_vel[1] * sin_a_hdg,
            -a_vel[0] * sin_a_hdg + a_vel[1] * cos_a_hdg,
            a_vel[2],
        ])

        t_vel_body = np.array([
            t_vel[0] * cos_a_hdg + t_vel[1] * sin_a_hdg,
            -t_vel[0] * sin_a_hdg + t_vel[1] * cos_a_hdg,
            t_vel[2],
        ])

        t_rpy = self.target_ac.rpy_rad
        a_ang_vel = self._compute_angular_velocity(self.pursuer.rpy_rad, self._prev_rpy)
        self._prev_rpy = self.pursuer.rpy_rad.copy()
        t_ang_vel = self._compute_angular_velocity(self.target_ac.rpy_rad, self._prev_target_rpy)
        self._prev_target_rpy = self.target_ac.rpy_rad.copy()

        a_forward = compute_forward_vector(a_rpy)
        t_forward = compute_forward_vector(t_rpy)
        _, los_dir, _ = compute_los(a_pos, t_pos)
        geo = compute_tactical_angles(a_forward, t_forward, los_dir)

        height_norm = a_pos[2] / MAX_HEIGHT
        airspeed = float(self.pursuer.state["airspeed_mps"])
        airspeed_norm = airspeed / MAX_VEL
        alpha_deg = float(self.pursuer.state["alpha_deg"])
        alpha_norm = alpha_deg / MAX_AOA
        ps = self._compute_specific_excess_power(airspeed, a_pos[2])
        ps_norm = ps / MAX_PS

        obs_array = np.array([
            rel_pos_body[0] / MAX_DIST,
            rel_pos_body[1] / MAX_DIST,
            rel_pos_body[2] / MAX_DIST,
            vel_body[0] / MAX_VEL,
            vel_body[1] / MAX_VEL,
            vel_body[2] / MAX_VEL,
            a_rpy[0] / np.pi,
            a_rpy[1] / (np.pi / 2),
            a_rpy[2] / np.pi,
            a_ang_vel[0] / MAX_ANG_VEL,
            a_ang_vel[1] / MAX_ANG_VEL,
            a_ang_vel[2] / MAX_ANG_VEL,
            height_norm,
            t_vel_body[0] / MAX_VEL,
            t_vel_body[1] / MAX_VEL,
            t_vel_body[2] / MAX_VEL,
            t_ang_vel[0] / MAX_ANG_VEL,
            t_ang_vel[1] / MAX_ANG_VEL,
            t_ang_vel[2] / MAX_ANG_VEL,
            geo["cos_ata"],
            geo["cos_aa"],
            geo["cos_hca"],
            alpha_norm,
            airspeed_norm,
            ps_norm,
        ], dtype=np.float32)

        return {
            "obs": np.clip(obs_array, -1.0, 1.0),
            "action_mask": self.action_masks().astype(np.float32),
        }

    def _compute_angular_velocity(self, current_rpy, prev_rpy):
        diff = current_rpy - prev_rpy
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff / PHYSICS_DT

    def _compute_specific_excess_power(self, airspeed, alt):
        d_v = airspeed - self._prev_airspeed
        self._prev_airspeed = airspeed
        ps = d_v * airspeed / 9.81 + (alt - 3000.0) * 9.81 / airspeed if airspeed > 1 else 0.0
        return float(np.clip(ps, -MAX_PS, MAX_PS))

    # ── Target movement (identical to SinglePursuitEnv) ────────────────────

    def _generate_target_profile(self, rng, initial_heading, spawn_alt_m=3000.0):
        d = self._difficulty
        return TargetProfile(
            alt_m=spawn_alt_m,
            speed_mps=160.0 + d * 30.0,
            heading_deg=initial_heading,
            heading_rate_dps=rng.uniform(-5.0, 5.0) * d,
            alt_rate_mps=rng.uniform(-3.0, 3.0) * d,
        )

    def _move_target(self, dt):
        tp = self._target_profile
        tp.heading_deg = (tp.heading_deg + tp.heading_rate_dps * dt) % 360.0
        tp.alt_m = np.clip(tp.alt_m + tp.alt_rate_mps * dt, 500.0, 11000.0)

        t_s = self.target_ac.state
        target_elev = self._target_alt_stab.compute(t_s["alt_m"], tp.alt_m, dt)
        self.target_ac.set_controls(throttle=0.80, elevator=target_elev, aileron=0.0, rudder=0.0)

    # ── Tacview (identical to SinglePursuitEnv) ────────────────────────────

    def _record_tacview_frame(self, t):
        ps = self.pursuer.state
        ts = self.target_ac.state
        self._tacview_frames.append({
            "time": t,
            "pursuer": {
                "lat_deg": ps["lat_deg"], "lon_deg": ps["lon_deg"],
                "alt_m": ps["alt_m"], "roll_deg": ps["roll_deg"],
                "pitch_deg": ps["pitch_deg"], "yaw_deg": ps["yaw_deg"],
                "airspeed_mps": ps["airspeed_mps"], "n_z_g": ps["n_z_g"],
                "alpha_deg": ps["alpha_deg"], "thrust_lbs": ps["thrust_lbs"],
            },
            "target": {
                "lat_deg": ts["lat_deg"], "lon_deg": ts["lon_deg"],
                "alt_m": ts["alt_m"], "roll_deg": ts["roll_deg"],
                "pitch_deg": ts["pitch_deg"], "yaw_deg": ts["yaw_deg"],
                "airspeed_mps": ts["airspeed_mps"],
            },
        })

    def export_tacview(self, path):
        with open(path, "w") as f:
            f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
            f.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")
            f.write("0,Name=Pursuer\n0,Color=Red\n")
            f.write("1,Name=Target\n1,Color=Blue\n")
            for fr in self._tacview_frames:
                t = fr["time"]
                for obj_id, key in [(0, "pursuer"), (1, "target")]:
                    d = fr[key]
                    f.write(f"#{t:.2f}\n")
                    f.write(f"{obj_id},T={d['lat_deg']}|{d['lon_deg']}|{d['alt_m']:.1f}"
                            f"|{d['roll_deg']:.1f}|{d['pitch_deg']:.1f}|{d['yaw_deg']:.1f}\n")
