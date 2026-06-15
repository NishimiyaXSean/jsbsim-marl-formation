"""Single-agent pursuit environment backed by JSBSim F-16.

The RL agent flies one F-16 and must intercept a scripted target aircraft.
The agent controls **high-level flight targets** (heading, altitude, speed)
through a stabilised ``FlightController``.

Target difficulty increases across 3 curriculum stages:
    1. Straight-and-level at constant speed/altitude
    2. Gentle weaving (sinusoidal heading + altitude variations)
    3. Evasive random maneuvers

Observations
------------
25-dim local observation:
    0-2:   target relative position in body frame (3)
    3-5:   own velocity in body frame (3)
    6-8:   own attitude rpy (3)
    9-11:  own angular velocity [p, q, r] (3)   [real JSBSim body rates, rad/s]
    12:    own height (1)
    13-15: target velocity in body frame (3)
    16-18: target angular velocity [roll, pitch, yaw] (3) [finite-diff]
    19-21: tactical geometry cos(ATA), cos(AA), cos(HCA) (3)
    22:    Angle of Attack (1)  [alpha_deg / 30°]
    23:    airspeed (1)  [m/s / 400]
    24:    Specific Excess Power (1)  [Ps / 300 m/s, clipped]

Actions
-------
3-dim continuous: ``[d_heading, d_alt, d_speed]`` ∈ [-1, 1]^3
    d_heading →  [-1, 1] maps to [-6°, +6°] heading change per decision (0.1 s = 60°/s)
    d_alt     →  [-1, 1] maps to [-3 m, +3 m] altitude change per decision (0.1 s = 30 m/s)
    d_speed   →  [-1, 1] maps to [-2, +2] m/s speed change per decision (0.1 s = 20 m/s²)

The FlightController adds these deltas to the persistent target state
(which the agent never sees directly — only through the observation).

Reward
------
r = progress       + tracking_bonus   + altitude_bonus   - energy_penalty   - altitude_alert
  = 0.3 × (-Δdist) + 1.5 × cos(ATA)  + 0.0005 × alt     - 0.1 × |d_thr|    - 1.0 × I(alt<800)

All terms are computed per micro-step and summed over the 30-step decision interval.

Termination
-----------
- Collision: dist < 50 m → success  (+100 reward)
- Target lost: dist > 8000 m → failure
- Ground crash: alt < 10 m → failure
- Ceiling: alt > 12000 m → failure
- Timeout: 120 s → truncation (no additional penalty)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.flight_controller import FlightController, FlightControlTargets, AltitudeStabilizer, ELEVATOR_TRIM
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.1           # 10 Hz — agent issues new targets every 0.1 s
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 6 micro-steps per decision

MAX_EPISODE_TIME = 120.0    # seconds before timeout

# Action scaling (raw [-1,1] → real units, scaled per DECISION_DT)
MAX_D_HEADING_DEG = 6.0     # per 0.1s decision — up to 60°/s commanded turn
MAX_D_ALT_M = 3.0            # per 0.1s decision — up to 30 m/s climb/descent
MAX_D_SPEED_MPS = 2.0        # per 0.1s decision — ±20 m/s² acceleration

# Observation normalisation constants
MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi
MAX_AOA = 30.0               # AoA in degrees — F-16 limit ~25-30°
MAX_PS = 300.0               # Specific Excess Power (m/s) — F-16 max ~300 m/s
LOW_SPEED_THRESHOLD = 100.0  # m/s — below this, large heading commands are penalised

# Reward weights — simple and effective (v5 baseline + staged proximity bonuses)
REWARD_PROGRESS = 5.0        # primary pursuit signal — closing distance
REWARD_ATA = 5.0             # pointing at target
REWARD_GROUND_WARNING = 2.0
REWARD_SUCCESS = 500.0       # capture bonus (dist < 200m)
REWARD_CRASH = -200.0        # strong negative for crashing
REWARD_LOST_TARGET = -200.0  # strong negative for losing target
REWARD_LOW_SPEED_TURN = 5.0  # penalty per decision for high turn rate at low speed

# Staged proximity bonuses — stepping stones to guide the agent closer.
# Awarded once per episode when the agent first crosses each threshold.
PROXIMITY_TIERS = [
    (800.0, 25.0),    # entering engagement zone
    (500.0, 50.0),    # terminal guidance zone
    (300.0, 100.0),   # near-capture
]


@dataclass
class TargetProfile:
    """Pre-scripted target aircraft motion profile."""

    alt_m: float = 3000.0
    speed_mps: float = 180.0
    heading_deg: float = 90.0
    heading_rate_dps: float = 0.0      # continuous heading change
    alt_rate_mps: float = 0.0          # continuous altitude change


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════════

class SinglePursuitEnv(gym.Env):
    """Single-agent F-16 pursuit environment.

    Action space:  Box(3,)  →  [d_heading, d_altitude, d_speed]  ∈  [-1, 1]^3
    Observation:   Box(19,) →  local body-frame relative state + tactical geometry
    """

    metadata = {"render_modes": ["human", "tacview"], "name": "single_pursuit_v0"}

    def __init__(
        self,
        curriculum_stage: float = 1.0,
        jsbsim_data_dir: Optional[str] = None,
        record_tacview: bool = False,
    ):
        super().__init__()

        self.curriculum_stage = curriculum_stage
        self.record_tacview = record_tacview
        self._tacview_frames: List[dict] = []
        self._ref_lla = (30.0, 120.0, 3000.0)

        # Aircraft
        self.pursuer = Aircraft(jsbsim_data_dir)
        self.target_ac = Aircraft(jsbsim_data_dir)

        # Flight controller (driven by agent actions)
        self.fc = FlightController()
        # Target gets its own altitude stabiliser for smooth profile tracking
        self._target_alt_stab = AltitudeStabilizer()

        # Observation / action spaces
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # Persistent flight target (agent modifies this via actions)
        self._target = FlightControlTargets()

        # Episode state
        self._step_counter = 0
        self._prev_dist = 0.0
        self._proximity_awarded: set = set()  # tiers already awarded this episode
        self._prev_rpy = np.zeros(3, dtype=np.float64)  # for pursuer angular velocity
        self._prev_target_rpy = np.zeros(3, dtype=np.float64)  # for target angular velocity
        self._prev_airspeed = 180.0  # m/s — for Specific Excess Power calculation

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)

        # --- Pursuer spawn ---
        pursuer_hdg = rng.uniform(0.0, 360.0)

        self.pursuer.reset(
            lat_deg=30.0, lon_deg=120.0,
            alt_ft=int(3000 * 3.28084),
            heading_deg=pursuer_hdg,
            speed_kts=400,  # start fast — engine maintains it now
            trim=False,
        )
        pursuer_ned = np.array([
            rng.uniform(-500, 500),
            rng.uniform(-500, 500),
            3000.0,
        ])
        self.pursuer.position_ned = pursuer_ned

        # --- Target spawn (5-stage float curriculum) ---
        stage = self.curriculum_stage

        if np.isclose(stage, 1.0):
            target_dist = rng.uniform(800, 1800)
            bearing_offset = 0.0
            target_alt_offset = rng.uniform(-50, 50)
            heading_diff = 0.0
        elif np.isclose(stage, 1.5):
            target_dist = rng.uniform(900, 2000)
            bearing_offset = rng.uniform(-7, 7)
            target_alt_offset = rng.uniform(-75, 75)
            heading_diff = rng.uniform(-10, 10)
        elif np.isclose(stage, 2.0):
            target_dist = rng.uniform(1000, 2500)
            bearing_offset = rng.uniform(-15, 15)
            target_alt_offset = rng.uniform(-150, 150)
            heading_diff = rng.uniform(-20, 20)
        elif np.isclose(stage, 2.5):
            target_dist = rng.uniform(1200, 2700)
            bearing_offset = rng.uniform(-30, 30)
            target_alt_offset = rng.uniform(-225, 225)
            heading_diff = rng.uniform(-25, 25)
        else:  # stage 3.0
            target_dist = rng.uniform(1500, 3000)
            bearing_offset = rng.uniform(-45, 45)
            target_alt_offset = rng.uniform(-300, 300)
            heading_diff = rng.uniform(-30, 30)

        target_bearing = (pursuer_hdg + bearing_offset) % 360.0
        target_bearing_rad = np.deg2rad(target_bearing)
        target_ned = np.array([
            pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
            pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
            pursuer_ned[2] + target_alt_offset,
        ])
        target_hdg = (pursuer_hdg + heading_diff) % 360.0

        self.target_ac.reset(
            lat_deg=30.0, lon_deg=120.0,
            alt_ft=int(3000 * 3.28084),
            heading_deg=target_hdg,
            speed_kts=350,  # 180 m/s cruise for target
            trim=False,
        )
        self.target_ac.position_ned = target_ned

        # Flight controller + target reset
        self.fc.reset()
        self._target_alt_stab.reset()
        self._target = FlightControlTargets(
            heading_deg=pursuer_hdg,
            altitude_m=float(target_ned[2]),
            speed_mps=180.0,  # cruise speed — agent can adjust ±10 m/s per decision
        )

        self._step_counter = 0

        # ── Warmup: 3s at trim to stabilise engine and aerodynamics ──
        # Without this, the aircraft starts in a transient state (low n_z,
        # engine spooling) and loses ~60m altitude before recovering.
        warmup_steps = int(3.0 * CTRL_FREQ)
        for _ in range(warmup_steps):
            self.pursuer.set_controls(throttle=0.80, elevator=-0.05, aileron=0.0, rudder=0.0)
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
        self._target_profile = self._generate_target_profile(rng, target_hdg,
                                                            spawn_alt_m=float(target_ned[2]))

        if self.record_tacview:
            self._record_tacview_frame(0.0)

        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        dt = PHYSICS_DT

        # Parse action: [d_heading, d_alt, d_speed] — the agent sets high-level
        # flight targets; FlightController handles the low-level aerodynamics.
        d_hdg = float(action[0]) * MAX_D_HEADING_DEG
        d_alt = float(action[1]) * MAX_D_ALT_M
        d_spd = float(action[2]) * MAX_D_SPEED_MPS

        # Update persistent flight targets
        self._target.heading_deg = (self._target.heading_deg + d_hdg) % 360.0
        self._target.altitude_m = np.clip(self._target.altitude_m + d_alt, 500.0, 11000.0)
        self._target.speed_mps = np.clip(self._target.speed_mps + d_spd, 100.0, 250.0)

        total_reward = 0.0

        # ── Low-speed turn penalty ─────────────────────────────────────────
        # When airspeed < 100 m/s, aggressive heading changes cause energy
        # loss and departure.  Penalise large d_heading at low speed to teach
        # the agent to "dive for speed before turning."
        airspeed = float(self.pursuer.state["airspeed_mps"])
        if airspeed < LOW_SPEED_THRESHOLD and abs(d_hdg) > 0.2 * MAX_D_HEADING_DEG:
            exceed_ratio = abs(d_hdg) / MAX_D_HEADING_DEG
            low_speed_ratio = (LOW_SPEED_THRESHOLD - airspeed) / LOW_SPEED_THRESHOLD
            total_reward -= REWARD_LOW_SPEED_TURN * exceed_ratio * low_speed_ratio
        terminated = False
        truncated = False
        reason = "timeout"
        min_dist = self._prev_dist

        for _ in range(DECISION_STEPS):
            # FlightController handles all 3 axes (heading + altitude + speed)
            thr, elev, ail, rud = self.fc.compute(self.pursuer.state, self._target, dt)
            self.pursuer.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)

            # --- Move target via scripted profile ---
            self._move_target(dt)

            # --- Step dynamics ---
            self.pursuer.run()
            self.target_ac.run()
            self._step_counter += 1

            # --- Update world positions ---
            self.pursuer.position_ned[0:2] += self.pursuer.velocity_ned[0:2] * dt
            self.pursuer.position_ned[2] = self.pursuer.state["alt_m"]
            self.target_ac.position_ned[0:2] += self.target_ac.velocity_ned[0:2] * dt
            self.target_ac.position_ned[2] = self.target_ac.state["alt_m"]

            # --- NaN guard ---
            if any(not np.isfinite(float(self.pursuer.state[k]))
                   for k in ["n_z_g", "airspeed_mps", "alt_m"]):
                total_reward += REWARD_CRASH
                terminated = True
                reason = "jsbsim_nan"
                break

            a_pos = self.pursuer.position_ned
            t_pos = self.target_ac.position_ned
            a_vel = self.pursuer.velocity_ned
            t_vel = self.target_ac.velocity_ned

            current_dist = float(np.linalg.norm(a_pos - t_pos))
            if current_dist < min_dist:
                min_dist = current_dist

            # --- Tactical geometry ---
            a_forward = compute_forward_vector(self.pursuer.rpy_rad)
            t_forward = compute_forward_vector(self.target_ac.rpy_rad)
            _, los_dir, _ = compute_los(a_pos, t_pos)
            geo = compute_tactical_angles(a_forward, t_forward, los_dir)

            # --- Micro-step rewards (v5 baseline) ---
            delta_dist = self._prev_dist - current_dist  # + when closing

            # Progress: closing distance
            total_reward += REWARD_PROGRESS * delta_dist
            # Terminal boost: extra reward within 500m
            if current_dist < 500.0:
                total_reward += REWARD_PROGRESS * delta_dist * 2.0

            # ATA: pointing at target
            total_reward += REWARD_ATA * max(geo["cos_ata"], -0.2) * dt

            # Time pressure
            time_ratio = self._step_counter / (CTRL_FREQ * MAX_EPISODE_TIME)
            total_reward -= 0.5 * time_ratio * dt

            # Ground warning
            if a_pos[2] < 800.0:
                total_reward -= REWARD_GROUND_WARNING * dt

            # --- Staged proximity bonuses (one-time per tier) ---
            for threshold, bonus in PROXIMITY_TIERS:
                if current_dist < threshold and threshold not in self._proximity_awarded:
                    total_reward += bonus
                    self._proximity_awarded.add(threshold)

            self._prev_dist = current_dist

            # --- Termination checks ---
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

        # --- Timeout ---
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"

        # --- Tacview ---
        if self.record_tacview:
            self._record_tacview_frame(current_time)

        return self._get_obs(), total_reward, terminated, truncated, {"reason": reason, "min_dist": min_dist}

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build 19-dim local observation."""
        a_pos = self.pursuer.position_ned
        t_pos = self.target_ac.position_ned
        a_rpy = self.pursuer.rpy_rad
        t_rpy = self.target_ac.rpy_rad
        a_vel = self.pursuer.velocity_ned
        t_vel = self.target_ac.velocity_ned

        a_forward = compute_forward_vector(a_rpy)
        t_forward = compute_forward_vector(t_rpy)
        _, los_dir, _ = compute_los(a_pos, t_pos)
        geo = compute_tactical_angles(a_forward, t_forward, los_dir)

        # Body-frame transform
        yaw = a_rpy[2]
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)

        def world_to_body(vec):
            return np.array([
                vec[0] * cos_y - vec[1] * sin_y,
                vec[0] * sin_y + vec[1] * cos_y,
                vec[2],
            ])

        rel_pos_body = world_to_body(t_pos - a_pos)
        own_vel_body = world_to_body(a_vel)
        tgt_vel_body = world_to_body(t_vel)

        # Pursuer angular velocity: REAL JSBSim body-frame rates [p, q, r] (rad/s)
        a_state = self.pursuer.state
        ang_vel = np.array([a_state["p_rps"], a_state["q_rps"], a_state["r_rps"]])

        # Target angular velocity: finite-difference from previous target rpy
        current_tgt_rpy = t_rpy.copy()
        d_tgt_roll = current_tgt_rpy[0] - self._prev_target_rpy[0]
        d_tgt_pitch = current_tgt_rpy[1] - self._prev_target_rpy[1]
        d_tgt_yaw = current_tgt_rpy[2] - self._prev_target_rpy[2]
        d_tgt_yaw = (d_tgt_yaw + np.pi) % (2 * np.pi) - np.pi  # unwrap
        tgt_ang_vel = np.array([d_tgt_roll, d_tgt_pitch, d_tgt_yaw]) / DECISION_DT
        self._prev_target_rpy = current_tgt_rpy

        # ── Energy / AoA features ──────────────────────────────────────────
        # Angle of Attack (deg) — tells the agent how hard the wing is working
        alpha_deg = float(a_state["alpha_deg"])

        # Explicit airspeed (m/s) — critical for energy-aware decisions
        airspeed_mps = float(a_state["airspeed_mps"])

        # Specific Excess Power: Ps = climb_rate + (V/g) * dV/dt
        #   climb_rate = h_dot (m/s), dV/dt from finite-diff airspeed change
        climb_rate = float(a_state["h_dot_fps"]) * 0.3048
        accel = (airspeed_mps - self._prev_airspeed) / DECISION_DT
        g = 9.81
        ps = climb_rate + (airspeed_mps / g) * accel if airspeed_mps > 1.0 else 0.0
        self._prev_airspeed = airspeed_mps

        obs = np.concatenate([
            rel_pos_body / MAX_DIST,           # 0-2
            own_vel_body / MAX_VEL,             # 3-5
            a_rpy / np.pi,                      # 6-8
            ang_vel / MAX_ANG_VEL,              # 9-11  (real JSBSim body rates)
            [a_pos[2] / MAX_HEIGHT],            # 12
            tgt_vel_body / MAX_VEL,             # 13-15
            tgt_ang_vel / MAX_ANG_VEL,          # 16-18
            [geo["cos_ata"], geo["cos_aa"],     # 19-21
             geo["cos_hca"]],
            [alpha_deg / MAX_AOA],              # 22  (AoA)
            [airspeed_mps / MAX_VEL],           # 23  (explicit airspeed)
            [np.clip(ps / MAX_PS, -1.0, 1.0)], # 24  (Specific Excess Power)
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Target motion ────────────────────────────────────────────────────────

    def _generate_target_profile(self, rng: np.random.Generator, spawn_heading: float = 90.0,
                                 spawn_alt_m: float = 3000.0) -> TargetProfile:
        """Generate stage-dependent target motion."""
        tp = TargetProfile()
        tp.alt_m = spawn_alt_m  # align with actual spawn altitude
        stage = self.curriculum_stage

        if np.isclose(stage, 1.0):
            tp.speed_mps = 130.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = 0.0
            tp.alt_rate_mps = 0.0
        elif np.isclose(stage, 1.5):
            tp.speed_mps = 135.0                     # gentle speed increase from 130
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-1.5, 1.5)  # very gentle weave
            tp.alt_rate_mps = rng.uniform(-0.5, 0.5)      # minimal altitude drift
        elif np.isclose(stage, 2.0):
            tp.speed_mps = 150.0                     # moderate speed
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-5, 5)       # moderate weave
            tp.alt_rate_mps = rng.uniform(-1.5, 1.5)       # gentle altitude changes
        elif np.isclose(stage, 2.5):
            tp.speed_mps = 165.0                     # faster but not max
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-8, 8)       # active weaving
            tp.alt_rate_mps = rng.uniform(-2.5, 2.5)       # moderate climbs
        else:  # stage 3.0
            tp.speed_mps = 180.0                     # near-max challenge
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-12, 12)     # aggressive evasion
            tp.alt_rate_mps = rng.uniform(-4, 4)           # significant altitude changes

        return tp

    def _move_target(self, dt: float) -> None:
        """Advance target along its scripted profile with smooth altitude control."""
        tp = self._target_profile
        tp.heading_deg = (tp.heading_deg + tp.heading_rate_dps * dt) % 360.0
        tp.alt_m = np.clip(tp.alt_m + tp.alt_rate_mps * dt, 500.0, 11000.0)

        from src.dynamics.flight_controller import THROTTLE_TRIM

        # Use the same AltitudeStabilizer as the pursuer for smooth tracking
        target_elev = self._target_alt_stab.compute(
            self.target_ac.state["alt_m"], tp.alt_m, dt,
        )

        self.target_ac.set_controls(
            throttle=THROTTLE_TRIM,
            elevator=np.clip(target_elev, -1.0, 1.0),
            aileron=0.0,
            rudder=0.0,
        )

    # ── Tacview ──────────────────────────────────────────────────────────────

    def _record_tacview_frame(self, time_sec: float) -> None:
        from src.utils.kinematics import ned_to_lla

        ref = self._ref_lla
        a_ned = self.pursuer.position_ned
        t_ned = self.target_ac.position_ned

        a_lla = ned_to_lla(
            np.array([a_ned[0], a_ned[1], ref[2] - a_ned[2]], dtype=np.float64),
            np.array(ref, dtype=np.float64),
        )
        t_lla = ned_to_lla(
            np.array([t_ned[0], t_ned[1], ref[2] - t_ned[2]], dtype=np.float64),
            np.array(ref, dtype=np.float64),
        )

        a_s = self.pursuer.state
        t_s = self.target_ac.state

        self._tacview_frames.append({
            "time": time_sec,
            "pursuer": {
                "lat_deg": float(a_lla[0]),
                "lon_deg": float(a_lla[1]),
                "alt_m": float(a_lla[2]),
                "roll_deg": a_s["roll_deg"],
                "pitch_deg": a_s["pitch_deg"],
                "yaw_deg": a_s["yaw_deg"],
            },
            "target": {
                "lat_deg": float(t_lla[0]),
                "lon_deg": float(t_lla[1]),
                "alt_m": float(t_lla[2]),
                "roll_deg": t_s["roll_deg"],
                "pitch_deg": t_s["pitch_deg"],
                "yaw_deg": t_s["yaw_deg"],
            },
        })

    def export_tacview(self, filepath: str) -> None:
        from src.logging.tacview_exporter import TacviewExporter

        # Re-map keys to match existing TacviewExporter (expects "attacker"/"evader")
        frames = []
        for f in self._tacview_frames:
            frames.append({
                "time": f["time"],
                "attacker": f["pursuer"],
                "evader": f["target"],
            })
        TacviewExporter(filepath).write(frames)
