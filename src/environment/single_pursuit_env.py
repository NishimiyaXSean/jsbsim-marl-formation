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
19-dim local observation (same as the original ``compute_obs``):
    0-2:   target relative position in body frame (3)
    3-5:   own velocity in body frame (3)
    6-8:   own attitude rpy (3)
    9-11:  own angular velocity in body frame (3)   [placeholder — always zero]
    12:    own height (1)
    13-15: target velocity in body frame (3)
    16-18: tactical geometry cos(ATA), cos(AA), cos(HCA) (3)

Actions
-------
3-dim continuous: ``[d_heading, d_alt, d_speed]`` ∈ [-1, 1]^3
    d_heading →  [-1, 1] maps to [-30°, +30°] heading change per decision (0.5 s)
    d_alt     →  [-1, 1] maps to [-50 m, +50 m] altitude change per decision
    d_speed   →  [-1, 1] maps to [-20, +20] m/s speed change per decision

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
from src.dynamics.flight_controller import FlightController, FlightControlTargets, ELEVATOR_TRIM
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.5          # agent issues new targets every 0.5 s
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 30 micro-steps per decision

MAX_EPISODE_TIME = 120.0   # seconds before timeout

# Action scaling (raw [-1,1] → real units)
MAX_D_HEADING_DEG = 30.0    # per decision (0.5s) — up to 60°/s commanded turn
MAX_D_ALT_M = 15.0           # per decision — up to 30 m/s climb/descent
MAX_D_SPEED_MPS = 10.0       # per decision — ±20 m/s/s acceleration

# Observation normalisation constants
MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi

# Reward weights — inspired by MARL BFM reward taxonomy (docs/marl_env.py)
# Categories: Range Rate | Tracking (ATA/AA/Collision/HCA) | Height | Energy | Terminal
REWARD_RANGE_RATE_CLOSING = 0.3    # per meter closed (was 5.0 — too high)
REWARD_RANGE_RATE_OPENING = -0.1   # per meter lost (new — penalize losing ground)
REWARD_ATA = 3.0                   # pointing at target (cos_ata > 0)
REWARD_ATA_PENALTY = 5.0           # penalty for nose-off (cos_ata < 0)
REWARD_AA = 2.0                    # behind target bonus (cos_aa > 0)
REWARD_COLLISION = 5.0             # collision course base weight
REWARD_HCA = 1.0                   # heading cross angle alignment
REWARD_HEIGHT_ADV = 0.001          # height advantage per meter above target
REWARD_HEIGHT_DISADV = 0.01        # penalty per meter below target
REWARD_ENERGY_LOW_SPEED = 0.5      # penalty per m/s below safe speed
REWARD_TERMINAL_CLOSING = 0.1      # terminal closing speed bonus (<500m)
REWARD_TERMINAL_BOOST = 5.0        # extra range-rate weight in terminal phase
REWARD_MACRO_ATA = 50.0            # macro ATA improvement per decision step
REWARD_GROUND_WARNING = 2.0        # ground proximity penalty
REWARD_SUCCESS = 500.0             # capture bonus (dist < 200m)
REWARD_CRASH = -200.0              # ground crash penalty
REWARD_LOST_TARGET = -200.0        # lost target penalty
REWARD_TIMEOUT = -500.0            # timeout penalty (new — was 0)

SAFE_SPEED_MPS = 180.0             # below this, energy penalty applies
TERMINAL_RADIUS = 500.0            # terminal guidance phase


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

        # Observation / action spaces
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(19,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # Persistent flight target (agent modifies this via actions)
        self._target = FlightControlTargets()

        # Episode state
        self._step_counter = 0
        self._prev_dist = 0.0
        self._last_cos_ata = 0.0  # for macro ATA trend reward

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
        self._last_cos_ata = self._get_cos_ata()
        self._tacview_frames = []
        self._target_profile = self._generate_target_profile(rng, target_hdg)

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

            # ═══════════════════════════════════════════════════════════════
            #  Multi-category reward (MARL-inspired, see docs/marl_env.py)
            # ═══════════════════════════════════════════════════════════════

            # ── Category 1: Range Rate ───────────────────────────────────
            delta_dist = self._prev_dist - current_dist  # + when closing
            if delta_dist > 0:
                total_reward += REWARD_RANGE_RATE_CLOSING * delta_dist
            else:
                total_reward += REWARD_RANGE_RATE_OPENING * delta_dist

            # ── Category 2: Tracking Guidance ─────────────────────────────
            cos_ata = geo["cos_ata"]
            cos_aa = geo["cos_aa"]
            cos_hca = geo["cos_hca"]

            # ATA: nose-on-target
            if cos_ata > 0.0:
                total_reward += REWARD_ATA * cos_ata * dt
                if cos_ata > 0.866:  # within 30° cone
                    total_reward += 2.0 * dt  # high-quality lock bonus
            else:
                total_reward += REWARD_ATA_PENALTY * cos_ata * dt

            # AA: behind target (tactical advantage)
            if cos_aa > 0.5:
                total_reward += REWARD_AA * cos_aa * dt

            # HCA: heading alignment (same direction)
            if cos_hca > 0.0:
                total_reward += REWARD_HCA * cos_hca * dt

            # Collision course: relative velocity pointing at target
            a_vel_arr = np.asarray(a_vel)
            t_vel_arr = np.asarray(t_vel)
            rel_vel = a_vel_arr - t_vel_arr
            rel_speed = float(np.linalg.norm(rel_vel))
            if rel_speed > 1e-6:
                rel_vel_dir = rel_vel / rel_speed
                cos_collision = float(np.clip(np.dot(rel_vel_dir, los_dir), -1.0, 1.0))
                if cos_collision > 0.0:
                    # Weight increases as distance decreases → prioritize terminal approach
                    dynamic_weight = REWARD_COLLISION + (500.0 / (current_dist + 100.0)) * 3.0
                    total_reward += cos_collision * dynamic_weight * dt
                else:
                    total_reward += cos_collision * 3.0 * dt  # mild penalty for bad approach

            # ── Category 3: Height Advantage ──────────────────────────────
            dz = a_pos[2] - t_pos[2]
            if dz > 50.0 and cos_ata > 0.5 and delta_dist > 0:
                # Above target and engaging: tactical energy advantage
                total_reward += min(dz, 1000.0) * REWARD_HEIGHT_ADV * dt
            elif dz < -100.0:
                # Below target: penalty scaled by depth
                penalty_scale = min(abs(dz) / 300.0, 4.0)
                total_reward -= abs(dz) * REWARD_HEIGHT_DISADV * penalty_scale * dt

            # ── Category 4: Energy Management ─────────────────────────────
            a_spd = float(self.pursuer.state["airspeed_mps"])
            vz = float(self.pursuer.state.get("h_dot_fps", 0.0)) * 0.3048  # ft/s → m/s
            if a_spd < SAFE_SPEED_MPS and vz < 10.0:
                # Low speed and not actively climbing → dangerous energy state
                total_reward -= (SAFE_SPEED_MPS - a_spd) * REWARD_ENERGY_LOW_SPEED * dt

            # ── Category 5: Terminal Guidance (< TERMINAL_RADIUS) ─────────
            if current_dist < TERMINAL_RADIUS:
                # Extra reward for closing fast in final phase
                closing_speed = float(np.dot(a_vel_arr, los_dir))
                if closing_speed > 0:
                    total_reward += min(closing_speed, 300.0) * REWARD_TERMINAL_CLOSING * dt
                # Boost range-rate weight for aggressive terminal attack
                if delta_dist > 0:
                    total_reward += REWARD_TERMINAL_BOOST * delta_dist

            # ── Category 6: Time Pressure ─────────────────────────────────
            time_ratio = self._step_counter / (CTRL_FREQ * MAX_EPISODE_TIME)
            total_reward -= (0.5 + time_ratio * 2.0) * dt  # grows from -0.5/s to -2.5/s

            # ── Category 7: Ground Warning ────────────────────────────────
            if a_pos[2] < 200.0:
                depth_ratio = (200.0 - a_pos[2]) / 200.0
                total_reward -= (depth_ratio ** 2) * 5.0 * dt
                if vz < -1.0:  # descending into ground
                    total_reward -= abs(vz) * 0.2 * dt

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

        # --- Macro ATA Trend Reward (per decision step) ---
        # Reward improvement in pointing accuracy over the decision interval.
        final_cos_ata = self._get_cos_ata()
        macro_delta_cos = final_cos_ata - self._last_cos_ata
        if macro_delta_cos > 0:
            total_reward += macro_delta_cos * REWARD_MACRO_ATA
        self._last_cos_ata = final_cos_ata

        # --- Timeout ---
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"
            total_reward += REWARD_TIMEOUT

        # --- Tacview ---
        if self.record_tacview:
            self._record_tacview_frame(current_time)

        return self._get_obs(), total_reward, terminated, truncated, {"reason": reason, "min_dist": min_dist}

    def _get_cos_ata(self) -> float:
        """Return current cos(ATA) for macro trend tracking."""
        a_forward = compute_forward_vector(self.pursuer.rpy_rad)
        _, los_dir, _ = compute_los(self.pursuer.position_ned, self.target_ac.position_ned)
        return float(np.clip(np.dot(a_forward, los_dir), -1.0, 1.0))

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

        obs = np.concatenate([
            rel_pos_body / MAX_DIST,
            own_vel_body / MAX_VEL,
            a_rpy / np.pi,
            np.zeros(3),                        # angular velocity placeholder
            [a_pos[2] / MAX_HEIGHT],
            tgt_vel_body / MAX_VEL,
            [geo["cos_ata"], geo["cos_aa"], geo["cos_hca"]],
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Target motion ────────────────────────────────────────────────────────

    def _generate_target_profile(self, rng: np.random.Generator, spawn_heading: float = 90.0) -> TargetProfile:
        """Generate stage-dependent target motion."""
        tp = TargetProfile()
        tp.alt_m = 3500.0
        stage = self.curriculum_stage

        if np.isclose(stage, 1.0):
            tp.speed_mps = 130.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = 0.0
            tp.alt_rate_mps = 0.0
        elif np.isclose(stage, 1.5):
            tp.speed_mps = 145.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-3, 3)
            tp.alt_rate_mps = rng.uniform(-1.5, 1.5)
        elif np.isclose(stage, 2.0):
            tp.speed_mps = 160.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-10, 10)
            tp.alt_rate_mps = rng.uniform(-3, 3)
        elif np.isclose(stage, 2.5):
            tp.speed_mps = 170.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-15, 15)
            tp.alt_rate_mps = rng.uniform(-5, 5)
        else:  # stage 3.0
            tp.speed_mps = 180.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-20, 20)
            tp.alt_rate_mps = rng.uniform(-8, 8)

        return tp

    def _move_target(self, dt: float) -> None:
        """Advance target along its scripted profile."""
        tp = self._target_profile
        tp.heading_deg = (tp.heading_deg + tp.heading_rate_dps * dt) % 360.0
        tp.alt_m = np.clip(tp.alt_m + tp.alt_rate_mps * dt, 500.0, 11000.0)

        from src.dynamics.flight_controller import THROTTLE_TRIM, ELEVATOR_TRIM

        # Altitude hold (same as before)
        alt_err = tp.alt_m - self.target_ac.state["alt_m"]
        target_elev = ELEVATOR_TRIM - 0.002 * alt_err

        # Heading: don't correct — let the target fly naturally straight.
        # Aileron=0 at trim gives wings-level flight.  The heading-correction
        # P-controller was causing unintended turns due to transient roll.
        target_ail = 0.0

        self.target_ac.set_controls(
            throttle=THROTTLE_TRIM,
            elevator=np.clip(target_elev, -1.0, 1.0),
            aileron=target_ail,
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
