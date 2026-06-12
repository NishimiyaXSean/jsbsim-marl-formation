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
# With engine running, F-16 can sustain ~267 m/s at 3000m — plenty of energy.
MAX_D_HEADING_DEG = 10.0    # per decision (0.5s) — up to 20°/s commanded turn
MAX_D_ALT_M = 15.0           # per decision — up to 30 m/s climb/descent
MAX_D_SPEED_MPS = 10.0       # per decision — faster speed changes

# Observation normalisation constants
MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi

# Reward weights
REWARD_PROGRESS = 2.0        # primary pursuit signal — closing distance
REWARD_ATA = 3.0             # pointing at target
REWARD_ALTITUDE_BONUS = 0.0  # disabled
REWARD_ENERGY_PENALTY = 0.0  # disabled
REWARD_GROUND_WARNING = 2.0
REWARD_SUCCESS = 500.0       # strong positive reinforcement for capture
REWARD_CRASH = -200.0        # strong negative for crashing
REWARD_LOST_TARGET = -200.0  # strong negative for losing target


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
        curriculum_stage: int = 1,
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

        # --- Target spawn (curriculum-aware) ---
        if self.curriculum_stage == 1:
            # Stage 1: target DEAD AHEAD — zero bearing offset, same heading.
            # At 260 m/s, even 1° offset creates 90m lateral miss after 20s.
            # Agent just needs throttle forward; steering comes in stage 2.
            target_dist = rng.uniform(800, 1800)
            target_bearing = pursuer_hdg  # exactly ahead
            target_bearing_rad = np.deg2rad(target_bearing)

            target_ned = np.array([
                pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
                pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
                pursuer_ned[2] + rng.uniform(-50, 50),  # small altitude delta
            ])
            target_hdg = pursuer_hdg  # exactly same direction
        elif self.curriculum_stage == 2:
            # Stage 2: moderate bearing (±15°), mild heading diff (±20°), weaving target
            target_dist = rng.uniform(1000, 2500)
            bearing_offset = rng.uniform(-15, 15)
            target_bearing = (pursuer_hdg + bearing_offset) % 360.0
            target_bearing_rad = np.deg2rad(target_bearing)
            target_ned = np.array([
                pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
                pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
                pursuer_ned[2] + rng.uniform(-150, 150),
            ])
            target_hdg = (pursuer_hdg + rng.uniform(-20, 20)) % 360.0
        else:
            # Stage 3: wide bearing (±45°), weaving target — combat-adjacent
            target_dist = rng.uniform(1500, 3000)
            bearing_offset = rng.uniform(-45, 45)
            target_bearing = (pursuer_hdg + bearing_offset) % 360.0
            target_bearing_rad = np.deg2rad(target_bearing)
            target_ned = np.array([
                pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
                pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
                pursuer_ned[2] + rng.uniform(-300, 300),
            ])
            target_hdg = (pursuer_hdg + rng.uniform(-30, 30)) % 360.0

        self.target_ac.reset(
            lat_deg=30.0, lon_deg=120.0,
            alt_ft=int(3000 * 3.28084),
            heading_deg=target_hdg,
            speed_kts=350,  # 180 m/s cruise for target
            trim=False,
        )
        self.target_ac.position_ned = target_ned

        # Flight controller + target reset — start fast to give speed advantage
        self.fc.reset()
        # Match target altitude so the pursuer can actually intercept in 3D
        self._target = FlightControlTargets(
            heading_deg=pursuer_hdg,
            altitude_m=float(target_ned[2]),  # match target altitude
            speed_mps=250.0,  # start fast — F-16 can sustain this
        )

        self._step_counter = 0
        self._prev_dist = float(np.linalg.norm(pursuer_ned - target_ned))
        self._tacview_frames = []
        self._target_profile = self._generate_target_profile(rng, target_hdg)

        if self.record_tacview:
            self._record_tacview_frame(0.0)

        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        dt = PHYSICS_DT

        # Parse action: [aileron, d_alt, d_speed]
        # Aileron: direct control, bypasses FC heading stabiliser
        raw_ail = float(np.clip(action[0], -1.0, 1.0))
        d_alt = float(action[1]) * MAX_D_ALT_M
        d_spd = float(action[2]) * MAX_D_SPEED_MPS

        # Update FC targets (altitude + speed only — heading is direct)
        self._target.altitude_m = np.clip(self._target.altitude_m + d_alt, 500.0, 11000.0)
        self._target.speed_mps = np.clip(self._target.speed_mps + d_spd, 100.0, 250.0)

        total_reward = 0.0
        terminated = False
        truncated = False
        reason = "timeout"
        min_dist = self._prev_dist

        for _ in range(DECISION_STEPS):
            # --- FC altitude + speed, direct aileron for heading ---
            elev = self.fc.alt.compute(self.pursuer.state["alt_m"], self._target.altitude_m, dt)
            # Bank compensation: when banked, boost elevator to maintain vertical lift
            import math
            roll_abs_rad = math.radians(abs(self.pursuer.state["roll_deg"]))
            cos_roll = max(math.cos(roll_abs_rad), 0.1)
            bank_factor = 1.0 / cos_roll
            elev = ELEVATOR_TRIM + (elev - ELEVATOR_TRIM) * bank_factor
            elev = float(np.clip(elev, -1.0, 1.0))

            thr = self.fc.spd.compute(self.pursuer.state["airspeed_mps"], self._target.speed_mps, dt)
            ail = raw_ail * 0.30  # direct aileron: ±0.3 range
            rud = 0.0
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

            # --- Micro-step rewards ---
            dz = a_pos[2] - t_pos[2]

            # Progress: closing distance (positive when closing)
            delta_dist = self._prev_dist - current_dist
            total_reward += REWARD_PROGRESS * delta_dist

            # ATA: pointing at target
            total_reward += REWARD_ATA * max(geo["cos_ata"], -0.2) * dt

            # Altitude: bonus for staying high (energy advantage)
            total_reward += REWARD_ALTITUDE_BONUS * a_pos[2] * dt

            # Energy: penalty for rapid throttle changes
            total_reward -= REWARD_ENERGY_PENALTY * abs(float(thr) - 0.8) * dt

            # Ground warning
            if a_pos[2] < 800.0:
                total_reward -= REWARD_GROUND_WARNING * dt

            self._prev_dist = current_dist

            # --- Termination checks ---
            if current_dist < 200.0:  # generous kill radius for training
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

        if self.curriculum_stage == 1:
            # Straight and level — easiest; slower target for easy catch
            tp.speed_mps = 130.0  # 250 kts — big speed advantage for pursuer
            tp.heading_deg = spawn_heading  # same direction as spawn
            tp.heading_rate_dps = 0.0
            tp.alt_rate_mps = 0.0
        elif self.curriculum_stage == 2:
            # Gentle weaving, moderate speed
            tp.speed_mps = 160.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(5, 15) * rng.choice([-1, 1])
            tp.alt_rate_mps = rng.uniform(-3, 3)
        else:
            # Evasive — random changes, faster
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
