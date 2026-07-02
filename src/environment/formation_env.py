"""Formation pursuit environment — N pursuers + M targets (SB3 prototype).

Phase 1: 2v1 fixed configuration with shared-policy single Gym Env.
Phase 2: NvM with RLlib MAPPO after validation.

Each pursuer flies its own F-16 via FlightController (Box(2) action),
inheriting the validated single-agent architecture from ContinuousPursuitEnv.
Observations include formation-mate state for cooperative tactics.

Usage (2v1):
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty=0.0)
    obs, _ = env.reset()
    action = model.predict(obs)  # Box(4) = [turn_p0, spd_p0, turn_p1, spd_p1]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.utils.units import kts_to_mps
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants (shared with ContinuousPursuitEnv)
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.5
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)
MAX_EPISODE_TIME = 120.0  # longer for formation tactics

MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi
MAX_AOA = 30.0
MAX_PS = 300.0
MAX_LOS_RATE = 0.5

# Reward weights (from ContinuousPursuitEnv Phase 3)
REWARD_PROGRESS = 1.5
REWARD_ATA = 8.0
REWARD_SUCCESS = 5000.0
REWARD_CRASH = -200.0
REWARD_LOST_TARGET = -200.0
REWARD_TIMEOUT = -500.0
STEP_PENALTY = 0.25
LOW_ENERGY_PENALTY = 2.0
ANTI_STALL_WINDOW = 35
ANTI_STALL_MIN_VC = 15.0
ANTI_STALL_MIN_DIST = 200.0
ANTI_STALL_PENALTY = 200.0
ANTI_STALL_SPEED_WARN = 130.0
ANTI_STALL_SPEED_WARN_WEIGHT = 1.0
REWARD_DELTA_ATA = 8.0
REWARD_CLOSURE_RATE = 6.0
CLOSURE_RATE_NORM = 30.0
VELOCITY_SHAPING_WEIGHT = 3.0
VELOCITY_SHAPING_ATA_THRESH = 0.95
PROXIMITY_TIERS = [(800.0, 25.0), (500.0, 50.0), (300.0, 100.0)]

# ── Phase 3 reward additions ─────────────────────────────────────────────
ATA_DEGRADATION_THRESH = 20.0     # |ATA| above this triggers penalty
ATA_DEGRADATION_WEIGHT = 1.0      # -1.0·dt per micro-step when degraded
TERMINAL_PULL_MAX = 50.0          # max per-micro-step terminal pull at 200m

# ── Formation-specific (Phase 4.1: piecewise spacing + weight annealing) ─────
FORMATION_COLLISION_DIST = 50.0   # below this: terminate episode
FORMATION_COLLISION_PENALTY = -3000.0

# Piecewise spacing zones (metres between pursuers)
SPACING_DANGER = 50.0     # < this: strong fixed penalty, terminate
SPACING_REPEL_MAX = 200.0  # 50–200: linear repulsion (Coulomb-like)
SPACING_IDEAL_LO = 200.0   # 200–500: small positive, gated on closing
SPACING_IDEAL_HI = 500.0
SPACING_REWARD_CAP = 2.0   # max ideal-zone reward per micro-step
SPACING_DANGER_PENALTY = -5.0  # penalty per micro-step in danger zone

# Weight annealing
FORMATION_WEIGHT = 0.0      # current dynamic weight (0→1 via annealing)

# ── Phase 5: Cooperative 2v1 (pincer + AND-gate + asymmetric resets) ─────
# Pincer angle reward: encourage pursuers to flank the target
PINCER_IDEAL_ANGLE_MIN = 60.0    # min pincer angle (degrees) for reward
PINCER_IDEAL_ANGLE_MAX = 150.0   # max pincer angle for reward (180=perfect flank)
PINCER_WEIGHT = 15.0             # reward weight for pincer angle (per micro-step)
PINCER_DIST_MAX = 2000.0         # only apply pincer reward when both within this range

# Cooperative interception (AND-gate) — curriculum-annealed
COOP_SUCCESS_DIST_EASY = 800.0   # relaxed: both within 800m
COOP_SUCCESS_DIST_HARD = 300.0   # strict: both within 300m
COOP_SUCCESS_ANGLE_EASY = 20.0   # relaxed: > 20 deg pincer
COOP_SUCCESS_ANGLE_HARD = 60.0   # strict: > 60 deg pincer
COOP_SUSTAIN_STEPS = 6           # must hold for N consecutive micro-steps
COOP_CURRICULUM_START = 0.0      # curriculum progress at which tightening begins
COOP_CURRICULUM_END = 0.6        # curriculum progress at which criteria are fully strict

# Dynamic role assignment
STRIKER_TRACKING_BONUS = 1.5     # multiplier on tracking reward for closer pursuer
INTERCEPTOR_PINCER_BONUS = 2.0   # multiplier on pincer reward for further pursuer

# Asymmetric resets
ASYMMETRIC_RESET_PROB = 0.7      # probability of asymmetric spawn per episode
ASYMMETRIC_DIST_FAR = 1500.0     # disadvantaged pursuer starts this far behind (m)
ASYMMETRIC_HEADING_OFF = 120.0   # disadvantaged pursuer faces away (degrees)

# Target evasive maneuvers (cooperative mode, difficulty > 0)
EVASION_DIST_TRIGGER = 800.0     # target starts evading when BOTH pursuers within this
EVASION_TURN_RATE = 5.0          # max evasion turn rate (deg/s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PursuerState:
    """Runtime state for one pursuer."""
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    ref_hdg: float = 0.0         # accumulating heading target
    ref_alt_m: float = 3000.0
    prev_dist: float = 0.0
    prev_ata_deg: Optional[float] = None
    prev_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_airspeed: float = 180.0
    proximity_awarded: set = field(default_factory=set)
    closure_rates: deque = field(default_factory=lambda: deque(maxlen=ANTI_STALL_WINDOW))
    zone_death_counter: int = 0
    loiter_time: float = 0.0
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    episode_start_dist: float = 0.0
    kill_zone_steps: int = 0   # consecutive micro-steps both pursuers in kill zone

    def reset_state(self):
        self.prev_ata_deg = None
        self.proximity_awarded.clear()
        self.closure_rates.clear()
        self.zone_death_counter = 0
        self.loiter_time = 0.0
        self.kill_zone_steps = 0


@dataclass
class TargetState:
    """Runtime state for one target."""
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    ref_hdg: float = 0.0
    ref_alt_m: float = 3000.0


class FormationEnv(gym.Env):
    """Multi-pursuer formation pursuit with Box(2N) shared-policy action.

    Phase 1: SB3 prototype with concatenated observations.
    Phase 2: RLlib MAPPO after Phase 1 validation.

    Observation (per pursuer, concatenated):
      [0:27]  = 27-dim single-pursuer features (same as ContinuousPursuitEnv)
      [27:30] = nearest mate relative position (body frame, /MAX_DIST)
      [30:33] = nearest mate relative velocity (body frame, /MAX_VEL)

    Total obs dim = num_pursuers × 33 for 2v1 = 66
    """

    metadata = {"name": "formation_pursuit_v0"}

    def __init__(
        self,
        num_pursuers: int = 2,
        num_targets: int = 1,
        difficulty_level: float = 0.0,
        lock_altitude: bool = True,
        jsbsim_data_dir: Optional[str] = None,
        record_tacview: bool = False,
        cooperative_mode: bool = False,
    ):
        super().__init__()
        self.N = num_pursuers
        self.M = num_targets
        self._difficulty = float(np.clip(difficulty_level, 0.0, 1.0))
        self._lock_altitude = lock_altitude
        self.record_tacview = record_tacview
        self.cooperative_mode = cooperative_mode
        self._ref_lla = (30.0, 120.0, 3000.0)
        self._tacview_frames: List[dict] = []
        self._ata_penalty_weight = 0.0
        self._formation_weight = FORMATION_WEIGHT
        # Cooperative state
        self._striker_idx: int = 0       # which pursuer is the striker this episode
        self._coop_sustain_counter: int = 0  # consecutive steps in cooperative kill zone
        # Dynamic AND-gate (curriculum-annealed)
        self._coop_curriculum_progress: float = 0.0
        self._coop_success_dist: float = COOP_SUCCESS_DIST_EASY
        self._coop_success_angle: float = COOP_SUCCESS_ANGLE_EASY

        # ── Build aircraft + controllers ──────────────────────────────
        self.pursuers: List[PursuerState] = []
        for _ in range(self.N):
            ac = Aircraft(jsbsim_data_dir)
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.pursuers.append(PursuerState(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap))

        self.targets: List[TargetState] = []
        for _ in range(self.M):
            ac = Aircraft(jsbsim_data_dir)
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.targets.append(TargetState(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap))

        # ── Observation / Action spaces ───────────────────────────────
        self._obs_per_pursuer = 33  # 27 base + 3 mate pos + 3 mate vel
        obs_dim = self.N * self._obs_per_pursuer

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2 * self.N,), dtype=np.float32)

        # ── Episode state ─────────────────────────────────────────────
        self._step_counter = 0

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        d = self._difficulty

        # Spawn: pursuers in a loose cluster, targets offset
        cluster_center = np.array([rng.uniform(-200, 200),
                                    rng.uniform(-200, 200), 3000.0])

        # ── Asymmetric reset (cooperative mode) ──────────────────────────
        asymmetric = False
        disadvantaged_idx = 0
        if self.cooperative_mode and self.N >= 2 and rng.random() < ASYMMETRIC_RESET_PROB:
            asymmetric = True
            disadvantaged_idx = rng.integers(0, self.N)  # randomly pick one pursuer
            self._striker_idx = 1 if disadvantaged_idx == 0 else 0

        for i, ps in enumerate(self.pursuers):
            if asymmetric and i == disadvantaged_idx:
                # Disadvantaged spawn: far behind, facing away
                behind_dir = rng.uniform(0, 2 * np.pi)
                far_offset = np.array([
                    ASYMMETRIC_DIST_FAR * np.cos(behind_dir),
                    ASYMMETRIC_DIST_FAR * np.sin(behind_dir), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster_center + far_offset
                # Face away from cluster center
                away_hdg = float(np.degrees(np.arctan2(-far_offset[1], -far_offset[0]))) % 360.0
                away_hdg += rng.uniform(-ASYMMETRIC_HEADING_OFF / 2, ASYMMETRIC_HEADING_OFF / 2)
                ps.ref_hdg = away_hdg % 360.0
            else:
                # Normal / advantaged spawn
                offset = np.array([rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster_center + offset
                ps.ref_hdg = float(ps.aircraft.state["yaw_deg"])

            ps.fc.reset()
            ps.envelope.reset()
            ps.autopilot.reset(initial_speed_mps=200.0)
            ps.ref_alt_m = 3000.0
            ps.reset_state()

        if not asymmetric:
            self._striker_idx = rng.integers(0, self.N)  # random striker for symmetric reset

        for j, ts in enumerate(self.targets):
            target_dist = rng.uniform(900 + d * 1100, 1300 + d * 1700)
            bearing_offset = rng.uniform(-d * 45.0, d * 45.0)
            heading_diff = rng.uniform(-d * 30.0, d * 30.0)

            # Target spawn relative to pursuer cluster center
            pursuer_hdg = float(self.pursuers[0].aircraft.state["yaw_deg"])
            target_bearing = (pursuer_hdg + bearing_offset) % 360.0
            target_hdg = (pursuer_hdg + heading_diff) % 360.0

            target_ned = cluster_center + np.array([
                target_dist * np.cos(np.radians(target_bearing)),
                target_dist * np.sin(np.radians(target_bearing)),
                0.0])
            target_ned[2] = 3000.0

            ts.aircraft.reset(
                lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                heading_deg=target_hdg, speed_kts=310, trim=False)
            ts.aircraft.position_ned = target_ned
            ts.fc.reset()
            ts.envelope.reset()
            ts.autopilot.reset(initial_speed_mps=160.0)
            ts.ref_hdg = target_hdg
            ts.ref_alt_m = 3000.0

        # ── Warmup: 3s level flight for all ───────────────────────────
        warmup_steps = int(3.0 * CTRL_FREQ)
        for _ in range(warmup_steps):
            for ps in self.pursuers:
                s = ps.aircraft.state
                tgt = FlightControlTargets(heading_deg=ps.ref_hdg, altitude_m=3000.0,
                                           speed_mps=kts_to_mps(400))
                thr, elev, ail, rud = ps.fc.compute(s, tgt, PHYSICS_DT)
                ps.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += ps.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ps.aircraft.position_ned[2] = s["alt_m"]

            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(heading_deg=ts.ref_hdg, altitude_m=3000.0,
                                           speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, PHYSICS_DT)
                ts.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += ts.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ts.aircraft.position_ned[2] = s["alt_m"]

        # ── Post-warmup init ──────────────────────────────────────────
        for ps in self.pursuers:
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - self.targets[0].aircraft.position_ned))
            ps.episode_start_dist = ps.prev_dist

        self._step_counter = 0
        self._tacview_frames = []

        if self.record_tacview:
            self._record_tacview_frame(0.0)

        return self._get_obs(), {}

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        dt = PHYSICS_DT
        action = np.clip(action, -1.0, 1.0)
        total_reward = 0.0

        # Parse per-pursuer actions
        pursuer_actions = []
        for i in range(self.N):
            a = action[2*i:2*i+2]
            pursuer_actions.append({
                'turn': float(a[0]),
                'speed': float(a[1]),
                'cmd_turn_rate': float(a[0] * 15.0),
                'cmd_speed': float(250.0 + a[1] * 100.0),
            })

        terminated = False
        truncated = False
        reason = "timeout"
        kill_info = {"killer": -1, "min_dist": 9999.0}

        # Track per-pursuer closure for formation reward conditioning
        initial_dists = [ps.prev_dist for ps in self.pursuers]

        # ═══════════════════════════════════════════════════════════════
        #  Micro-step loop
        # ═══════════════════════════════════════════════════════════════
        for _ in range(DECISION_STEPS):
            # ── Control all pursuers ──────────────────────────────────
            for i, (ps, pa) in enumerate(zip(self.pursuers, pursuer_actions)):
                s = ps.aircraft.state
                ps.ref_hdg = (ps.ref_hdg + pa['cmd_turn_rate'] * dt) % 360.0
                fc_tgt = FlightControlTargets(
                    heading_deg=ps.ref_hdg, altitude_m=ps.ref_alt_m,
                    speed_mps=pa['cmd_speed'])
                thr, elev, ail, rud = ps.fc.compute(s, fc_tgt, dt)
                ps.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ps.last_action = np.array([pa['turn'], pa['speed']], dtype=np.float32)

            # ── Control all targets (straight-and-level in 2D) ────────
            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ts.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, dt)
                ts.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)

            # ── Physics step ──────────────────────────────────────────
            for ps in self.pursuers:
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += ps.aircraft.velocity_ned[0:2] * dt
                ps.aircraft.position_ned[2] = ps.aircraft.state["alt_m"]
            for ts in self.targets:
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += ts.aircraft.velocity_ned[0:2] * dt
                ts.aircraft.position_ned[2] = ts.aircraft.state["alt_m"]

            self._step_counter += 1

            # ── NaN guard ─────────────────────────────────────────────
            for ps in self.pursuers:
                if any(not np.isfinite(float(ps.aircraft.state[k]))
                       for k in ["n_z_g", "airspeed_mps", "alt_m"]):
                    total_reward += REWARD_CRASH
                    terminated = True
                    reason = "jsbsim_nan"
                    break
            if terminated:
                break

            # ── Per-pursuer reward + target-relative geometry ─────────
            pursuer_dists = []
            pursuer_geos = []
            all_closing = True

            for i, ps in enumerate(self.pursuers):
                t_pos = self.targets[0].aircraft.position_ned
                a_pos = ps.aircraft.position_ned
                current_dist = float(np.linalg.norm(a_pos - t_pos))
                pursuer_dists.append(current_dist)

                # Track minimum distance
                if current_dist < kill_info["min_dist"]:
                    kill_info["min_dist"] = current_dist

                delta_dist = ps.prev_dist - current_dist
                if delta_dist <= 0:
                    all_closing = False

                # ── Progress reward ────────────────────────────────
                prog = REWARD_PROGRESS * delta_dist * 0.5
                total_reward += prog
                if current_dist < 500.0:
                    total_reward += REWARD_PROGRESS * delta_dist * 5.0

                # ── ATA reward (distance-gated) ────────────────────
                a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
                t_fwd = compute_forward_vector(self.targets[0].aircraft.rpy_rad)
                _, los_dir, _ = compute_los(a_pos, t_pos)
                geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
                pursuer_geos.append(geo)

                dist_factor = max(0.0, 1.0 - current_dist / 3000.0)
                ata_r = REWARD_ATA * max(geo["cos_ata"], -0.2) * dt * dist_factor
                total_reward += ata_r

                # ── Terminal pull (ATA-gated) ──────────────────────
                if 200.0 <= current_dist <= 500.0:
                    ata_gate = max(0.0, float(geo["cos_ata"]) ** 3)
                    terminal_pull = (500.0 - current_dist) / 300.0 * TERMINAL_PULL_MAX * dt * ata_gate
                    total_reward += terminal_pull

                # ── ATA degradation penalty ────────────────────────
                if current_dist < 1000.0:
                    ata_deg_now = float(np.degrees(
                        np.arccos(np.clip(geo["cos_ata"], -1.0, 1.0))))
                    if ata_deg_now > ATA_DEGRADATION_THRESH:
                        total_reward -= ATA_DEGRADATION_WEIGHT * dt * self._ata_penalty_weight

                # ── Low-speed warning ──────────────────────────────
                spd = float(ps.aircraft.state["airspeed_mps"])
                if spd < ANTI_STALL_SPEED_WARN:
                    spd_deficit = (ANTI_STALL_SPEED_WARN - spd) / ANTI_STALL_SPEED_WARN
                    total_reward -= ANTI_STALL_SPEED_WARN_WEIGHT * spd_deficit * dt

                # ── Baseline bleed ─────────────────────────────────
                total_reward -= 1.0 * dt

                # ── Proximity milestones ───────────────────────────
                for threshold, bonus in PROXIMITY_TIERS:
                    if current_dist < threshold and threshold not in ps.proximity_awarded:
                        total_reward += bonus
                        ps.proximity_awarded.add(threshold)

                # ── Lost target check ──────────────────────────────
                if current_dist > 10000.0:
                    total_reward += REWARD_LOST_TARGET
                    terminated = True
                    reason = "lost_target"
                    break

                ps.prev_dist = current_dist

            if terminated:
                break

            # ═══════════════════════════════════════════════════════════
            #  Phase 5: Cooperative 2v1 (pincer + dynamic roles)
            # ═══════════════════════════════════════════════════════════
            if self.cooperative_mode and self.N >= 2:
                d0, d1 = pursuer_dists[0], pursuer_dists[1]

                # ── Pincer angle: angle between the two LOS vectors ──
                p0_pos = self.pursuers[0].aircraft.position_ned
                p1_pos = self.pursuers[1].aircraft.position_ned
                t_pos = self.targets[0].aircraft.position_ned
                los0 = t_pos - p0_pos; los1 = t_pos - p1_pos
                los0_h = los0[:2]; los1_h = los1[:2]
                norm0 = float(np.linalg.norm(los0_h)); norm1 = float(np.linalg.norm(los1_h))

                if norm0 > 1.0 and norm1 > 1.0:
                    cos_pincer = np.clip(float(np.dot(los0_h, los1_h)) / (norm0 * norm1), -1.0, 1.0)
                    pincer_angle = float(np.degrees(np.arccos(cos_pincer)))
                else:
                    pincer_angle = 0.0

                both_in_range = (d0 < PINCER_DIST_MAX and d1 < PINCER_DIST_MAX)

                # Reward pincer angles in IDEAL range (wider = better flanking)
                if both_in_range and PINCER_IDEAL_ANGLE_MIN <= pincer_angle <= PINCER_IDEAL_ANGLE_MAX:
                    # Normalize angle to [0, 1] within reward range
                    angle_quality = (pincer_angle - PINCER_IDEAL_ANGLE_MIN) / \
                                    (PINCER_IDEAL_ANGLE_MAX - PINCER_IDEAL_ANGLE_MIN)
                    pincer_r = PINCER_WEIGHT * angle_quality * dt
                    total_reward += pincer_r

                # ── Dynamic role assignment ─────────────────────────
                closer_idx = 0 if d0 <= d1 else 1
                further_idx = 1 if closer_idx == 0 else 0

                # Striker (closer): bonus on tracking/target-relative rewards
                striker_geo = pursuer_geos[closer_idx]
                striker_ata = float(striker_geo["cos_ata"])
                striker_dist = pursuer_dists[closer_idx]
                striker_factor = max(0.0, 1.0 - striker_dist / 3000.0)
                striker_bonus = STRIKER_TRACKING_BONUS * max(striker_ata, -0.2) * dt * striker_factor
                total_reward += striker_bonus

                # Interceptor (further): bonus on pincer angle maintenance
                if both_in_range and pincer_angle >= PINCER_IDEAL_ANGLE_MIN:
                    interceptor_bonus = INTERCEPTOR_PINCER_BONUS * (pincer_angle / 180.0) * dt
                    total_reward += interceptor_bonus

                # ── Cooperative success (AND-gate, curriculum-annealed) ─
                both_in_kill_zone = (d0 < self._coop_success_dist and d1 < self._coop_success_dist
                                     and pincer_angle >= self._coop_success_angle)

                if both_in_kill_zone:
                    self._coop_sustain_counter += 1
                else:
                    self._coop_sustain_counter = 0

                if self._coop_sustain_counter >= COOP_SUSTAIN_STEPS:
                    total_reward += REWARD_SUCCESS
                    # Extra cooperative bonus: bigger for better pincer
                    coop_bonus = 2000.0 * (pincer_angle / 180.0)
                    total_reward += coop_bonus
                    terminated = True
                    reason = "cooperative_success"
                    kill_info["killer"] = closer_idx
                    break

            else:
                # ── Legacy: single-pursuer success (any within 200m) ──
                for i, ps in enumerate(self.pursuers):
                    if pursuer_dists[i] < 200.0 and ps.episode_start_dist > 400.0:
                        total_reward += REWARD_SUCCESS
                        terminated = True
                        reason = "success"
                        kill_info["killer"] = i
                        break

            # ── Collision between pursuers ────────────────────────────
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    pi = self.pursuers[i].aircraft.position_ned
                    pj = self.pursuers[j].aircraft.position_ned
                    mate_dist = float(np.linalg.norm(pi - pj))
                    if mate_dist < FORMATION_COLLISION_DIST:
                        total_reward += FORMATION_COLLISION_PENALTY
                        terminated = True
                        reason = "formation_collision"
                        break
                if terminated:
                    break
            if terminated:
                break

            # ── Formation spacing (Phase 4.1: piecewise + gated) ────
            if self.N >= 2 and self._formation_weight > 0.0:
                pi = self.pursuers[0].aircraft.position_ned
                pj = self.pursuers[1].aircraft.position_ned
                mate_dist = float(np.linalg.norm(pi - pj))

                if mate_dist < SPACING_DANGER:
                    # Danger zone: strong fixed penalty
                    total_reward += SPACING_DANGER_PENALTY * self._formation_weight * dt
                elif mate_dist < SPACING_REPEL_MAX:
                    # Repulsion buffer: linear decay from max penalty to zero
                    frac = (mate_dist - SPACING_DANGER) / (SPACING_REPEL_MAX - SPACING_DANGER)
                    penalty = SPACING_DANGER_PENALTY * (1.0 - frac)
                    total_reward += penalty * self._formation_weight * dt
                elif mate_dist <= SPACING_IDEAL_HI and all_closing:
                    # Ideal zone: small positive reward, GATED on both closing
                    frac = 1.0 - abs(mate_dist - (SPACING_IDEAL_LO + SPACING_IDEAL_HI) / 2.0) / \
                        ((SPACING_IDEAL_HI - SPACING_IDEAL_LO) / 2.0)
                    bonus = SPACING_REWARD_CAP * max(0.0, frac)
                    total_reward += bonus * self._formation_weight * dt
                # d > 500m: no reward/penalty (spread too far)

            # ── Ground / ceiling checks ───────────────────────────────
            for ps in self.pursuers:
                alt = ps.aircraft.position_ned[2]
                if alt < 10.0:
                    total_reward += REWARD_CRASH
                    terminated = True
                    reason = "ground_crash"
                    break
                if alt > 12000.0:
                    terminated = True
                    reason = "out_of_bounds"
                    break
            if terminated:
                break

        # ── Post-loop: timeout ────────────────────────────────────────
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and not truncated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"
            total_reward += REWARD_TIMEOUT

        if self.record_tacview:
            self._record_tacview_frame(current_time)

        info = {
            "reason": reason,
            "termination_reason": reason,
            "kill_info": kill_info,
            "total_reward": total_reward,
        }
        return self._get_obs(), total_reward, terminated, truncated, info

    # ── Observation ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Concatenated per-pursuer observations with mate info."""
        all_obs = []
        target_pos = self.targets[0].aircraft.position_ned
        target_vel = self.targets[0].aircraft.velocity_ned

        for i, ps in enumerate(self.pursuers):
            # ── 27-dim base obs (same logic as ContinuousPursuitEnv) ──
            a_pos = ps.aircraft.position_ned
            a_rpy = ps.aircraft.rpy_rad
            a_vel = ps.aircraft.velocity_ned
            t_vel = target_vel

            rel_pos_world = target_pos - a_pos
            cos_hdg = np.cos(a_rpy[2])
            sin_hdg = np.sin(a_rpy[2])
            rel_pos_body = np.array([
                rel_pos_world[0] * cos_hdg + rel_pos_world[1] * sin_hdg,
                -rel_pos_world[0] * sin_hdg + rel_pos_world[1] * cos_hdg,
                -rel_pos_world[2],
            ])
            vel_body = np.array([
                a_vel[0] * cos_hdg + a_vel[1] * sin_hdg,
                -a_vel[0] * sin_hdg + a_vel[1] * cos_hdg,
                a_vel[2],
            ])
            t_vel_body = np.array([
                t_vel[0] * cos_hdg + t_vel[1] * sin_hdg,
                -t_vel[0] * sin_hdg + t_vel[1] * cos_hdg,
                t_vel[2],
            ])
            a_ang_vel = self._compute_ang_vel(ps.aircraft.rpy_rad, ps.prev_rpy)
            ps.prev_rpy = ps.aircraft.rpy_rad.copy()

            a_fwd = compute_forward_vector(a_rpy)
            t_fwd = compute_forward_vector(self.targets[0].aircraft.rpy_rad)
            _, los_dir, _ = compute_los(a_pos, target_pos)
            geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)

            airspeed = float(ps.aircraft.state["airspeed_mps"])
            alpha = float(ps.aircraft.state["alpha_deg"])

            # LOS rate
            r_h = target_pos[:2] - a_pos[:2]
            dist_h = float(np.linalg.norm(r_h))
            if dist_h > 1.0:
                v_rel_h = target_vel[:2] - a_vel[:2]
                lambda_dot = float(np.cross(r_h, v_rel_h)) / (dist_h * dist_h)
                lambda_dot_norm = float(np.clip(lambda_dot / MAX_LOS_RATE, -1.0, 1.0))
            else:
                lambda_dot_norm = 0.0
            bearing_deg = float(np.degrees(np.arctan2(r_h[1], r_h[0]))) % 360.0
            hdg_deg = float(ps.aircraft.state["yaw_deg"]) % 360.0
            bearing_err = (bearing_deg - hdg_deg + 180.0) % 360.0 - 180.0
            bearing_err_norm = float(np.clip(bearing_err / 180.0, -1.0, 1.0))

            base_obs = np.array([
                rel_pos_body[0] / MAX_DIST, rel_pos_body[1] / MAX_DIST,
                rel_pos_body[2] / MAX_DIST,
                vel_body[0] / MAX_VEL, vel_body[1] / MAX_VEL, vel_body[2] / MAX_VEL,
                a_rpy[0] / np.pi, a_rpy[1] / (np.pi / 2), a_rpy[2] / np.pi,
                a_ang_vel[0] / MAX_ANG_VEL, a_ang_vel[1] / MAX_ANG_VEL,
                a_ang_vel[2] / MAX_ANG_VEL,
                a_pos[2] / MAX_HEIGHT,
                t_vel_body[0] / MAX_VEL, t_vel_body[1] / MAX_VEL, t_vel_body[2] / MAX_VEL,
                0.0, 0.0, 0.0,  # target ang_vel (not tracked)
                geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
                alpha / MAX_AOA, airspeed / MAX_VEL, 0.0,  # Ps placeholder
                lambda_dot_norm, bearing_err_norm,
            ], dtype=np.float32)

            # ── Mate observation (6 dims: rel pos + rel vel) ──────────
            if self.N >= 2:
                # Find nearest mate
                mate_idx = 1 if i == 0 else 0
                mate_pos = self.pursuers[mate_idx].aircraft.position_ned
                mate_vel = self.pursuers[mate_idx].aircraft.velocity_ned

                mate_rel_world = mate_pos - a_pos
                mate_rel_body = np.array([
                    mate_rel_world[0] * cos_hdg + mate_rel_world[1] * sin_hdg,
                    -mate_rel_world[0] * sin_hdg + mate_rel_world[1] * cos_hdg,
                    -mate_rel_world[2],
                ])
                mate_vel_rel = mate_vel - a_vel
                mate_vel_body = np.array([
                    mate_vel_rel[0] * cos_hdg + mate_vel_rel[1] * sin_hdg,
                    -mate_vel_rel[0] * sin_hdg + mate_vel_rel[1] * cos_hdg,
                    mate_vel_rel[2],
                ])
                mate_obs = np.array([
                    mate_rel_body[0] / MAX_DIST, mate_rel_body[1] / MAX_DIST,
                    mate_rel_body[2] / MAX_DIST,
                    mate_vel_body[0] / MAX_VEL, mate_vel_body[1] / MAX_VEL,
                    mate_vel_body[2] / MAX_VEL,
                ], dtype=np.float32)
            else:
                mate_obs = np.zeros(6, dtype=np.float32)

            all_obs.append(np.concatenate([base_obs, mate_obs]))

        return np.clip(np.concatenate(all_obs), -1.0, 1.0).astype(np.float32)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _compute_ang_vel(self, current_rpy, prev_rpy):
        diff = current_rpy - prev_rpy
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff / PHYSICS_DT

    def _record_tacview_frame(self, t):
        """Record all aircraft state for Tacview export."""
        frame = {"time": t, "aircraft": []}
        for i, ps in enumerate(self.pursuers):
            s = ps.aircraft.state
            p_lat, p_lon = self._ned_to_latlon(ps.aircraft.position_ned)
            frame["aircraft"].append({
                "id": 101 + i, "name": f"F-16 Pursuer {i}",
                "lat_deg": p_lat, "lon_deg": p_lon,
                "alt_m": ps.aircraft.position_ned[2],
                "roll_deg": s["roll_deg"], "pitch_deg": s["pitch_deg"],
                "yaw_deg": s["yaw_deg"],
            })
        for j, ts in enumerate(self.targets):
            s = ts.aircraft.state
            t_lat, t_lon = self._ned_to_latlon(ts.aircraft.position_ned)
            frame["aircraft"].append({
                "id": 201 + j, "name": f"F-16 Target {j}",
                "lat_deg": t_lat, "lon_deg": t_lon,
                "alt_m": ts.aircraft.position_ned[2],
                "roll_deg": s["roll_deg"], "pitch_deg": s["pitch_deg"],
                "yaw_deg": s["yaw_deg"],
            })
        self._tacview_frames.append(frame)

    def _ned_to_latlon(self, ned):
        ref_lat, ref_lon, _ = self._ref_lla
        lat = ref_lat + ned[0] / 111320.0
        lon = ref_lon + ned[1] / (111320.0 * np.cos(np.radians(ref_lat)))
        return float(lat), float(lon)

    def export_tacview(self, path):
        """Export recorded frames as Tacview ACMI 2.2 file.

        Writes object definitions (Name, Color, Type) BEFORE any telemetry,
        in compliance with the ACMI specification. Without this ordering,
        Tacview cannot instantiate the aircraft objects.
        """
        if not self._tacview_frames:
            print(f"  [WARN] export_tacview: no frames recorded (use record_tacview=True)")
            return

        with open(path, "w", encoding="utf-8-sig") as f:
            # ── Global header ──────────────────────────────────────────
            f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
            f.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")

            # ── Object definitions (MUST precede all telemetry) ─────────
            seen_ids = set()
            for frame in self._tacview_frames:
                for ac in frame["aircraft"]:
                    aid = ac["id"]
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        color = "Red" if "Pursuer" in ac["name"] else "Blue"
                        f.write(f"{aid},Name={ac['name']}\n")
                        f.write(f"{aid},Color={color}\n")
                        if "Pursuer" in ac["name"]:
                            f.write(f"{aid},Coalition=Enemies\n")

            # ── Telemetry ───────────────────────────────────────────────
            for frame in self._tacview_frames:
                f.write(f"#{frame['time']:.2f}\n")
                for ac in frame["aircraft"]:
                    # T = longitude|latitude|altitude_m|roll|pitch|yaw
                    f.write(f"{ac['id']},T={ac['lon_deg']:.8f}|{ac['lat_deg']:.8f}|{ac['alt_m']:.1f}"
                            f"|{ac['roll_deg']:.1f}|{ac['pitch_deg']:.1f}|{ac['yaw_deg']:.1f}\n")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def difficulty_level(self) -> float:
        return self._difficulty

    def set_ata_penalty_weight(self, w: float) -> None:
        self._ata_penalty_weight = float(np.clip(w, 0.0, 1.0))

    def set_formation_weight(self, w: float) -> None:
        """Dynamic formation spacing weight (0→1 via annealing)."""
        self._formation_weight = float(np.clip(w, 0.0, 1.0))

    def set_coop_curriculum(self, progress: float) -> None:
        """Anneal AND-gate criteria from easy→strict.

        progress=0.0: dist<800m, angle>20deg (easy to discover)
        progress=1.0: dist<300m, angle>60deg (strict pincer)

        The curriculum window is [CURRICULUM_START, CURRICULUM_END].
        Before START: full easy. After END: full strict.
        """
        self._coop_curriculum_progress = float(np.clip(progress, 0.0, 1.0))
        if progress < COOP_CURRICULUM_START:
            frac = 0.0
        elif progress > COOP_CURRICULUM_END:
            frac = 1.0
        else:
            frac = (progress - COOP_CURRICULUM_START) / (COOP_CURRICULUM_END - COOP_CURRICULUM_START)
        # Cosine easing for smooth transition
        frac = (1.0 - np.cos(np.pi * frac)) / 2.0
        self._coop_success_dist = COOP_SUCCESS_DIST_EASY + frac * (COOP_SUCCESS_DIST_HARD - COOP_SUCCESS_DIST_EASY)
        self._coop_success_angle = COOP_SUCCESS_ANGLE_EASY + frac * (COOP_SUCCESS_ANGLE_HARD - COOP_SUCCESS_ANGLE_EASY)
