"""Formation RLlib MultiAgentEnv — Phase 5 Cooperative 2v1 Pursuit.

RLlib-native MultiAgentEnv for CTDE MAPPO training.  Each pursuer is an
independent agent with Box(2) action and 33-dim local observation.
A centralized critic sees 21-dim global state (3 entities × 7 features).

Key differences from FormationEnv (SB3 prototype):
  - Returns Dict[str, ...] for obs, rewards, terminateds, truncateds, infos
  - Per-agent rewards (not global total)
  - "__all__" key in terminateds/truncateds for RLlib episode termination
  - Action clamping on every step() call (DiagGaussian unbounded sampling fix)

Agent IDs: "p0", "p1"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.utils.units import kts_to_mps
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants (shared with formation_env.py)
# ═══════════════════════════════════════════════════════════════════════════════

CTRL_FREQ = 60.0
PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.5
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)
MAX_EPISODE_TIME = 120.0

MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi
MAX_AOA = 30.0
MAX_LOS_RATE = 0.5

# Reward weights
REWARD_PROGRESS = 1.5
REWARD_ATA = 8.0
REWARD_SUCCESS = 5000.0
REWARD_CRASH = -200.0
REWARD_LOST_TARGET = -200.0
REWARD_TIMEOUT = -500.0
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

# Formation collision
FORMATION_COLLISION_DIST = 50.0
FORMATION_COLLISION_PENALTY = -3000.0

# ── Phase 5: Cooperative 2v1 ───────────────────────────────────────────────
PINCER_IDEAL_ANGLE_MIN = 60.0
PINCER_IDEAL_ANGLE_MAX = 150.0
PINCER_WEIGHT = 15.0
PINCER_DIST_MAX = 2000.0

COOP_PHASE_OR = 0
COOP_PHASE_AND = 1
COOP_PHASE1_OR_DIST = 200.0
COOP_PHASE2_AND_DIST = 800.0
COOP_PHASE2_AND_ANGLE = 30.0
COOP_SUSTAIN_STEPS = 6

STRIKER_TRACKING_BONUS = 1.5
INTERCEPTOR_PINCER_BONUS = 2.0

ASYMMETRIC_RESET_PROB = 0.7
ASYMMETRIC_DIST_FAR = 1500.0
ASYMMETRIC_HEADING_OFF = 120.0

# Global state: per-entity features
GLOBAL_DIM_PER_AIRCRAFT = 7  # pos(3) + vel(3) + heading(1)
OBS_PER_PURSUER = 33


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal state dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Pursuer:
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    ref_hdg: float = 0.0
    ref_alt_m: float = 3000.0
    prev_dist: float = 0.0
    prev_ata_deg: Optional[float] = None
    prev_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_airspeed: float = 180.0
    proximity_awarded: set = field(default_factory=set)
    closure_rates: deque = field(default_factory=lambda: deque(maxlen=ANTI_STALL_WINDOW))
    zone_death_counter: int = 0
    loiter_time: float = 0.0
    episode_start_dist: float = 0.0

    def reset_state(self):
        self.prev_ata_deg = None
        self.proximity_awarded.clear()
        self.closure_rates.clear()
        self.zone_death_counter = 0
        self.loiter_time = 0.0


@dataclass
class _Target:
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    ref_hdg: float = 0.0
    ref_alt_m: float = 3000.0


# ═══════════════════════════════════════════════════════════════════════════════
#  RLlib MultiAgentEnv
# ═══════════════════════════════════════════════════════════════════════════════

class FormationRLlibEnv(MultiAgentEnv):
    """2v1 cooperative formation pursuit for RLlib MAPPO (CTDE).

    Agents: "p0", "p1"
    Action per agent: Box(2) [turn_rate_factor, speed_factor]
    Obs per agent: Dict {"obs": Box(33), "global_state": Box(21)}

    Cooperative features (Phase 5):
      - Pincer angle reward (60°–150°)
      - Dynamic Striker/Interceptor role assignment
      - AND-gate success (800m/30°, 6-step sustain)
      - OR-gate Phase 1 warmup (200m single-pursuer)
      - Asymmetric resets (70% prob, 1500m behind)
      - Two-phase training via set_coop_phase()
    """

    metadata = {"name": "formation_rllib_v0"}

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        config = env_config or {}
        self._difficulty = float(np.clip(config.get("difficulty_level", 0.0), 0.0, 1.0))
        self._lock_altitude = config.get("lock_altitude", True)
        self._record_tacview = config.get("record_tacview", False)
        self._ref_lla = (30.0, 120.0, 3000.0)
        self._tacview_frames: List[dict] = []

        self.N = 2  # pursuers
        self.M = 1  # targets
        self._agent_ids = ["p0", "p1"]

        # Cooperative state
        self._striker_idx: int = 0
        self._coop_sustain_counter: int = 0
        self._coop_phase: int = COOP_PHASE_OR

        # ── Build aircraft ──────────────────────────────────────────────
        self.pursuers: List[_Pursuer] = []
        for _ in range(self.N):
            ac = Aircraft(config.get("jsbsim_data_dir"))
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.pursuers.append(_Pursuer(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap))

        self.targets: List[_Target] = []
        for _ in range(self.M):
            ac = Aircraft(config.get("jsbsim_data_dir"))
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.targets.append(_Target(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap))

        # ── Spaces ──────────────────────────────────────────────────────
        self._global_dim = (self.N + self.M) * GLOBAL_DIM_PER_AIRCRAFT  # 21

        single_obs = gym.spaces.Dict({
            "obs": gym.spaces.Box(-1.0, 1.0, (OBS_PER_PURSUER,), dtype=np.float32),
            "global_state": gym.spaces.Box(-1.0, 1.0, (self._global_dim,), dtype=np.float32),
        })
        single_act = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            aid: single_obs for aid in self._agent_ids
        })
        self.action_space = gym.spaces.Dict({
            aid: single_act for aid in self._agent_ids
        })

        # RLlib requires _agent_ids to be a property or accessible list
        self._agent_ids = list(self._agent_ids)
        self._step_counter = 0

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Reset environment. Returns (obs_dict, info_dict) tuple."""
        rng = np.random.default_rng(seed)
        d = self._difficulty

        # Cluster center for spawns
        cluster = np.array([rng.uniform(-200, 200),
                           rng.uniform(-200, 200), 3000.0])

        # ── Asymmetric reset (cooperative mode) ────────────────────────
        asymmetric = False
        disadvantaged_idx = 0
        if self.N >= 2 and rng.random() < ASYMMETRIC_RESET_PROB:
            asymmetric = True
            disadvantaged_idx = rng.integers(0, self.N)
            self._striker_idx = 1 if disadvantaged_idx == 0 else 0

        for i, ps in enumerate(self.pursuers):
            if asymmetric and i == disadvantaged_idx:
                behind_dir = rng.uniform(0, 2 * np.pi)
                far_offset = np.array([
                    ASYMMETRIC_DIST_FAR * np.cos(behind_dir),
                    ASYMMETRIC_DIST_FAR * np.sin(behind_dir), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster + far_offset
                away_hdg = float(np.degrees(np.arctan2(
                    -far_offset[1], -far_offset[0]))) % 360.0
                away_hdg += rng.uniform(-ASYMMETRIC_HEADING_OFF / 2,
                                       ASYMMETRIC_HEADING_OFF / 2)
                ps.ref_hdg = away_hdg % 360.0
            else:
                offset = np.array([rng.uniform(-100, 100),
                                  rng.uniform(-100, 100), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster + offset
                ps.ref_hdg = float(ps.aircraft.state["yaw_deg"])

            ps.fc.reset()
            ps.envelope.reset()
            ps.autopilot.reset(initial_speed_mps=200.0)
            ps.ref_alt_m = 3000.0
            ps.reset_state()

        if not asymmetric:
            self._striker_idx = rng.integers(0, self.N)

        # Target spawn
        for j, ts in enumerate(self.targets):
            target_dist = rng.uniform(900 + d * 1100, 1300 + d * 1700)
            bearing_offset = rng.uniform(-d * 45.0, d * 45.0)
            heading_diff = rng.uniform(-d * 30.0, d * 30.0)

            pursuer_hdg = float(self.pursuers[0].aircraft.state["yaw_deg"])
            target_bearing = (pursuer_hdg + bearing_offset) % 360.0
            target_hdg = (pursuer_hdg + heading_diff) % 360.0

            target_ned = cluster + np.array([
                target_dist * np.cos(np.radians(target_bearing)),
                target_dist * np.sin(np.radians(target_bearing)), 0.0])
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

        # Warmup: 3s level flight
        warmup = int(3.0 * CTRL_FREQ)
        for _ in range(warmup):
            for ps in self.pursuers:
                s = ps.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ps.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(400))
                thr, elev, ail, rud = ps.fc.compute(s, tgt, PHYSICS_DT)
                ps.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += \
                    ps.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ps.aircraft.position_ned[2] = s["alt_m"]

            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ts.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, PHYSICS_DT)
                ts.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += \
                    ts.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ts.aircraft.position_ned[2] = s["alt_m"]

        # Post-warmup init
        for ps in self.pursuers:
            ps.prev_dist = float(np.linalg.norm(
                ps.aircraft.position_ned - self.targets[0].aircraft.position_ned))
            ps.episode_start_dist = ps.prev_dist

        self._step_counter = 0
        self._coop_sustain_counter = 0
        self._tacview_frames = []

        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action_dict: dict):
        """Execute one macro-action (0.5s, 30 physics sub-steps).

        Args:
            action_dict: {"p0": np.array([turn, speed]), "p1": ...}

        Returns:
            (obs, rewards, terminateds, truncateds, infos) — all Dict[str, ...]
        """
        dt = PHYSICS_DT

        # 🔴 CRITICAL: Clamp actions to [-1, 1] — DiagGaussian produces unbounded samples
        for aid in list(action_dict.keys()):
            action_dict[aid] = np.clip(
                np.asarray(action_dict[aid], dtype=np.float32), -1.0, 1.0)

        # Parse per-pursuer actions
        actions = {}
        for i, aid in enumerate(self._agent_ids):
            a = action_dict.get(aid, np.zeros(2, dtype=np.float32))
            a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
            actions[aid] = {
                'turn': float(a[0]),
                'speed': float(a[1]),
                'cmd_turn_rate': float(a[0] * 15.0),
                'cmd_speed': float(250.0 + a[1] * 100.0),
            }

        terminated = False
        truncated = False
        reason = "timeout"
        kill_aid = None

        # Per-agent reward accumulators
        rewards = {aid: 0.0 for aid in self._agent_ids}

        # ═══════════════════════════════════════════════════════════════
        #  Micro-step loop (30 steps × 1/60s)
        # ═══════════════════════════════════════════════════════════════
        for _ in range(DECISION_STEPS):
            # ── Control pursuers ──────────────────────────────────────
            for i, (ps, aid) in enumerate(zip(self.pursuers, self._agent_ids)):
                ac = actions[aid]
                s = ps.aircraft.state
                ps.ref_hdg = (ps.ref_hdg + ac['cmd_turn_rate'] * dt) % 360.0
                fc_tgt = FlightControlTargets(
                    heading_deg=ps.ref_hdg, altitude_m=ps.ref_alt_m,
                    speed_mps=ac['cmd_speed'])
                thr, elev, ail, rud = ps.fc.compute(s, fc_tgt, dt)
                ps.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)

            # ── Control targets (straight-and-level) ──────────────────
            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ts.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, dt)
                ts.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)

            # ── Physics step ──────────────────────────────────────────
            for ps in self.pursuers:
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += \
                    ps.aircraft.velocity_ned[0:2] * dt
                ps.aircraft.position_ned[2] = ps.aircraft.state["alt_m"]
            for ts in self.targets:
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += \
                    ts.aircraft.velocity_ned[0:2] * dt
                ts.aircraft.position_ned[2] = ts.aircraft.state["alt_m"]
            self._step_counter += 1

            # ── NaN guard ─────────────────────────────────────────────
            for ps in self.pursuers:
                if any(not np.isfinite(float(ps.aircraft.state[k]))
                       for k in ["n_z_g", "airspeed_mps", "alt_m"]):
                    for aid in self._agent_ids:
                        rewards[aid] += REWARD_CRASH
                    terminated = True
                    reason = "jsbsim_nan"
                    break
            if terminated:
                break

            # ── Per-pursuer rewards ───────────────────────────────────
            pursuer_dists = []
            pursuer_geos = []

            for i, (ps, aid) in enumerate(zip(self.pursuers, self._agent_ids)):
                t_pos = self.targets[0].aircraft.position_ned
                a_pos = ps.aircraft.position_ned
                cur_dist = float(np.linalg.norm(a_pos - t_pos))
                pursuer_dists.append(cur_dist)

                delta = ps.prev_dist - cur_dist

                # Progress
                rewards[aid] += REWARD_PROGRESS * delta * 0.5
                if cur_dist < 500.0:
                    rewards[aid] += REWARD_PROGRESS * delta * 5.0

                # ATA (distance-gated)
                a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
                t_fwd = compute_forward_vector(
                    self.targets[0].aircraft.rpy_rad)
                _, los_dir, _ = compute_los(a_pos, t_pos)
                geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
                pursuer_geos.append(geo)

                dist_factor = max(0.0, 1.0 - cur_dist / 3000.0)
                rewards[aid] += REWARD_ATA * max(
                    geo["cos_ata"], -0.2) * dt * dist_factor

                # Terminal pull (ATA-gated)
                if 200.0 <= cur_dist <= 500.0:
                    ata_gate = max(0.0, float(geo["cos_ata"]) ** 3)
                    terminal_pull = ((500.0 - cur_dist) / 300.0 *
                                    TERMINAL_PULL_MAX * dt * ata_gate)
                    rewards[aid] += terminal_pull

                # ATA degradation penalty
                if cur_dist < 1000.0:
                    ata_deg = float(np.degrees(
                        np.arccos(np.clip(geo["cos_ata"], -1.0, 1.0))))
                    if ata_deg > ATA_DEGRADATION_THRESH:
                        rewards[aid] -= ATA_DEGRADATION_WEIGHT * dt

                # Low-speed warning
                spd = float(ps.aircraft.state["airspeed_mps"])
                if spd < ANTI_STALL_SPEED_WARN:
                    deficit = (ANTI_STALL_SPEED_WARN - spd) / ANTI_STALL_SPEED_WARN
                    rewards[aid] -= ANTI_STALL_SPEED_WARN_WEIGHT * deficit * dt

                # Baseline bleed
                rewards[aid] -= 1.0 * dt

                # Proximity milestones
                for thresh, bonus in PROXIMITY_TIERS:
                    if cur_dist < thresh and thresh not in ps.proximity_awarded:
                        rewards[aid] += bonus
                        ps.proximity_awarded.add(thresh)

                # Lost target
                if cur_dist > 10000.0:
                    rewards[aid] += REWARD_LOST_TARGET
                    terminated = True
                    reason = "lost_target"
                    break

                ps.prev_dist = cur_dist

            if terminated:
                break

            # ════════════════════════════════════════════════════════════
            #  Phase 5: Cooperative 2v1 (pincer + dynamic roles)
            # ════════════════════════════════════════════════════════════
            if self.N >= 2:
                d0, d1 = pursuer_dists[0], pursuer_dists[1]

                # Pincer angle: angle between LOS vectors (horizontal)
                p0_pos = self.pursuers[0].aircraft.position_ned
                p1_pos = self.pursuers[1].aircraft.position_ned
                t_pos = self.targets[0].aircraft.position_ned
                los0_h = (t_pos - p0_pos)[:2]
                los1_h = (t_pos - p1_pos)[:2]
                n0 = float(np.linalg.norm(los0_h))
                n1 = float(np.linalg.norm(los1_h))

                if n0 > 1.0 and n1 > 1.0:
                    cos_pincer = np.clip(
                        float(np.dot(los0_h, los1_h)) / (n0 * n1), -1.0, 1.0)
                    pincer_angle = float(np.degrees(np.arccos(cos_pincer)))
                else:
                    pincer_angle = 0.0

                both_in_range = (d0 < PINCER_DIST_MAX and d1 < PINCER_DIST_MAX)

                # Pincer reward (both in range + angle in ideal range)
                if both_in_range and PINCER_IDEAL_ANGLE_MIN <= pincer_angle <= PINCER_IDEAL_ANGLE_MAX:
                    angle_qual = ((pincer_angle - PINCER_IDEAL_ANGLE_MIN) /
                                  (PINCER_IDEAL_ANGLE_MAX - PINCER_IDEAL_ANGLE_MIN))
                    pincer_r = PINCER_WEIGHT * angle_qual * dt
                    # Split pincer reward between both agents
                    for aid in self._agent_ids:
                        rewards[aid] += pincer_r * 0.5

                # Dynamic role assignment
                closer_idx = 0 if d0 <= d1 else 1
                further_idx = 1 if closer_idx == 0 else 0

                # Striker (closer): tracking bonus
                striker_geo = pursuer_geos[closer_idx]
                striker_ata = float(striker_geo["cos_ata"])
                striker_dist = pursuer_dists[closer_idx]
                striker_factor = max(0.0, 1.0 - striker_dist / 3000.0)
                striker_bonus = (STRIKER_TRACKING_BONUS *
                                max(striker_ata, -0.2) * dt * striker_factor)
                rewards[self._agent_ids[closer_idx]] += striker_bonus

                # Interceptor (further): pincer bonus
                if both_in_range and pincer_angle >= PINCER_IDEAL_ANGLE_MIN:
                    interceptor_bonus = (INTERCEPTOR_PINCER_BONUS *
                                        (pincer_angle / 180.0) * dt)
                    rewards[self._agent_ids[further_idx]] += interceptor_bonus

                # Cooperative success (phase-aware)
                if self._coop_phase == COOP_PHASE_AND:
                    both_in_kill = (d0 < COOP_PHASE2_AND_DIST and
                                   d1 < COOP_PHASE2_AND_DIST and
                                   pincer_angle >= COOP_PHASE2_AND_ANGLE)
                    if both_in_kill:
                        self._coop_sustain_counter += 1
                    else:
                        self._coop_sustain_counter = 0

                    if self._coop_sustain_counter >= COOP_SUSTAIN_STEPS:
                        for aid in self._agent_ids:
                            rewards[aid] += REWARD_SUCCESS
                        coop_bonus = 2000.0 * (pincer_angle / 180.0)
                        # Split coop bonus
                        for aid in self._agent_ids:
                            rewards[aid] += coop_bonus * 0.5
                        terminated = True
                        reason = "cooperative_success"
                        kill_aid = self._agent_ids[closer_idx]
                        break
                else:
                    # Phase 1: OR-gate
                    for i, ps in enumerate(self.pursuers):
                        if (pursuer_dists[i] < COOP_PHASE1_OR_DIST and
                                ps.episode_start_dist > 400.0):
                            for aid in self._agent_ids:
                                rewards[aid] += REWARD_SUCCESS
                            # Light pincer guidance in OR phase
                            if pincer_angle >= 30.0:
                                pincer_bonus = 500.0 * (pincer_angle / 180.0)
                                for aid in self._agent_ids:
                                    rewards[aid] += pincer_bonus * 0.5
                            terminated = True
                            reason = "success"
                            kill_aid = self._agent_ids[i]
                            break

            # ── Collision between pursuers ────────────────────────────
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    pi = self.pursuers[i].aircraft.position_ned
                    pj = self.pursuers[j].aircraft.position_ned
                    if float(np.linalg.norm(pi - pj)) < FORMATION_COLLISION_DIST:
                        for aid in self._agent_ids:
                            rewards[aid] += FORMATION_COLLISION_PENALTY
                        terminated = True
                        reason = "formation_collision"
                        break
                if terminated:
                    break
            if terminated:
                break

            # ── Ground / ceiling checks ────────────────────────────────
            for ps in self.pursuers:
                alt = ps.aircraft.position_ned[2]
                if alt < 10.0:
                    for aid in self._agent_ids:
                        rewards[aid] += REWARD_CRASH
                    terminated = True
                    reason = "ground_crash"
                    break
                if alt > 12000.0:
                    terminated = True
                    reason = "out_of_bounds"
                    break
            if terminated:
                break

        # ── Timeout ──────────────────────────────────────────────────────
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and not truncated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"
            for aid in self._agent_ids:
                rewards[aid] += REWARD_TIMEOUT

        # ── Build RLlib-format returns ────────────────────────────────────
        obs = self._get_obs()
        info = {"reason": reason, "kill_agent": kill_aid}

        # Per-agent info
        infos = {aid: info for aid in self._agent_ids}

        # Termination dicts with "__all__" for RLlib
        terminateds = {aid: terminated for aid in self._agent_ids}
        terminateds["__all__"] = terminated or truncated

        truncateds = {aid: truncated for aid in self._agent_ids}
        truncateds["__all__"] = terminated or truncated

        return obs, rewards, terminateds, truncateds, infos

    # ── Observation ────────────────────────────────────────────────────────

    def _get_obs(self):
        """Build per-agent observation dict.

        Returns:
            {"p0": {"obs": ndarray(33), "global_state": ndarray(21)},
             "p1": {"obs": ndarray(33), "global_state": ndarray(21)}}
        """
        target_pos = self.targets[0].aircraft.position_ned
        target_vel = self.targets[0].aircraft.velocity_ned

        # Global state: all aircraft pos(3)+vel(3)+heading(1)
        # Token order for Critic: [Self, Mate, Target]
        # But global_state is the same for all — absolute, not ego-centric
        global_parts = []
        for ps in self.pursuers:
            p = ps.aircraft.position_ned / np.array(
                [MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ps.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ps.aircraft.state["yaw_deg"]) / 180.0])
            global_parts.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        for ts in self.targets:
            p = ts.aircraft.position_ned / np.array(
                [MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ts.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ts.aircraft.state["yaw_deg"]) / 180.0])
            global_parts.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        global_state = np.array(global_parts, dtype=np.float32)

        obs = {}
        for i, (ps, aid) in enumerate(zip(self.pursuers, self._agent_ids)):
            local = self._build_local_obs(i, ps, target_pos, target_vel)
            obs[aid] = {
                "obs": local.astype(np.float32),
                "global_state": global_state,
            }

        return obs

    def _build_local_obs(self, idx, ps, target_pos, target_vel):
        """33-dim per-pursuer local observation (matches FormationEnv)."""
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
        t_fwd = compute_forward_vector(
            self.targets[0].aircraft.rpy_rad)
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
            lambda_dot_norm = float(np.clip(lambda_dot / MAX_LOS_RATE, -1, 1))
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
            ang_vel[0] / MAX_ANG_VEL, ang_vel[1] / MAX_ANG_VEL,
            ang_vel[2] / MAX_ANG_VEL,
            a_pos[2] / MAX_HEIGHT,
            t_vel_body[0] / MAX_VEL, t_vel_body[1] / MAX_VEL,
            t_vel_body[2] / MAX_VEL,
            0.0, 0.0, 0.0,  # target ang_vel placeholder
            geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
            alpha / MAX_AOA, spd / MAX_VEL, 0.0,  # Ps placeholder
            lambda_dot_norm, berr_norm,
        ], dtype=np.float32)

        # Mate observation (indices 27-32)
        if self.N >= 2:
            mate_idx = 1 if idx == 0 else 0
            mp = self.pursuers[mate_idx].aircraft.position_ned
            mv = self.pursuers[mate_idx].aircraft.velocity_ned
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
            mate = np.array([
                mate_body_pos[0] / MAX_DIST, mate_body_pos[1] / MAX_DIST,
                mate_body_pos[2] / MAX_DIST,
                mate_body_vel[0] / MAX_VEL, mate_body_vel[1] / MAX_VEL,
                mate_body_vel[2] / MAX_VEL,
            ], dtype=np.float32)
        else:
            mate = np.zeros(6, dtype=np.float32)

        return np.clip(np.concatenate([base, mate]), -1, 1)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ang_vel(self, cur, prev):
        d = cur - prev
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return d / PHYSICS_DT

    def set_coop_phase(self, phase: int) -> None:
        """Switch cooperative success criteria.

        Args:
            phase: COOP_PHASE_OR (0) or COOP_PHASE_AND (1)
        """
        self._coop_phase = int(phase)

    @property
    def cooperation_phase(self) -> int:
        return self._coop_phase
