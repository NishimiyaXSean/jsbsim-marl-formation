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
DECISION_DT = 0.2        # 5 Hz decision rate (was 2 Hz)
DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)  # 12 micro-steps
MAX_EPISODE_TIME = 120.0

MAX_DIST = 10000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi
MAX_AOA = 30.0
MAX_LOS_RATE = 0.5

# Reward weights
REWARD_PROGRESS = 1.0       # was 1.5 — reduced so pincer cooperation outweighs solo rushing
REWARD_ATA = 8.0
REWARD_SUCCESS = 5000.0
REWARD_CRASH = -3000.0        # was -200 — suicide must be worse than engagement
REWARD_LOST_TARGET = -3000.0  # was -200 — suicide must never be profitable
REWARD_OOB = -3000.0          # new — out-of-bounds must equal crash/ lost penalty
REWARD_TIMEOUT = -500.0
REWARD_OR_FALLBACK = 1000.0   # one-shot in AND phase: pursuer reaches 200m (no episode termination)
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
# Potential pincer shaping: linear reward c * min(theta, AND_angle) when
# both in range. Pushes pursuers apart to build pincer angle, caps at the
# AND-gate threshold so the real payoff comes from cooperative_success.
PINCER_SHAPING_COEFF = 35.0  # c in: c * min(pincer_angle, and_angle) * dt
PINCER_DIST_MAX = 2000.0     # legacy — replaced by dynamic AND_dist in pincer gate

COOP_PHASE_OR = 0
COOP_PHASE_AND = 1
COOP_PHASE1_OR_DIST = 200.0
COOP_PHASE2_AND_DIST = 800.0       # final target (after annealing)
COOP_PHASE2_AND_DIST_INIT = 2000.0  # initial value for dynamic annealing
COOP_PHASE2_AND_ANGLE = 30.0
COOP_SUSTAIN_STEPS = 6

# Early termination: if BOTH pursuers are beyond this distance, episode is hopeless
LOST_PURSUER_DIST = 6000.0      # min(d0, d1) > 6km → irrecoverable
LOST_PURSUER_STEPS = 30         # sustain for 30 decision steps (6s) before terminating

# Out-of-Combat (OOC) penalty: per-agent anti-camping pressure.
# If a single pursuer hangs back beyond AND_dist + margin for too long,
# it gets penalized individually. This breaks the "one works, one watches" stalemate.
OOC_MARGIN = 400.0              # threshold = AND_dist + 400m (Stage 1: 1600m)
OOC_PENALTY_STEPS = 30          # sustain for 30 decision steps (6s) before penalizing
OOC_PENALTY_PER_STEP = 2.0      # penalty per agent per decision step while OOC

# Closing velocity penalty: anti-loiter outside AND envelope.
# If a pursuer is beyond AND_dist and closing slower than this threshold
# (or moving away), penalize proportional to the deficit.
CLOSING_VEL_THRESHOLD = 30.0    # m/s — must close at least 30 m/s toward target
CLOSING_VEL_PENALTY = 0.5       # penalty coefficient per m/s below threshold

STRIKER_TRACKING_BONUS = 1.5
INTERCEPTOR_PINCER_BONUS = 2.0

ASYMMETRIC_RESET_PROB = 0.7
ASYMMETRIC_DIST_FAR = 1500.0
ASYMMETRIC_HEADING_OFF = 120.0

# Distance asymmetry penalty (continuous shaping — prevents free-riding)
# Relaxed for exploration phase: wider threshold + softer weight to let policy
# experiment with more aggressive asymmetric tactics without being over-penalized.
DIST_ASYMMETRY_THRESH = 800.0   # start penalizing when dist diff > 800m (was 500)
DIST_ASYMMETRY_WEIGHT = 0.3     # penalty coefficient (was 0.5)
DIST_ASYMMETRY_NORM = 1000.0    # normalization factor

# Time-sync pacing penalty (prevents Striker from rushing ahead alone)
# Also relaxed: reduced weight so Striker can push forward more aggressively.
SYNC_PACING_STRIKER_DIST = 1200.0   # Striker within this range triggers sync check
SYNC_PACING_INTERCEPTOR_DIST = 1500.0  # Interceptor beyond this range → too far behind
SYNC_PACING_WEIGHT = 0.5              # penalty weight (was 1.0)

# Global state: per-entity features
GLOBAL_DIM_PER_AIRCRAFT = 7  # pos(3) + vel(3) + heading(1)
OBS_PER_PURSUER = 39  # was 33 → 37 (+4 broadcast) → 39 (+2 agent onehot)

# ── Discrete action space ──────────────────────────────────────────────────
# MultiDiscrete([5, 3]) = 15 tactical primitives
# Turn:  0=HardLeft, 1=SoftLeft, 2=Straight, 3=SoftRight, 4=HardRight
# Speed: 0=Slow(180m/s), 1=Cruise(250m/s), 2=Fast(320m/s)
# Turn rates are speed-dependent: slower speeds allow more aggressive turning
# because the airframe can sustain higher angular rates at lower g-loading.
#   At 180 m/s: 20°/s = 6.2g  (F-16 can sustain 7g+)
#   At 250 m/s: 15°/s = 6.7g  (near optimal corner speed)
#   At 320 m/s: 12°/s = 6.9g  (approaching structural limits)
SPEEDS     = [180.0, 250.0, 320.0]
N_SPEED = len(SPEEDS)       # 3

# Speed-dependent turn rate scaling factors (applied to base grid)
_TURN_SCALE = {0: 1.33, 1: 1.0, 2: 0.8}  # Slow: +33%, Cruise: nominal, Fast: -20%

def _get_turn_rates(speed_idx: int) -> list[float]:
    """Return turn rates scaled for the given speed band."""
    base = [-15.0, -5.0, 0.0, 5.0, 15.0]
    scale = _TURN_SCALE.get(speed_idx, 1.0)
    return [r * scale for r in base]

# Nominal turn rates (for action space definition — Cruise-speed baseline)
TURN_RATES = _get_turn_rates(1)  # [-15, -5, 0, 5, 15]
N_TURN = len(TURN_RATES)    # 5
N_ACTIONS = N_TURN + N_SPEED  # 8 total logits


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
        self.cooperative_mode = config.get("cooperative_mode", True)
        self._ref_lla = (30.0, 120.0, 3000.0)
        self._tacview_frames: List[dict] = []

        self.N = 2  # pursuers
        self.M = 1  # targets
        self._agent_ids = ["p0", "p1"]

        # Cooperative state
        self._striker_idx: int = 0
        self._coop_sustain_counter: int = 0
        self._sustain_required: int = COOP_SUSTAIN_STEPS  # dynamic per curriculum stage
        self._coop_phase: int = COOP_PHASE_OR
        self._and_dist: float = COOP_PHASE2_AND_DIST_INIT  # dynamic, annealed from outside
        self._and_angle: float = COOP_PHASE2_AND_ANGLE     # dynamic, adjusted per curriculum stage
        self._init_bearing_range: tuple = (-180.0, 180.0)   # dynamic, per-stage constraint
        self._target_dist_range: tuple = (900.0, 1300.0)    # dynamic, per-stage target spawn distance
        self._curriculum_stage: int = 0                      # 0=pre-curriculum, 1/2/3=stages
        self._last_termination_reason: str = "none"          # for sync-rate tracking
        self._reward_breakdown: dict = {}                    # per-episode reward component totals (diagnostic)
        self._lost_pursuer_steps: int = 0                    # counter for early termination
        self._ooc_counters: list[int] = [0, 0]               # per-pursuer OOC step counters
        self._or_triggered: list[bool] = [False, False]      # per-pursuer one-shot OR fallback flag
        self._last_actions: dict = {}                         # store last action per agent for broadcast

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
            "action_mask": gym.spaces.Box(0.0, 1.0, (N_ACTIONS,), dtype=np.float32),
        })
        single_act = gym.spaces.MultiDiscrete([N_TURN, N_SPEED])

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
        # Expose for sensitivity analysis scripts
        self._last_asymmetric = asymmetric
        self._last_disadvantaged = disadvantaged_idx

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
            dist_min, dist_max = self._target_dist_range
            target_dist = rng.uniform(dist_min + d * 200, dist_max + d * 500)
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

        # ── Curriculum: constrain initial bearing error ──────────────────
        if self._init_bearing_range[0] > -180.0 or self._init_bearing_range[1] < 180.0:
            self._apply_bearing_constraint()

        self._step_counter = 0
        self._coop_sustain_counter = 0
        self._ooc_counters = [0, 0]
        self._or_triggered = [False, False]
        self._tacview_frames = []

        return self._get_obs(), {}

    def _apply_bearing_constraint(self) -> None:
        """Constrain initial bearing error of each pursuer to the configured range.

        After target spawn, computes the bearing error for each pursuer. If it
        falls outside [_init_bearing_range[0], _init_bearing_range[1]], rotates
        the pursuer's heading toward the target until within range.
        """
        t_pos = self.targets[0].aircraft.position_ned
        bearing_min, bearing_max = self._init_bearing_range

        for ps in self.pursuers:
            p_pos = ps.aircraft.position_ned
            p_yaw = float(ps.aircraft.state["yaw_deg"])

            # Compute bearing to target
            vec = t_pos - p_pos
            target_bearing = float(np.degrees(np.arctan2(vec[1], vec[0]))) % 360.0

            # Compute bearing error (signed, normalized to [-180, 180])
            bearing_err = (target_bearing - p_yaw + 180.0) % 360.0 - 180.0

            if bearing_err < bearing_min:
                # Rotate pursuer to bearing_min (toward target)
                new_yaw = (target_bearing - bearing_min) % 360.0
                ps.aircraft.state["yaw_deg"] = new_yaw
                ps.ref_hdg = new_yaw
            elif bearing_err > bearing_max:
                # Rotate pursuer to bearing_max (toward target)
                new_yaw = (target_bearing - bearing_max) % 360.0
                ps.aircraft.state["yaw_deg"] = new_yaw
                ps.ref_hdg = new_yaw
            # else: bearing error within range, no adjustment needed

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action_dict: dict):
        """Execute one macro-action (0.5s, 30 physics sub-steps).

        Args:
            action_dict: {"p0": np.array([turn, speed]), "p1": ...}

        Returns:
            (obs, rewards, terminateds, truncateds, infos) — all Dict[str, ...]
        """
        dt = PHYSICS_DT

        # Parse discrete actions → physical flight control targets
        actions = {}
        for i, aid in enumerate(self._agent_ids):
            a = action_dict.get(aid, np.array([2, 1], dtype=np.int64))  # default: straight+cruise
            a = np.asarray(a, dtype=np.int64)
            turn_idx = int(np.clip(a[0], 0, N_TURN - 1))
            speed_idx = int(np.clip(a[1], 0, N_SPEED - 1))
            turn_rates = _get_turn_rates(speed_idx)  # speed-dependent scaling
            actions[aid] = {
                'turn_idx': turn_idx,
                'speed_idx': speed_idx,
                'cmd_turn_rate': turn_rates[turn_idx],
                'cmd_speed': SPEEDS[speed_idx],
            }
        self._last_actions = actions  # store for mate broadcast in observation

        terminated = False
        truncated = False
        reason = "timeout"
        kill_aid = None

        # Per-agent reward accumulators
        rewards = {aid: 0.0 for aid in self._agent_ids}

        # ═══════════════════════════════════════════════════════════════
        # ── Per-step state for closing-velocity penalty ──────────────────
        _step_start_dists = [float(np.linalg.norm(
            ps.aircraft.position_ned - self.targets[0].aircraft.position_ned))
            for ps in self.pursuers]

        # ── AND-gate sustain tracking (per decision step, not per physics sub-step) ─
        and_met_this_step = False

        #  Micro-step loop (12 steps × 1/60s)
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
            if self.cooperative_mode and self.N >= 2:
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

                # ── Pincer shaping: exponential distance-decay (no hard gate) ─
                # R = c * min(theta, AND_angle) * exp(-d0/tau) * exp(-d1/tau)
                # tau = AND_dist. At d=tau: 37%, at d=2*tau: 14% (effectively zero).
                # Continuous gradient everywhere — no dead zone, no binary cliff.
                # Far-distance angle farming is economically impossible.
                if pincer_angle > 0:
                    tau = max(self._and_dist, 100.0)  # avoid div-by-zero
                    decay = np.exp(-d0 / tau) * np.exp(-d1 / tau)
                    shaped_angle = min(pincer_angle, self._and_angle)
                    pincer_r = decay * PINCER_SHAPING_COEFF * shaped_angle * dt
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

                # Interceptor (further): pincer bonus with same exponential decay
                if pincer_angle > 0:
                    interceptor_bonus = (decay * INTERCEPTOR_PINCER_BONUS *
                                        min(pincer_angle, self._and_angle) / 180.0 * dt)
                    rewards[self._agent_ids[further_idx]] += interceptor_bonus

                # Distance asymmetry penalty (continuous — both pay when one lags)
                dist_diff = abs(d0 - d1)
                if dist_diff > DIST_ASYMMETRY_THRESH:
                    asymmetry_penalty = (DIST_ASYMMETRY_WEIGHT *
                                        (dist_diff - DIST_ASYMMETRY_THRESH) /
                                        DIST_ASYMMETRY_NORM * dt)
                    for aid in self._agent_ids:
                        rewards[aid] -= asymmetry_penalty

                # Time-sync pacing penalty — prevent striker rushing ahead alone
                # If striker is within kill zone but interceptor is far behind,
                # penalize the team for tactical desynchronization
                d_striker = pursuer_dists[self._striker_idx]
                d_interceptor = pursuer_dists[1 - self._striker_idx]
                if (d_striker < SYNC_PACING_STRIKER_DIST and
                        d_interceptor > SYNC_PACING_INTERCEPTOR_DIST):
                    sync_penalty = (SYNC_PACING_WEIGHT *
                                   (d_interceptor - d_striker) /
                                   1000.0 * dt)
                    for aid in self._agent_ids:
                        rewards[aid] -= sync_penalty

                # Cooperative success check (flag at decision-step level)
                if self._coop_phase == COOP_PHASE_AND:
                    both_in_kill = (d0 < self._and_dist and
                                   d1 < self._and_dist and
                                   pincer_angle >= self._and_angle)
                    if both_in_kill:
                        and_met_this_step = True

                    # ── One-shot OR fallback (event trigger, NOT continuous) ──
                    # Each pursuer gets +1000 ONCE per episode for reaching 200m.
                    # Flag locks after trigger — no looping, no camping at 199m.
                    for i, dist in enumerate(pursuer_dists):
                        if dist < COOP_PHASE1_OR_DIST and not self._or_triggered[i]:
                            rewards[self._agent_ids[i]] += REWARD_OR_FALLBACK
                            self._or_triggered[i] = True
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
                    for aid in self._agent_ids:
                        rewards[aid] += REWARD_OOB
                    terminated = True
                    reason = "out_of_bounds"
                    break
            if terminated:
                break

        # ── AND-gate sustain counter (per decision step, NOT per physics sub-step) ─
        if self._coop_phase == COOP_PHASE_AND and not terminated:
            if and_met_this_step:
                self._coop_sustain_counter += 1
            else:
                self._coop_sustain_counter = 0

            if self._coop_sustain_counter >= self._sustain_required:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_SUCCESS
                coop_bonus = 2000.0 * (pincer_angle / 180.0)
                for aid in self._agent_ids:
                    rewards[aid] += coop_bonus * 0.5
                terminated = True
                reason = "cooperative_success"
                kill_aid = self._agent_ids[closer_idx]

        # ── Closing velocity penalty: anti-loiter outside AND envelope ───
        # If a pursuer is beyond AND_dist and not closing fast enough (or
        # moving away), penalize. Directly targets the '79% Slow' loiter
        # strategy observed in V6 P1 autopsy.
        if not terminated:
            for i, dist in enumerate(pursuer_dists):
                if dist > self._and_dist and self._coop_phase == COOP_PHASE_AND:
                    closing = (_step_start_dists[i] - dist) / DECISION_DT  # m/s, + closing
                    if closing < CLOSING_VEL_THRESHOLD:
                        deficit = CLOSING_VEL_THRESHOLD - closing
                        penalty = CLOSING_VEL_PENALTY * deficit * DECISION_DT
                        rewards[self._agent_ids[i]] -= penalty

        # ── OOC penalty: per-agent anti-camping pressure ────────────────
        # If a pursuer hangs back beyond threshold, increment its OOC counter.
        # Once past OOC_PENALTY_STEPS, penalize every decision step.
        # Counters are independent — P1 can't hide behind P0's progress.
        # Runs in BOTH phases: warmup uses 2000m threshold, AND uses AND_dist.
        if not terminated:
            if self._coop_phase == COOP_PHASE_AND:
                ooc_threshold = self._and_dist + OOC_MARGIN
            else:
                ooc_threshold = PINCER_DIST_MAX + OOC_MARGIN  # 2400m during OR warmup
            for i, dist in enumerate(pursuer_dists):
                if dist > ooc_threshold:
                    self._ooc_counters[i] += 1
                else:
                    self._ooc_counters[i] = 0

                if self._ooc_counters[i] >= OOC_PENALTY_STEPS:
                    rewards[self._agent_ids[i]] -= OOC_PENALTY_PER_STEP

        # ── Early termination: single pursuer hopelessly lost ────────────
        if self._coop_phase == COOP_PHASE_AND and not terminated:
            min_dist = min(pursuer_dists[0], pursuer_dists[1])
            if min_dist > LOST_PURSUER_DIST:
                self._lost_pursuer_steps += 1
            else:
                self._lost_pursuer_steps = 0

            if self._lost_pursuer_steps >= LOST_PURSUER_STEPS:
                terminated = True
                reason = "lost_target"
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_TIMEOUT

        # ── Timeout ──────────────────────────────────────────────────────
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and not truncated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"
            for aid in self._agent_ids:
                rewards[aid] += REWARD_TIMEOUT

        # Store for curriculum scheduler (sync-rate tracking)
        self._last_termination_reason = reason

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
            mask = self._build_action_mask(ps)
            obs[aid] = {
                "obs": local.astype(np.float32),
                "global_state": global_state,
                "action_mask": mask.astype(np.float32),
            }

        return obs

    def _build_action_mask(self, ps) -> np.ndarray:
        """Build action mask [8] based on flight safety constraints.

        Returns binary mask where 1=allowed, 0=forbidden.
        Layout: [turn_0, ..., turn_4, speed_0, speed_1, speed_2]
        """
        mask = np.ones(N_ACTIONS, dtype=np.float32)

        airspeed = float(ps.aircraft.state["airspeed_mps"])
        alt_m = float(ps.aircraft.state["alt_m"])
        nz_g = float(ps.aircraft.state.get("n_z_g", 1.0))

        # ── Low-speed protection ──────────────────────────────────────────
        # Below stall warning: forbid slow speed and hard turns (high AoA risk)
        if airspeed < ANTI_STALL_SPEED_WARN:
            mask[5] = 0.0   # forbid speed=Slow (need energy)
            mask[0] = 0.0   # forbid HardLeft (high drag)
            mask[4] = 0.0   # forbid HardRight (high drag)

        # ── Ground proximity protection ───────────────────────────────────
        # Very low altitude: forbid hard turns (risk of spiral into ground)
        if alt_m < 200.0:
            mask[0] = 0.0   # forbid HardLeft
            mask[4] = 0.0   # forbid HardRight
            if alt_m < 100.0 and nz_g > 2.0:
                mask[0] = 0.0
                mask[1] = 0.0  # forbid SoftLeft
                mask[3] = 0.0  # forbid SoftRight
                mask[4] = 0.0  # only allow Straight

        # ── Overspeed protection ──────────────────────────────────────────
        # Near max speed: forbid Fast (structural limits)
        if airspeed > MAX_VEL * 0.95:
            mask[7] = 0.0   # forbid speed=Fast

        return mask

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

        # Agent one-hot ID: breaks symmetry so the shared policy can develop
        # consistent role preferences ("I'm P0, I flank left; I'm P1, I go right")
        agent_onehot = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float32)

        base = np.concatenate([base, agent_onehot])  # base: 27→29 dims

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

            # ── Broadcast: mate's tactical intent (4 dims) ────────────────
            mate_aid = self._agent_ids[mate_idx]
            mate_act = self._last_actions.get(mate_aid, {})
            mate_cmd_turn = mate_act.get('cmd_turn_rate', 0.0) / 20.0  # [-1, 1]
            mate_cmd_spd  = (mate_act.get('cmd_speed', 250.0) - 180.0) / 140.0  # [0, 1]
            mate_ref_hdg  = np.deg2rad(self.pursuers[mate_idx].ref_hdg)
            mate_broadcast = np.array([
                mate_cmd_turn,
                mate_cmd_spd,
                np.cos(mate_ref_hdg),   # heading intent (dim 1)
                np.sin(mate_ref_hdg),   # heading intent (dim 2)
            ], dtype=np.float32)
            mate = np.concatenate([mate, mate_broadcast])
        else:
            mate = np.zeros(10, dtype=np.float32)

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

    def set_and_distance(self, dist: float) -> None:
        """Dynamically adjust AND-gate distance threshold (for curriculum annealing).

        Args:
            dist: New AND-gate distance threshold in meters (clamped ≥ 800)
        """
        self._and_dist = max(COOP_PHASE2_AND_DIST, float(dist))

    def set_and_angle(self, angle: float) -> None:
        """Dynamically adjust AND-gate pincer angle threshold.

        Args:
            angle: New AND-gate angle threshold in degrees (clamped >= 10)
        """
        self._and_angle = max(float(angle), 10.0)

    def set_initial_bearing_range(self, min_deg: float, max_deg: float) -> None:
        """Constrain initial bearing error range for curriculum learning.

        Stage 1 (greenhouse): bearing error in [-20, +20]
        Stage 2 (envelope):   bearing error in [-45, +45]
        Stage 3 (full):       bearing error unrestricted [-180, +180]

        Args:
            min_deg: Minimum initial bearing error in degrees
            max_deg: Maximum initial bearing error in degrees
        """
        self._init_bearing_range = (float(min_deg), float(max_deg))

    def set_curriculum_stage(self, stage: int, and_dist: float, and_angle: float,
                              bearing_min: float, bearing_max: float) -> None:
        """Set all curriculum parameters atomically for a given stage.

        Args:
            stage: Curriculum stage number (1, 2, or 3)
            and_dist: AND-gate distance threshold (m)
            and_angle: AND-gate pincer angle threshold (deg)
            bearing_min: Initial bearing error lower bound (deg)
            bearing_max: Initial bearing error upper bound (deg)
        """
        self._curriculum_stage = int(stage)
        self._and_dist = max(float(and_dist), COOP_PHASE2_AND_DIST)
        self._and_angle = max(float(and_angle), 10.0)
        self._init_bearing_range = (float(bearing_min), float(bearing_max))

    def set_curriculum_stage_full(self, stage: int, and_dist: float, and_angle: float,
                                   bearing_min: float, bearing_max: float,
                                   target_dist_min: float, target_dist_max: float,
                                   sustain_steps: int = 6) -> None:
        """Set all curriculum parameters including target spawn distance.

        This extended version also controls how far the target spawns,
        ensuring pursuers start OUTSIDE the AND envelope and must
        actively close distance to succeed.

        sustain_steps: AND-gate consecutive decision steps required (Stage 1=2, 2=4, 3=6)
        """
        self._curriculum_stage = int(stage)
        self._and_dist = max(float(and_dist), COOP_PHASE2_AND_DIST)
        self._and_angle = max(float(and_angle), 10.0)
        self._init_bearing_range = (float(bearing_min), float(bearing_max))
        self._target_dist_range = (float(target_dist_min), float(target_dist_max))
        self._sustain_required = int(sustain_steps)

    @property
    def and_distance(self) -> float:
        return self._and_dist

    @property
    def cooperation_phase(self) -> int:
        return self._coop_phase
