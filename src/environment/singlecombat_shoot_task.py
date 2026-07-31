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
from .formation_task import PHYSICS_DT
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
# ── Global state (for centralized critic) ────────────────────────────────────
GLOBAL_PER_AIRCRAFT = 8    # alt, roll_sin, roll_cos, pitch_sin, pitch_cos, vc, heading_sin, heading_cos
GLOBAL_DIM = (N_PURSUERS + N_TARGETS) * GLOBAL_PER_AIRCRAFT  # 16

# ── Action space ─────────────────────────────────────────────────────────────
N_SPEED_DELTA = 3
N_HEADING_DELTA = 5
N_ALT_DELTA = 1     # frozen altitude — missile phase, not rate-fight
N_FIRE = 2
N_ACTIONS = N_SPEED_DELTA + N_HEADING_DELTA + N_ALT_DELTA + N_FIRE  # 11

OBS_DIM = SELF_DIM + TARGET_DIM + MISSILE_DIM + N_ACTIONS  # 12+7+6+11=36

DELTA_SPEEDS    = [-20.0,   0.0,  20.0]       # m/s
DELTA_HEADINGS  = [-10.0, -5.0, 0.0, 5.0, 10.0]  # degrees — gentler BFM
DELTA_ALTITUDES = [0.0]                            # frozen — missile phase, not rate-fight

# ── Missile launch parameters (WEZ: Weapons Engagement Zone) ──────────────────
MAX_ATTACK_ANGLE = 15.0         # degrees — must be precisely on-target
MAX_ATTACK_DISTANCE = 8000.0    # meters — missile effective range
MIN_ATTACK_DISTANCE = 1500.0    # meters — avoid self-destruct at point-blank
MIN_ATTACK_INTERVAL = 30        # decision steps (6s at 5Hz) — conserve ammo
NUM_MISSILES = 4                # per aircraft — limited, force precision

# ── Reward weights ───────────────────────────────────────────────────────────
REWARD_HIT_BASE = 1000.0        # guaranteed for any hit within lethal radius
REWARD_HIT_BONUS = 1000.0       # scaled by accuracy: 0m→+1000, 300m→+0
REWARD_SHOTDOWN = -2000.0       # hit by enemy missile
REWARD_CRASH = -2000.0          # low altitude / overstress
REWARD_SHOOT_PENALTY = -10.0    # dry-fire penalty (should never happen with mask)

# ── Shaping weight overrides ─────────────────────────────────────────────────
PROGRESS_WEIGHT = 0.2           # reduced — hit reward dominates
ATA_WEIGHT = 1.5                # reduced — hit reward dominates


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
        # Flat observation = obs(25) + action_mask(11) = 36-dim Box
        single_obs = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        single_act = gym.spaces.MultiDiscrete(
            [N_SPEED_DELTA, N_HEADING_DELTA, N_ALT_DELTA, N_FIRE])

        self._observation_space = gym.spaces.Dict({aid: single_obs for aid in AGENT_IDS})
        self._action_space = gym.spaces.Dict({aid: single_act for aid in AGENT_IDS})

        # ── Reward modules (reuse existing with reduced shaping weights) ────
        self._progress = ProgressReward({**config, "progress_weight": PROGRESS_WEIGHT})
        self._ata = ATAAlignmentReward({**config, "ata_weight": ATA_WEIGHT})
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
        """Initialize missile state, target evasion, and combat geometry.

        Overrides BaseEnv default reset to create proper medium-range
        tail-chase or head-on geometry (2-5km range), suitable for
        learning fire discipline and approach tactics.
        """
        self._step_count = 0
        self._last_termination_reason = "none"

        # ── Missile state ───────────────────────────────────────────────────
        self.remaining_missiles = {aid: NUM_MISSILES for aid in AGENT_IDS}
        self._last_shoot_step = {aid: -MIN_ATTACK_INTERVAL for aid in AGENT_IDS}

        # ── Reward tracking ─────────────────────────────────────────────────
        self._prev_alive = {aid: True for aid in AGENT_IDS}
        self._prev_missile_count = {aid: NUM_MISSILES for aid in AGENT_IDS}
        self._has_launched_this_step: Dict[str, bool] = {aid: False for aid in AGENT_IDS}
        self._commanded_fire: Dict[str, bool] = {aid: False for aid in AGENT_IDS}

        # ── Launch quality tracking ────────────────────────────────────────
        self._launch_stats: Dict[str, list] = {
            "bad": [], "good": [], "premium": [],
            "closure_vals": [], "ata_vals": [], "range_vals": [],
        }
        self._episode_wez_first_step: Dict[str, int] = {aid: 0 for aid in AGENT_IDS}
        self._episode_fire_first_step: int = 0
        self._episodes_completed: int = 0

        # ── Reward breakdown for diagnostics ────────────────────────────────
        self._reward_breakdown: Dict[str, Dict[str, float]] = {}

        # ── Combat geometry: tail-chase at 2-5km range ──────────────────────
        # KEY: set JSBSim lat/lon directly for ACMI consistency
        rng = np.random.default_rng()
        p0 = env.pursuers[0]; t0 = env.targets[0]

        t_alt = 3000.0
        t_hdg = float(rng.uniform(0, 360))
        t_spd = float(rng.uniform(180, 240))
        chase_dist = float(rng.uniform(2000, 5000))
        lateral = float(rng.uniform(-500, 500))
        t_hdg_rad = np.radians(t_hdg)

        # WGS84 conversion constants at ~30°N
        M_PER_DEG_LAT = 111132.0
        M_PER_DEG_LON = 96420.0  # 111320 * cos(30°)

        # Target at ~2km north of reference (30°N, 120°E)
        t_north_m = 2000.0 + rng.uniform(-200, 200)
        t_east_m = rng.uniform(-200, 200)
        t_lat = 30.0 + t_north_m / M_PER_DEG_LAT
        t_lon = 120.0 + t_east_m / M_PER_DEG_LON

        # Pursuer BEHIND target (opposite of heading direction)
        p_north_m = t_north_m - chase_dist * np.cos(t_hdg_rad) + lateral * np.sin(t_hdg_rad)
        p_east_m = t_east_m - chase_dist * np.sin(t_hdg_rad) - lateral * np.cos(t_hdg_rad)
        p_lat = 30.0 + p_north_m / M_PER_DEG_LAT
        p_lon = 120.0 + p_east_m / M_PER_DEG_LON
        p_spd = float(rng.uniform(240, 300))

        # Reset JSBSim at CORRECT coordinates (ACMI will show these)
        t0.aircraft.reset(lat_deg=t_lat, lon_deg=t_lon, alt_ft=int(t_alt * 3.28084),
                          heading_deg=t_hdg, speed_kts=int(t_spd / 0.5144), trim=False)
        t0.aircraft.position_ned = np.array([t_north_m, t_east_m, t_alt])
        t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt

        # Initial heading offset: force agent to MANEUVER into WEZ first
        heading_bias = float(rng.uniform(30, 60) * rng.choice([-1, 1]))
        p0_hdg = float((t_hdg + heading_bias) % 360.0)

        p0.aircraft.reset(lat_deg=p_lat, lon_deg=p_lon, alt_ft=int(t_alt * 3.28084),
                          heading_deg=p0_hdg, speed_kts=int(p_spd / 0.5144), trim=False)
        p0.aircraft.position_ned = np.array([p_north_m, p_east_m, t_alt])
        p0.ref_hdg, p0.ref_alt_m = p0_hdg, t_alt
        p0._cmd_speed = p_spd

        # Warmup JSBSim — P0 flies at its OWN offset heading, T0 at its heading
        for _ in range(int(1.0 * 60)):
            for ac, h, a, s in [(p0, p0_hdg, t_alt, p_spd), (t0, t_hdg, t_alt, t_spd)]:
                st = ac.aircraft.state
                tgt = FlightControlTargets(heading_deg=h, altitude_m=a, speed_mps=s)
                thr, elev, ail, rud = ac.fc.compute(st, tgt, PHYSICS_DT)
                ac.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ac.aircraft.run()
                ac.aircraft.position_ned[0:2] += ac.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ac.aircraft.position_ned[2] = st["alt_m"]

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
            # Clamp altitude to target ± 2000m engagement cylinder
            t_alt = float(env.targets[0].aircraft.state["alt_m"]) if env.M > 0 else 3000.0
            ps.ref_alt_m = float(np.clip(
                ps.ref_alt_m + DELTA_ALTITUDES[alt_idx],
                max(500.0, t_alt - 2000.0),
                t_alt + 2000.0))
            ps._cmd_speed = float(np.clip(
                getattr(ps, '_cmd_speed', 200.0) + DELTA_SPEEDS[speed_idx],
                120.0, 400.0))

            # ── Fire logic (WEZ-masked: fire only available in kill position) ──
            self._has_launched_this_step[aid] = False
            self._commanded_fire[aid] = (fire == 1)
            if fire == 1 and ps.is_alive and self.remaining_missiles[aid] > 0:
                self._try_launch_missile(env, ps, aid, step)

        # ── Target: rule-based evasion ──────────────────────────────────────
        for ts in env.targets:
            if ts.is_alive:
                self._update_target_evasion(env, ts)

    def step(self, env) -> None:
        """Task-level per-step: track missile hits, update target HP.

        Missiles mark themselves HIT/MISS internally.  We detect new hits,
        increment the target's hit counter, and issue rewards.
        The target only dies when hits_taken >= max_hits.
        """
        self._step_count += 1
        self._hit_this_step: Dict[str, int] = {aid: 0 for aid in AGENT_IDS}
        target = env.targets[0] if env.M > 0 else None

        for i, (ps, aid) in enumerate(zip(env.pursuers, AGENT_IDS)):
            for m in list(ps.launch_missiles):
                if m.is_success and not getattr(m, '_hit_rewarded', False):
                    m._hit_rewarded = True
                    self._hit_this_step[aid] += 1  # counter, not bool (twin missiles)
                    if target is not None:
                        target.hits_taken += 1
                    # Log the hit
                    print(f"\n[HIT!] {m.uid} hit target!  HP: {target.hits_taken}/{target.max_hits if target else '?'}")

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
            # Flatten: obs + action_mask concatenated → single Box
            mask = self.get_action_mask(env, aid)
            flat_obs = np.concatenate([obs, mask]).astype(np.float32)
            obs_dict[aid] = flat_obs

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

            # WEZ maintenance: reward for staying in kill position
            r_wez = 2.0 if self._is_valid_launch_envelope(ps, target) else 0.0

            r += r_progress + r_ata + r_alt + r_wez

            # ── Fire-spam penalty ──────────────────────────────────────────
            r_spam = 0.0
            if self._commanded_fire.get(aid, False) and not self._has_launched_this_step.get(aid, False):
                r_spam = -1.0
            r += r_spam

            # ── Closure shaping at launch ──────────────────────────────────
            r_closure = 0.0
            if self._has_launched_this_step.get(aid, False):
                # Compute closure speed when the agent actually fired
                p_pos = ps.aircraft.position_ned
                t_pos = target.aircraft.position_ned
                los = t_pos - p_pos
                los_dir = los / max(np.linalg.norm(los), 1e-6)
                rel_vel = target.aircraft.velocity_ned - ps.aircraft.velocity_ned
                closure = float(np.dot(rel_vel, los_dir))
                # Reward positive closure, penalize negative (target escaping)
                r_closure = np.clip(closure / 50.0, -5.0, 5.0)
            r += r_closure

            # ── Quality launch bonus ───────────────────────────────────────
            r_quality = 0.0
            if self._has_launched_this_step.get(aid, False):
                dist = float(np.linalg.norm(ps.aircraft.position_ned - target.aircraft.position_ned))
                p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
                los_dir = (target.aircraft.position_ned - ps.aircraft.position_ned) / max(dist, 1e-6)
                cos_ata = float(np.dot(p_fwd, los_dir))
                ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))
                if closure > 0 and ata_deg < 10.0 and 2000 < dist < 4000:
                    r_quality = 20.0  # premium launch window
            r += r_quality

            # ── Event-driven rewards ────────────────────────────────────────
            r_event = 0.0

            # Hit: per-missile accuracy-based reward
            for m in list(ps.launch_missiles):
                if m.is_success and not getattr(m, '_accuracy_rewarded', False):
                    m._accuracy_rewarded = True
                    actual_dist = getattr(m, 'miss_distance', 300.0)
                    accuracy_score = max(0.0, 1.0 - (actual_dist / 300.0))
                    # Launch range multiplier: 1500m→1.0, 4000m→~0.3
                    l_dist = getattr(m, 'launch_dist', 3000.0)
                    range_mult = np.clip(1.0 - (l_dist - 1500.0) / 3500.0, 0.3, 1.0)
                    missile_reward = (REWARD_HIT_BASE + REWARD_HIT_BONUS * accuracy_score) * range_mult
                    r_event += missile_reward

            # Shot down by enemy (one-shot)
            if self._prev_alive.get(aid, True) and not ps.is_alive:
                r_event += REWARD_SHOTDOWN

            # Fled combat — altitude desertion (one-shot, same severity as crash)
            if (target.is_alive and
                abs(float(ps.aircraft.state["alt_m"]) - float(target.aircraft.state["alt_m"])) > 3000.0):
                r_event += REWARD_CRASH

            r += r_event

            self._reward_breakdown = {
                "ProgressReward": {"p0": r_progress},
                "ATAAlignmentReward": {"p0": r_ata},
                "AltitudeDeviationPenalty": {"p0": r_alt},
                "WEZ_Maintenance": {"p0": r_wez},
                "FireSpamPenalty": {"p0": r_spam},
                "ClosureShaping": {"p0": r_closure},
                "QualityBonus": {"p0": r_quality},
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

        max_steps = 1500  # allow longer multi-hit engagements
        low_alt = 1000.0  # tolerance for vertical manoeuvres when chasing diving target

        for ps, aid in zip(env.pursuers, AGENT_IDS):
            s = ps.aircraft.state
            alt_m = float(s["alt_m"])
            target = env.targets[0] if env.M > 0 else None

            # Pursuer shot down
            if not ps.is_alive:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "pursuer_shotdown"}
                self._last_termination_reason = "pursuer_shotdown"
            # Low altitude crash
            elif alt_m < low_alt:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "low_altitude"}
                self._last_termination_reason = "low_altitude"
            # Fled combat — altitude deviation > 3000m from target
            elif (target is not None and target.is_alive
                  and abs(alt_m - float(target.aircraft.state["alt_m"])) > 3000.0):
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "fled_combat_altitude"}
                self._last_termination_reason = "fled_combat_altitude"
            # Target HP depleted (all hits landed)
            elif target is not None and target.hits_taken >= target.max_hits:
                terminateds[aid] = True
                infos[aid] = {"termination_reason": "target_killed"}
                self._last_termination_reason = "target_killed"
            # All ammo spent and all missiles resolved
            elif (self.remaining_missiles.get(aid, 0) == 0
                  and all(m.is_done for m in ps.launch_missiles)):
                terminateds[aid] = True
                hits = target.hits_taken if target else 0
                infos[aid] = {"termination_reason": f"ammo_exhausted_hits={hits}"}
                self._last_termination_reason = "ammo_exhausted"
            # Lost target (too far)
            elif target is not None and target.is_alive:
                dist = float(np.linalg.norm(
                    ps.aircraft.position_ned - target.aircraft.position_ned))
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

        if terminateds.get("__all__") or truncateds.get("__all__"):
            self._episodes_completed += 1
            wez_s = self._episode_wez_first_step.get("p0", 0)
            fire_s = self._episode_fire_first_step if self._episode_fire_first_step > 0 else 0
            if wez_s > 0:
                print(f"[TIMING] ep={self._episodes_completed} WEZ_first={wez_s} fire_first={fire_s} "
                      f"fire_after_WEZ={fire_s - wez_s if fire_s > 0 else 'never'} "
                      f"ep_steps={self._step_count}")
            # Reset per-episode trackers
            self._episode_wez_first_step = {aid: 0 for aid in AGENT_IDS}
            self._episode_fire_first_step = 0

        self._last_episode_steps = self._step_count if (terminateds.get("__all__") or truncateds.get("__all__")) else 0

        return terminateds, truncateds, infos

    # ══════════════════════════════════════════════════════════════════════════
    #  Action masking
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_mask(self, env, agent_id: str) -> np.ndarray:
        """Action mask with WEZ + Dynamic Launch Zone (DLZ) gating.

        Flat mask layout: [speed(3), heading(5), altitude(3), fire(2)]
        Fire = index 12, only unmasked when ALL conditions met.

        Dynamic DLZ: max range depends on Aspect Angle (AA)
          - Head-on  (AA≈180°): missile+target closing fast → 8km
          - Tail-chase (AA≈0°):  missile must catch up    → 3km
          - Linear interpolation between
        """
        mask = np.ones(N_ACTIONS, dtype=np.float32)
        fire_start = N_SPEED_DELTA + N_HEADING_DELTA + N_ALT_DELTA  # 3+5+1=9

        if env.M == 0:
            mask[fire_start + 1] = 0.0
            return mask

        target = env.targets[0]
        aid_idx = env._agent_ids.index(agent_id) if agent_id in env._agent_ids else 0
        ps = env.pursuers[aid_idx]

        can_fire = True

        # Condition 1: missiles remaining
        if self.remaining_missiles.get(agent_id, 0) <= 0:
            can_fire = False

        # Condition 2: target alive
        if not target.is_alive:
            can_fire = False

        # Condition 3: nose on target (ATA < MAX_ATTACK_ANGLE)
        if can_fire:
            p_pos = ps.aircraft.position_ned
            t_pos = target.aircraft.position_ned
            dist = float(np.linalg.norm(p_pos - t_pos))
            los_dir = (t_pos - p_pos) / max(dist, 1e-6)
            p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            cos_ata = float(np.dot(p_fwd, los_dir))
            ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))
            if ata_deg > MAX_ATTACK_ANGLE:
                can_fire = False

        # Condition 4: within Dynamic DLZ range
        if can_fire:
            t_fwd = compute_forward_vector(target.aircraft.rpy_rad)
            # AA: angle between target nose and LOS (0=target facing away, 180=target facing toward)
            cos_aa = float(np.dot(t_fwd, los_dir))
            aa_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_aa))))
            # Dynamic max range: 3km (tail) → 8km (head-on)
            dynamic_max_dist = 3000.0 + 5000.0 * (aa_deg / 180.0)
            if dist < MIN_ATTACK_DISTANCE or dist > dynamic_max_dist:
                can_fire = False

        if not can_fire:
            mask[fire_start + 1] = 0.0
        else:
            # Track first WEZ entry this episode
            if self._episode_wez_first_step.get(aid, 0) == 0:
                self._episode_wez_first_step[aid] = self._step_count

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
        # Hex ID: 301, 302, 303, ... (Tacview object IDs as hex)
        uid = f"{0x301 + env._missile_uid_counter:X}"
        env._missile_uid_counter += 1

        # Create and launch — pass parent's ACMI ID (101 + pursuer_index)
        parent_uid = str(101 + env.pursuers.index(ps))
        m = MissileSimulator.create(
            parent=ps, target=target, uid=uid, dt=1.0 / 60.0,
            parent_uid=parent_uid)
        # Record launch geometry for diagnostics
        t_pos = target.aircraft.position_ned
        p_pos = ps.aircraft.position_ned
        m.launch_dist = float(np.linalg.norm(p_pos - t_pos))
        p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
        t_fwd = compute_forward_vector(target.aircraft.rpy_rad)
        los_dir = (t_pos - p_pos) / max(m.launch_dist, 1e-6)
        cos_ata = float(np.dot(p_fwd, los_dir))
        cos_aa = float(np.dot(t_fwd, los_dir))
        closure = float(np.dot(target.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
        print(f"[LAUNCH] {uid}: range={m.launch_dist:.0f}m ATA={math.degrees(math.acos(max(-1,min(1,cos_ata)))):.0f}deg "
              f"AA={math.degrees(math.acos(max(-1,min(1,cos_aa)))):.0f}deg closure={closure:.0f}m/s "
              f"alt={ps.aircraft.state['alt_m']:.0f}m roll={ps.aircraft.state['roll_deg']:.0f}deg "
              f"hdg_err={(ps.ref_hdg-target.ref_hdg+180)%360-180:.0f}deg")
        env.add_temp_simulator(m)

        # ── Classify launch quality ────────────────────────────────────────
        closure = float(np.dot(target.aircraft.velocity_ned - ps.aircraft.velocity_ned,
                               (t_pos - p_pos) / max(m.launch_dist, 1e-6)))
        self._launch_stats["closure_vals"].append(closure)
        self._launch_stats["ata_vals"].append(ata_deg if 'ata_deg' in dir() else 0)
        self._launch_stats["range_vals"].append(m.launch_dist)
        if closure > 0 and 'ata_deg' in dir() and ata_deg < 10 and 2000 < m.launch_dist < 4000:
            self._launch_stats["premium"].append(m.launch_dist)
            tag = "PREMIUM"
        elif closure > 0:
            self._launch_stats["good"].append(closure)
            tag = "GOOD"
        else:
            self._launch_stats["bad"].append(closure)
            tag = "BAD"
        # Write launch stats to disk every 20 launches (stdout lost inside Ray workers)
        total = len(self._launch_stats["good"]) + len(self._launch_stats["bad"]) + len(self._launch_stats["premium"])
        if total % 20 == 0:
            import os as _os
            stats_path = _os.environ.get("LAUNCH_STATS_FILE", "/tmp/launch_stats.log")
            with open(stats_path, "a") as f:
                f.write(f"[LAUNCH-STATS] total={total} good={len(self._launch_stats['good'])} "
                        f"bad={len(self._launch_stats['bad'])} premium={len(self._launch_stats['premium'])} "
                        f"good_ratio={len(self._launch_stats['good'])/max(total,1)*100:.0f}% "
                        f"premium_ratio={len(self._launch_stats['premium'])/max(total,1)*100:.0f}% "
                        f"closure_mean={np.mean(self._launch_stats['closure_vals']):.0f}m/s\n")

        # Track first fire step this episode
        if self._episode_fire_first_step == 0:
            self._episode_fire_first_step = self._step_count

        # Update state
        self.remaining_missiles[aid] -= 1
        self._last_shoot_step[aid] = step
        self._has_launched_this_step[aid] = True

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal: target evasion
    # ══════════════════════════════════════════════════════════════════════════

    def _update_target_evasion(self, env, ts) -> None:
        """Rule-based target control — S-turns + missile evasion reaction.

        difficulty=0.0: straight-and-level (no evasion).  Easy target for
                       Stage 1 curriculum — RL learns fire→hit→+1000.

        difficulty>0.0: S-turn evasion + missile threat reaction.
                       Higher difficulty → harder evasion → forced to close
                       range before firing for a reliable hit.
        """
        self._target_evasion_step += 1
        t = self._target_evasion_step * DECISION_DT
        d = self._difficulty

        # ── Baseline S-turn ────────────────────────────────────────────────
        hdg_var = d * 30.0 * math.sin(t * 0.3)
        new_hdg = float((self._target_base_hdg + hdg_var) % 360.0)
        new_alt = 3000.0

        # ── Missile threat reaction (difficulty > 0 only) ──────────────────
        if d > 0.0 and len(ts.under_missiles) > 0:
            # Find the closest incoming missile
            closest_dist = float('inf')
            closest_missile = None
            for m in ts.under_missiles:
                if m.is_alive:
                    dist = m.target_distance
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_missile = m

            if closest_missile is not None:
                # Hard break turn away from missile + dive for energy
                m_pos = closest_missile.get_absolute_position()
                t_pos = ts.aircraft.position_ned
                los_to_missile = np.arctan2(m_pos[1] - t_pos[1],
                                             m_pos[0] - t_pos[0])

                # Turn perpendicular to incoming missile (beam defence)
                # More aggressive at higher difficulty and closer ranges
                break_strength = d * min(1.0, 3000.0 / max(closest_dist, 1.0))
                side = 1.0 if math.sin(t * 0.5) > 0 else -1.0  # alternate direction
                evade_hdg = float(np.degrees(los_to_missile) + side * 90.0 * break_strength)
                new_hdg = float((new_hdg + evade_hdg) * break_strength +
                                new_hdg * (1.0 - break_strength)) % 360.0

                # Dive to trade altitude for speed (energy manoeuvre)
                if closest_dist < 5000.0:
                    new_alt = 3000.0 - d * 800.0

        ts.ref_hdg = new_hdg
        ts.ref_alt_m = max(2000.0, new_alt)  # floor at 2000m — safe margin above crash threshold

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
        reward = PROGRESS_WEIGHT * delta * 0.5 * DECISION_STEPS * dist_factor
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
        return float(ATA_WEIGHT * cos_ata * dist_factor * DECISION_STEPS)

    def _alt_reward(self, ps, target) -> float:
        """Penalty for altitude deviation + low-altitude soft warning.

        - 300-1500m: mild linear penalty (deviation from target)
        - >1500m:    severe quadratic penalty (discourages fleeing to space)
        - <1500m:    soft penalty encouraging higher altitude (AIM-9 needs energy)
        """
        t_alt = float(target.aircraft.state["alt_m"]) if target is not None else 3000.0
        p_alt = float(ps.aircraft.state["alt_m"])
        alt_diff = abs(p_alt - t_alt)
        penalty = 0.0
        if alt_diff > 300.0:
            penalty -= 0.1 * (alt_diff - 300.0) / 1000.0 * DECISION_STEPS
        if alt_diff > 1500.0:
            penalty -= 2.0 * ((alt_diff - 1500.0) / 1000.0) ** 2 * DECISION_STEPS
        # Soft low-altitude penalty — discourage energy loss, not hard termination
        if p_alt < 1500.0:
            penalty -= 0.5 * (1500.0 - p_alt) / 500.0 * DECISION_STEPS
        return float(penalty)

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
