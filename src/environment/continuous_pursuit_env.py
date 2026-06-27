"""Continuous-action pursuit environment — Phase 2 (Box(2) → FlightController).

Inherits from BFMPursuitEnv to reuse observation computation, reset logic,
Tacview export, reward formulas, and termination conditions.  Only the
action space and the control routing change:

    Discrete(9) BFM  →  Box(2) [turn_rate_factor, speed_factor]
    BFMAutopilot     →  FlightController (3-channel stabilisation)

This preserves the research baseline while isolating the single independent
variable: action-space granularity.

Action space
------------
Box(2, [-1, 1]):
  dim[0] = turn_rate_factor  →  cmd_turn_rate = factor × 15.0 °/s
  dim[1] = speed_factor      →  cmd_speed     = 250 + factor × 100 m/s

The FlightController heading target integrates the turn-rate command each
micro-step; altitude is always locked at 3000m.  This gives the agent
continuous, energy-aware control — it can choose 5° banks for energy
conservation or 15° banks for aggressive tracking, rather than being
forced into 0° or 60° by a discrete action table.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import gymnasium as gym
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.bfm_actions import PURSUIT_ACTIONS, NUM_PURSUIT_ACTIONS
from src.utils.units import kts_to_mps
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles

from src.environment.bfm_pursuit_env import (
    BFMPursuitEnv, TargetProfile,
    # Re-import all constants for clarity — same values as parent
    CTRL_FREQ, PHYSICS_DT, DECISION_DT, DECISION_STEPS,
    MAX_EPISODE_TIME,
    MAX_DIST, MAX_HEIGHT, MAX_VEL, MAX_ANG_VEL, MAX_AOA, MAX_PS,
    LOW_SPEED_THRESHOLD,
    REWARD_PROGRESS, REWARD_ATA, REWARD_GROUND_WARNING,
    REWARD_SUCCESS, REWARD_CRASH, REWARD_LOST_TARGET, REWARD_TIMEOUT,
    REWARD_LOW_SPEED_TURN, STEP_PENALTY, LOW_ENERGY_PENALTY,
    ANTI_STALL_WINDOW, ANTI_STALL_MIN_VC, ANTI_STALL_MIN_DIST,
    ANTI_STALL_PENALTY, ANTI_STALL_SPEED_WARN, ANTI_STALL_SPEED_WARN_WEIGHT,
    ZONE_DEATH_DIST_LO, ZONE_DEATH_DIST_HI, ZONE_DEATH_DIST_HI_SCALE,
    ZONE_DEATH_MIN_VC, ZONE_DEATH_WINDOW, ZONE_DEATH_PENALTY,
    VELOCITY_SHAPING_WEIGHT, VELOCITY_SHAPING_ATA_THRESH,
    REWARD_DELTA_ATA, REWARD_CLOSURE_RATE, CLOSURE_RATE_NORM,
    PROXIMITY_TIERS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Continuous action mapping constants
# ═══════════════════════════════════════════════════════════════════════════════

MAX_TURN_RATE_DPS = 15.0       # °/s  — max heading-rate magnitude
SPEED_BASE = 250.0             # m/s  — centre of speed range (~486 kts)
SPEED_RANGE = 100.0            # m/s  — ± offset → [150, 350] m/s


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousPursuitEnv(BFMPursuitEnv):
    """Single-agent continuous pursuit with Box(2) → FlightController.

    Inherits all observation computation, reset logic, Tacview export,
    reward formulas, and termination conditions from BFMPursuitEnv.
    Only the action space and pursuer control routing differ.

    Action: Box(2) [turn_rate_factor, speed_factor] both in [-1, 1]
      → heading integrates each micro-step
      → speed is a constant target for the macro-action hold
      → altitude locked to 3000m via FlightController
    """

    metadata = {"render_modes": ["human", "tacview"], "name": "continuous_pursuit_v0"}

    def __init__(self, **kwargs):
        # ── Parent init — builds aircraft, autopilot, FlightController,
        #    sets up difficulty, lock_altitude, tacview, etc. ────────────
        super().__init__(**kwargs)

        # ── Override action space ──────────────────────────────────────
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        # ── Override observation space — plain Box(25), no action_mask ─
        # The parent's Dict obs includes an action_mask key for discrete
        # safety masking.  Continuous policies don't use masks; the
        # FlightController's internal limits (bank compensation, GPWS)
        # provide the safety envelope instead.
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32,
        )

    # ── Observation (plain array, not Dict) ────────────────────────────────

    def _get_obs(self):
        """Return 25-dim observation array (no action_mask)."""
        parent_obs = super()._get_obs()
        return parent_obs["obs"]

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one macro-action (0.5 s hold, 30 micro-steps).

        The continuous action is held constant for DECISION_STEPS micro-steps.
        Heading integrates each micro-step::

            hdg_target += turn_rate * dt

        Speed is a constant target for the full macro-action period.
        Altitude is always locked to 3000m by FlightController.
        """
        dt = PHYSICS_DT

        # ── Parse continuous action ────────────────────────────────────
        action = np.clip(action, -1.0, 1.0)
        cmd_turn_rate = float(action[0] * MAX_TURN_RATE_DPS)   # °/s
        cmd_speed = float(SPEED_BASE + action[1] * SPEED_RANGE)  # m/s

        _airspeed = float(self.pursuer.state["airspeed_mps"])
        _target_spd = float(self.target_ac.state.get("airspeed_mps", 180.0))
        _energy_ok = _airspeed >= _target_spd
        self._last_action = action.copy()

        # ── Reward accumulators (identical to parent) ──────────────────
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

        # ═══════════════════════════════════════════════════════════════════
        #  Micro-step loop (30 steps @ 60 Hz = 0.5 s macro-action hold)
        # ═══════════════════════════════════════════════════════════════════
        for _ in range(DECISION_STEPS):
            s = self.pursuer.state

            # ── Integrate heading target ───────────────────────────────
            self._ref_hdg = (self._ref_hdg + cmd_turn_rate * dt) % 360.0

            # ── Pursuer: pure FlightController (3-channel stabilisation)
            fc_tgt = FlightControlTargets(
                heading_deg=self._ref_hdg,
                altitude_m=self._ref_alt_m,
                speed_mps=cmd_speed,
            )
            thr, elev, ail, rud = self._pursuer_fc.compute(s, fc_tgt, dt)
            self.pursuer.set_controls(throttle=thr, elevator=elev,
                                      aileron=ail, rudder=rud)

            # ── Target control (inherited from parent) ─────────────────
            self._control_target(dt)

            # ── Physics + position update ──────────────────────────────
            self.pursuer.run()
            self.target_ac.run()
            self._step_counter += 1

            self.pursuer.position_ned[0:2] += self.pursuer.velocity_ned[0:2] * dt
            self.pursuer.position_ned[2] = self.pursuer.state["alt_m"]
            self.target_ac.position_ned[0:2] += self.target_ac.velocity_ned[0:2] * dt
            self.target_ac.position_ned[2] = self.target_ac.state["alt_m"]

            # ── NaN guard ──────────────────────────────────────────────
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

            # ── Reward: progress ───────────────────────────────────────
            prog = REWARD_PROGRESS * delta_dist * 0.5
            total_reward += prog
            _r_progress += prog
            if current_dist < 500.0:
                boost = REWARD_PROGRESS * delta_dist * 5.0
                total_reward += boost
                _r_terminal_boost += boost

            # ── Reward: progressive low-speed warning ──────────────────
            _cur_spd = float(self.pursuer.state["airspeed_mps"])
            if _cur_spd < ANTI_STALL_SPEED_WARN:
                spd_deficit = (ANTI_STALL_SPEED_WARN - _cur_spd) / ANTI_STALL_SPEED_WARN
                total_reward -= ANTI_STALL_SPEED_WARN_WEIGHT * spd_deficit * dt

            if not _energy_ok:
                le_penalty = LOW_ENERGY_PENALTY * dt
                total_reward -= le_penalty
                _r_low_speed_penalty -= le_penalty

            # ── Reward: distance-gated ATA ─────────────────────────────
            dist_factor = max(0.0, 1.0 - current_dist / 3000.0)

            ata_r = REWARD_ATA * max(geo["cos_ata"], -0.2) * dt * dist_factor
            total_reward += ata_r
            _r_ata += ata_r

            # ── Reward: delta-ATA (potential-based) ────────────────────
            ata_deg_cur = float(np.degrees(np.arccos(np.clip(geo["cos_ata"], -1.0, 1.0))))
            if self._prev_ata_deg is not None:
                pot_cur = np.exp(-ata_deg_cur / 30.0)
                pot_prev = np.exp(-self._prev_ata_deg / 30.0)
                delta_ata = REWARD_DELTA_ATA * (pot_cur - pot_prev) * dt * dist_factor
                total_reward += delta_ata
                _r_ata += delta_ata
            self._prev_ata_deg = ata_deg_cur

            # ── Reward: closure rate ───────────────────────────────────
            closure_rate_ms = (self._prev_dist - current_dist) / dt if dt > 0 else 0.0
            if closure_rate_ms > 0:
                total_reward += REWARD_CLOSURE_RATE * (closure_rate_ms / CLOSURE_RATE_NORM) * dt

            # ── Reward: velocity shaping when nose-on ──────────────────
            if geo["cos_ata"] > VELOCITY_SHAPING_ATA_THRESH:
                aspd = float(self.pursuer.state["airspeed_mps"])
                vel_bonus = (aspd / MAX_VEL) * VELOCITY_SHAPING_WEIGHT * dt
                total_reward += vel_bonus
                _r_ata += vel_bonus

            # ── Reward: baseline bleed ─────────────────────────────────
            total_reward -= 1.0 * dt
            _r_time_pressure -= 1.0 * dt

            # ── Reward: ground warning ─────────────────────────────────
            if a_pos[2] < 800.0:
                gw = -REWARD_GROUND_WARNING * dt
                total_reward += gw
                _r_ground_warning += gw

            # ── Reward: proximity tiers ────────────────────────────────
            for threshold, bonus in PROXIMITY_TIERS:
                if current_dist < threshold and threshold not in self._proximity_awarded:
                    total_reward += bonus
                    _r_proximity += bonus
                    self._proximity_awarded.add(threshold)

            self._prev_dist = current_dist

            # ── Termination checks ─────────────────────────────────────
            if current_dist < 200.0 and start_dist > 400.0:
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

        # ── Post-loop: anti-stall + zone-of-death (identical to parent) ──
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

        # ── Timeout ────────────────────────────────────────────────────
        current_time = self._step_counter / CTRL_FREQ
        if not terminated and not truncated and current_time >= MAX_EPISODE_TIME:
            truncated = True
            reason = "timeout"
            total_reward += REWARD_TIMEOUT

        # ── Tacview ────────────────────────────────────────────────────
        if self.record_tacview:
            self._record_tacview_frame(current_time)

        end_dist = self._prev_dist
        _closure_rate = (start_dist - end_dist) / DECISION_DT

        info = {
            "reason": reason,
            "termination_reason": reason,
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
            "cmd_turn_rate_dps": cmd_turn_rate,
            "cmd_speed_mps": cmd_speed,
            "ref_hdg_deg": self._ref_hdg,
        }
        return self._get_obs(), total_reward, terminated, truncated, info
