"""Autopilot layer: BFM commands → JSBSim control surfaces.

Two tiers are provided:

1. **Simple testing autopilots** (``AltitudeHoldAP``, ``SpeedHoldAP``,
   ``TurnCoordinator``) — single-channel PID controllers useful for
   verification scripts and fixed-target tracking demos.

2. **BFMAutopilot** — the production λ-g flight control law that converts
   high-level BFM commands ``(n_x, n_n, mu)`` into the four JSBSim control
   surface inputs ``(throttle, elevator, aileron, rudder)``.  This is the
   bridge described in **Decision 2** of the migration plan.

Channel mapping
---------------

============ ====================== ===================== ======================
Channel      Tracks                 JSBSim input           PID type
============ ====================== ===================== ======================
Elevator     ``n_z_g`` (body-Z G)   ``elevator-cmd-norm`` Nz-tracking PID
Aileron      Roll angle ``mu``      ``aileron-cmd-norm``  Roll-tracking PID
Throttle     Airspeed               ``throttle-cmd-norm`` Speed-hold PID
Rudder       Sideslip ``beta``      ``rudder-cmd-norm``   Beta-suppression PID
============ ====================== ===================== ======================

The Nz channel is the core of the **λ-g flight control law**: it closes the
loop around the aircraftʼs normal acceleration rather than pitch attitude,
which is how real F-16-class flight control systems work  (Stevens & Lewis,
*Aircraft Control and Simulation*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Speed-dependent trim schedule (Phase 3: replaces hardcoded constants)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrimSchedule:
    """Speed-to-elevator-trim lookup using 1/V² scaling law.

    trim(V) = ref_elevator * (V_ref / V)²

    This follows from: lift required = weight (constant), lift ∝ q·CL,
    and CL ∝ elevator deflection in the linear regime.  Since q ∝ V²,
    elevator_trim ∝ 1/V².

    Reference values calibrated from Phase 1 open-loop sweep
    (scripts/sweep_elevator.py) at 3000 m / 400 kts.
    """

    ref_speed_mps: float = 176.0
    ref_elevator: float = -0.05
    ref_throttle: float = 0.80
    min_speed_mps: float = 80.0   # below this, clamp (avoid division by zero)

    def get_elevator_trim(self, speed_mps: float) -> float:
        """Return trim elevator for level flight at *speed_mps*."""
        V = max(speed_mps, self.min_speed_mps)
        return self.ref_elevator * (self.ref_speed_mps / V) ** 2

    def get_throttle_trim(self, speed_mps: float) -> float:
        """Return trim throttle.  First-order constant; may refine later."""
        return self.ref_throttle


# ═══════════════════════════════════════════════════════════════════════════════
#  PID controller building-block
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PIDController:
    """Discrete PID with integral anti-windup and output clamping."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_min: float = -1.0
    integral_max: float = 1.0

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: Optional[float] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        """Advance one time step, return control output."""
        # Integral
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, self.integral_min, self.integral_max))

        # Derivative  (on measurement, not error, to avoid derivative kick)
        derivative = 0.0
        if self._prev_error is not None and dt > 1e-8:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))


# ═══════════════════════════════════════════════════════════════════════════════
#  Simple testing autopilots  (keep from original skeleton)
# ═══════════════════════════════════════════════════════════════════════════════

class AltitudeHoldAP:
    """Altitude-hold via elevator — for verification / demo scripts."""

    def __init__(self) -> None:
        self.pid = PIDController(kp=0.0005, ki=0.00005, kd=0.0001,
                                 output_min=-1.0, output_max=1.0)

    def compute(self, current_alt_m: float, target_alt_m: float, dt: float) -> float:
        error = target_alt_m - current_alt_m
        return self.pid.step(error, dt)


class SpeedHoldAP:
    """Speed-hold via throttle — for verification / demo scripts."""

    def __init__(self) -> None:
        self.pid = PIDController(kp=0.005, ki=0.001, kd=0.0,
                                 output_min=0.0, output_max=1.0)

    def compute(self, current_speed_mps: float, target_speed_mps: float, dt: float) -> float:
        error = target_speed_mps - current_speed_mps
        return self.pid.step(error, dt)


class TurnCoordinator:
    """Coordinated turn: aileron tracks roll, rudder kills sideslip."""

    def __init__(self) -> None:
        self.roll_pid = PIDController(kp=0.05, ki=0.01, kd=0.02,
                                      output_min=-1.0, output_max=1.0)
        self.rudder_pid = PIDController(kp=0.1, ki=0.0, kd=0.0,
                                        output_min=-1.0, output_max=1.0)

    def compute(
        self,
        current_roll_rad: float,
        target_roll_rad: float,
        sideslip_deg: float,
        dt: float,
    ) -> tuple[float, float]:
        roll_error = target_roll_rad - current_roll_rad
        roll_error = (roll_error + np.pi) % (2 * np.pi) - np.pi
        aileron = self.roll_pid.step(roll_error, dt)
        rudder = self.rudder_pid.step(-sideslip_deg, dt)
        return aileron, rudder


# ═══════════════════════════════════════════════════════════════════════════════
#  BFM autopilot — λ-g flight control law
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BFMAutopilotConfig:
    """Gains for the four-channel BFM autopilot.

    Phase 3 values: PD from Phase 2 single-channel tuning, small integral
    terms re-enabled with conservative anti-windup limits (~30 % of output
    range) to prevent windup during aggressive manoeuvres.
    """

    # ── Nz (elevator) channel ─────────────────────────────────────────
    nz_kp: float = 0.18      # Phase 2: kp=0.18 (slower rise, stable)
    nz_ki: float = 0.02      # small integral to kill steady-state G error
    nz_kd: float = 0.012     # Phase 2: kd=0.012
    nz_integral_min: float = -0.3
    nz_integral_max: float = 0.3

    # ── Roll (aileron) channel ────────────────────────────────────────
    roll_kp: float = 1.5     # Phase 2: V10 default (fast roll, overshoot acceptable)
    roll_ki: float = 0.05    # small integral for precise bank-angle hold
    roll_kd: float = 0.08    # Phase 2: kd=0.08 (dampens overshoot)
    roll_integral_min: float = -0.2
    roll_integral_max: float = 0.2

    # ── Speed (throttle) channel ──────────────────────────────────────
    speed_kp: float = 0.02   # Phase 2: highest kp without oscillation
    speed_ki: float = 0.005  # small integral (slow dynamics)
    speed_kd: float = 0.0
    speed_integral_min: float = -0.3
    speed_integral_max: float = 0.3
    min_target_speed_mps: float = 80.0
    max_target_speed_mps: float = 400.0

    # ── Sideslip (rudder) channel ─────────────────────────────────────
    beta_kp: float = 0.06
    beta_ki: float = 0.005   # small integral for persistent sideslip correction
    beta_kd: float = 0.0


class BFMAutopilot:
    """λ-g flight control law: BFM ``(n_x, n_n, mu)`` → control surfaces.

    Each aircraft instance needs its own ``BFMAutopilot`` because the
    autopilot maintains per-channel PID integral states and the throttle-
    channel target-speed memory.

    Usage inside the environment micro-step loop::

        thr, elev, ail, rud = autopilot.step(
            n_x, n_n, mu, dt,
            n_z_g=ac.state["n_z_g"],
            roll_rad=ac.state["roll_rad"],
            airspeed_mps=ac.state["airspeed_mps"],
            beta_deg=ac.state["beta_deg"],
        )
        ac.set_controls(thr, elev, ail, rud)
    """

    def __init__(self, config: Optional[BFMAutopilotConfig] = None,
                 trim: Optional[TrimSchedule] = None) -> None:
        cfg = config or BFMAutopilotConfig()
        self._trim = trim or TrimSchedule()

        # ── Per-channel PIDs ──────────────────────────────────────────
        self._nz_pid = PIDController(
            kp=cfg.nz_kp, ki=cfg.nz_ki, kd=cfg.nz_kd,
            output_min=-1.0, output_max=1.0,
            integral_min=cfg.nz_integral_min, integral_max=cfg.nz_integral_max,
        )
        self._roll_pid = PIDController(
            kp=cfg.roll_kp, ki=cfg.roll_ki, kd=cfg.roll_kd,
            output_min=-1.0, output_max=1.0,
            integral_min=cfg.roll_integral_min, integral_max=cfg.roll_integral_max,
        )
        self._speed_pid = PIDController(
            kp=cfg.speed_kp, ki=cfg.speed_ki, kd=cfg.speed_kd,
            output_min=0.0, output_max=1.0,
            integral_min=cfg.speed_integral_min, integral_max=cfg.speed_integral_max,
        )
        self._beta_pid = PIDController(
            kp=cfg.beta_kp, ki=cfg.beta_ki, kd=cfg.beta_kd,
            output_min=-1.0, output_max=1.0,
        )

        self._min_target_speed = cfg.min_target_speed_mps
        self._max_target_speed = cfg.max_target_speed_mps

        # Throttle channel has memory: where we want the speed to be
        self._target_speed_mps: Optional[float] = None

        # Alpha / G-limiter constants
        self._max_alpha_deg = 25.0
        self._max_nz_g = 9.0
        self._alpha_limiter_gain = 0.5

    # ── Public API ────────────────────────────────────────────────────

    def reset(self, initial_speed_mps: Optional[float] = None) -> None:
        """Reset all PID integrators.

        Args:
            initial_speed_mps: If given, prime the speed-hold target so
                the autopilot doesnʼt hunt on the first step.
        """
        self._nz_pid.reset()
        self._roll_pid.reset()
        self._speed_pid.reset()
        self._beta_pid.reset()
        self._target_speed_mps = initial_speed_mps

    def step(
        self,
        n_x: float,
        n_n: float,
        mu: float,
        dt: float,
        *,
        n_z_g: float,
        roll_rad: float,
        airspeed_mps: float,
        beta_deg: float,
        alpha_deg: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """One autopilot iteration.

        Args:
            n_x:           Tangential acceleration target (G, + = speed up).
            n_n:           Normal acceleration target (G, + = pull up).
            mu:            Bank-angle target (rad).
            dt:            Time step (s).
            n_z_g:         Current body-Z acceleration at pilot station (G).
                           JSBSim property ``accelerations/n-pilot-z-norm``.
                           **Negative** when pulling positive G.
            roll_rad:      Current roll angle (rad).  JSBSim ``attitude/roll-rad``.
            airspeed_mps:  Current calibrated airspeed (m/s).
            beta_deg:      Sideslip angle (deg).  JSBSim ``aero/beta-deg``.
            alpha_deg:     Angle of attack (deg).  JSBSim ``aero/alpha-deg``.
                           Used for alpha limiter (Phase 3).

        Returns:
            ``(throttle, elevator, aileron, rudder)`` — all in [-1, 1].
        """
        # ── Dynamic trim (Phase 3: speed-dependent) ──────────────────
        elevator_trim = self._trim.get_elevator_trim(airspeed_mps)

        # ── Elevator: track body-Z normal acceleration ───────────────
        # JSBSim convention: n_z_g < 0 means positive-G (pilot pushed into seat).
        # BFM convention:   n_n > 0 means positive-G pull-up.
        # Therefore:        target_n_z_g = -(n_n).
        #
        # Bank compensation (Phase 3): in a banked turn at angle μ,
        # vertical lift = cos(μ).  To maintain altitude, total lift must
        # be weight / cos(μ), i.e. n_z = 1 / cos(μ) G.
        # Extra G needed beyond 1G: bank_extra = 1/cos(μ) - 1.
        # This feedforward is added to the Nz target so the PID doesn't
        # have to discover basic physics.
        cos_roll = math.cos(abs(roll_rad))
        bank_extra_g = (1.0 / max(cos_roll, 0.1)) - 1.0
        target_n_z_g = -(n_n + bank_extra_g)

        nz_error = n_z_g - target_n_z_g   # + → need more negative n_z (more pull)
        elevator = elevator_trim - self._nz_pid.step(nz_error, dt)

        # ── Alpha / G-limiter (Phase 3: last safety net) ─────────────
        # Overrides PID output when limits are violated.  The JSBSim
        # native FCS has its own limiter at 28-30° alpha; this Python-
        # level limiter catches our PID outputs before they reach JSBSim.
        alpha_excess = abs(alpha_deg) - self._max_alpha_deg
        if alpha_excess > 0:
            # Push nose down proportionally as alpha increases
            alpha_override = self._alpha_limiter_gain * alpha_excess / (35.0 - self._max_alpha_deg)
            elevator = max(elevator, 0.0)             # don't pull any more
            elevator = min(elevator + alpha_override, 1.0)  # push nose down

        # G-limiter: if already at max positive G, prevent more pull
        if abs(n_z_g) > self._max_nz_g and n_z_g < 0:
            elevator = max(elevator, elevator_trim - 0.2)  # don't go more negative than trim-0.2

        # ── Aileron: track bank angle ────────────────────────────────
        roll_error = roll_rad - mu   # + when need right roll
        roll_error = (roll_error + np.pi) % (2 * np.pi) - np.pi
        aileron = self._roll_pid.step(roll_error, dt)

        # ── Throttle: convert n_x → speed target → PID ──────────────
        throttle = self._throttle_step(n_x, airspeed_mps, dt)

        # ── Rudder: suppress sideslip (turn coordination) ───────────
        rudder = self._beta_pid.step(beta_deg, dt)

        return throttle, elevator, aileron, rudder

    # ── Throttle-channel internals ────────────────────────────────────

    def _throttle_step(self, n_x: float, airspeed_mps: float, dt: float) -> float:
        """Drive airspeed toward a target that drifts with *n_x*.

        * n_x > 0  →  target speed increases  (open throttle to accelerate)
        * n_x < 0  →  target speed decreases  (close throttle to decelerate)
        * n_x = 0  →  hold current target

        Includes a speed-dependent feed-forward bias (Phase 3: dynamic trim).
        """
        throttle_bias = self._trim.get_throttle_trim(airspeed_mps)

        # Bootstrap target on first call
        if self._target_speed_mps is None:
            self._target_speed_mps = airspeed_mps

        # Drift the target based on tangential acceleration command
        self._target_speed_mps += n_x * 9.81 * dt
        self._target_speed_mps = float(np.clip(
            self._target_speed_mps,
            self._min_target_speed,
            self._max_target_speed,
        ))

        speed_error = self._target_speed_mps - airspeed_mps  # + when need more speed
        return float(np.clip(throttle_bias + self._speed_pid.step(speed_error, dt), 0.0, 1.0))
