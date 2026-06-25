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

    ref_speed_mps: float = 206.0   # calibrated at 400 kts (Phase 1 measured)
    # Preserving the Phase 1 calibrated value (verified 2026-06-25:
    # the 1/V^2 fit is correct — trim was NOT the root cause of the
    # step-response issues; those were PID-tuning problems now fixed).
    ref_elevator: float = -0.0492
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
#  Gain scheduler (Phase 3.5: speed + target-Nz adaptive PID gains)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GainScheduler:
    """1-D speed-scheduled gains for the Nz channel, with target-Nz boost.

    Control effectiveness ∝ dynamic pressure ∝ V².  To maintain constant
    loop gain across the speed envelope, kp scales as 1/V²:

        kp(V) = kp_ref · (V_ref / V)²

    The integral gain ki is boosted for large Nz commands to rapidly kill
    steady-state error during aggressive manoeuvres.

    Roll, speed, and beta channels use fixed gains for now.
    """

    # ── Speed LUT: reference points for 1/V² interpolation ────────────
    ref_speed_mps: float = 206.0     # calibration point (400 kts)
    # Porpoising fix (2026-06-25): kp_ref halved from 0.18 → 0.09.
    # At 0.18 the Nz channel was under-damped, causing σ=1.55G oscillation
    # during step commands (see scripts/verify_autopilot_channels.py).
    kp_ref: float = 0.09             # halved from 0.18 for critical damping
    kd_ref: float = 0.025            # stable kd at reference speed
    kp_min: float = 0.04             # halved from 0.08
    kp_max: float = 0.15             # halved from 0.30
    kd_min: float = 0.010
    kd_max: float = 0.040

    # ── Target-Nz boost for integral ──────────────────────────────────
    # 2026-06-25: ki_base 0.08→0.12 to eliminate 0.39G steady-state
    # offset in level flight.  With the back-calculation anti-windup
    # protecting against saturation, we can safely use a higher base ki.
    ki_base: float = 0.12            # boosted from 0.08 for trim-bias correction
    ki_boost: float = 0.16           # boosted from 0.14
    nz_boost_threshold: float = 1.5  # |target_nz - 1.0| above this → begin boost
    nz_boost_slope: float = 2.0      # tanh slope for smooth transition

    def schedule_nz(self, airspeed_mps: float, target_nz: float
                    ) -> tuple[float, float, float]:
        """Return (kp, ki, kd) for the Nz channel at current flight condition."""
        V = max(airspeed_mps, 80.0)

        # 1/V² scaling for kp and kd
        ratio_sq = (self.ref_speed_mps / V) ** 2
        kp = float(np.clip(self.kp_ref * ratio_sq, self.kp_min, self.kp_max))
        kd = float(np.clip(self.kd_ref * ratio_sq, self.kd_min, self.kd_max))

        # ki boost: smoothly transition from baseline to boosted for large commands
        delta_nz = abs(target_nz - 1.0)  # deviation from level flight
        boost_factor = 0.5 * (1.0 + math.tanh(self.nz_boost_slope * (delta_nz - self.nz_boost_threshold)))
        ki = self.ki_base + (self.ki_boost - self.ki_base) * boost_factor

        return kp, ki, kd


# ═══════════════════════════════════════════════════════════════════════════════
#  BFM autopilot — λ-g flight control law
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BFMAutopilotConfig:
    """Gains for the four-channel BFM autopilot.

    Phase 3.5 (conservative): boost integral + derivative within cascaded
    stability limits.  Outer-loop kp must stay ≤ 0.5× inner-loop kp to
    avoid fighting the JSBSim native FCS PIDs.
    """

    # ── Nz (elevator) channel ─────────────────────────────────────────
    # JSBSim inner G-load PID: kp=0.3.  Keep outer kp ≤ 0.18 (0.6× margin).
    # NOTE: When GainScheduler is active, kp/ki/kd are overridden at runtime.
    nz_kp: float = 0.18      # hold at Phase 2 (stable response)
    nz_ki: float = 0.05      # 2.5x from 0.02 — kill steady-state G error
    nz_kd: float = 0.025     # 2x from 0.012 — dampen turn-induced pitch jitter
    # Q-damping gain (2026-06-25): inner-loop pitch-rate feedback.
    # Positive Kq = nose-up rate → elevator push-down, damping the rotation.
    # Tuned for F-16 at 180–350 m/s: provides critical damping without
    # fighting the outer Nz loop.
    nz_kq: float = 0.15
    # Integral clamping tightened from ±0.4 → ±0.3 (anti-windup, 2026-06-25).
    nz_integral_min: float = -0.3
    nz_integral_max: float = 0.3

    # ── Roll (aileron) channel ────────────────────────────────────────
    # JSBSim inner roll-rate PID: kp=3.0.  Keep outer kp ≤ 1.5 (0.5× margin).
    # 2026-06-25: roll inertia overshoot fix — doubled kp (1.5→3.0) because
    # at small errors (~1°) the original kp produced negligible aileron
    # (<0.04), unable to stop the aircraft's roll momentum past 60° target.
    # Also boosted ki for faster steady-state convergence.
    roll_kp: float = 3.0     # doubled from 1.5 for roll momentum arrest
    roll_ki: float = 0.15    # boosted from 0.10
    roll_kd: float = 0.15    # boosted from 0.10 for better rate damping
    roll_integral_min: float = -0.4
    roll_integral_max: float = 0.4

    # ── Speed (throttle) channel ──────────────────────────────────────
    speed_kp: float = 0.02
    speed_ki: float = 0.005
    speed_kd: float = 0.0
    speed_integral_min: float = -0.3
    speed_integral_max: float = 0.3
    min_target_speed_mps: float = 80.0
    max_target_speed_mps: float = 400.0

    # ── Sideslip (rudder) channel ─────────────────────────────────────
    beta_kp: float = 0.06
    beta_ki: float = 0.005
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
                 trim: Optional[TrimSchedule] = None,
                 scheduler: Optional[GainScheduler] = None) -> None:
        cfg = config or BFMAutopilotConfig()
        self._trim = trim or TrimSchedule()
        self._scheduler = scheduler  # None → use fixed cfg gains

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

        # Q-damping gain (inner-loop pitch-rate feedback, 2026-06-25)
        self._kq = cfg.nz_kq

        # Command pre-filter state (low-pass on target_nz to avoid
        # step shocks that cause integral windup, 2026-06-25)
        self._filtered_target_nz: Optional[float] = None

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
        self._filtered_target_nz = None

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
        q_rps: float = 0.0,
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
            q_rps:         Body-Y pitch rate (rad/s).  JSBSim ``velocities/q-rad_sec``.
                           Used for inner-loop Q-damping (2026-06-25).

        Returns:
            ``(throttle, elevator, aileron, rudder)`` — all in [-1, 1].
        """
        # ── Elevator: track body-Z normal acceleration ───────────────
        # 1. Dynamic trim: 1/V² speed-scaling from calibrated reference
        elevator_trim = self._trim.get_elevator_trim(airspeed_mps)

        # 2. Target: BFM n_n → body-Z, with bank feedforward + pre-filter
        cos_roll = math.cos(abs(roll_rad))
        bank_extra_g = (1.0 / max(cos_roll, 0.1)) - 1.0
        raw_target = -(n_n + bank_extra_g)

        # Command pre-filter (2026-06-25): first-order low-pass prevents
        # square-wave step shocks from slamming the PID with instantaneous
        # large errors that cause integral windup and porpoising.
        # tau ~ dt / (1 - 0.85) = 0.017 / 0.15 ~ 0.11 s smoothing at 60 Hz.
        if self._filtered_target_nz is None:
            self._filtered_target_nz = raw_target
        else:
            alpha_lpf = 0.15
            self._filtered_target_nz = ((1.0 - alpha_lpf) * self._filtered_target_nz
                                        + alpha_lpf * raw_target)
        target_n_z_g = self._filtered_target_nz

        # 3. Gain scheduling (Phase 3.5): adapt Nz gains to speed + target
        if self._scheduler is not None:
            kp_s, ki_s, kd_s = self._scheduler.schedule_nz(
                airspeed_mps, abs(target_n_z_g))
            self._nz_pid.kp = kp_s
            self._nz_pid.ki = ki_s
            self._nz_pid.kd = kd_s

        # 4. Error
        nz_error = n_z_g - target_n_z_g   # + → need more negative n_z (more pull)

        # 5. Derivative on Nz error (computed BEFORE integral update
        #    so the D-term reflects the raw error rate of change)
        derivative = 0.0
        if self._nz_pid._prev_error is not None and dt > 1e-8:
            derivative = (nz_error - self._nz_pid._prev_error) / dt
        self._nz_pid._prev_error = nz_error

        # 6. Q-damping (inner-loop pitch-rate feedback, 2026-06-25)
        q_damping = self._kq * q_rps

        # 7. Assemble PID output and elevator with back-calculation
        #    anti-windup (2026-06-25).
        #
        #    First compute the full output using the CURRENT integral.
        #    Then clip the elevator.  If it saturates, solve for the
        #    integral that would put the elevator exactly at the limit.
        #    This is the gold-standard "back-calculation" method and
        #    avoids the chicken-and-egg problem of predicting saturation.
        pid_out = (self._nz_pid.kp * nz_error
                   + self._nz_pid.ki * self._nz_pid._integral
                   + self._nz_pid.kd * derivative
                   + q_damping)
        elev_unclipped = elevator_trim - pid_out
        elevator = float(np.clip(elev_unclipped, -1.0, 1.0))

        if abs(elevator - elev_unclipped) > 1e-8:
            # Saturation occurred.  Back-calculate integral so that the
            # controller output is consistent with the achievable limit.
            # elevator_limit = elevator_trim - (kp*e + ki*integral + kd*de + kq*q)
            # → ki*integral = elevator_trim - elevator_limit - kp*e - kd*de - kq*q
            if abs(self._nz_pid.ki) > 1e-8:
                integral_limit = (elevator_trim - elevator
                                  - self._nz_pid.kp * nz_error
                                  - self._nz_pid.kd * derivative
                                  - q_damping) / self._nz_pid.ki
                self._nz_pid._integral = float(np.clip(
                    integral_limit,
                    self._nz_pid.integral_min, self._nz_pid.integral_max))
        else:
            # Not saturated — normal integral update with hard clamp
            self._nz_pid._integral += nz_error * dt
            self._nz_pid._integral = float(np.clip(
                self._nz_pid._integral,
                self._nz_pid.integral_min, self._nz_pid.integral_max))

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
        # Convention bridge (2026-06-25):
        #   JSBSim attitude/roll-rad:  positive = right bank
        #   BFM action mu / Envelope:  positive = left  bank
        # Convert mu from BFM → JSBSim so both operands share one frame.
        # Original formula  roll_rad - mu  exploited the sign mismatch
        # for direct BFM targets but gave positive feedback (wrong sign)
        # when recovering to wings-level (mu=0, roll≠0).
        mu_jsbsim = -mu
        roll_error = mu_jsbsim - roll_rad   # + when need right roll
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
