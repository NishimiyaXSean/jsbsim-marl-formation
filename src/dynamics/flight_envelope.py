"""F-16 flight envelope protection for JSBSim-powered air combat.

Migrated from the PyBullet point-mass model in marl_env.py and adapted for
JSBSim's 6-DOF dynamics.  Provides:

- V-n diagram limits (structural + aerodynamic)
- Corner-speed-based lift-limited G availability
- Stall / overspeed tangential acceleration clamps
- GPWS (Ground Proximity Warning System) override
- First-order G-onset lag for realistic engine / airframe response
- Roll-angle P-controller with rate limiting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EnvelopeConfig:
    """F-16-class flight envelope parameters.

    All speeds in m/s, altitudes in m, accelerations in G.
    Values are the same as those used in the original PyBullet 1v1 marl_env.
    """

    max_g: float = 9.0            # Structural positive G limit
    min_g: float = -3.0           # Structural negative G limit  (asymmetric — jets can't push as hard as they can pull)
    corner_speed: float = 150.0   # Corner speed (m/s, ≈ 540 km/h) — minimum speed at which max G is aerodynamically achievable
    max_speed: float = 400.0      # Absolute maximum level speed (m/s, ≈ M1.2 at altitude)
    stall_speed: float = 60.0     # Base stall speed (m/s)
    g: float = 9.81               # Gravitational acceleration

    # GPWS hard-coded pull-up altitudes  (attacker has more freedom to go low;
    # evader gets a higher floor to keep the engagement inside the arena)
    gpws_trigger_alt_attacker: float = 500.0   # raised from 300 for training safety
    gpws_trigger_alt_evader: float = 800.0
    gpws_vz_threshold: float = -5.0        # must be descending faster than this (m/s)

    # Ceiling / hard-deck protection (2026-06-25)
    ceiling_trigger_alt: float = 5800.0   # force nose-down above this
    ceiling_target_g: float = 0.5         # max G when above ceiling
    hard_deck_trigger_alt: float = 1500.0 # force pull-up below this
    hard_deck_target_g: float = 3.0       # min G when below hard deck

    # First-order lag for G-onset  (reduced from 0.4s for better autopilot tracking)
    tau_g: float = 0.15

    # Altitude-hold gain (2026-06-25): mild P-correction on Nz when the
    # aircraft drifts from the reference altitude.  Prevents the excess-thrust
    # climb that occurs when n_n=1.0G is tracked perfectly but the throttle
    # channel cannot fully compensate for the thrust surplus.
    alt_hold_kp: float = 0.0003  # ~0.3G correction per 1000m altitude error
    alt_hold_max_correction: float = 0.2  # max G correction (gentle)

    # Roll control
    max_roll_rate: float = np.pi           # 180 deg/s
    roll_gain: float = 4.0                 # P-controller proportional gain


# ── Flight envelope processor ────────────────────────────────────────────────

class FlightEnvelope:
    """Per-aircraft flight envelope protection and BFM-command smoothing.

    Each aircraft (attacker / evader) needs its own instance because the
    envelope maintains per-aircraft state (smoothed G, roll angle).

    Usage inside the environment step loop::

        envelope = FlightEnvelope()
        for each micro-step:
            n_x_sm, n_n_sm, mu_cmd = envelope.step(
                n_x_cmd, n_n_cmd, target_mu, speed_mps, alt_m, vz_mps,
                current_roll_rad, dt, is_attacker=True, g_scale=1.0, speed_scale=1.0,
            )
            # feed n_x_sm, n_n_sm, mu_cmd to BFMAutopilot → control surfaces
    """

    def __init__(self, config: Optional[EnvelopeConfig] = None) -> None:
        self.cfg = config or EnvelopeConfig()
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, ref_alt_m: float = 3048.0) -> None:
        """Reset per-episode smoothing state.

        Args:
            ref_alt_m: Reference altitude for altitude-hold (m MSL).
        """
        self._n_x_sm: float = 0.0    # smoothed tangential G
        self._n_n_sm: float = 1.0    # smoothed normal G  (1 G = level trim)
        self._mu_sm: float = 0.0     # smoothed bank angle (rad)
        self._ref_alt_m: float = ref_alt_m  # altitude-hold reference

    def step(
        self,
        n_x_cmd: float,
        n_n_cmd: float,
        target_mu: float,
        *,
        speed_mps: float,
        alt_m: float,
        vz_mps: float,
        current_roll_rad: float,
        dt: float,
        is_attacker: bool = True,
        g_scale: float = 1.0,
        speed_scale: float = 1.0,
    ) -> tuple[float, float, float]:
        """Process raw BFM commands through the full flight envelope.

        Args:
            n_x_cmd:        Desired tangential acceleration (G).  + = speed up.
            n_n_cmd:        Desired normal acceleration (G).  + = pull up.
            target_mu:      Desired bank angle (rad).
            speed_mps:      Current true airspeed (m/s).
            alt_m:          Current altitude MSL (m).
            vz_mps:         Current vertical speed (m/s, positive = climbing).
            current_roll_rad: Current roll angle (rad).
            dt:             Simulation time step (s).
            is_attacker:    True → use attacker GPWS floor; False → evader floor.
            g_scale:        Curriculum multiplier on G limits (1.0 = full).
            speed_scale:    Curriculum multiplier on speed limits (1.0 = full).

        Returns:
            (n_x_smoothed, n_n_smoothed, mu_cmd) — the physically realisable
            targets to send to the autopilot.
        """
        # 1.  V-n envelope  ────────────────────────────────────────────────
        n_n_cmd = self._apply_vn_limits(n_n_cmd, speed_mps, g_scale)

        # 1b. Altitude-hold correction (2026-06-25): mild P-bias on Nz
        #      to prevent the BFMAutopilot's excess-thrust climb when
        #      n_n=1.0G is tracked but no altitude feedback exists.
        n_n_cmd = self._apply_altitude_hold(n_n_cmd, alt_m)

        # 2.  Speed-based tangential clamp  ─────────────────────────────────
        n_x_cmd = self._clamp_tangential(n_x_cmd, speed_mps, speed_scale)

        # 3.  First-order G-onset lag  ──────────────────────────────────────
        n_x_sm, n_n_sm = self._smooth_g(n_x_cmd, n_n_cmd, dt)

        # 4.  Roll target — pass-through with convention bridge ───────────
        # _roll_step provides rate-limited trajectory tracking, BUT its
        # output stays too close to current_roll when near the target,
        # starving the BFMAutopilot of error signal.  For the autopilot
        # we pass the RAW target (with convention conversion) so the PID
        # sees the full error.  Roll rate is inherently limited by the
        # aileron PID gains + JSBSim's own roll damping.
        mu_cmd = target_mu  # already in BFM convention (positive = left)

        # 5.  Altitude protection: ceiling + hard deck (2026-06-25) ─────
        n_n_sm, mu_cmd = self._apply_altitude_limits(n_n_sm, mu_cmd, alt_m)

        # 6.  GPWS override  (hard safety — runs LAST to guarantee pull-up) ─
        n_n_sm, mu_cmd = self._apply_gpws(n_n_sm, mu_cmd, alt_m, vz_mps,
                                          g_scale, is_attacker)

        return n_x_sm, n_n_sm, mu_cmd

    # ── Internal steps ────────────────────────────────────────────────────

    def _apply_vn_limits(self, n_n: float, V: float, g_scale: float) -> float:
        """Clamp normal-G command to the V-n envelope.

        The available lift (and therefore G) scales with V².
        At the corner speed the aircraft can *just* reach its structural G limit;
        below corner speed the aerodynamic limit bites first.
        """
        V = max(V, 1e-3)
        scaled_max_g = self.cfg.max_g * g_scale
        scaled_min_g = self.cfg.min_g * g_scale

        available_n_lift = ((V / self.cfg.corner_speed) ** 2) * scaled_max_g

        actual_max_n = min(scaled_max_g, available_n_lift)
        actual_min_n = max(scaled_min_g, -available_n_lift)

        return float(np.clip(n_n, actual_min_n, actual_max_n))

    def _clamp_tangential(self, n_x: float, V: float, speed_scale: float) -> float:
        """Block acceleration past max speed and deceleration below stall."""
        scaled_max_speed = self.cfg.max_speed * speed_scale

        if V > scaled_max_speed and n_x > 0:
            return 0.0   # drag wall — can't push through
        if V < self.cfg.stall_speed and n_x < 0:
            return 0.0   # stall protection — don't bleed the last knots
        return n_x

    def _smooth_g(self, n_x: float, n_n: float, dt: float) -> tuple[float, float]:
        """First-order exponential lag on G commands.

        tau_g = 0.4 s means the aircraft reaches 63 % of the commanded G
        in 0.4 s, and 95 % in ~1.2 s — realistic for F-16 FCS + engine.
        """
        alpha = dt / (self.cfg.tau_g + dt)
        self._n_x_sm += (n_x - self._n_x_sm) * alpha
        self._n_n_sm += (n_n - self._n_n_sm) * alpha
        return self._n_x_sm, self._n_n_sm

    def _roll_step(self, target_mu: float, current_roll: float, dt: float) -> float:
        """P-controller for roll angle with hard rate limiting.

        All computation is done in the aircraft-native JSBSim convention
        (positive = right bank).  The caller (:meth:`step`) is responsible
        for the BFM ↔ JSBSim convention bridge before passing the result
        to the autopilot.
        """
        error = target_mu - current_roll
        error = (error + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

        roll_rate = np.clip(
            error * self.cfg.roll_gain,
            -self.cfg.max_roll_rate,
            self.cfg.max_roll_rate,
        )
        raw_output = float(current_roll + roll_rate * dt)
        # Wrap to [-pi, pi] — keeps the angle bounded even when the aircraft
        # rolls through multiple revolutions (cosmetic; the PID uses the
        # wrapped error, so the control action is identical either way).
        return float((raw_output + np.pi) % (2 * np.pi) - np.pi)

    def _apply_altitude_hold(self, n_n: float, alt_m: float) -> float:
        """Mild altitude-hold correction on the Nz command (2026-06-25).

        Only activates when the raw Nz command is near 1.0G (level-flight
        intent).  During climbs, descents, or turns the altitude hold
        stays quiet to avoid fighting the manoeuvre.
        """
        # Only correct when level-flight intent (n_n within 0.3G of 1.0)
        if abs(n_n - 1.0) > 0.3:
            return n_n

        alt_error = self._ref_alt_m - alt_m  # + when below target (need climb)
        correction = alt_error * self.cfg.alt_hold_kp
        correction = float(np.clip(correction, -self.cfg.alt_hold_max_correction,
                                   self.cfg.alt_hold_max_correction))
        return n_n + correction

    def _apply_altitude_limits(
        self, n_n: float, mu: float, alt_m: float,
    ) -> tuple[float, float]:
        """Ceiling and hard-deck protection (2026-06-25).

        Above the ceiling, force nose-down to prevent "escape to space".
        Below the hard deck, force wings-level pull-up to prevent CFIT.
        These run BEFORE GPWS so that GPWS can further override if needed.
        """
        # Ceiling: force nose-down to stop climb
        if alt_m > self.cfg.ceiling_trigger_alt:
            n_n = min(n_n, self.cfg.ceiling_target_g)
            mu = 0.0  # wings level — don't turn while descending

        # Hard deck: force pull-up, wings level for max vertical lift
        if alt_m < self.cfg.hard_deck_trigger_alt:
            n_n = max(n_n, self.cfg.hard_deck_target_g)
            mu = 0.0  # wings level

        return n_n, mu

    def _apply_gpws(
        self, n_n: float, mu: float, alt_m: float, vz_mps: float,
        g_scale: float, is_attacker: bool,
    ) -> tuple[float, float]:
        """Ground Proximity Warning System — hard override.

        If the aircraft is *descending* below its trigger altitude,
        forcibly command max-G wings-level pull-up.  This is the
        highest-priority safety interlock in the envelope.
        """
        trigger = (self.cfg.gpws_trigger_alt_attacker if is_attacker
                   else self.cfg.gpws_trigger_alt_evader)

        if alt_m < trigger and vz_mps < self.cfg.gpws_vz_threshold:
            return self.cfg.max_g, 0.0  # PULL UP, wings level — always full G for safety

        return n_n, mu
