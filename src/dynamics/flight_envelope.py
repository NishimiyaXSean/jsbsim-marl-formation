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

    # First-order lag for G-onset  (reduced from 0.4s for better autopilot tracking)
    tau_g: float = 0.15

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

    def reset(self) -> None:
        """Reset per-episode smoothing state."""
        self._n_x_sm: float = 0.0    # smoothed tangential G
        self._n_n_sm: float = 1.0    # smoothed normal G  (1 G = level trim)
        self._mu_sm: float = 0.0     # smoothed bank angle (rad)

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

        # 2.  Speed-based tangential clamp  ─────────────────────────────────
        n_x_cmd = self._clamp_tangential(n_x_cmd, speed_mps, speed_scale)

        # 3.  First-order G-onset lag  ──────────────────────────────────────
        n_x_sm, n_n_sm = self._smooth_g(n_x_cmd, n_n_cmd, dt)

        # 4.  Roll P-controller with rate limiting  ─────────────────────────
        # _roll_step works entirely in JSBSim convention (positive = right).
        # Negate target_mu from BFM (positive = left) → JSBSim, then negate
        # the output for the BFMAutopilot which also works in BFM convention.
        target_mu_jsbsim = -target_mu
        mu_cmd_jsbsim = self._roll_step(target_mu_jsbsim, current_roll_rad, dt)
        mu_cmd = -mu_cmd_jsbsim  # back to BFM convention for autopilot

        # 5.  GPWS override  (hard safety — runs LAST to guarantee pull-up) ─
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
