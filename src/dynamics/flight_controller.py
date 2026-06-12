"""Stabilized flight controller for JSBSim F-16.

Provides three independently-tested PID stabilisers around a fixed trim
baseline.  Each stabiliser outputs small control-surface deltas to be
superimposed on top of the trim point.

Trim state
----------
Verified on JSBSim F-16 at 3000 m (9842 ft) / 400 kts (206 m/s):
    throttle = 0.8,  elevator = -0.05
    → level flight, pitch ≈ 0.8°,  n_z ≈ -0.95 G,  equilibrium speed ≈ 176 m/s

Elevator sign convention (JSBSim F-16)
---------------------------------------
- elevator > 0  →  nose goes DOWN  (n_z > -1.0 G, altitude decreases)
- elevator < 0  →  nose goes UP    (n_z < -1.0 G, altitude increases)

All altitude/vertical-speed PIDs use **negative gains** so that a positive
error ("need to climb") produces a negative elevator command ("pull up").

Control authority at trim
-------------------------
- elevator  delta ±0.05  → ±15 m/s vertical speed
- aileron   delta ±0.05  → ±2.6 °/s heading rate, ≈50° bank
- throttle  delta ±0.20  → speed range 140–206 m/s
- rudder    delta ±0.05  → marginal sideslip correction (~0.3° beta reduction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.dynamics.autopilot import PIDController


# ═══════════════════════════════════════════════════════════════════════════════
#  Trim constants
# ═══════════════════════════════════════════════════════════════════════════════

THROTTLE_TRIM = 0.80
ELEVATOR_TRIM = -0.05


# ═══════════════════════════════════════════════════════════════════════════════
#  Altitude stabiliser  (alt_m → elevator)
# ═══════════════════════════════════════════════════════════════════════════════

class AltitudeStabilizer:
    """Hold altitude target via elevator.

    Gains tuned for F-16 at 3000 m / 176–206 m/s.
    """

    def __init__(self) -> None:
        # Negative kp: positive alt error → need climb → negative elevator (pull)
        self._pid = PIDController(
            kp=0.008, ki=0.0005, kd=0.002,
            output_min=-0.30, output_max=0.30,
            integral_min=-0.15, integral_max=0.15,
        )

    def reset(self) -> None:
        self._pid.reset()

    def compute(self, alt_m: float, target_alt_m: float, dt: float) -> float:
        """Return *elevator* in [-1, 1] — trim + delta applied by caller."""
        error = target_alt_m - alt_m
        return ELEVATOR_TRIM - self._pid.step(error, dt)


# ═══════════════════════════════════════════════════════════════════════════════
#  Speed stabiliser  (airspeed → throttle)
# ═══════════════════════════════════════════════════════════════════════════════

class SpeedStabilizer:
    """Hold airspeed target via throttle.

    Gains tuned for F-16 at 3000 m.  Equilibrium speed with thr=0.8 is
    ≈ 176 m/s, so the target should be in the 140–190 m/s range.
    """

    def __init__(self) -> None:
        self._pid = PIDController(
            kp=0.015, ki=0.010, kd=0.0,
            output_min=-0.30, output_max=0.20,
            integral_min=-0.15, integral_max=0.20,
        )

    def reset(self) -> None:
        self._pid.reset()

    def compute(self, airspeed_mps: float, target_speed_mps: float, dt: float) -> float:
        """Return *throttle* in [0, 1] — trim + delta applied by caller."""
        error = target_speed_mps - airspeed_mps
        return THROTTLE_TRIM + self._pid.step(error, dt)


# ═══════════════════════════════════════════════════════════════════════════════
#  Heading stabiliser  (heading → aileron + rudder)
# ═══════════════════════════════════════════════════════════════════════════════

class HeadingStabilizer:
    """Hold heading target via aileron (roll-to-turn) + rudder.

    The outer loop converts heading error into a desired bank angle;
    the inner loop tracks that bank angle via aileron.  Rudder provides
    mild turn coordination.

    During sustained turns, the altitude channel receives a bank-angle
    feed-forward boost to compensate for the reduced vertical lift component.
    """

    ROLL_PER_DEG_HEADING = 2.5      # deg bank per deg heading error (aggressive for pursuit)
    MAX_BANK_DEG = 70.0              # allow up to 70° bank for fast turns (altitude channel compensates)

    def __init__(self) -> None:
        self._roll_pid = PIDController(
            kp=0.10, ki=0.03, kd=0.03,
            output_min=-0.50, output_max=0.50,
            integral_min=-0.10, integral_max=0.10,
        )
        self._rudder_pid = PIDController(
            kp=0.08, ki=0.02, kd=0.0,
            output_min=-0.10, output_max=0.10,
        )

    def reset(self) -> None:
        self._roll_pid.reset()
        self._rudder_pid.reset()

    def compute(
        self,
        heading_deg: float,
        target_heading_deg: float,
        roll_deg: float,
        sideslip_deg: float,
        dt: float,
    ) -> tuple[float, float]:
        """Return *(aileron, rudder)* — deltas around zero (trim=0)."""
        # Heading error wrapped to [-180, 180]
        hdg_err = (target_heading_deg - heading_deg + 180.0) % 360.0 - 180.0

        # Desired bank angle (limited to MAX_BANK)
        desired_roll_deg = np.clip(
            hdg_err * self.ROLL_PER_DEG_HEADING,
            -self.MAX_BANK_DEG, self.MAX_BANK_DEG,
        )

        # Bank error → aileron (sign: positive error = need right roll)
        roll_err = desired_roll_deg - roll_deg
        # Wrap to [-180, 180]
        roll_err = (roll_err + 180.0) % 360.0 - 180.0

        aileron = self._roll_pid.step(roll_err, dt)

        # Rudder kills sideslip (beta > 0 = nose left in JSBSim → rudder > 0)
        rudder = -self._rudder_pid.step(sideslip_deg, dt)

        return aileron, rudder


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined flight controller
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlightControlTargets:
    """High-level flight targets set by the RL agent."""

    heading_deg: float = 90.0     # target heading (0=North, CW)
    altitude_m: float = 3000.0     # target MSL altitude (m)
    speed_mps: float = 180.0       # target calibrated airspeed (m/s)


class FlightController:
    """Combined altitude + speed + heading stabiliser for JSBSim F-16.

    Usage inside the environment micro-step loop::

        fc = FlightController()
        fc.reset()
        for each micro_step:
            target = FlightControlTargets(heading_deg=..., altitude_m=..., speed_mps=...)
            thr, elev, ail, rud = fc.compute(aircraft.state, target, dt)
            aircraft.set_controls(thr, elev, ail, rud)
    """

    def __init__(self) -> None:
        self.alt = AltitudeStabilizer()
        self.spd = SpeedStabilizer()
        self.hdg = HeadingStabilizer()

    def reset(self) -> None:
        self.alt.reset()
        self.spd.reset()
        self.hdg.reset()

    def compute(
        self,
        state: dict,               # Aircraft.state dict
        target: FlightControlTargets,
        dt: float,
    ) -> tuple[float, float, float, float]:
        """Return ``(throttle, elevator, aileron, rudder)`` in valid ranges.

        All three channels operate independently.
        During banking turns the altitude channel receives a boost to
        compensate for the reduced vertical lift component.
        """
        # Bank-compensated altitude target: when banked, the vertical
        # component of lift is cos(bank).  We need more elevator to pull
        # higher total G to maintain the same vertical component.
        import math
        roll_abs_rad = math.radians(abs(state["roll_deg"]))
        cos_roll = math.cos(roll_abs_rad)
        if cos_roll > 0.1:
            bank_factor = 1.0 / cos_roll  # e.g. 1.0 at 0°, 2.0 at 60°, 2.9 at 70°
        else:
            bank_factor = 10.0  # near 90° bank — max compensation

        elevator_base = self.alt.compute(state["alt_m"], target.altitude_m, dt)
        # Re-center around trim and apply bank boost
        d_elev = (elevator_base - ELEVATOR_TRIM) * bank_factor
        elevator = np.clip(ELEVATOR_TRIM + d_elev, -1.0, 1.0)

        throttle = self.spd.compute(state["airspeed_mps"], target.speed_mps, dt)
        aileron, rudder = self.hdg.compute(
            state["yaw_deg"], target.heading_deg,
            state["roll_deg"], state["beta_deg"], dt,
        )
        return throttle, elevator, aileron, rudder
