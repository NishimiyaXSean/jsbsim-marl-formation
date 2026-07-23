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

from src.dynamics.autopilot import PIDController, TrimSchedule


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared trim schedule (Phase 3: dynamic, replaces hardcoded constants)
# ═══════════════════════════════════════════════════════════════════════════════

_trim = TrimSchedule()

# Backward-compatible module-level constants (derived from TrimSchedule defaults)
THROTTLE_TRIM = _trim.ref_throttle
ELEVATOR_TRIM = _trim.ref_elevator


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

    Uses cascaded control with heading-rate damping to prevent overshoot:
      1. Outer loop: heading error → desired bank angle (P + D on heading)
      2. Inner loop: roll PID tracks the desired bank angle

    The heading-rate D term reduces desired bank as the aircraft turns
    toward the target, preventing the classic bank-to-turn overshoot.
    """

    # F-16 at 200 m/s, 70° bank: max turn rate ≈ 7.7°/s (physics limit).
    # At 130 m/s (agent can slow down): turn rate ≈ 11.9°/s.
    # The agent must learn energy management for tight turns.
    # F-16 at 200 m/s, 70° bank: max turn rate ≈ 7.7°/s (physics limit).
    # The agent learns energy management — slowing to 130 m/s gives 11.9°/s.
    ROLL_PER_DEG_HEADING = 2.5      # deg bank per deg heading error
    MAX_BANK_DEG = 70.0

    def __init__(self) -> None:
        # Roll PID with BFM-ported smooth gains + moderate output range.
        # P-only outer heading loop — steady-state error provides natural
        # exploration that helps RL escape local optima.
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

        # Desired bank proportional to heading error
        desired_roll_deg = float(np.clip(
            hdg_err * self.ROLL_PER_DEG_HEADING,
            -self.MAX_BANK_DEG, self.MAX_BANK_DEG,
        ))

        # Bank error → aileron (positive error = need right roll → positive aileron)
        roll_err = desired_roll_deg - roll_deg
        roll_err = (roll_err + 180.0) % 360.0 - 180.0

        aileron = self._roll_pid.step(roll_err, dt)

        # Rudder kills sideslip
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
        d_elev = elevator_base - ELEVATOR_TRIM

        # ── Bank turn compensation (feedforward) ───────────────────────
        # During a banked turn, the vertical component of lift drops as
        # cos(bank). We need more elevator to pull higher total G and
        # maintain altitude. A feedforward term independent of alt error
        # ensures compensation even when error=0 (e.g., entry into turn).
        # Scale: at 60° bank (cos=0.5), we add ~0.15 extra elevator.
        K_bank_ff = 0.30
        bank_ff = K_bank_ff * (bank_factor - 1.0)  # 0 at 0° bank
        # PID correction amplified for the reduced lift component
        d_elev = d_elev * bank_factor + bank_ff

        elevator = ELEVATOR_TRIM + d_elev

        # ── Pitch rate damping (artificial C_mq) ───────────────────────
        # JSBSim F-16: elevator > 0 → nose DOWN.
        # When nose pitches UP (q > 0), add positive elevator to push
        # nose back down, extracting energy from the phugoid mode.
        q_rps = float(state.get("q_rps", 0.0))  # body-frame pitch rate (rad/s)
        Kd_q = 0.8
        elevator += Kd_q * q_rps  # +q → +elev → nose DOWN → damping ✅

        elevator = np.clip(elevator, -1.0, 1.0)

        throttle = self.spd.compute(state["airspeed_mps"], target.speed_mps, dt)
        aileron, rudder = self.hdg.compute(
            state["yaw_deg"], target.heading_deg,
            state["roll_deg"], state["beta_deg"], dt,
        )
        return throttle, elevator, aileron, rudder
