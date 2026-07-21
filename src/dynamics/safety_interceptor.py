"""SafetyInterceptor — flight envelope protection for any controller.

Wraps a BaseController and applies hard action masking based on flight safety
constraints. Designed to work with both PID and Neural controllers.

Hard masking rules (matching FormationRLlibEnv._build_action_mask):
  - airspeed < 130 m/s:  block slow speed + hard turns
  - altitude < 200 m:    block hard turns
  - altitude < 100 m + high G:  block all turns except straight
  - airspeed > 95% Vmax: block fast speed

Reserved interface for smooth blending (future):
  - blend_factor: float [0, 1] — 0 = full hard mask, 1 = soft blend
  - safe_fallback: ControlSurfaces — default safe action when masked
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .controller_base import BaseController, ControlSurfaces, FlightTarget


class SafetyInterceptor:
    """Wraps a BaseController with flight envelope protection.

    Applies hard action masking on the controller output. If the controller
    requests a forbidden action, it is replaced by a safe fallback.
    """

    # ── Thresholds (same as FormationRLlibEnv constants) ────────────────
    STALL_SPEED_WARN = 130.0       # m/s — below this, block slow + hard turns
    GROUND_PROXIMITY = 200.0       # m   — below this, block hard turns
    GROUND_CRITICAL = 100.0        # m   — below this, block all turns
    OVERSPEED_FACTOR = 0.95        # × Vmax (380 m/s)
    VMAX = 400.0                   # m/s

    def __init__(self, inner: BaseController, blend_factor: float = 0.0):
        self._inner = inner
        self._blend_factor = blend_factor  # 0 = hard mask, 1 = smooth (reserved)

    @property
    def controller_type(self) -> str:
        return f"safe({self._inner.controller_type})"

    def reset(self, initial_speed_mps: float = 200.0) -> None:
        self._inner.reset(initial_speed_mps)

    def predict(self, state: dict, target: FlightTarget, dt: float) -> ControlSurfaces:
        """Get raw controller output, then apply safety masking."""
        raw = self._inner.predict(state, target, dt)

        airspeed = float(state.get("airspeed_mps", 200.0))
        alt_m = float(state.get("alt_m", 3000.0))
        nz_g = float(state.get("n_z_g", 1.0))

        # Build safety mask
        mask_slow = airspeed < self.STALL_SPEED_WARN
        mask_low = alt_m < self.GROUND_PROXIMITY
        mask_critical = alt_m < self.GROUND_CRITICAL and nz_g > 2.0
        mask_fast = airspeed > self.VMAX * self.OVERSPEED_FACTOR

        ail = raw.aileron
        elev = raw.elevator
        rud = raw.rudder
        thr = raw.throttle

        # Hard mask: override forbidden actions with safe fallback
        if mask_slow:
            # Block slow speed → force at least cruise throttle
            thr = max(thr, 0.65)
            # Block hard turns
            if abs(ail) > 0.8:
                ail = 0.0
            if abs(rud) > 0.8:
                rud = 0.0

        if mask_low:
            # Block hard turns near ground
            if abs(ail) > 0.8:
                ail = 0.0
            if abs(rud) > 0.8:
                rud = 0.0

        if mask_critical:
            # Only allow straight flight
            ail = 0.0
            rud = 0.0

        if mask_fast:
            # Block fast speed → force cruise at most
            thr = min(thr, 0.7)

        return ControlSurfaces(
            throttle=float(np.clip(thr, 0.0, 1.0)),
            elevator=float(np.clip(elev, -1.0, 1.0)),
            aileron=float(np.clip(ail, -1.0, 1.0)),
            rudder=float(np.clip(rud, -1.0, 1.0)),
        )
