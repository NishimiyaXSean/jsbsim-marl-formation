"""PIDFlightController — wraps existing FlightController + BFMAutopilot.

This is the default controller. It uses:
  - FlightController (heading/altitude/speed PID loops)
  - BFMAutopilot (trim + gain scheduling)

to convert FlightTarget → ControlSurfaces.

Usage:
    ctrl = PIDFlightController()
    surfaces = ctrl.predict(aircraft.state, target, dt)
    aircraft.set_controls(surfaces.throttle, surfaces.elevator, ...)
"""

from __future__ import annotations

from typing import Optional

from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler

from .controller_base import BaseController, ControlSurfaces, FlightTarget


class PIDFlightController(BaseController):
    """PID-based flight controller wrapping FlightController + BFMAutopilot."""

    def __init__(self):
        self._fc = FlightController()
        self._ap = BFMAutopilot(
            BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())

    @property
    def controller_type(self) -> str:
        return "pid"

    def predict(self, state: dict, target: FlightTarget, dt: float) -> ControlSurfaces:
        """Run PID loops → return control surfaces."""
        fc_target = FlightControlTargets(
            heading_deg=target.heading_deg,
            altitude_m=target.altitude_m,
            speed_mps=target.speed_mps,
        )
        thr, elev, ail, rud = self._fc.compute(state, fc_target, dt)
        return ControlSurfaces(
            throttle=float(thr),
            elevator=float(elev),
            aileron=float(ail),
            rudder=float(rud),
        )

    def reset(self, initial_speed_mps: float = 200.0) -> None:
        self._fc.reset()
        self._ap.reset(initial_speed_mps=initial_speed_mps)
