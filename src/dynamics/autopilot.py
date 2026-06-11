"""Simple PID autopilot to map high-level commands to JSBSim control surfaces.

For the initial demo we use direct continuous control (throttle/elevator/aileron/rudder).
This module provides optional PID controllers for altitude-hold, speed-hold, and coordinated turns.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PIDController:
    """Discrete PID controller with anti-windup clamping."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_min: float = -1.0
    integral_max: float = 1.0

    _integral: float = 0.0
    _prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = np.clip(self._integral, self.integral_min, self.integral_max)

        derivative = 0.0
        if self._prev_error is not None and dt > 1e-8:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))


class AltitudeHoldAP:
    """Altitude-hold autopilot: uses elevator to track a target altitude."""

    def __init__(self):
        self.pid = PIDController(kp=0.0005, ki=0.00005, kd=0.0001, output_min=-1.0, output_max=1.0)

    def compute(self, current_alt_m: float, target_alt_m: float, dt: float) -> float:
        error = target_alt_m - current_alt_m
        return self.pid.step(error, dt)


class SpeedHoldAP:
    """Speed-hold autopilot: uses throttle to track a target airspeed."""

    def __init__(self):
        self.pid = PIDController(kp=0.005, ki=0.001, kd=0.0, output_min=0.0, output_max=1.0)

    def compute(self, current_speed_mps: float, target_speed_mps: float, dt: float) -> float:
        error = target_speed_mps - current_speed_mps
        return self.pid.step(error, dt)


class TurnCoordinator:
    """Coordinated turn: aileron to track roll, rudder to cancel sideslip."""

    def __init__(self):
        self.roll_pid = PIDController(kp=0.05, ki=0.01, kd=0.02, output_min=-1.0, output_max=1.0)
        self.rudder_pid = PIDController(kp=0.1, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0)

    def compute(
        self,
        current_roll_rad: float,
        target_roll_rad: float,
        sideslip_deg: float,
        dt: float,
    ) -> tuple:
        roll_error = target_roll_rad - current_roll_rad
        # Wrap to [-pi, pi]
        roll_error = (roll_error + np.pi) % (2 * np.pi) - np.pi
        aileron = self.roll_pid.step(roll_error, dt)
        rudder = self.rudder_pid.step(-sideslip_deg, dt)
        return aileron, rudder
