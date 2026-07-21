"""Abstract flight controller interface — pluggable PID or Neural control.

Design philosophy (Priority 2):
  - PIDFlightController: engineering safety net, hand-tuned, action-masked
  - NeuralFlightController: LAG-trained MLP+GRU, high-maneuver potential
  - SafetyInterceptor: external hard-masking layer, smooth-override interface reserved

All controllers receive a unified high-level target (heading, altitude, speed)
and return control surfaces (throttle, elevator, aileron, rudder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlSurfaces:
    """Normalized control surface deflections.

    All values in [-1, 1] except throttle which is in [0, 1].
    """
    throttle: float   # [0, 1]
    elevator: float   # [-1, 1]
    aileron: float    # [-1, 1]
    rudder: float     # [-1, 1]


@dataclass
class FlightTarget:
    """High-level flight command — what the RL policy outputs."""
    heading_deg: float    # target heading [0, 360)
    altitude_m: float     # target altitude (meters)
    speed_mps: float      # target speed (m/s)


class BaseController(ABC):
    """Abstract flight controller.

    Takes current aircraft state + high-level target, returns control surfaces.
    """

    @abstractmethod
    def predict(self, state: dict, target: FlightTarget, dt: float) -> ControlSurfaces:
        """Compute control surfaces from aircraft state and flight target.

        Args:
            state: Aircraft.state dict (position, velocity, attitude, etc.)
            target: Desired flight state (heading, altitude, speed).
            dt: Physics time step (1/60 s).

        Returns:
            Normalized control surface deflections.
        """

    @abstractmethod
    def reset(self, initial_speed_mps: float = 200.0) -> None:
        """Reset controller internal state (e.g., integrators, RNN hidden states)."""

    @property
    @abstractmethod
    def controller_type(self) -> str:
        """Identifier string: 'pid' or 'neural'."""
