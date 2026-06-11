"""JSBSim flight dynamics layer — aircraft, autopilot, flight envelope, BFM actions."""

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import (
    PIDController,
    AltitudeHoldAP,
    SpeedHoldAP,
    TurnCoordinator,
    BFMAutopilot,
    BFMAutopilotConfig,
)
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.bfm_actions import (
    BFM_ACTION_MAPPING,
    NUM_BFM_ACTIONS,
    get_bfm_action,
    describe_bfm_action,
)

__all__ = [
    "Aircraft",
    "PIDController",
    "AltitudeHoldAP",
    "SpeedHoldAP",
    "TurnCoordinator",
    "BFMAutopilot",
    "BFMAutopilotConfig",
    "FlightEnvelope",
    "EnvelopeConfig",
    "BFM_ACTION_MAPPING",
    "NUM_BFM_ACTIONS",
    "get_bfm_action",
    "describe_bfm_action",
]
