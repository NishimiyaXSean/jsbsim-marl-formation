"""JSBSim F-16 aircraft wrapper.

Each Aircraft instance owns one FGFDMExec and exposes a clean Pythonic interface.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

import jsbsim

from src.utils.units import ft_to_m, kts_to_mps, rad_to_deg, deg_to_rad


class Aircraft:
    """Wrapper around JSBSim FGFDMExec for a single F-16."""

    MODEL = "f16"
    DEFAULT_ALT_FT = 10000.0   # ~3048 m
    DEFAULT_SPEED_KTS = 400.0  # ~206 m/s

    def __init__(self, jsbsim_data_dir: Optional[str] = None):
        """
        Args:
            jsbsim_data_dir: Path to JSBSim data root (contains aircraft/, engines/, systems/).
                             If None, uses JSBSIM_DATA_DIR env var, then jsbsim's default.
        """
        self._resolve_data_dir(jsbsim_data_dir)
        # Pass None (not "None" string) when data_root is None
        root_arg = str(self._data_root) if self._data_root else None
        self.fdm = jsbsim.FGFDMExec(root_arg, None)

        # Load the F-16 model
        self.fdm.load_model(self.MODEL)

        # Set simulation time step (60 Hz default)
        self.fdm.set_dt(1.0 / 60.0)

        # Cached state vector (populated after run())
        self._state = {}

    def _resolve_data_dir(self, jsbsim_data_dir: Optional[str]) -> None:
        """Find JSBSim aircraft/engines/systems/ data directory."""
        self._data_root = None

        if jsbsim_data_dir:
            candidate = Path(jsbsim_data_dir)
            if (candidate / "aircraft").exists():
                self._data_root = candidate
                return

        env_dir = os.environ.get("JSBSIM_DATA_DIR", "")
        if env_dir:
            candidate = Path(env_dir)
            if (candidate / "aircraft").exists():
                self._data_root = candidate
                return

        # Search common locations
        _project_root = Path(__file__).resolve().parent.parent.parent
        search_paths = [
            Path.cwd() / "data" / "jsbsim",
            _project_root / "data" / "jsbsim",
            Path.home() / "jsbsim-data",
        ]
        for opt in search_paths:
            if (opt / "aircraft").exists():
                self._data_root = opt
                return

    def reset(
        self,
        lat_deg: float = 30.0,
        lon_deg: float = 120.0,
        alt_ft: float = 10000.0,
        heading_deg: float = 0.0,
        speed_kts: float = 400.0,
        trim: bool = True,
    ) -> None:
        """Reset aircraft to specified initial conditions.

        Args:
            lat_deg: WGS-84 latitude (degrees).
            lon_deg: WGS-84 longitude (degrees).
            alt_ft: Altitude above sea level (feet).
            heading_deg: True heading (degrees, 0=North).
            speed_kts: Initial calibrated airspeed (knots).
            trim: If True, auto-trim for steady-state flight.
        """
        fdm = self.fdm

        fdm["ic/lat-geod-deg"] = lat_deg
        fdm["ic/long-gc-deg"] = lon_deg
        fdm["ic/h-sl-ft"] = alt_ft
        fdm["ic/psi-true-deg"] = heading_deg
        fdm["ic/vc-kts"] = speed_kts

        if trim:
            fdm["simulation/do_simple_trim"] = 1

        fdm.run_ic()

        self._state = {}

    def set_controls(self, throttle: float, elevator: float, aileron: float, rudder: float) -> None:
        """Apply normalized control surface commands.

        Args:
            throttle:   [0, 1] Throttle position.
            elevator:  [-1, 1] Elevator deflection (positive = pitch up).
            aileron:   [-1, 1] Aileron deflection (positive = roll right).
            rudder:    [-1, 1] Rudder deflection (positive = yaw right).
        """
        self.fdm["fcs/throttle-cmd-norm"] = np.clip(throttle, 0.0, 1.0)
        self.fdm["fcs/elevator-cmd-norm"] = np.clip(elevator, -1.0, 1.0)
        self.fdm["fcs/aileron-cmd-norm"] = np.clip(aileron, -1.0, 1.0)
        self.fdm["fcs/rudder-cmd-norm"] = np.clip(rudder, -1.0, 1.0)

    def run(self) -> None:
        """Advance the simulation by one time step (dt)."""
        self.fdm.run()
        self._state = {}  # Invalidate cache

    def _read_state(self) -> dict:
        """Extract state from JSBSim property tree into a flat dict."""
        fdm = self.fdm
        return {
            "lat_deg": float(fdm["position/lat-geod-deg"]),
            "lon_deg": float(fdm["position/long-gc-deg"]),
            "alt_ft": float(fdm["position/h-sl-ft"]),
            "alt_m": ft_to_m(float(fdm["position/h-sl-ft"])),
            "roll_deg": rad_to_deg(float(fdm["attitude/roll-rad"])),
            "pitch_deg": rad_to_deg(float(fdm["attitude/pitch-rad"])),
            "yaw_deg": rad_to_deg(float(fdm["attitude/heading-true-rad"])),
            "airspeed_kts": float(fdm["velocities/vc-kts"]),
            "airspeed_mps": kts_to_mps(float(fdm["velocities/vc-kts"])),
            "mach": float(fdm["velocities/mach"]),
            "alpha_deg": float(fdm["aero/alpha-deg"]),
            "beta_deg": float(fdm["aero/beta-deg"]),
            # Body-frame accelerations (G)
            "n_x_g": float(fdm["accelerations/n-pilot-x-norm"]),
            "n_y_g": float(fdm["accelerations/n-pilot-y-norm"]),
            "n_z_g": float(fdm["accelerations/n-pilot-z-norm"]),
            # Body-frame velocities (ft/s)
            "u_fps": float(fdm["velocities/u-fps"]),
            "v_fps": float(fdm["velocities/v-fps"]),
            "w_fps": float(fdm["velocities/w-fps"]),
            # Body-frame angular velocities (rad/s)
            "p_rps": float(fdm["velocities/p-rad_sec"]),
            "q_rps": float(fdm["velocities/q-rad_sec"]),
            "r_rps": float(fdm["velocities/r-rad_sec"]),
            # Engine
            "thrust_lbs": float(fdm["propulsion/engine[0]/thrust-lbs"]),
        }

    @property
    def state(self) -> dict:
        """Cached access to current aircraft state."""
        if not self._state:
            self._state = self._read_state()
        return self._state

    @property
    def position_ned(self) -> np.ndarray:
        """Placeholder NED position — set externally by orchestrator after lla→ned conversion.

        Returns (3,) array [north_m, east_m, down_m] relative to scenario origin.
        """
        if "ned" in self._state:
            return self._state["ned"]
        return np.zeros(3)

    @position_ned.setter
    def position_ned(self, value: np.ndarray) -> None:
        self._state["ned"] = np.asarray(value, dtype=np.float64)

    @property
    def velocity_ned(self) -> np.ndarray:
        """Approximate velocity in NED frame [vn, ve, vd] (m/s)."""
        s = self.state
        yaw_rad = deg_to_rad(s["yaw_deg"])
        spd = s["airspeed_mps"]
        return np.array([
            spd * np.cos(yaw_rad),
            spd * np.sin(yaw_rad),
            0.0,
        ])

    @property
    def rpy_rad(self) -> np.ndarray:
        """Roll-pitch-yaw in radians."""
        s = self.state
        return np.array([
            deg_to_rad(s["roll_deg"]),
            deg_to_rad(s["pitch_deg"]),
            deg_to_rad(s["yaw_deg"]),
        ])

    def get_sim_time(self) -> float:
        """Return elapsed simulation time in seconds."""
        return float(self.fdm.get_sim_time())
