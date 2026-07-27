"""MissileSimulator — 3-DOF point-mass missile with proportional navigation guidance.

All dynamics are integrated in a local NED frame anchored at the launch point.
No JSBSim dependency — pure Python physics.

Reference: LAG envs/JSBSim/core/simulatior.py → MissileSimulator
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  AIM-9L physical parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MissileParams:
    """Physical constants for AIM-9L Sidewinder."""
    g: float = 9.81           # gravitational acceleration (m/s²)
    t_max: float = 60.0       # max flight time (s)
    t_thrust: float = 3.0     # engine burn time (s)
    Isp: float = 120.0        # specific impulse (s) — reduced from 240 in original
    length: float = 2.87      # missile length (m)
    diameter: float = 0.127   # missile diameter (m)
    cD: float = 0.4           # drag coefficient
    m0: float = 84.0          # initial mass (kg)
    dm: float = 6.0           # mass flow rate (kg/s)
    K: float = 3.0            # PN guidance constant
    nyz_max: float = 30.0     # max lateral overload (g)
    Rc: float = 300.0         # lethal radius (m)
    v_min: float = 150.0      # minimum speed before self-destruct (m/s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Status enum
# ═══════════════════════════════════════════════════════════════════════════════

class MissileStatus:
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2


# ═══════════════════════════════════════════════════════════════════════════════
#  MissileSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class MissileSimulator:
    """3-DOF point-mass missile with proportional navigation guidance.

    Physics are integrated in a local NED frame anchored at the launch point:
      - _position: [north, east, down] relative to launch point (m)
      - _velocity: [vn, ve, vd] (m/s)
      - _posture: [roll(always 0), pitch(θ), yaw(φ)] (rad)

    The launch point is the parent aircraft's NED position at the moment of
    launch.  Target position is queried from the linked target aircraft's
    position_ned at each step — no WGS84⇔NED conversion needed because
    our Aircraft already track NED externally.
    """

    _params: MissileParams = MissileParams()  # class-level default, can override per instance

    def __init__(
        self,
        uid: str,
        color: str = "Red",
        model: str = "AIM-9L",
        dt: float = 1.0 / 60.0,
        params: MissileParams | None = None,
    ):
        self.uid = uid
        self.color = color
        self.model = model
        self.dt = dt
        if params is not None:
            self._params = params

        self._status = MissileStatus.INACTIVE
        self._t: float = 0.0
        self._m: float = self._params.m0

        # NED state (relative to launch origin)
        self._position: np.ndarray = np.zeros(3)
        self._velocity: np.ndarray = np.zeros(3)
        self._posture: np.ndarray = np.zeros(3)  # [roll, pitch, yaw]

        # Guidance state
        self._dtheta: float = 0.0
        self._dphi: float = 0.0
        self._distance_pre: float = np.inf

        # Launch origin (NED of parent at launch time)
        self._launch_origin: np.ndarray = np.zeros(3)

        # WGS84 reference for ACMI rendering (set at launch from parent aircraft state)
        self._launch_lat: float = 30.0
        self._launch_lon: float = 120.0

        # Links
        self.parent_aircraft: Optional[object] = None   # _Pursuer or _Target
        self.target_aircraft: Optional[object] = None   # _Pursuer or _Target

        # ACMI render tracking
        self.render_explosion: bool = False
        self._first_log: bool = True   # first log needs Type= declaration

        # Distance increment queue — self-destruct if distance increases for 5s
        self._dist_increment: deque = deque(maxlen=int(5.0 / self.dt))

        # Countdown after destruction (1s before removal)
        self._left_t: int = int(1.0 / self.dt)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._status == MissileStatus.LAUNCHED

    @property
    def is_success(self) -> bool:
        return self._status == MissileStatus.HIT

    @property
    def is_done(self) -> bool:
        return self._status in (MissileStatus.HIT, MissileStatus.MISS)

    @property
    def is_inactive(self) -> bool:
        return self._status == MissileStatus.INACTIVE

    @property
    def Isp(self) -> float:
        """Specific impulse — zero after engine burnout."""
        return self._params.Isp if self._t < self._params.t_thrust else 0.0

    @property
    def K(self) -> float:
        """PN guidance constant — linearly decays to prevent terminal oscillation."""
        return max(self._params.K * (self._params.t_max - self._t) / self._params.t_max, 0.0)

    @property
    def S(self) -> float:
        """Cross-sectional area including angle-of-attack contribution (m²)."""
        S0 = np.pi * (self._params.diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * \
              self._params.diameter * self._params.length
        return S0

    @property
    def rho(self) -> float:
        """Air density at current altitude (kg/m³)."""
        # _launch_origin[2] is altitude (our project stores altitude in position_ned[2])
        # _position[2] is missile's altitude deviation from launch
        h = self._launch_origin[2] + self._position[2]
        return 1.225 * np.exp(-h / 9300.0)

    @property
    def target_distance(self) -> float:
        if self.target_aircraft is None:
            return np.inf
        target_ned = self.target_aircraft.aircraft.position_ned
        return float(np.linalg.norm(self.get_absolute_position() - target_ned))

    # ── Position helpers ─────────────────────────────────────────────────────

    def get_position(self) -> np.ndarray:
        """Position relative to launch origin (NED)."""
        return self._position.copy()

    def get_absolute_position(self) -> np.ndarray:
        """Absolute NED position (launch origin + relative)."""
        return self._launch_origin + self._position

    def get_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def get_rpy(self) -> np.ndarray:
        return self._posture.copy()

    def get_absolute_geodetic(self) -> tuple:
        """Approximate WGS84 for ACMI rendering.

        Uses parent's JSBSim lat/lon as the anchor point and adds ONLY
        the missile's relative displacement from launch.  This avoids
        double-counting the parent's pre-launch displacement.

        NOTE: _launch_origin[2] = parent altitude at launch (from state["alt_m"])
              _position[2] = missile's altitude CHANGE since launch
        """
        # Missile displacement relative to launch point (metres)
        rel_north = self._position[0]
        rel_east = self._position[1]

        # WGS84 metres per degree at launch latitude
        lat_rad = np.radians(self._launch_lat)
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

        lat = self._launch_lat + rel_north / m_per_deg_lat
        lon = self._launch_lon + rel_east / m_per_deg_lon
        alt = self._launch_origin[2] + self._position[2]

        return lat, lon, alt

    # ── Lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        parent,      # _Pursuer or _Target
        target,      # _Pursuer or _Target
        uid: str,
        dt: float = 1.0 / 60.0,
        missile_model: str = "AIM-9L",
        **kwargs,
    ) -> "MissileSimulator":
        """Factory: create a missile, inherit parent state, and link to target."""
        parent_ned = parent.aircraft.position_ned
        parent_rpy = parent.aircraft.rpy_rad
        parent_vel = parent.aircraft.velocity_ned
        # Capture WGS84 for ACMI rendering
        parent_state = parent.aircraft.state
        parent_lat = float(parent_state["lat_deg"])
        parent_lon = float(parent_state["lon_deg"])

        missile = cls(uid=uid, color="Red", model=missile_model, dt=dt)
        missile.launch(parent, parent_ned, parent_rpy, parent_vel, parent_lat, parent_lon)
        missile.target(target)
        # Store parent ACMI ID for Tacview Parent= attribute
        missile._parent_uid = kwargs.get("parent_uid", "101")
        return missile

    def launch(
        self,
        parent,               # _Pursuer or _Target
        parent_ned: np.ndarray,
        parent_rpy: np.ndarray,
        parent_vel: np.ndarray,
        parent_lat: float = 30.0,
        parent_lon: float = 120.0,
    ) -> None:
        """Activate the missile at the parent's current NED position."""
        self.parent_aircraft = parent

        # Set launch origin to parent's current absolute NED
        self._launch_origin = parent_ned.copy()

        # Store WGS84 reference for ACMI rendering
        self._launch_lat = parent_lat
        self._launch_lon = parent_lon

        # Missile starts at launch origin (relative position = 0)
        self._position = np.zeros(3)

        # Inherit velocity from parent
        self._velocity = parent_vel.copy()

        # Inherit posture (but zero roll — missile is symmetric)
        self._posture = parent_rpy.copy()
        self._posture[0] = 0.0

        # Reset state
        self._t = 0.0
        self._m = self._params.m0
        self._dtheta = 0.0
        self._dphi = 0.0
        self._distance_pre = np.inf
        self._dist_increment.clear()
        self._left_t = int(1.0 / self.dt)
        self.render_explosion = False
        self._first_log = True

        self._status = MissileStatus.LAUNCHED

    def target(self, target) -> None:
        """Link to target aircraft."""
        self.target_aircraft = target

    def run(self) -> None:
        """Advance the missile by one dt step (must be called at 60 Hz)."""
        if not self.is_alive:
            if self.is_done:
                self._left_t -= 1
            return

        self._t += self.dt

        # ── Guidance ──────────────────────────────────────────────────────
        action, distance = self._guidance()

        # ── Distance tracking ────────────────────────────────────────────
        self._dist_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # ── Hit / Miss detection ──────────────────────────────────────────
        target_alive = getattr(self.target_aircraft, 'is_alive', True) if self.target_aircraft is not None else False
        if distance < self._params.Rc and target_alive:
            self._status = MissileStatus.HIT
        elif (
            self._t > self._params.t_max
            or np.linalg.norm(self._velocity) < self._params.v_min
            or sum(self._dist_increment) >= self._dist_increment.maxlen
            or not target_alive
        ):
            self._status = MissileStatus.MISS
        else:
            # ── State transition ──────────────────────────────────────────
            self._state_trans(action)

    # ── Guidance law (proportional navigation) ───────────────────────────────

    def _guidance(self) -> tuple:
        """Compute PN lateral overload commands.

        Returns:
            (ny, nz): lateral overload in yaw/pitch channels, and current distance.
        """
        if self.target_aircraft is None:
            return (0.0, 0.0), np.inf

        x_m, y_m, z_m = self.get_absolute_position()
        dx_m, dy_m, dz_m = self._velocity
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(np.clip(dz_m / max(v_m, 1e-6), -1.0, 1.0))

        target_ned = self.target_aircraft.aircraft.position_ned
        x_t, y_t, z_t = target_ned

        # Target velocity from the linked aircraft
        target_vel = self.target_aircraft.aircraft.velocity_ned
        dx_t, dy_t, dz_t = target_vel

        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])

        if Rxy < 1e-6 or Rxyz < 1e-6:
            return (0.0, 0.0), Rxyz

        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / (Rxy ** 2)
        deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy)

        ny = self.K * v_m / self._params.g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._params.g * deps + np.cos(theta_m)

        return np.clip([ny, nz], -self._params.nyz_max, self._params.nyz_max), Rxyz

    # ── State transition ─────────────────────────────────────────────────────

    def _state_trans(self, action: np.ndarray) -> None:
        """Integrate one time step: update position, velocity, posture, mass."""
        ny, nz = action
        p = self._params
        v = np.linalg.norm(self._velocity)
        theta, phi = self._posture[1], self._posture[2]

        # Forces
        T = p.g * self.Isp * p.dm   # thrust (N)
        D = 0.5 * p.cD * self.S * self.rho * v ** 2  # drag (N)

        # Axial load factor
        nx = (T - D) / (self._m * p.g)

        # Kinematic derivatives
        dv = p.g * (nx - np.sin(theta))
        self._dphi = p.g / max(v, 1e-6) * (ny / max(np.cos(theta), 1e-6))
        self._dtheta = p.g / max(v, 1e-6) * (nz - np.cos(theta))

        # Integrate velocity
        v_new = v + self.dt * dv
        phi_new = phi + self.dt * self._dphi
        theta_new = theta + self.dt * self._dtheta

        # Update velocity vector
        self._velocity = np.array([
            v_new * np.cos(theta_new) * np.cos(phi_new),
            v_new * np.cos(theta_new) * np.sin(phi_new),
            v_new * np.sin(theta_new),
        ])

        # Update posture
        self._posture = np.array([0.0, theta_new, phi_new])

        # Update position (relative to launch origin)
        self._position += self.dt * self._velocity

        # Mass depletion (engine burn phase only)
        if self._t < p.t_thrust:
            self._m -= self.dt * p.dm

        # Floor mass
        self._m = max(self._m, 0.1)

    # ── ACMI log ─────────────────────────────────────────────────────────────

    def _get_tacview_position(self) -> tuple:
        """Return (lon, lat, alt) in WGS84 for Tacview T= string."""
        lat, lon, alt = self.get_absolute_geodetic()
        return lon, lat, alt  # Tacview order: lon, lat, alt

    def get_parent_uid(self) -> str:
        """Return the parent aircraft's ACMI object ID."""
        return getattr(self, '_parent_uid', '101')

    def log(self) -> str | None:
        """Tacview-compatible log line.

        Uses hex object IDs (301+ for missiles, 401+ for explosions),
        Type=Air+FixedWing for rendering, and Parent= attribute to link
        the missile to its launching aircraft.
        """
        if self.is_alive:
            lon, lat, alt = self._get_tacview_position()
            roll, pitch, yaw = self._posture * 180.0 / np.pi
            parent_id = self.get_parent_uid()
            if self._first_log:
                self._first_log = False
                return (
                    f"{self.uid},T={lon:.6f}|{lat:.6f}|{alt:.1f}|"
                    f"{roll:.1f}|{pitch:.1f}|{yaw:.1f},"
                    f"Name={self.model},Type=Air+FixedWing,Color=Red,"
                    f"Parent={parent_id}"
                )
            return (
                f"{self.uid},T={lon:.6f}|{lat:.6f}|{alt:.1f}|"
                f"{roll:.1f}|{pitch:.1f}|{yaw:.1f},"
                f"Name={self.model},Color=Red"
            )

        if self.is_done and not self.render_explosion:
            self.render_explosion = True
            lon, lat, alt = self._get_tacview_position()

            # Generate explosion ID from missile ID (missile 301 → explosion 401)
            try:
                explosion_id = f"{int(self.uid, 16) + 0x100:X}"
            except (ValueError, TypeError):
                explosion_id = f"{int(self.uid) + 100}"

            msg = f"-{self.uid}\n"

            if self._status == MissileStatus.HIT:
                # Yellow explosion at impact point
                msg += (
                    f"{explosion_id},T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
                    f"Type=Misc+Explosion,Color=Yellow,Radius={self._params.Rc}"
                )
            else:
                # MISS: final Grey marker + Grey explosion
                msg += (
                    f"{self.uid},T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
                    f"Name={self.model},Color=Grey\n"
                )
                msg += (
                    f"{explosion_id},T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
                    f"Type=Misc+Explosion,Color=Grey,Radius={self._params.Rc}"
                )
            return msg

        return None

    def should_remove(self) -> bool:
        """Signal to the environment that this missile can be garbage-collected."""
        return self.is_done and self._left_t <= 0

    def close(self) -> None:
        self.target_aircraft = None
        self.parent_aircraft = None
