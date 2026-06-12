"""Randomized initial conditions for 1v1 air combat scenarios."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.units import m_to_ft, mps_to_kts, rad_to_deg


@dataclass
class Bounds:
    """Spatial and kinematic bounds for spawn randomization."""

    d_min: float   # min initial separation (m)
    d_max: float   # max initial separation (m)
    z_min: float   # min altitude (m)
    z_max: float   # max altitude (m)
    speed_min_mps: float
    speed_max_mps: float


STAGE_BOUNDS = {
    1: Bounds(d_min=600, d_max=1000, z_min=2500, z_max=3500, speed_min_mps=150, speed_max_mps=200),
    2: Bounds(d_min=1000, d_max=1500, z_min=2500, z_max=3500, speed_min_mps=180, speed_max_mps=260),
    3: Bounds(d_min=1500, d_max=2500, z_min=3000, z_max=4000, speed_min_mps=260, speed_max_mps=340),
}


def generate_spawn(stage: int, rng: Optional[np.random.Generator] = None) -> dict:
    """Generate randomized spawn positions, headings, and speeds for a 1v1 engagement.

    Returns dict with keys:
        attacker: {lat_deg, lon_deg, alt_ft, heading_deg, speed_kts}
        evader:   {lat_deg, lon_deg, alt_ft, heading_deg, speed_kts}
        ref_lla:  (lat, lon, alt_m) NED origin for this episode.
    """
    if rng is None:
        rng = np.random.default_rng()

    bounds = STAGE_BOUNDS.get(stage, STAGE_BOUNDS[3])

    # Random quadrant for attacker
    sign_x = rng.choice([-1, 1])
    sign_y = rng.choice([-1, 1])

    attacker_x = sign_x * rng.uniform(bounds.d_min, bounds.d_max)
    attacker_y = sign_y * rng.uniform(bounds.d_min, bounds.d_max)
    attacker_z = rng.uniform(bounds.z_min, bounds.z_max)

    # Evader in opposite quadrant
    evader_x = -sign_x * rng.uniform(bounds.d_min, bounds.d_max)
    evader_y = -sign_y * rng.uniform(bounds.d_min, bounds.d_max)
    evader_z = np.clip(attacker_z + rng.uniform(-500.0, 500.0), 1000.0, 3600.0)

    # Headings: attacker points toward origin, evader gets tactical offset
    base_heading_deg = np.rad2deg(np.arctan2(-attacker_y, -attacker_x)) % 360.0
    attacker_heading = base_heading_deg

    tactical_offset = rng.choice([0.0, 90.0, -90.0, 180.0])
    noise = rng.uniform(-15.0, 15.0)
    evader_heading = (attacker_heading + tactical_offset + noise) % 360.0

    # Speeds
    attacker_speed = rng.uniform(bounds.speed_min_mps, bounds.speed_max_mps)
    evader_speed = rng.uniform(bounds.speed_min_mps, bounds.speed_max_mps)

    # Reference point = midpoint (for NED conversion)
    ref_lat = 30.0
    ref_lon = 120.0
    ref_alt_m = (attacker_z + evader_z) / 2.0

    return {
        "attacker": {
            "lat_deg": ref_lat,   # approximation: spawn at reference
            "lon_deg": ref_lon,
            "alt_ft": m_to_ft(attacker_z),
            "heading_deg": attacker_heading,
            "speed_kts": mps_to_kts(attacker_speed),
            "ned": np.array([attacker_x, attacker_y, attacker_z]),  # Z = positive UP
        },
        "evader": {
            "lat_deg": ref_lat,
            "lon_deg": ref_lon,
            "alt_ft": m_to_ft(evader_z),
            "heading_deg": evader_heading,
            "speed_kts": mps_to_kts(evader_speed),
            "ned": np.array([evader_x, evader_y, evader_z]),  # Z = positive UP
        },
        "ref_lla": (ref_lat, ref_lon, ref_alt_m),
    }


def generate_pursuit_spawn(stage: int, rng: Optional[np.random.Generator] = None) -> dict:
    """Generate spawn for single-agent pursuit: target in front of pursuer.

    Stage 1: target straight ahead, same direction, slow — just close distance.
    Stage 2: target with mild heading offset — learn to turn and track.

    Returns same dict schema as ``generate_spawn``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pursuer at origin-ish
    pursuer_z = rng.uniform(2800.0, 3200.0)
    pursuer_hdg = rng.uniform(0.0, 360.0)
    pursuer_ned = np.array([rng.uniform(-300, 300), rng.uniform(-300, 300), pursuer_z])

    if stage == 1:
        # Stage 1: target in front (±30°), same direction, slow
        bearing_offset = rng.uniform(-30.0, 30.0)
        dist = rng.uniform(1000.0, 2500.0)
        target_hdg_offset = rng.uniform(-15.0, 15.0)
        target_speed = 130.0  # m/s — easy catch
    else:
        # Stage 2: wider bearing, more heading variance, faster target
        bearing_offset = rng.uniform(-45.0, 45.0)
        dist = rng.uniform(1500.0, 3000.0)
        target_hdg_offset = rng.uniform(-30.0, 30.0)
        target_speed = 160.0

    target_bearing = (pursuer_hdg + bearing_offset) % 360.0
    target_bearing_rad = np.deg2rad(target_bearing)
    target_hdg = (pursuer_hdg + target_hdg_offset) % 360.0
    target_z = np.clip(pursuer_z + rng.uniform(-200.0, 200.0), 1500.0, 4000.0)

    target_ned = np.array([
        pursuer_ned[0] + dist * np.cos(target_bearing_rad),
        pursuer_ned[1] + dist * np.sin(target_bearing_rad),
        target_z,
    ])

    ref_alt_m = (pursuer_z + target_z) / 2.0

    return {
        "attacker": {
            "lat_deg": 30.0,
            "lon_deg": 120.0,
            "alt_ft": m_to_ft(pursuer_z),
            "heading_deg": pursuer_hdg,
            "speed_kts": mps_to_kts(180.0),  # trim-equilibrium speed
            "ned": pursuer_ned,
        },
        "evader": {
            "lat_deg": 30.0,
            "lon_deg": 120.0,
            "alt_ft": m_to_ft(target_z),
            "heading_deg": target_hdg,
            "speed_kts": mps_to_kts(180.0),  # same trim speed as attacker
            "ned": target_ned,
        },
        "ref_lla": (30.0, 120.0, ref_alt_m),
    }
