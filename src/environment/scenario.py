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
