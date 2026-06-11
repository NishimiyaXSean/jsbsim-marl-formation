"""Coordinate transformations: NED <-> WGS-84, quaternion conversions."""

import numpy as np

# WGS-84 constants
WGS84_A = 6378137.0  # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def ned_to_lla(ned: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """Convert local NED coordinates to WGS-84 lat/lon/alt.

    Flat-earth approximation valid within ~50km of reference point.

    Args:
        ned: (3,) array of [north, east, down] in meters from reference.
        ref_lla: (3,) array of [lat_deg, lon_deg, alt_m] of NED origin.
    Returns:
        (3,) array of [lat_deg, lon_deg, alt_m].
    """
    ref_lat_rad = np.deg2rad(ref_lla[0])
    # Meridional radius of curvature
    rn = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(ref_lat_rad) ** 2)
    rm = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * np.sin(ref_lat_rad) ** 2) ** 1.5

    dlat_rad = ned[0] / (rm + ref_lla[2])
    dlon_rad = ned[1] / ((rn + ref_lla[2]) * np.cos(ref_lat_rad))

    lat = ref_lla[0] + np.rad2deg(dlat_rad)
    lon = ref_lla[1] + np.rad2deg(dlon_rad)
    alt = ref_lla[2] - ned[2]  # down -> altitude

    return np.array([lat, lon, alt])
