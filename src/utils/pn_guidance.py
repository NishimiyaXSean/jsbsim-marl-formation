"""Proportional Navigation (PN) guidance law for air-to-air pursuit.

PN is the most widely used missile guidance law. It commands a turn rate
proportional to the line-of-sight (LOS) angular rate:

    omega_cmd = N * lambda_dot

where N is the navigation constant (typically 3–5) and lambda_dot is the
angular velocity of the line-of-sight vector.

In discrete time, the desired heading is updated as:

    desired_heading += N * lambda_dot * dt

This "lead pursuit" behaviour naturally produces an intercept course without
requiring speed advantage — unlike pure pursuit, which requires the pursuer
to be faster than the target.
"""

from __future__ import annotations

import numpy as np


def compute_pn_heading(
    pursuer_ned: np.ndarray,
    pursuer_vel: np.ndarray,
    target_ned: np.ndarray,
    target_vel: np.ndarray,
    current_heading_deg: float,
    dt: float,
    nav_constant: float = 3.0,
    max_turn_rate_dps: float = 20.0,
) -> float:
    """Compute desired heading using augmented proportional navigation.

    Pure PN only commands turn-rate proportional to LOS-rate.  This works for
    terminal homing but cannot acquire a target that is far off-boresight.
    We therefore augment it with a *bearing bias* that pulls the nose toward
    the target initially, then lets the PN lead term take over as the
    engagement closes.

    Args:
        pursuer_ned:  (3,) pursuer position [north, east, down] in meters.
        pursuer_vel:  (3,) pursuer velocity in NED frame (m/s).
        target_ned:   (3,) target position in NED frame (m).
        target_vel:   (3,) target velocity in NED frame (m/s).
        current_heading_deg: Current pursuer heading in degrees.
        dt:           Time step (seconds) since last heading update.
        nav_constant: Navigation constant (N).  3–5 typical.
        max_turn_rate_dps: Maximum turn rate in deg/s.

    Returns:
        Desired heading in degrees [0, 360), wrapped.
    """
    # ── Line-of-sight (horizontal plane) ───────────────────────────────
    r_h = target_ned[:2] - pursuer_ned[:2]
    dist_h = float(np.linalg.norm(r_h))
    if dist_h < 1.0:
        return current_heading_deg

    r_h_dir = r_h / dist_h
    bearing_deg = float(np.degrees(np.arctan2(r_h_dir[1], r_h_dir[0]))) % 360.0

    # ── LOS angular rate ───────────────────────────────────────────────
    v_rel_h = target_vel[:2] - pursuer_vel[:2]
    lambda_dot = float(np.cross(r_h, v_rel_h)) / (dist_h * dist_h)  # rad/s

    # ── PN lead angle ──────────────────────────────────────────────────
    lead_angle_deg = float(np.degrees(nav_constant * lambda_dot * dt))
    lead_angle_deg = np.clip(lead_angle_deg, -30.0, 30.0)

    # ── Desired heading = bearing + lead ───────────────────────────────
    # The bearing term pulls nose onto target initially.
    # The lead term provides PN intercept geometry.
    desired_heading = (bearing_deg + lead_angle_deg) % 360.0

    return desired_heading
