"""Tactical geometry calculations: ATA, AA, HCA, LOS, closing speed."""

import numpy as np


def compute_forward_vector(rpy: np.ndarray) -> np.ndarray:
    """Compute unit forward vector from roll-pitch-yaw Euler angles (radians).

    Args:
        rpy: (3,) array of [roll, pitch, yaw] in radians.
    Returns:
        (3,) unit vector in the direction the aircraft nose points.
    """
    pitch, yaw = rpy[1], rpy[2]
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def compute_los(own_pos: np.ndarray, target_pos: np.ndarray) -> tuple:
    """Compute line-of-sight vector and distance.

    Args:
        own_pos: (3,) self position in NED [x, y, z].
        target_pos: (3,) target position in NED [x, y, z].
    Returns:
        (los_vec, los_dir, dist): los vector, normalized direction, distance.
    """
    los_vec = target_pos - own_pos
    dist = np.linalg.norm(los_vec)
    los_dir = los_vec / (dist + 1e-6)
    return los_vec, los_dir, dist


def compute_tactical_angles(
    own_forward: np.ndarray,
    target_forward: np.ndarray,
    los_dir: np.ndarray,
) -> dict:
    """Compute ATA, AA, HCA tactical angles.

    Args:
        own_forward: (3,) self nose direction.
        target_forward: (3,) target nose direction.
        los_dir: (3,) normalized line-of-sight from self to target.
    Returns:
        dict with keys: cos_ata, cos_aa, cos_hca (all in [-1, 1]).
    """
    cos_ata = np.clip(np.dot(own_forward, los_dir), -1.0, 1.0)
    cos_aa = np.clip(np.dot(target_forward, los_dir), -1.0, 1.0)
    cos_hca = np.clip(np.dot(own_forward, target_forward), -1.0, 1.0)
    return {"cos_ata": cos_ata, "cos_aa": cos_aa, "cos_hca": cos_hca}


def compute_closing_speed(own_vel: np.ndarray, los_dir: np.ndarray) -> float:
    """Project own velocity onto line-of-sight direction.

    Positive = closing, negative = separating.
    """
    return float(np.dot(own_vel, los_dir))
