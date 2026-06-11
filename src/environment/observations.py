"""Observation computation for 1v1 air combat environment.

Each agent receives a 19-dim local observation (body-frame relative state + tactical geometry).
The critic receives a 26-dim global state (both aircraft absolute states).
"""

import numpy as np

from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles
from src.utils.units import deg_to_rad


# Normalization constants
MAX_DIST = 8000.0
MAX_HEIGHT = 5000.0
MAX_VEL = 400.0
MAX_ANG_VEL = np.pi


def compute_obs(own_pos: np.ndarray, own_rpy: np.ndarray, own_vel: np.ndarray,
                own_ang_vel: np.ndarray, enemy_pos: np.ndarray, enemy_vel: np.ndarray,
                enemy_rpy: np.ndarray) -> np.ndarray:
    """Compute 19-dim first-person local observation.

    Features (all normalized to [-1, 1]):
        0-2:   enemy relative position in body frame (3)
        3-5:   own velocity in body frame (3)
        6-8:   own attitude rpy (3)
        9-11:  own angular velocity in body frame (3)
        12:    own height (1)
        13-15: enemy velocity in body frame (3)
        16-18: tactical geometry cos(ATA), cos(AA), cos(HCA) (3)
    """
    # --- Tactical geometry ---
    own_forward = compute_forward_vector(own_rpy)
    enemy_forward = compute_forward_vector(enemy_rpy)
    los_vec, los_dir, dist = compute_los(own_pos, enemy_pos)
    geo = compute_tactical_angles(own_forward, enemy_forward, los_dir)

    # --- Body-frame transforms (simplified: use heading-only rotation) ---
    yaw = own_rpy[2]
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)

    def world_to_body(vec):
        x = vec[0] * cos_y - vec[1] * sin_y
        y = vec[0] * sin_y + vec[1] * cos_y
        return np.array([x, y, vec[2]])

    local_rel_pos = world_to_body(enemy_pos - own_pos)
    local_vel = world_to_body(own_vel)
    local_ang_vel = own_ang_vel.copy()  # simplification
    local_enemy_vel = world_to_body(enemy_vel)

    # --- Normalize and assemble ---
    obs = np.concatenate([
        local_rel_pos / MAX_DIST,
        local_vel / MAX_VEL,
        own_rpy / np.pi,
        local_ang_vel / MAX_ANG_VEL,
        [own_pos[2] / MAX_HEIGHT],
        local_enemy_vel / MAX_VEL,
        [geo["cos_ata"], geo["cos_aa"], geo["cos_hca"]],
    ]).astype(np.float32)

    return np.clip(obs, -1.0, 1.0)


def compute_global_state(own_pos: np.ndarray, own_quat: np.ndarray,
                         own_vel: np.ndarray, own_ang_vel: np.ndarray,
                         enemy_pos: np.ndarray, enemy_quat: np.ndarray,
                         enemy_vel: np.ndarray, enemy_ang_vel: np.ndarray) -> np.ndarray:
    """Compute 26-dim global state (13 dims per aircraft).

    Per aircraft: position(3) + quaternion(4) + velocity(3) + angular_velocity(3).
    Dead aircraft are zero-padded.
    """
    own_state = np.concatenate([
        own_pos / MAX_DIST,
        own_quat,                         # quaternion is already in [-1,1]
        own_vel / MAX_VEL,
        own_ang_vel / MAX_ANG_VEL,
    ])

    enemy_state = np.concatenate([
        enemy_pos / MAX_DIST,
        enemy_quat,
        enemy_vel / MAX_VEL,
        enemy_ang_vel / MAX_ANG_VEL,
    ])

    return np.clip(np.concatenate([own_state, enemy_state]), -1.0, 1.0).astype(np.float32)
