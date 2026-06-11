"""Simple baseline agents for comparison."""

import numpy as np


def random_agent(rng: np.random.Generator | None = None) -> np.ndarray:
    """Random policy: output uniform [-1, 1] for each of 4 controls."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)


def pure_pursuit(los_dir: np.ndarray, own_forward: np.ndarray) -> np.ndarray:
    """Pure pursuit guidance: point nose directly at target.

    Returns controls: [throttle, elevator, aileron, rudder].
    """
    # Cross product to determine turn direction
    cross = np.cross(own_forward, los_dir)
    turn_error = cross[2]  # Z-component = horizontal error

    # Pitch error
    pitch_error = -los_dir[2]  # negative = target above

    throttle = 0.8  # constant high throttle
    elevator = np.clip(pitch_error * 2.0, -1.0, 1.0)
    aileron = np.clip(turn_error * 3.0, -1.0, 1.0)
    rudder = 0.0

    return np.array([throttle, elevator, aileron, rudder], dtype=np.float32)
