"""Termination condition checks for 1v1 air combat."""

import numpy as np


def check_collision(dist: float, macro_step: int, min_steps: int = 2) -> bool:
    """Physical collision: distance < 50m after warm-up steps."""
    return dist < 50.0 and macro_step > min_steps


def check_cpa(dist: float, prev_dist: float, cpa_radius: float, macro_step: int,
              min_steps: int = 2) -> bool:
    """Closest-point-of-approach trigger: within CPA radius and now separating."""
    return dist <= cpa_radius and (dist - prev_dist) > 0 and macro_step > min_steps


def check_ground_crash(alt_m: float, death_floor: float) -> bool:
    """Aircraft hit the ground."""
    return alt_m < death_floor


def check_out_of_bounds(alt_m: float, ceiling: float) -> bool:
    """Aircraft exceeded altitude ceiling."""
    return alt_m > ceiling


def check_timeout(sim_time: float, episode_len_sec: float) -> bool:
    """Episode time limit reached."""
    return sim_time > episode_len_sec
