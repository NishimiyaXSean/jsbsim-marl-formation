"""Reward functions for 1v1 air combat.

Each reward component is a standalone function with signature:
    reward_xxx(state_dict, dt) -> float

This makes ablation studies trivial — just remove a component from the sum.
"""

from dataclasses import dataclass, field
from typing import Dict, Callable, Optional

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward component weights."""

    progress_weight: float = 0.3
    progress_penalty_weight: float = 0.08

    ata_weight: float = 6.0
    ata_lock_bonus: float = 3.0
    ata_rear_penalty: float = 8.0
    ata_lock_threshold: float = 0.866  # cos(30°)

    aa_weight: float = 3.0
    aa_threshold: float = 0.5

    hca_weight: float = 1.5
    hca_threshold: float = 0.0

    collision_weight: float = 10.0
    collision_penalty_weight: float = 6.0
    collision_near_weight_extra: float = 5.0
    collision_near_dist: float = 1000.0

    closing_speed_weight: float = 0.15
    closing_speed_terminal_weight: float = 0.25
    terminal_radius: float = 400.0

    time_base_penalty: float = 0.15
    time_max_penalty: float = 1.0
    episode_len_sec: float = 240.0

    z_advantage_weight: float = 0.003
    z_disadvantage_weight: float = 0.03
    z_advantage_min: float = 50.0
    z_disadvantage_threshold: float = -100.0

    ground_warning_alt: float = 800.0      # warn well above death floor
    ground_warning_base: float = 5.0
    ground_warning_vz_factor: float = 0.5
    altitude_bonus: float = 0.0             # disabled — too dominant over tracking signal

    stall_speed: float = 150.0
    stall_penalty_weight: float = 0.5
    stall_vz_threshold: float = 10.0

    sideslip_factor: float = 2.0

    kill_reward: float = 500.0
    cpa_radius: float = 300.0
    cpa_base: float = 200.0
    cpa_extra_max: float = 300.0

    survival_reward: float = 0.5
    escape_weight: float = 2.0
    spoofing_weight: float = 5.0
    spoofing_threshold: float = 0.5

    timeout_attacker_penalty: float = 100.0   # light penalty — motivates engagement
    timeout_evader_bonus: float = 0.0


# ─── Attacker reward components ────────────────────────────────────────────


def reward_progress(micro_delta_dist: float, dt: float, cfg: RewardConfig) -> float:
    """Reward for closing distance, penalty for letting it grow."""
    if micro_delta_dist < 0:
        return abs(micro_delta_dist) * cfg.progress_weight
    else:
        return -micro_delta_dist * cfg.progress_penalty_weight


def reward_time_pressure(step_counter: int, ctrl_freq: float, dt: float, cfg: RewardConfig) -> float:
    """Increasing time penalty to encourage fast engagement."""
    time_ratio = (step_counter / ctrl_freq) / cfg.episode_len_sec
    return -(cfg.time_base_penalty + time_ratio * (cfg.time_max_penalty - cfg.time_base_penalty)) * dt


def reward_z_advantage(dz: float, cos_ata: float, micro_delta_dist: float, dt: float, cfg: RewardConfig) -> float:
    """Altitude advantage: reward being above target when engaging, penalize being below."""
    r = 0.0
    if dz > cfg.z_advantage_min and cos_ata > 0.5 and micro_delta_dist < 0:
        r += np.clip(dz, 0.0, 1000.0) * cfg.z_advantage_weight * dt
    elif dz < cfg.z_disadvantage_threshold:
        penalty_scale = np.clip(abs(dz) / 300.0, 1.0, 4.0)
        r -= abs(dz) * cfg.z_disadvantage_weight * penalty_scale * dt
    return r


def reward_energy_loss(vel_norm: float, vz: float, dt: float, cfg: RewardConfig) -> float:
    """Penalize dangerously low speed (kinetic energy), unless climbing."""
    if vel_norm < cfg.stall_speed and vz < cfg.stall_vz_threshold:
        return -(cfg.stall_speed - vel_norm) * cfg.stall_penalty_weight * dt
    return 0.0


def reward_ground_warning(alt: float, vz: float, dt: float, cfg: RewardConfig) -> float:
    """Exponential penalty near ground, amplified by downward velocity."""
    r = 0.0
    if alt < cfg.ground_warning_alt:
        depth_ratio = (cfg.ground_warning_alt - alt) / cfg.ground_warning_alt
        r -= (depth_ratio ** 2) * cfg.ground_warning_base * dt
        if vz < -1.0:
            r -= abs(vz) * cfg.ground_warning_vz_factor * dt
    return r


def reward_tracking(
    cos_ata: float, cos_aa: float, cos_hca: float,
    cos_collision: float, sideslip_vel: float,
    dist: float, dz: float, micro_delta_dist: float,
    warning_radius: float, max_speed: float,
    dt: float, cfg: RewardConfig,
) -> float:
    """Combined tracking reward: ATA + AA + HCA + collision angle + sideslip penalty."""
    r = 0.0

    # Collision course reward
    if cos_collision > 0.0:
        dynamic_weight = cfg.collision_weight + (cfg.collision_near_dist / (dist + 100.0)) * cfg.collision_near_weight_extra
        r += cos_collision * dynamic_weight * dt
    else:
        r += cos_collision * cfg.collision_penalty_weight * dt

    # ATA reward
    base_ata_reward = 0.0
    if cos_ata > 0.0:
        base_ata_reward = cos_ata * cfg.ata_weight * dt
        if cos_ata > cfg.ata_lock_threshold:
            base_ata_reward += cfg.ata_lock_bonus * dt

        if dist < warning_radius:
            z_error = max(0.0, abs(dz) - 100.0)
            z_penalty_factor = np.clip(z_error / 500.0, 0.0, 1.0)
            if micro_delta_dist < 0:
                r += base_ata_reward * (1.0 - z_penalty_factor)
        else:
            if micro_delta_dist < 0:
                r += base_ata_reward
            else:
                r += base_ata_reward * 0.1
    else:
        r += cos_ata * cfg.ata_rear_penalty * dt

    # AA (aspect angle) reward
    if cos_aa > cfg.aa_threshold:
        r += cos_aa * cfg.aa_weight * dt

    # HCA reward
    if cos_hca > cfg.hca_threshold:
        r += cos_hca * cfg.hca_weight * dt

    # Sideslip penalty
    r -= (abs(sideslip_vel) / max_speed) * cfg.sideslip_factor * dt

    return r


def reward_closing_speed(closing_speed: float, dz: float, dist: float,
                         terminal_radius: float, dt: float, cfg: RewardConfig) -> float:
    """Reward velocity vector pointing toward target (lead pursuit)."""
    r = 0.0
    if closing_speed > 0:
        capped = np.clip(closing_speed, 0.0, 300.0)
        if dz < -50.0:
            ramping_mult = np.clip((250.0 + dz) / 200.0, 0.0, 1.0)
            r += capped * cfg.closing_speed_weight * dt * ramping_mult
        else:
            r += capped * cfg.closing_speed_weight * dt
        if dist <= terminal_radius:
            r += closing_speed * cfg.closing_speed_terminal_weight * dt
    return r


# ─── Evader reward components ──────────────────────────────────────────────


def reward_survival(dt: float, cfg: RewardConfig) -> float:
    """Small per-second reward for staying alive."""
    return cfg.survival_reward * dt


def reward_escape(micro_delta_dist: float, cfg: RewardConfig) -> float:
    """Reward for increasing distance when within warning radius."""
    if micro_delta_dist > 0:
        return micro_delta_dist * cfg.escape_weight
    return 0.0


def reward_spoofing(cos_ata: float, dt: float, cfg: RewardConfig) -> float:
    """Penalty for being locked by attacker (ATA in forward hemisphere)."""
    if cos_ata > cfg.spoofing_threshold:
        return -(cos_ata ** 2) * cfg.spoofing_weight * dt
    return 0.0
