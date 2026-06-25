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

    # ── Air Combat Geometry shaping (2026-06-25) ──────────────────────
    # ATA Gaussian kernel: smooth reward centred on boresight
    ata_gaussian_sigma: float = 15.0          # degrees — narrower = more precise tracking
    ata_gaussian_weight: float = 3.0          # scale factor for Gaussian reward

    # Closure rate gating: prevent head-on suicide intercepts
    closure_gate_dist: float = 1000.0         # m — below this, penalise closure
    closure_gate_vc_threshold: float = 50.0   # m/s — above this closing speed triggers penalty
    closure_gate_weight: float = 1.5          # penalty scale

    # Energy conservation: reward specific energy maintenance
    energy_weight: float = 0.001              # small per-step reward for energy management
    energy_ref_alt: float = 3000.0            # reference altitude (m) for energy normalisation
    energy_ref_spd: float = 200.0             # reference speed (m/s) for energy normalisation

    # Action smoothness / jerk penalty (2026-06-25)
    action_rate_weight: float = 0.5           # penalty for large action-to-action changes
    action_rate_discrete_penalty: float = 2.0 # penalty for switching discrete actions too fast


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


# ─── Air Combat Geometry reward components (2026-06-25) ─────────────────────


def reward_ata_gaussian(cos_ata: float, dt: float, cfg: RewardConfig) -> float:
    """Gaussian-kernel reward for keeping the target in the boresight.

    R_ATA = w * exp(-ATA² / (2*σ²)) * dt

    where ATA = acos(cos_ata) in degrees and σ controls the width.
    Narrower σ demands more precise nose-pointing; wider σ is more
    forgiving and suitable for early training.

    This replaces the linear cos_ata reward with a smooth kernel that
    peaks at ATA=0° (perfect boresight alignment) and falls off
    gracefully, avoiding the sharp gradient of linear rewards.
    """
    ata_deg = np.degrees(np.arccos(np.clip(cos_ata, -1.0, 1.0)))
    sigma = cfg.ata_gaussian_sigma
    return cfg.ata_gaussian_weight * np.exp(-0.5 * (ata_deg / sigma) ** 2) * dt


def reward_closure_rate_gating(
    dist_m: float, closing_speed_mps: float, dt: float, cfg: RewardConfig,
) -> float:
    """Penalise high closure rate at close range to prevent suicide intercepts.

    When the pursuer is within *closure_gate_dist* and closing faster than
    *closure_gate_vc_threshold*, a penalty proportional to the excess closure
    rate is applied.  This teaches the agent to slow down and establish a
    tail-chase (low Vc at close range) rather than simply charging head-on
    for a high-speed intercept.

    The penalty is gated: zero when far away or when already slow.
    """
    if dist_m < cfg.closure_gate_dist and closing_speed_mps > cfg.closure_gate_vc_threshold:
        excess_vc = closing_speed_mps - cfg.closure_gate_vc_threshold
        return -cfg.closure_gate_weight * (excess_vc / 100.0) * dt
    return 0.0


def reward_energy_conservation(
    alt_m: float, airspeed_mps: float, dt: float, cfg: RewardConfig,
) -> float:
    """Small positive reward for maintaining specific energy (Es).

    Es = h + V²/(2g)  (energy height)

    The reward is proportional to how close the current Es is to a
    reference value (3000 m + (200 m/s)² / (2*9.81) ≈ 5040 m energy height).
    This gently encourages the agent to preserve altitude and airspeed,
    enabling sustained manoeuvring rather than bleeding all energy in
    a single turn.

    The weight is intentionally very small (0.001) — this is a shaping
    reward that should not dominate the tactical ATA/progress signals.
    """
    g = 9.81
    es_current = alt_m + (airspeed_mps ** 2) / (2.0 * g)
    es_reference = cfg.energy_ref_alt + (cfg.energy_ref_spd ** 2) / (2.0 * g)
    # Normalise: 1.0 at reference Es, decays slowly away from it
    es_ratio = np.clip(es_current / max(es_reference, 1.0), 0.0, 2.0)
    # Small reward for staying above 70% of reference energy
    if es_ratio > 0.7:
        return cfg.energy_weight * es_ratio * dt
    return 0.0


def reward_action_smoothness(
    prev_action: tuple, curr_action: tuple, dt: float, cfg: RewardConfig,
    is_discrete: bool = False,
) -> float:
    """Penalise jerky / high-frequency action switching.

    For continuous actions (n_x, n_n, mu): penalises the L2 norm of the
    action delta, encouraging the agent to make smooth, gradual adjustments
    rather than thrashing the controls every 0.1 s.

    For discrete actions: penalises any switch at all — the agent should
    commit to macro-actions and hold them for the full hold time.

    This is the RL equivalent of "pilot smoothness": real pilots don't
    jerk the stick at 10 Hz; they make deliberate, sustained inputs.
    """
    if is_discrete:
        # Discrete: any change = penalty (macro-action commitment)
        if prev_action is not None and prev_action != curr_action:
            return -cfg.action_rate_discrete_penalty * dt
        return 0.0

    # Continuous: L2 norm of action delta
    if prev_action is None:
        return 0.0
    delta = np.array(curr_action) - np.array(prev_action)
    jerk = float(np.sqrt(np.sum(delta ** 2)))
    return -cfg.action_rate_weight * jerk * dt
