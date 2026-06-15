"""Ablation experiment wrappers for single-pursuit training.

Each wrapper modifies exactly one concern, composes with ResidualExpertWrapper,
and is independently testable.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from src.utils.geometry import compute_forward_vector, compute_los
from src.environment.single_pursuit_env import PHYSICS_DT as SINGLE_PURSUIT_PHYSICS_DT


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame Stack Wrapper — temporal awareness via stacked observations
# ═══════════════════════════════════════════════════════════════════════════════

class FrameStackWrapper(gym.Wrapper):
    """Stack the last N observations into a flat vector.

    The policy sees [obs_{t-N+1}, ..., obs_t] giving it implicit velocity and
    inertia information through consecutive position changes.

    On reset, the buffer is filled with copies of the first observation.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        base_shape = env.observation_space.shape
        base_dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(base_shape[0] * n_frames,),
            dtype=base_dtype,
        )
        self._buffer: deque = deque(maxlen=n_frames)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        # Fill the buffer with copies of the initial observation
        self._buffer.clear()
        for _ in range(self.n_frames):
            self._buffer.append(obs.copy())
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        self._buffer.append(obs.copy())
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self._buffer)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Blended Action Wrapper — linear-cubic mix to eliminate the origin dead-zone
# ═══════════════════════════════════════════════════════════════════════════════

class BlendedActionWrapper(gym.Wrapper):
    """Linear-cubic blended action mapping: α·a + (1-α)·a³.

    Pure cubic (a³) creates a dead-zone near zero where tiny policy outputs
    produce essentially no physical effect.  This kills the exploration gradient
    and encourages the policy to collapse to a fixed trim command.

    The blend guarantees a linear floor (α·a) so that even small network
    outputs produce perceptible physical changes, while the cubic component
    still provides precision near the origin.

        a=0.0 → 0.000    a=0.1 → 0.0208   a=0.5 → 0.2188
        a=1.0 → 1.000    a=-0.3 → -0.078

    Args:
        env:  Gym environment to wrap.
        alpha: Linear blend coefficient (default 0.2).
               alpha=0.0 → pure cubic,  alpha=1.0 → pure linear.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.2):
        super().__init__(env)
        self.alpha = alpha

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        cubic = np.sign(action) * np.power(np.abs(action), 3.0)
        mapped = self.alpha * action + (1.0 - self.alpha) * cubic
        return self.env.step(mapped)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cubic Action Wrapper — deprecated, use BlendedActionWrapper for new work
# ═══════════════════════════════════════════════════════════════════════════════

class CubicActionWrapper(gym.Wrapper):
    """Pure cubic mapping a³ — deprecated in favour of BlendedActionWrapper.

    Kept for backward compatibility with saved models and past experiments.
    """

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        mapped = np.sign(action) * np.power(np.abs(action), 3.0)
        return self.env.step(mapped)


# ═══════════════════════════════════════════════════════════════════════════════
#  Lead Pursuit Reward Wrapper — guide toward predicted intercept point
# ═══════════════════════════════════════════════════════════════════════════════

class LeadPursuitRewardWrapper(gym.Wrapper):
    """Add lead pursuit reward terms on top of the base environment reward.

    Two new components:
    1. Velocity alignment — cos(pursuer_vel_dir, LOS_dir) × 2.0 × dt
       Rewards the aircraft actually MOVING toward the target (not just
       pointing at it — accounts for AoA/sideslip).

    2. Lead prediction — cos(pursuer_forward, LOS_to_future) × 3.0 × dt
       Rewards pointing at where the target WILL be (1 second ahead),
       not where it currently is. This is the core of lead pursuit.
    """

    VEL_ALIGN_WEIGHT = 15.0      # velocity alignment — moving toward target
    LEAD_PREDICT_WEIGHT = 25.0   # lead prediction — pointing at future position
    LOS_RATE_WEIGHT = 20.0       # LOS-rate damping — maintaining collision course
    LOS_RATE_SCALE = 5.0         # sensitivity: higher = sharper decay around λ̇≈0
    LEAD_TIME_SEC = 1.0          # look-ahead time for lead point

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only add lead pursuit bonus during normal flight (not on termination)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Access underlying SinglePursuitEnv state via .unwrapped (works through
        # any wrapper chain, e.g. CubicActionWrapper or ResidualExpertWrapper).
        env = self.unwrapped

        pursuer_pos = env.pursuer.position_ned
        pursuer_vel = env.pursuer.velocity_ned
        pursuer_rpy = env.pursuer.rpy_rad
        target_pos = env.target_ac.position_ned
        target_vel = env.target_ac.velocity_ned

        dt = SINGLE_PURSUIT_PHYSICS_DT

        # 1. Velocity alignment: is the aircraft MOVING toward the target?
        _, los_dir, _ = compute_los(pursuer_pos, target_pos)
        vel_norm = float(np.linalg.norm(pursuer_vel))
        r_vel_align = 0.0
        if vel_norm > 1.0:
            vel_dir = pursuer_vel / vel_norm
            cos_vel_los = float(np.clip(np.dot(vel_dir, los_dir), -0.5, 1.0))
            r_vel_align = cos_vel_los * self.VEL_ALIGN_WEIGHT * dt
            reward += r_vel_align

        # 2. Lead prediction: point at future target position
        future_pos = target_pos + target_vel * self.LEAD_TIME_SEC
        _, future_los_dir, _ = compute_los(pursuer_pos, future_pos)
        pursuer_forward = compute_forward_vector(pursuer_rpy)
        cos_lead = float(np.clip(np.dot(pursuer_forward, future_los_dir), -0.5, 1.0))
        r_lead_pred = cos_lead * self.LEAD_PREDICT_WEIGHT * dt
        reward += r_lead_pred

        # 3. LOS-rate damping — the core guidance metric
        # λ̇ = |v_rel_perp| / dist  (rad/s)
        # λ̇ ≈ 0  →  pursuer is on a perfect collision course
        # Reward decays exponentially with |λ̇|, creating a strong gradient
        # toward the collision-course manifold.
        los_vec = target_pos - pursuer_pos
        los_dist = float(np.linalg.norm(los_vec))
        if los_dist > 10.0:
            los_dir = los_vec / los_dist
            rel_vel = target_vel - pursuer_vel
            # Component of relative velocity perpendicular to LOS
            rel_vel_parallel = float(np.dot(rel_vel, los_dir)) * los_dir
            rel_vel_perp = rel_vel - rel_vel_parallel
            los_rate_mag = float(np.linalg.norm(rel_vel_perp)) / los_dist
            # Exponential reward: max at λ̇=0, decays to zero for large λ̇
            r_los_rate = np.exp(-los_rate_mag * self.LOS_RATE_SCALE) * self.LOS_RATE_WEIGHT * dt
            reward += r_los_rate
        else:
            r_los_rate = 0.0

        # Append lead pursuit components to info for diagnostics
        info["r_lead_vel_align"] = r_vel_align
        info["r_lead_pred"] = r_lead_pred
        info["r_los_rate"] = r_los_rate

        return obs, reward, terminated, truncated, info
