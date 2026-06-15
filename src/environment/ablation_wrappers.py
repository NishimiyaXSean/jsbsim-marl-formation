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
#  Cubic Action Wrapper — nonlinear action mapping for origin precision
# ═══════════════════════════════════════════════════════════════════════════════

class CubicActionWrapper(gym.Wrapper):
    """Map raw policy output a ∈ [-1,1] through a cubic function: a³.

    This gives the policy fine-grained control near the origin while preserving
    full authority at the extremes:

        a=0.0 → 0.000  (dead zone for small jitter)
        a=0.1 → 0.001  (~1000 steps to max, extremely precise)
        a=0.5 → 0.125  (~8 steps to max)
        a=1.0 → 1.000  (full authority preserved)

    The mapping applies to all action dimensions uniformly. Exploration noise
    from PPO's log_std also passes through this mapping.
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

    VEL_ALIGN_WEIGHT = 2.0
    LEAD_PREDICT_WEIGHT = 3.0
    LEAD_TIME_SEC = 1.0          # look-ahead time for lead point

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only add lead pursuit bonus during normal flight (not on termination)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Access underlying SinglePursuitEnv state via unwrapped chain
        env = self.env

        pursuer_pos = env.pursuer.position_ned
        pursuer_vel = env.pursuer.velocity_ned
        pursuer_rpy = env.pursuer.rpy_rad
        target_pos = env.target_ac.position_ned
        target_vel = env.target_ac.velocity_ned

        dt = SINGLE_PURSUIT_PHYSICS_DT

        # 1. Velocity alignment: is the aircraft MOVING toward the target?
        _, los_dir, _ = compute_los(pursuer_pos, target_pos)
        vel_norm = float(np.linalg.norm(pursuer_vel))
        if vel_norm > 1.0:
            vel_dir = pursuer_vel / vel_norm
            cos_vel_los = float(np.clip(np.dot(vel_dir, los_dir), -0.5, 1.0))
            reward += cos_vel_los * self.VEL_ALIGN_WEIGHT * dt

        # 2. Lead prediction: point at future target position
        future_pos = target_pos + target_vel * self.LEAD_TIME_SEC
        _, future_los_dir, _ = compute_los(pursuer_pos, future_pos)
        pursuer_forward = compute_forward_vector(pursuer_rpy)
        cos_lead = float(np.clip(np.dot(pursuer_forward, future_los_dir), -0.5, 1.0))
        reward += cos_lead * self.LEAD_PREDICT_WEIGHT * dt

        return obs, reward, terminated, truncated, info
