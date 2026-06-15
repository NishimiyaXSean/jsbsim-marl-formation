"""Ablation experiment wrappers for single-pursuit training.

Each wrapper modifies exactly one concern, composes with ResidualExpertWrapper,
and is independently testable.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np


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
