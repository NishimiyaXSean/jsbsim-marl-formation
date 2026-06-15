"""Unit tests for ablation wrappers."""
import numpy as np
import pytest
import gymnasium as gym

from src.environment.ablation_wrappers import FrameStackWrapper


# ── FrameStackWrapper tests ─────────────────────────────────────────────

class DummyEnvForFrameStack(gym.Env):
    """Minimal env that returns obs equal to step count."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.observation_space = gym.spaces.Box(-1, 1, (3,))
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return np.full((3,), float(self._step), dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full((3,), float(self._step), dtype=np.float32)
        reward = float(self._step)
        terminated = self._step >= 5
        truncated = False
        info = {"step": self._step}
        return obs, reward, terminated, truncated, info


def test_frame_stack_output_shape():
    """FrameStackWrapper outputs (obs_dim * N,) stacked observations."""
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    assert env.observation_space.shape == (12,)
    assert env.observation_space.dtype == np.float32


def test_frame_stack_reset_fills_buffer():
    """On reset, all N frames equal the initial observation."""
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    obs, _ = env.reset()
    assert obs.shape == (12,)
    # All 4 frames should be [0, 0, 0]
    expected = np.zeros(12, dtype=np.float32)
    np.testing.assert_array_equal(obs, expected)


def test_frame_stack_step_returns_stacked():
    """After step, observation is concatenation of last N frames."""
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    _, _ = env.reset()
    # first stacked obs = [0,0,0, 0,0,0, 0,0,0, 0,0,0]
    obs1, _, _, _, _ = env.step(np.zeros(2))
    # obs1 should be [0,0,0, 0,0,0, 0,0,0, 1,1,1] — 3 frames of 0s + 1 frame of 1s
    expected = np.array([0,0,0, 0,0,0, 0,0,0, 1,1,1], dtype=np.float32)
    np.testing.assert_array_equal(obs1, expected)


def test_frame_stack_3_frames():
    """n_frames=3 produces correct output shape."""
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=3)
    assert env.observation_space.shape == (9,)
    obs, _ = env.reset()
    assert obs.shape == (9,)


def test_frame_stack_preserves_info():
    """Info dict from base env is passed through unchanged."""
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["step"] == 1
