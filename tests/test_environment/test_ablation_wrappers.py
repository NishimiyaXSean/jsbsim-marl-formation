"""Unit tests for ablation wrappers."""
import numpy as np
import pytest
import gymnasium as gym

from src.environment.ablation_wrappers import CubicActionWrapper, FrameStackWrapper, LeadPursuitRewardWrapper


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


# ── CubicActionWrapper tests ────────────────────────────────────────────

class DummyEnvForCubicAction(gym.Env):
    """Captures the action received by the base env."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (4,))
        self.last_action = None

    def reset(self, seed=None, options=None):
        self.last_action = None
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_cubic_action_zero_passes_zero():
    """a=0 maps to 0 through cubic."""
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [0.0, 0.0, 0.0])


def test_cubic_action_half_maps_to_eighth():
    """a=0.5 maps to 0.125 through cubic."""
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [0.125, 0.125, 0.125])


def test_cubic_action_one_passes_one():
    """a=1.0 maps to 1.0 through cubic."""
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [1.0, 1.0, 1.0])


def test_cubic_action_negative_preserves_sign():
    """a=-0.5 maps to -0.125 through cubic — sign preserved."""
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([-0.5, -1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [-0.125, -1.0, 0.0])


def test_cubic_action_space_unchanged():
    """CubicActionWrapper preserves the action space definition."""
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    assert env.action_space.shape == (3,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


# ── LeadPursuitRewardWrapper tests ──────────────────────────────────────

class DummyEnvForLeadPursuit(gym.Env):
    """Minimal env that exposes pursuer/target NED state for reward calc."""

    REWARD_PROGRESS = 5.0
    REWARD_ATA = 5.0
    REWARD_GROUND_WARNING = 2.0
    REWARD_SUCCESS = 500.0
    REWARD_CRASH = -200.0
    REWARD_LOST_TARGET = -200.0
    PROXIMITY_TIERS = []
    MAX_DIST = 10000.0
    MAX_VEL = 400.0
    CTRL_FREQ = 60.0
    DECISION_STEPS = 30
    PHYSICS_DT = 1.0 / 60.0
    MAX_EPISODE_TIME = 120.0

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (3,))
        self.observation_space = gym.spaces.Box(-1, 1, (19,))
        self.pursuer = None
        self.target_ac = None
        self._step_counter = 0
        self._prev_dist = 1000.0
        self._proximity_awarded = set()
        self._tacview_frames = []
        self._record_tacview_frames = False

    def reset(self, seed=None, options=None):
        # Create mock aircraft-like objects with .position_ned and .velocity_ned
        self._step_counter = 0
        self._prev_dist = 1000.0
        self._proximity_awarded.clear()

        # Mock pursuer
        self.pursuer = type('obj', (object,), {})()
        self.pursuer.position_ned = np.array([0.0, 0.0, 3000.0], dtype=np.float64)
        self.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
        self.pursuer.rpy_rad = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.pursuer.state = {
            "n_z_g": 1.0, "airspeed_mps": 180.0, "alt_m": 3000.0,
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
            "beta_deg": 0.0,
        }

        # Mock target
        self.target_ac = type('obj', (object,), {})()
        self.target_ac.position_ned = np.array([1000.0, 0.0, 3000.0], dtype=np.float64)
        self.target_ac.velocity_ned = np.array([180.0, 10.0, 0.0], dtype=np.float64)  # moving slightly right
        self.target_ac.rpy_rad = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target_ac.state = {
            "n_z_g": 1.0, "airspeed_mps": 180.0, "alt_m": 3000.0,
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
            "beta_deg": 0.0,
        }

        obs = np.zeros(19, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Move pursuer forward (simple translation)
        self.pursuer.position_ned = self.pursuer.position_ned + np.array([3.0, 0.0, 0.0])

        # Target moves as well (same as initial velocity)
        self.target_ac.position_ned = self.target_ac.position_ned + np.array([3.0, 0.167, 0.0])

        self._step_counter += 1
        prev = self._prev_dist
        current = float(np.linalg.norm(self.pursuer.position_ned - self.target_ac.position_ned))
        self._prev_dist = current

        # Compute simple base reward matching SinglePursuitEnv pattern
        reward = 0.0
        delta_dist = prev - current
        reward += self.REWARD_PROGRESS * delta_dist

        from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles
        a_forward = compute_forward_vector(self.pursuer.rpy_rad)
        t_forward = compute_forward_vector(self.target_ac.rpy_rad)
        _, los_dir, _ = compute_los(self.pursuer.position_ned, self.target_ac.position_ned)
        geo = compute_tactical_angles(a_forward, t_forward, los_dir)
        reward += self.REWARD_ATA * max(geo["cos_ata"], -0.2) * self.PHYSICS_DT

        terminated = self._step_counter >= 30
        obs = np.zeros(19, dtype=np.float32)
        return obs, reward, terminated, False, {"reason": "timeout" if terminated else ""}


def test_lead_pursuit_wrapper_shape_unchanged():
    """LeadPursuitRewardWrapper preserves observation space."""
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    assert env.observation_space.shape == (19,)


def test_lead_pursuit_wrapper_adds_reward():
    """Wrapped reward exceeds base reward when velocity points at target."""
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    base.reset()
    base.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
    env.reset()
    env.unwrapped.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
    _, base_reward, _, _, _ = base.step(np.zeros(3))
    base.reset()
    base.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
    env.reset()
    env.unwrapped.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
    _, wrapped_reward, _, _, _ = env.step(np.zeros(3))
    # Pursuer moving [180,0,0], target directly ahead → velocity aligns with LOS
    # Lead pursuit terms should be positive (vel_align=~1.0, lead_pred=~1.0)
    assert wrapped_reward > base_reward, f"wrapped={wrapped_reward} <= base={base_reward}"


def test_lead_pursuit_wrapper_lead_prediction_contributes():
    """Lead prediction term varies as target moves laterally."""
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    base.reset()
    env.reset()
    # First step: both moving east
    _, r1, _, _, _ = env.step(np.zeros(3))
    # Second step: target drifts further right, changing lead point
    _, r2, _, _, _ = env.step(np.zeros(3))
    # Lead point should differ between steps → different lead prediction reward
    assert np.isfinite(r1) and np.isfinite(r2)
    # The base reward is identical both steps (same delta_dist, same ATA)
    # so any difference comes from the wrapper
    assert abs(r2 - r1) > 1e-6, f"Lead prediction not varying: r1={r1:.6f}, r2={r2:.6f}"
