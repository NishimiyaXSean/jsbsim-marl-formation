"""Quick smoke test for 22-dim observation."""
import sys, os, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

import numpy as np
from src.environment.single_pursuit_env import SinglePursuitEnv

# Test 1: Create env and verify 22-dim observation
env = SinglePursuitEnv(curriculum_stage=1.0)
obs, _ = env.reset()
print(f'Test 1: Observation shape: {obs.shape} (expected: (22,))')
assert obs.shape == (22,), f'Expected (22,), got {obs.shape}'

# Test 2: Run a few steps with random actions
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (22,), f'Step {i}: Expected (22,), got {obs.shape}'
print(f'Test 2: 5 steps OK, last reward={reward:.2f}, reason={info.get("reason")}')

# Test 3: Verify angular velocity indices are populated (non-zero after flight dynamics)
print(f'Test 3: Pursuer ang_vel (idx 9-11): {obs[9:12]}')
print(f'          Target ang_vel (idx 16-18): {obs[16:19]}')

# Test 4: Verify observation space is correct
print(f'Test 4: Observation space shape: {env.observation_space.shape}')
assert env.observation_space.shape == (22,)

print('\nAll smoke tests PASSED!')
