"""Test that AirCombatEnv.reset() returns valid observations."""

import numpy as np

from src.environment.air_combat_env import AirCombatEnv


def test_reset_returns_valid_obs():
    env = AirCombatEnv()
    obs, info = env.reset(seed=42)

    assert "attacker_0" in obs
    assert "evader_0" in obs

    for agent_id in ["attacker_0", "evader_0"]:
        assert "obs" in obs[agent_id]
        assert "global_state" in obs[agent_id]
        assert obs[agent_id]["obs"].shape == (19,)
        assert obs[agent_id]["global_state"].shape == (26,)
        assert obs[agent_id]["obs"].dtype == np.float32
        assert np.all(np.abs(obs[agent_id]["obs"]) <= 1.0)


def test_step_returns_valid_data():
    env = AirCombatEnv()
    env.reset(seed=42)

    actions = {
        "attacker_0": np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        "evader_0": np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    obs, rewards, terminated, truncated, infos = env.step(actions)

    assert isinstance(rewards["attacker_0"], float)
    assert isinstance(rewards["evader_0"], float)
    assert "__all__" in terminated
    assert "__all__" in truncated
