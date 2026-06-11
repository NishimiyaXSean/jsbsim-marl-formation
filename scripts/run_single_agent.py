"""Run a single-agent tracking task using PPO (Stable-Baselines3).

This is the simplest demo: one F-16 tries to track a moving target.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.environment.air_combat_env import AirCombatEnv

# Wrap 1v1 env as single-agent (only attacker RL, evader uses fixed policy)
class SingleAgentWrapper(gym.Wrapper):
    """Wrap AirCombatEnv so SB3 sees it as single-agent."""

    def __init__(self, env: AirCombatEnv):
        super().__init__(env)
        self.observation_space = env.observation_spaces["attacker_0"]["obs"]
        self.action_space = env.action_spaces["attacker_0"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["attacker_0"]["obs"], info

    def step(self, action):
        # Evader: straight and level flight
        evader_action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, infos = self.env.step({
            "attacker_0": action,
            "evader_0": evader_action,
        })
        return (
            obs.get("attacker_0", {}).get("obs", np.zeros(19, dtype=np.float32)),
            rewards.get("attacker_0", 0.0),
            terminated.get("attacker_0", terminated.get("__all__", False)),
            truncated.get("attacker_0", truncated.get("__all__", False)),
            infos.get("attacker_0", {}),
        )


def main():
    print("Single-agent tracking demo (JSBSim F-16)")
    print("=" * 50)

    base_env = AirCombatEnv(gui=False, record_tacview=False)
    env = SingleAgentWrapper(base_env)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, n_steps=2048)

    print("Training for 50,000 steps...")
    model.learn(total_timesteps=50000)

    model.save("data/checkpoints/single_agent_tracking")
    print("Model saved to data/checkpoints/single_agent_tracking.zip")

    # Quick eval
    print("\nEval (10 episodes):")
    successes = 0
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("reason") == "success":
            successes += 1
        print(f"  Episode {ep+1}: reason={info.get('reason', 'unknown')}")

    print(f"\nSuccess rate: {successes}/10 = {successes*10}%")


if __name__ == "__main__":
    main()
