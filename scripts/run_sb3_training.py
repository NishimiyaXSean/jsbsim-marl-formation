"""Train 1v1 MAPPO-style air combat using Stable-Baselines3 PPO with a self-play wrapper.

This avoids Ray/RLlib subprocess issues on Windows while maintaining the self-play
training paradigm (attacker and evader both learning).
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym

from src.environment.air_combat_env import AirCombatEnv


# ── Simple MLP policy for 4-dim continuous actions ──────────────────────

class MLPPolicy(nn.Module):
    """Separate actor-critic compatible with SB3's policy interface."""

    def __init__(self, obs_dim=19, act_dim=4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.actor(obs)
        value = self.critic(obs)
        return mean, self.log_std, value


# ── Self-play training ──────────────────────────────────────────────────

class SelfPlayCallback(BaseCallback):
    """Periodically swap the evader's policy with an older attacker snapshot."""

    def __init__(self, attacker_model, evader_model, swap_freq=50):
        super().__init__()
        self.attacker = attacker_model
        self.evader = evader_model
        self.swap_freq = swap_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.swap_freq == 0:
            # Copy attacker weights to evader (self-play update)
            self.evader.set_parameters(self.attacker.get_parameters())
        return True


def train():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.abspath(f"./marl_runs/sb3_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # ── Create environment ──────────────────────────────────────────────
    base_env = AirCombatEnv(gui=False, record_tacview=False)
    base_env.set_curriculum_stage(1)

    # ── Wrap as single-agent (attacker learns; evader uses fixed policy for now) ──
    class AttackerEnv(gym.Env):
        """Wrapper: attacker learns via SB3, evader uses a fixed policy."""

        def __init__(self, base_env):
            self._env = base_env
            self.observation_space = base_env.observation_spaces["attacker_0"]["obs"]
            self.action_space = base_env.action_spaces["attacker_0"]
            self._evader_policy = None  # set later

        def set_evader_policy(self, model):
            self._evader_policy = model

        def reset(self, **kwargs):
            obs, info = self._env.reset(**kwargs)
            return obs["attacker_0"]["obs"], info

        def step(self, action):
            # Evader: simple heuristic (random with bias toward straight flight)
            if self._evader_policy is not None:
                obs_e = self._env._get_obs_dict("evader_0")["obs"]
                evader_action, _ = self._evader_policy.predict(obs_e, deterministic=False)
            else:
                # Default: light left turn
                evader_action = np.array([0.6, 0.0, -0.3, 0.05], dtype=np.float32)

            actions = {"attacker_0": action, "evader_0": evader_action}
            obs, rewards, terminated, truncated, infos = self._env.step(actions)

            return (
                obs.get("attacker_0", {}).get("obs", np.zeros(19, dtype=np.float32)),
                rewards.get("attacker_0", 0.0),
                terminated.get("attacker_0", terminated.get("__all__", False)),
                truncated.get("attacker_0", truncated.get("__all__", False)),
                infos.get("attacker_0", {}),
            )

    env = AttackerEnv(base_env)

    # ── Train attacker ──────────────────────────────────────────────────
    print(f"{'='*50}")
    print("SB3 PPO 1v1 Air Combat Training (JSBSim F-16)")
    print(f"Log dir: {log_dir}")
    print(f"{'='*50}\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        device="cpu",
    )

    TOTAL_TIMESTEPS = 100_000
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...\n")
    print("Monitor with: tensorboard --logdir", log_dir)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving model...")

    # ── Save ────────────────────────────────────────────────────────────
    model_path = os.path.join(log_dir, "attacker_policy")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # ── Quick eval ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Final Evaluation (20 episodes)")
    successes = 0
    for ep in range(20):
        obs, _ = env.reset()
        done = False
        reason = "timeout"
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if "reason" in info:
                reason = info["reason"]
        if reason == "success":
            successes += 1
        print(f"  Ep {ep+1:2d}: {reason:15s}  reward={reward:+.0f}")

    print(f"\nSuccess rate: {successes}/20 = {successes * 5:.0f}%")
    print("Training complete.")


if __name__ == "__main__":
    train()
