"""Train 1v1 air combat using Stable-Baselines3 PPO.

Supports two action modes (set via ACTION_MODE below):
- "continuous": 4-dim [throttle, elevator, aileron, rudder]
- "bfm":        Discrete(13) Basic Fighter Maneuvers

Avoids Ray/RLlib subprocess issues on Windows.  Single-agent: attacker
learns via PPO; evader flies a fixed straight-and-level pattern (Stage 1).
"""

from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym

from src.environment.air_combat_env import AirCombatEnv
from src.environment.rewards import RewardConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration — tweak these
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_MODE: str = "continuous"   # Phase A: "continuous" → Phase B: "bfm"
TOTAL_TIMESTEPS: int = 500_000    # 500k steps — crash vs survive gradient needs more samples
CURRICULUM_STAGE: int = 1         # 1 = easy evader, 2 = medium, 3 = hard


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom callback — logs per-episode metrics to TensorBoard
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodeLoggingCallback(BaseCallback):
    """Log episode reward, length, and termination reason to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_count: int = 0

    def _on_step(self) -> bool:
        # Check each env for episode end
        for env_idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[env_idx]
                ep_info = info.get("episode", {})
                r = ep_info.get("r", 0.0)
                length = ep_info.get("l", 0)
                reason = ep_info.get("reason", info.get("reason", "?"))

                self._ep_rewards.append(r)
                self._ep_lengths.append(length)
                self._ep_count += 1

                if self.logger is not None:
                    self.logger.record("episode/rew", r)
                    self.logger.record("episode/len", length)
                    self.logger.record("episode/count", self._ep_count)
        return True

    def _on_training_end(self) -> None:
        if self._ep_rewards:
            print(f"\n[Callback] {self._ep_count} episodes completed "
                  f"| avg reward: {np.mean(self._ep_rewards):.0f} "
                  f"| avg length: {np.mean(self._ep_lengths):.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-agent wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class AttackerEnv(gym.Env):
    """Wrap 1v1 AirCombatEnv so SB3 sees only the attacker's observation.

    The evader uses a simple altitude-hold autopilot to stay airborne,
    forcing the attacker to actively learn pursuit and altitude control.
    """

    def __init__(self, base_env: AirCombatEnv):
        self._env = base_env
        self.observation_space = base_env.observation_spaces["attacker_0"]["obs"]
        self.action_space = base_env.action_spaces["attacker_0"]
        self._evader_target_alt_m: float = 3000.0

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        # Set evader's altitude target to its initial altitude
        self._evader_target_alt_m = float(self._env.evader.state["alt_m"])
        return obs["attacker_0"]["obs"], info

    def step(self, action):
        # Evader: simple altitude-hold PID to stay alive and challenge the attacker
        if self._env.action_mode == "bfm":
            evader_action = 0  # BFM: level flight
        else:
            evader_alt = self._env.evader.state["alt_m"]
            alt_error = self._evader_target_alt_m - evader_alt
            # P-controller: 100m error → 0.1 elevator
            elevator = float(np.clip(alt_error * 0.001, -0.5, 0.5))
            evader_action = np.array([0.8, elevator, 0.0, 0.0], dtype=np.float32)

        # ── Attacker: clip actions to prevent instant self-destruction ──
        # Prevent full nose-down (elevator < -0.3) but allow full pull-up (elevator up to 1.0).
        # Aileron/rudder limited to avoid spins.  Throttle unrestricted.
        action = np.array(action, dtype=np.float32)
        action = np.clip(action,
            [-1.0, -0.3, -0.5, -0.3],
            [ 1.0,  1.0,  0.5,  0.3],
        )

        actions = {"attacker_0": action, "evader_0": evader_action}
        obs, rewards, terminated, truncated, infos = self._env.step(actions)

        reward = float(rewards.get("attacker_0", 0.0))
        done = bool(terminated.get("attacker_0", False)
                    or terminated.get("__all__", False))
        trunc = bool(truncated.get("attacker_0", False)
                     or truncated.get("__all__", False))
        info_out = infos.get("attacker_0", {})
        obs_out = obs.get("attacker_0", {}).get("obs",
                   np.zeros(self.observation_space.shape, dtype=np.float32))

        return obs_out, reward, done, trunc, info_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Training entry point
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.abspath(f"./marl_runs/sb3_{ACTION_MODE}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # ── Create environment ──────────────────────────────────────────────
    base_env = AirCombatEnv(
        gui=False, record_tacview=False,
        action_mode=ACTION_MODE,
        reward_config=RewardConfig(),  # uses (already-tuned) defaults
    )
    base_env.set_curriculum_stage(CURRICULUM_STAGE)

    env = AttackerEnv(base_env)

    # ── Print setup ─────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"SB3 PPO 1v1 Air Combat  |  JSBSim F-16")
    print(f"  Action mode:  {ACTION_MODE}  ({env.action_space})")
    print(f"  Observation:  {env.observation_space.shape}")
    print(f"  Curriculum:   stage {CURRICULUM_STAGE}")
    print(f"  Timesteps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Log dir:      {log_dir}")
    print(f"{'='*55}\n")

    # ── Build PPO ───────────────────────────────────────────────────────
    if ACTION_MODE == "bfm":
        # Discrete actions: moderate entropy to allow convergence from 13 options
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=1e-4,
            n_steps=1024,             # longer rollouts for discrete actions
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,           # lower — let policy exploit after exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            device="cpu",
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            ),
        )
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            device="cpu",
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            ),
        )

    print(f"Monitor:  tensorboard --logdir {log_dir}")
    print(f"Training {TOTAL_TIMESTEPS:,} timesteps ...\n")

    callback = EpisodeLoggingCallback()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                    progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint ...")

    # ── Save ────────────────────────────────────────────────────────────
    model_path = os.path.join(log_dir, "attacker_policy")
    model.save(model_path)
    print(f"\nModel saved → {model_path}.zip")

    # ── Final eval ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final Evaluation (20 episodes)")
    successes = 0
    episodes = []
    for ep in range(20):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        reason = "timeout"
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward
            if "reason" in info:
                reason = info["reason"]
        if reason == "success":
            successes += 1
        episodes.append((reason, total_r))
        print(f"  Ep {ep+1:2d}: {reason:15s}  total_reward={total_r:+.0f}")

    print(f"\nSuccess rate: {successes}/20 = {successes*5:.0f}%")
    avg_r = np.mean([r for _, r in episodes])
    print(f"Avg total reward: {avg_r:+.0f}")
    print("Training complete.")


if __name__ == "__main__":
    # Suppress JSBSim startup noise during training
    import logging
    import warnings
    import os as _os
    _os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("ray").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)

    train()
