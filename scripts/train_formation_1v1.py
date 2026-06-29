"""Quick 1v1 validation: FormationEnv vs ContinuousPursuitEnv baseline.

Runs a short PPO training (100K steps) on FormationEnv(num_pursuers=1)
to verify it achieves comparable performance to the validated
ContinuousPursuitEnv single-agent pipeline.

Usage:
    python scripts/train_formation_1v1.py
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque, defaultdict

from src.environment.formation_env import FormationEnv

TOTAL_TIMESTEPS = 100_000


class SimpleCallback(BaseCallback):
    """Minimal termination-rate logger."""

    def __init__(self, total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self._total_steps = total_steps
        self._term_counts = defaultdict(int)
        self._term_total = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    self._term_total += 1
                    reason = infos[i].get("termination_reason", "unknown")
                    self._term_counts[reason] += 1

        if self.n_calls % 10_000 == 0:
            pct = 100.0 * self.num_timesteps / self._total_steps
            parts = [f"{k}={v}" for k, v in self._term_counts.items()]
            print(f"[1v1] step={self.num_timesteps:>7d}  "
                  f"progress={pct:.0f}%  terms:[{' '.join(parts)}]")
            for k in self._term_counts:
                self.logger.record(f"terms/{k}", self._term_counts[k] / max(self._term_total, 1))
            self._term_counts.clear()
            self._term_total = 0
        return True


def train(seed: int = 42, total_steps: int = TOTAL_TIMESTEPS, difficulty: float = 0.0):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"formation_1v1_{ts}_s{seed}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[1v1] FormationEnv validation — 1 pursuer + 1 target")
    print(f"  Steps: {total_steps:,}  Difficulty: {difficulty:.2f}  Log: {log_dir}")

    env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)
    env = Monitor(env)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                         activation_fn=torch.nn.Tanh)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
                verbose=1, seed=seed, device="auto")

    callback = SimpleCallback(total_steps, verbose=1)
    model.learn(total_timesteps=total_steps, callback=callback,
                tb_log_name="formation_1v1", progress_bar=False)

    model.save(os.path.join(log_dir, "formation_1v1_final"))
    print(f"[1v1] Done. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(seed=args.seed, total_steps=args.steps, difficulty=args.difficulty)
