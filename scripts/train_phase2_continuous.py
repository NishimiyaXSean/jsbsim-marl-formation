"""Phase 2 SB3 PPO — Continuous Box(2) pursuit with FlightController.

Key differences from Phase 1 (train_phase1_discrete.py):
1. ContinuousPursuitEnv: Box(2) action → FlightController (no BFM routing)
2. Standard ActorCriticPolicy (no action masking needed)
3. No BlendedActionWrapper (discrete-specific)
4. Lower ent_coef: continuous actions explore naturally via Gaussian noise
5. LeadPursuitRewardWrapper provides L2 action smoothness + lead guidance

Usage:
    python scripts/train_phase2_continuous.py
    python scripts/train_phase2_continuous.py --seed 42 --steps 500000 --difficulty 0.0
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque, defaultdict

from src.environment.continuous_pursuit_env import ContinuousPursuitEnv
from src.environment.ablation_wrappers import LeadPursuitRewardWrapper
from scripts.train_single_pursuit import EVAL_EPISODES, EVAL_FREQ

# ── Phase 2 hyperparameters ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
DECISION_HZ = 2  # 0.5 s macro-action hold


class ContinuousPursuitCallback(BaseCallback):
    """Auto-curriculum + termination-rate logging for continuous pursuit.

    Simpler than Phase1Callback — no smoothness curriculum (the
    LeadPursuitRewardWrapper handles L2 action smoothness natively).
    """

    MIN_STEPS_PER_LEVEL = 20_000
    MIN_DIFFICULTY = 0.0          # continuous can start from straight-line
    CONSECUTIVE_ADVANCE_REQUIRED = 1
    COLLAPSE_WIN_RATE = 0.15
    CHECKPOINT_INTERVAL = 200_000

    def __init__(self, eval_env, log_dir: str, total_steps: int,
                 train_env=None, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._train_env = train_env
        self._log_dir = log_dir
        self._total_steps = total_steps
        self._difficulty = self.MIN_DIFFICULTY
        self._best_capture_rate = -1.0
        self._last_difficulty_change = 0
        self._recent_outcomes: deque = deque(maxlen=50)
        self._consecutive_good_evals = 0
        self._last_healthy_checkpoint: str | None = None
        self._healthy_params = None
        self._peak_difficulty = self.MIN_DIFFICULTY

        # Termination-rate tracking
        self._term_counts: dict = defaultdict(int)
        self._term_total = 0
        self._TERM_CATEGORIES = [
            "success", "stall", "timeout", "lost_target",
            "ground_crash", "out_of_bounds", "jsbsim_nan",
        ]

    def _on_step(self) -> bool:
        # ── Termination-rate tracking ──────────────────────────────────
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    self._term_total += 1
                    reason = infos[i].get("termination_reason", "unknown")
                    self._term_counts[reason] += 1

        # ── Periodic evaluation ────────────────────────────────────────
        if self.n_calls % EVAL_FREQ == 0 and self.n_calls > 0:
            self._evaluate_and_adjust()

        # ── Periodic checkpointing ─────────────────────────────────────
        if (self.num_timesteps % self.CHECKPOINT_INTERVAL == 0
                and self.num_timesteps > 0):
            ckpt_path = os.path.join(
                self._log_dir, f"checkpoint_{self.num_timesteps:07d}_steps.zip")
            self.model.save(ckpt_path)
            if self.verbose > 0:
                print(f"[Phase2] Checkpoint saved: {ckpt_path}")

        # ── Logging ────────────────────────────────────────────────────
        if self.n_calls % 10_000 == 0:
            progress_pct = 100.0 * self.num_timesteps / self._total_steps
            if self._term_total > 0:
                for cat in self._TERM_CATEGORIES:
                    rate = self._term_counts.get(cat, 0) / self._term_total
                    self.logger.record(f"termination_rate/{cat}", rate)
                self.logger.record("termination_rate/total", self._term_total)

            self.logger.record("phase2/difficulty", self._difficulty)
            self.logger.record("phase2/progress_pct", progress_pct)
            if self.verbose > 0:
                term_parts = []
                for cat in self._TERM_CATEGORIES:
                    c = self._term_counts.get(cat, 0)
                    if c > 0:
                        term_parts.append(f"{cat}={c}")
                term_str = " ".join(term_parts) if term_parts else "no_terms"
                print(f"[Phase2] step={self.num_timesteps:>8d}  "
                      f"progress={progress_pct:.1f}%  "
                      f"difficulty={self._difficulty:.2f}  "
                      f"terms:[{term_str}]")
            self._term_counts.clear()
            self._term_total = 0

        return True

    def _evaluate_and_adjust(self):
        """Run eval episodes and adjust difficulty via spring mechanism."""
        successes = 0
        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done, truncated = False, False
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = self._eval_env.step(action)
                if done and info.get("reason") == "success":
                    successes += 1
                    break
                if done or truncated:
                    break
            if successes >= 15:
                break

        win_rate = successes / EVAL_EPISODES
        self._recent_outcomes.extend([True] * successes + [False] * (EVAL_EPISODES - successes))
        sliding_wr = sum(self._recent_outcomes) / max(len(self._recent_outcomes), 1)

        if sliding_wr > self._best_capture_rate:
            self._best_capture_rate = sliding_wr

        steps_since_change = self.num_timesteps - self._last_difficulty_change
        if steps_since_change < self.MIN_STEPS_PER_LEVEL:
            return

        new_diff = self._difficulty
        if sliding_wr >= 0.50:
            self._consecutive_good_evals += 1
            if self._consecutive_good_evals >= self.CONSECUTIVE_ADVANCE_REQUIRED:
                self._healthy_params = self.model.get_parameters()
                self._last_healthy_checkpoint = os.path.join(
                    self._log_dir, f"healthy_diff_{self._difficulty:.2f}.zip")
                self.model.save(self._last_healthy_checkpoint)
                new_diff = min(self._difficulty + 0.05, 1.0)
                self._peak_difficulty = max(self._peak_difficulty, new_diff)
                self._consecutive_good_evals = 0
                print(f"[Phase2] ADVANCE: difficulty {self._difficulty:.2f} -> {new_diff:.2f}  "
                      f"(sliding WR={sliding_wr:.1%})")
        elif sliding_wr >= 0.40:
            self._consecutive_good_evals += 1
            if self._consecutive_good_evals >= self.CONSECUTIVE_ADVANCE_REQUIRED:
                new_diff = min(self._difficulty + 0.02, 1.0)
                self._consecutive_good_evals = 0
        elif sliding_wr < 0.10:
            self._consecutive_good_evals = 0
            if sliding_wr < 0.05:
                new_diff = max(self._difficulty - 0.02, self.MIN_DIFFICULTY)
            else:
                new_diff = max(self._difficulty - 0.01, self.MIN_DIFFICULTY)
        else:
            self._consecutive_good_evals = 0

        # Rollback on cliff collapse
        if (sliding_wr < self.COLLAPSE_WIN_RATE
                and self._difficulty > self.MIN_DIFFICULTY + 0.05
                and self._healthy_params is not None):
            self.model.set_parameters(self._healthy_params)
            new_diff = max(self._difficulty - 0.10, self.MIN_DIFFICULTY)
            print(f"[Phase2] ROLLBACK: difficulty {self._difficulty:.2f} -> {new_diff:.2f}  "
                  f"(collapse WR={sliding_wr:.1%}, restoring checkpoint)")
            self._consecutive_good_evals = 0

        if abs(new_diff - self._difficulty) > 1e-9:
            self._difficulty = new_diff
            self._last_difficulty_change = self.num_timesteps
            if self._train_env is not None:
                try:
                    self._train_env.set_difficulty(new_diff)
                except AttributeError:
                    pass


def build_env(difficulty: float = 0.0, lock_altitude: bool = True):
    """Build the Phase 2 continuous pursuit env chain.

    No BlendedActionWrapper (discrete-specific).
    LeadPursuitRewardWrapper works with Box(2) via L2 action norms.
    """
    base = ContinuousPursuitEnv(difficulty_level=difficulty, record_tacview=False,
                                lock_altitude=lock_altitude)
    base = LeadPursuitRewardWrapper(base)
    return base


def train(seed: int = 0, total_steps: int = TOTAL_TIMESTEPS, difficulty: float = 0.0,
          load_ckpt: str | None = None):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"phase2_continuous_{timestamp}_s{seed}"
    if load_ckpt:
        run_name += "_bc"
    if difficulty != 0.0:
        run_name += f"_d{int(difficulty*100):02d}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Phase2] Starting continuous Box(2) pursuit training")
    if load_ckpt:
        print(f"  BC checkpoint: {load_ckpt}")
    print(f"  Action space: Box(2) → [turn_rate (-15..+15 °/s), speed (150..350 m/s)]")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Difficulty:  {difficulty:.2f}")
    print(f"  Decision rate: {DECISION_HZ} Hz")
    print(f"  Log: {log_dir}")

    # ── Build envs ────────────────────────────────────────────────────
    train_env = build_env(difficulty=difficulty)
    train_env = Monitor(train_env)

    eval_env = build_env(difficulty=difficulty)
    eval_env = Monitor(eval_env)

    # ── Model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPO(
        "MlpPolicy",  # standard policy for Box observations
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # lower than discrete — Gaussian exploration is built-in
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
        device="auto",
    )

    # ── Load BC-pretrained weights (optional) ────────────────────────
    if load_ckpt:
        print(f"  Loading BC-pretrained weights from {load_ckpt} ...")
        bc_model = PPO.load(load_ckpt, device="cpu")
        model.policy.load_state_dict(bc_model.policy.state_dict())
        print(f"  BC weights loaded.  Resuming with PPO fine-tuning.")

    # ── Callback ──────────────────────────────────────────────────────
    callback = ContinuousPursuitCallback(
        eval_env, log_dir, total_steps,
        train_env=train_env, verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        tb_log_name="phase2",
        progress_bar=False,
    )

    # ── Save ──────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "phase2_final")
    model.save(final_path)
    print(f"[Phase2] Final model saved: {final_path}")

    print(f"\n[Phase2] Training complete")
    print(f"  Peak difficulty:    {callback._peak_difficulty:.2f}")
    print(f"  Best capture rate:  {callback._best_capture_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Initial difficulty (0.0 = straight target)")
    parser.add_argument("--load-ckpt", type=str, default=None,
                        help="Path to BC-pretrained model .zip for warm-start")
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    train(seed=args.seed, total_steps=args.steps, difficulty=args.difficulty,
          load_ckpt=args.load_ckpt)
