"""Phase 1 SB3 PPO — Discrete(9) BFM actions with curriculum for smoothness.

Key features over the baseline train_bfm_pursuit.py:
1. **Curriculum action smoothness**: The action-switch penalty ramps from 0
   (first 20% of steps) to full weight (by 80%), preventing the "lazy agent"
   that learns to never turn to avoid the penalty.
2. **Macro-action hold time**: 2 Hz decisions (0.5s hold) via BFMPursuitEnv's
   built-in DECISION_DT.  The agent commits to each BFM action for 30 physics
   frames before the next decision.
3. **Full reward stack**: LeadPursuitReward + progress + ATA + energy + closure
   gate + action smoothness.
4. **Auto-curriculum with rollback**: difficulty breathes with capture rate.

Usage:
    python scripts/train_phase1_discrete.py
    python scripts/train_phase1_discrete.py --seed 0 --steps 3000000
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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from collections import deque, defaultdict

from src.environment.bfm_pursuit_env import BFMPursuitEnv
from src.environment.ablation_wrappers import (
    ActionRepeatWrapper, BlendedActionWrapper, LeadPursuitRewardWrapper,
)
from src.environment.masked_policy import MaskableActorCriticPolicy
from scripts.train_single_pursuit import EVAL_EPISODES, EVAL_FREQ

# ── Phase 1 hyperparameters ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 3_000_000
DECISION_HZ = 2  # 0.5 s macro-action hold time

# Action smoothness curriculum (prevents lazy-agent collapse)
SMOOTHNESS_FULL_WEIGHT = 0.5      # final penalty weight
SMOOTHNESS_RAMP_START = 0.20      # first 20% of training: no penalty
SMOOTHNESS_RAMP_END = 0.80        # by 80%: full penalty


class Phase1Callback(BaseCallback):
    """Auto-curriculum + action-smoothness ramping for Phase 1 training.

    Extends the standard auto-curriculum with a dynamic action-rate
    penalty that starts at zero (encouraging exploration) and ramps
    up as the agent gains competence (enforcing smoothness).

    Smoothness ramp:
        steps 0 – SMOOTHNESS_RAMP_START * total:  w = 0
        steps RAMP_START – RAMP_END:              w = linear interpolation
        steps RAMP_END – end:                     w = FULL_WEIGHT
    """

    # ── Difficulty spring (same as AutoCurriculumCallback) ─────────────
    MIN_STEPS_PER_LEVEL = 20_000
    MIN_DIFFICULTY = 0.15
    CONSECUTIVE_ADVANCE_REQUIRED = 1
    COLLAPSE_WIN_RATE = 0.15

    # ── Periodic checkpointing ─────────────────────────────────────────
    CHECKPOINT_INTERVAL = 200_000  # save model every N steps

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
        self._current_smoothness_weight = 0.0

        # ── Termination-rate tracking (2026-06-26) ────────────────────
        self._term_counts: dict = defaultdict(int)
        self._term_total = 0
        self._TERM_CATEGORIES = [
            "success", "stall", "timeout", "lost_target",
            "ground_crash", "out_of_bounds", "jsbsim_nan",
        ]

    def _get_smoothness_weight(self) -> float:
        """Compute smoothness weight based on training progress."""
        progress = self.num_timesteps / max(self._total_steps, 1)
        if progress < SMOOTHNESS_RAMP_START:
            return 0.0
        elif progress < SMOOTHNESS_RAMP_END:
            frac = (progress - SMOOTHNESS_RAMP_START) / (
                SMOOTHNESS_RAMP_END - SMOOTHNESS_RAMP_START)
            return SMOOTHNESS_FULL_WEIGHT * frac
        else:
            return SMOOTHNESS_FULL_WEIGHT

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

        # ── Update smoothness weight ───────────────────────────────────
        new_w = self._get_smoothness_weight()
        if abs(new_w - self._current_smoothness_weight) > 1e-6:
            self._current_smoothness_weight = new_w
            # Push to env if it supports it
            if self._train_env is not None:
                try:
                    self._train_env.set_action_rate_weight(new_w)
                except AttributeError:
                    pass  # env doesn't support dynamic weight

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
                print(f"[Phase1] Checkpoint saved: {ckpt_path}")

        # ── Logging ────────────────────────────────────────────────────
        if self.n_calls % 10_000 == 0:
            progress_pct = 100.0 * self.num_timesteps / self._total_steps
            # ── Log termination rates ──────────────────────────────
            if self._term_total > 0:
                for cat in self._TERM_CATEGORIES:
                    rate = self._term_counts.get(cat, 0) / self._term_total
                    self.logger.record(f"termination_rate/{cat}", rate)
                self.logger.record("termination_rate/total", self._term_total)

            self.logger.record("phase1/difficulty", self._difficulty)
            self.logger.record("phase1/smoothness_weight", self._current_smoothness_weight)
            self.logger.record("phase1/progress_pct", progress_pct)
            if self.verbose > 0:
                # Build termination summary string
                term_parts = []
                for cat in self._TERM_CATEGORIES:
                    c = self._term_counts.get(cat, 0)
                    if c > 0:
                        term_parts.append(f"{cat}={c}")
                term_str = " ".join(term_parts) if term_parts else "no_terms"
                print(f"[Phase1] step={self.num_timesteps:>8d}  "
                      f"progress={progress_pct:.1f}%  "
                      f"difficulty={self._difficulty:.2f}  "
                      f"smoothness_w={self._current_smoothness_weight:.3f}  "
                      f"terms:[{term_str}]")
            # Reset counters for next interval
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
                if done and info.get("capture", False):
                    successes += 1
                    break
                if done or truncated:
                    break
            if successes >= 15:
                break  # early exit if clearly passing

        win_rate = successes / EVAL_EPISODES
        self._recent_outcomes.extend([True] * successes + [False] * (EVAL_EPISODES - successes))
        sliding_wr = sum(self._recent_outcomes) / max(len(self._recent_outcomes), 1)

        if sliding_wr > self._best_capture_rate:
            self._best_capture_rate = sliding_wr

        # Spring mechanism
        steps_since_change = self.num_timesteps - self._last_difficulty_change
        if steps_since_change < self.MIN_STEPS_PER_LEVEL:
            return

        new_diff = self._difficulty
        if sliding_wr >= 0.50:
            self._consecutive_good_evals += 1
            if self._consecutive_good_evals >= self.CONSECUTIVE_ADVANCE_REQUIRED:
                # Save checkpoint before advancing
                self._healthy_params = self.model.get_parameters()
                self._last_healthy_checkpoint = os.path.join(
                    self._log_dir, f"healthy_diff_{self._difficulty:.2f}.zip")
                self.model.save(self._last_healthy_checkpoint)
                new_diff = min(self._difficulty + 0.05, 1.0)
                self._peak_difficulty = max(self._peak_difficulty, new_diff)
                self._consecutive_good_evals = 0
                print(f"[Phase1] ADVANCE: difficulty {self._difficulty:.2f} -> {new_diff:.2f}  "
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
            self._consecutive_good_evals = 0  # flow zone — no change

        # Rollback on cliff collapse
        if (sliding_wr < self.COLLAPSE_WIN_RATE
                and self._difficulty > self.MIN_DIFFICULTY + 0.05
                and self._healthy_params is not None):
            self.model.set_parameters(self._healthy_params)
            new_diff = max(self._difficulty - 0.10, self.MIN_DIFFICULTY)
            print(f"[Phase1] ROLLBACK: difficulty {self._difficulty:.2f} -> {new_diff:.2f}  "
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


def build_env(difficulty: float = 0.15, lock_altitude: bool = True):
    """Build the Phase 1 BFM pursuit env chain.

    lock_altitude=True (2D mode): both aircraft use FlightController to
    maintain 3000m.  Agent learns horizontal-plane pursuit without the
    energy/altitude coupling trap.
    """
    base = BFMPursuitEnv(difficulty_level=difficulty, record_tacview=False,
                         lock_altitude=lock_altitude)
    base = BlendedActionWrapper(base, alpha=0.02)
    base = LeadPursuitRewardWrapper(base)
    return base


def train(seed: int = 0, total_steps: int = TOTAL_TIMESTEPS):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"phase1_discrete_{timestamp}_s{seed}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Phase1] Starting discrete BFM pursuit training")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Decision rate: {DECISION_HZ} Hz (macro-action hold)")
    print(f"  Smoothness curriculum: ramp {SMOOTHNESS_RAMP_START:.0%}-{SMOOTHNESS_RAMP_END:.0%}")
    print(f"  Log: {log_dir}")

    # ── Build envs ────────────────────────────────────────────────────
    train_env = build_env(difficulty=0.15)
    train_env = Monitor(train_env)
    # VecNormalize: prevents reward-scale explosions from collapsing entropy
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                              clip_reward=100.0)

    # Raw eval env for callback (no VecEnv wrapping — callback does manual loops)
    eval_env = build_env(difficulty=0.15)
    eval_env = Monitor(eval_env)

    # ── Model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPO(
        MaskableActorCriticPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.08,  # heavy exploration — prevents Decelerate-only collapse
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
        device="auto",
    )

    # ── Callback ──────────────────────────────────────────────────────
    callback = Phase1Callback(
        eval_env, log_dir, total_steps,
        train_env=train_env, verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        tb_log_name="phase1",
        progress_bar=False,
    )

    # ── Save ──────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "phase1_final")
    model.save(final_path)
    print(f"[Phase1] Final model saved: {final_path}")

    # Summary
    print(f"\n[Phase1] Training complete")
    print(f"  Peak difficulty: {callback._peak_difficulty:.2f}")
    print(f"  Best capture rate: {callback._best_capture_rate:.1%}")
    print(f"  Final smoothness w: {callback._current_smoothness_weight:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    train(seed=args.seed, total_steps=args.steps)
