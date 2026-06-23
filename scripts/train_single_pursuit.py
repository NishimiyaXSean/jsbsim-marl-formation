"""Train a single-agent PPO policy for F-16 pursuit using Stable-Baselines3.

The agent controls high-level flight targets (heading, altitude, speed)
through a stabilised FlightController.  Training follows a 5-stage
curriculum with increasing target difficulty.

Usage:
    conda activate jsbsim_rl
    JSBSIM_DEBUG=0 python scripts/train_single_pursuit.py
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.utils.pn_guidance import compute_pn_heading
import gymnasium as gym

# ═══════════════════════════════════════════════════════════════════════════════
#  Residual wrapper: agent learns correction on top of expert baseline
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualExpertWrapper(gym.Wrapper):
    """Agent learns residual on top of a PN guidance expert.

    Expert reads world-frame positions/velocities directly from the underlying
    SinglePursuitEnv and uses proportional navigation to compute desired heading.
    A P-controller on heading error produces aileron commands.
    Agent adds ±0.5 residual corrections (scaled by RESIDUAL_SCALE).

    This guarantees the expert guidance is never lost — RL only fine-tunes.
    """
    RESIDUAL_SCALE = 1.0  # Agent has full ±1.0 authority (no expert baseline)

    def __init__(self, env):
        super().__init__(env)
        self._base_env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Agent controls all three channels directly (no expert overlay).
        # This gives full authority over aileron, altitude delta, and speed delta.
        combined = np.asarray(action, dtype=np.float32)
        obs, rew, term, trunc, info = self.env.step(combined)
        return obs, rew, term, trunc, info

    def _compute_expert(self) -> np.ndarray:
        """Neutral expert — agent has full control.

        Kept as a hook for future expert re-integration (e.g. curriculum-based
        PN guidance blending).
        """
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Delegate curriculum_stage / difficulty_level to underlying env
    @property
    def curriculum_stage(self):
        return self._base_env.curriculum_stage

    @curriculum_stage.setter
    def curriculum_stage(self, value):
        self._base_env.curriculum_stage = value

    @property
    def difficulty_level(self):
        return self.unwrapped.difficulty_level

    @difficulty_level.setter
    def difficulty_level(self, value):
        self.unwrapped.difficulty_level = value


# ═══════════════════════════════════════════════════════════════════════════════
#  Training config
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 5_000_000
CURRICULUM_STAGES = [1.0, 1.5, 2.0, 2.5, 3.0]
STAGE_TIMESTEPS = TOTAL_TIMESTEPS // len(CURRICULUM_STAGES)  # 100k per stage

EVAL_EPISODES = 30
EVAL_FREQ = 15_000
TARGET_CAPTURE_RATE_STAGE_1_2 = 0.40   # stage 1.0→1.5 and 1.5→2.0
TARGET_CAPTURE_RATE_STAGE_2_3 = 0.50   # stage 2.0→2.5 and 2.5→3.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class AutoCurriculumCallback(BaseCallback):
    """Adaptive auto-curriculum with conservative advancement and fallback rollback.

    Replaces fixed-stage curriculum with a continuous ``difficulty_level``
    that breathes with the agent's real capability.  A deque of the last
    50 episode outcomes provides a responsive win-rate estimate, and
    difficulty is adjusted via a spring mechanism.

    **Advancement** requires *consecutive* good evaluations (win_rate >= 50%
    for N consecutive evals) before difficulty is increased — this prevents
    a single lucky evaluation from pushing the agent beyond its true ability.

    **Rollback** fires when difficulty has been previously advanced but the
    agent subsequently suffers a cliff collapse (win_rate < 15%).  In that
    case both the environment difficulty AND the model weights are restored
    to the last healthy pre-advance checkpoint, giving the agent a clean
    second chance.

    Spring thresholds:
        win_rate >= 50%  →  difficulty += 0.05   (aggressive push)*
        40% <= wr < 50%  →  difficulty += 0.02   (gentle push)*
        10% <= wr < 40%  →  no change            (wide flow zone)
         5% <= wr < 10%  →  difficulty -= 0.01   (gentle nudge back)
        wr < 5%          →  difficulty -= 0.02   (moderate retreat)

        * Requires CONSECUTIVE_ADVANCE_REQUIRED consecutive qualifying evals.
    """

    MIN_STEPS_PER_LEVEL = 20_000          # minimum steps between difficulty changes
    MIN_DIFFICULTY = 0.15                 # curriculum floor — no trivial straight-line targets
    CONSECUTIVE_ADVANCE_REQUIRED = 1      # consecutive good evals to unlock advancement
    COLLAPSE_WIN_RATE = 0.15              # wr below this + difficulty above floor = rollback

    def __init__(self, eval_env, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._log_dir = log_dir
        self._difficulty = self.MIN_DIFFICULTY
        self._best_capture_rate = -1.0
        self._best_capture_rate_per_level: dict[float, float] = {}  # best CR at each difficulty
        self._last_difficulty_change = 0
        self._eval_metrics: list = []
        # Sliding window: True = success, False = failure (50 episodes for responsive spring)
        from collections import deque
        self._recent_outcomes: deque = deque(maxlen=50)
        # Conservative advancement trackers
        self._consecutive_good_evals = 0   # consecutive evals with wr >= 50%
        self._last_healthy_checkpoint: str | None = None  # disk path to pre-advance model
        self._healthy_params = None         # in-memory copy of pre-advance weights
        self._peak_difficulty = self.MIN_DIFFICULTY  # highest difficulty achieved this run

    @property
    def difficulty_level(self) -> float:
        return self._difficulty

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Evaluate at EVAL_FREQ intervals (same as CurriculumCallback)
        if self.num_timesteps % EVAL_FREQ > 2048:
            return

        # Update eval env to current difficulty
        self._eval_env.difficulty_level = self._difficulty

        # ── Evaluation ─────────────────────────────────────────────────
        successes, min_dists, intercept_times = 0, [], []
        # Termination counters
        term_counts = {"success": 0, "timeout": 0, "lost_target": 0,
                       "ground_crash": 0, "out_of_bounds": 0, "jsbsim_nan": 0,
                       "stall": 0}
        # Reward component accumulators (averaged over eval episodes)
        r_component_sums: dict = {}
        r_component_keys = ["r_progress", "r_terminal_boost", "r_ata",
                           "r_time_pressure", "r_ground_warning", "r_proximity",
                           "r_low_speed_penalty", "r_step_penalty",
                           "r_lead_vel_align", "r_lead_pred", "r_los_rate",
                           "r_energy_gated", "r_smoothness", "r_vc_coupled"]

        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done = False
            ep_min_dist = 8000.0
            ep_intercept_time = 120.0
            ep_r_components: dict = {k: 0.0 for k in r_component_keys}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if "min_dist" in info:
                    ep_min_dist = min(ep_min_dist, info["min_dist"])
                # Accumulate reward components from this step
                for k in r_component_keys:
                    if k in info:
                        ep_r_components[k] += float(info[k])
            reason = info.get("reason", "timeout")
            is_success = reason == "success"
            if is_success:
                successes += 1
                base_env = self._eval_env.unwrapped
                ep_intercept_time = base_env._step_counter / 60.0  # CTRL_FREQ
            # Track termination distribution
            if reason in term_counts:
                term_counts[reason] += 1
            else:
                term_counts[reason] = 1
            # Accumulate episode reward components
            for k in r_component_keys:
                r_component_sums[k] = r_component_sums.get(k, 0.0) + ep_r_components[k]
            self._recent_outcomes.append(is_success)
            min_dists.append(ep_min_dist)
            intercept_times.append(ep_intercept_time)

        capture_rate = successes / EVAL_EPISODES
        n_ep = EVAL_EPISODES

        # ── Win rate from sliding window ───────────────────────────────
        if len(self._recent_outcomes) > 0:
            win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        else:
            win_rate = 0.0

        avg_min_dist = np.mean(min_dists)
        avg_intercept_time = np.mean(intercept_times)

        # ── Reward components: average per episode ──────────────────────
        r_avg = {k: r_component_sums.get(k, 0.0) / n_ep for k in r_component_keys}

        # Physical telemetry: verify the difficulty actually reached the env
        physical_diff = self._eval_env.unwrapped.difficulty_level

        print(f"\n  [Eval @ {self.num_timesteps:,} steps] "
              f"diff={self._difficulty:.2f}(phys={physical_diff:.2f}) "
              f"capture_rate={capture_rate:.0%} "
              f"win_rate(50ep)={win_rate:.0%} "
              f"avg_min_dist={avg_min_dist:.0f}m "
              f"avg_intercept={avg_intercept_time:.1f}s "
              f"consec_good={self._consecutive_good_evals}")
        # Termination distribution
        print(f"    Terms: succ={term_counts['success']} "
              f"timeout={term_counts['timeout']} "
              f"lost={term_counts['lost_target']} "
              f"crash={term_counts['ground_crash']} "
              f"stall={term_counts['stall']} "
              f"oob={term_counts['out_of_bounds']}")
        # Reward decomposition
        print(f"    Rewards: progress={r_avg['r_progress']:+.1f} "
              f"ATA={r_avg['r_ata']:+.1f} "
              f"lead_vel={r_avg['r_lead_vel_align']:+.2f} "
              f"lead_pred={r_avg['r_lead_pred']:+.2f} "
              f"los_rate={r_avg['r_los_rate']:+.2f} "
              f"step={r_avg['r_step_penalty']:+.1f} "
              f"prox={r_avg['r_proximity']:+.1f}")

        # ── Logging ────────────────────────────────────────────────────
        self.logger.record("eval/capture_rate", capture_rate)
        self.logger.record("eval/avg_min_dist", avg_min_dist)
        self.logger.record("curriculum/difficulty", self._difficulty)
        self.logger.record("curriculum/difficulty_physical", physical_diff)
        self.logger.record("curriculum/win_rate_50ep", win_rate)
        self.logger.record("curriculum/consecutive_good", self._consecutive_good_evals)
        for k in r_component_keys:
            self.logger.record(f"reward/{k}", r_avg[k])

        self._eval_metrics.append({
            "timesteps": self.num_timesteps,
            "difficulty": self._difficulty,
            "difficulty_physical": physical_diff,
            "capture_rate": capture_rate,
            "win_rate_100ep": win_rate,
            "avg_min_dist": avg_min_dist,
            "avg_intercept_time": avg_intercept_time,
            "term_timeout": term_counts["timeout"],
            "term_lost": term_counts["lost_target"],
            "term_crash": term_counts["ground_crash"],
            "term_success": term_counts["success"],
            **{f"r_{k}": r_avg[k] for k in r_component_keys},
        })

        # ── Save best model (only when capture_rate > 0) ───────────────
        if capture_rate > 0 and capture_rate > self._best_capture_rate:
            self._best_capture_rate = capture_rate
            best_path = os.path.join(self._log_dir, "best_model")
            self.model.save(best_path)
            print(f"  -> New best model saved: {best_path}  "
                  f"(capture_rate={capture_rate:.0%})")

        # ── Save per-difficulty best model ─────────────────────────────
        diff_key = round(self._difficulty, 2)
        prev_best = self._best_capture_rate_per_level.get(diff_key, -1.0)
        if capture_rate > prev_best:
            self._best_capture_rate_per_level[diff_key] = capture_rate
            per_lvl_path = os.path.join(self._log_dir, f"best_model_diff_{diff_key:.2f}")
            self.model.save(per_lvl_path)
            print(f"  -> Per-level best saved: {per_lvl_path}  "
                  f"(capture_rate={capture_rate:.0%} @ diff={diff_key:.2f})")

        # ── Track consecutive good evaluations ─────────────────────────
        if win_rate >= 0.50:
            self._consecutive_good_evals += 1
        else:
            self._consecutive_good_evals = 0

        # ── Difficulty adjustment (conservative spring with rollback) ───
        steps_since_change = self.num_timesteps - self._last_difficulty_change
        old_difficulty = self._difficulty

        if steps_since_change >= self.MIN_STEPS_PER_LEVEL:
            # Check for advancement (requires consecutive good evals)
            if self._consecutive_good_evals >= self.CONSECUTIVE_ADVANCE_REQUIRED:
                if win_rate >= 0.50:
                    self._advance_difficulty(0.05, win_rate)
                elif win_rate >= 0.40:
                    self._advance_difficulty(0.02, win_rate)
            elif win_rate >= 0.10:
                # Flow zone — maintain current difficulty
                pass
            elif win_rate >= 0.05:
                self._difficulty = max(self.MIN_DIFFICULTY, self._difficulty - 0.01)
            else:
                self._difficulty = max(self.MIN_DIFFICULTY, self._difficulty - 0.02)

            # Log any difficulty change
            if abs(self._difficulty - old_difficulty) > 1e-6:
                self._last_difficulty_change = self.num_timesteps
                direction = "▲" if self._difficulty > old_difficulty else "▼"
                print(f"  >> Difficulty: {old_difficulty:.2f} → {self._difficulty:.2f} {direction}  "
                      f"(win_rate={win_rate:.0%}, consec_good={self._consecutive_good_evals}, "
                      f"recent={len(self._recent_outcomes)}eps)")

            # ── Collapse detection & rollback ───────────────────────────
            # If difficulty was ever advanced above the floor and win_rate
            # has now collapsed, restore the healthy checkpoint.
            if (self._peak_difficulty > self.MIN_DIFFICULTY
                    and win_rate < self.COLLAPSE_WIN_RATE
                    and self._healthy_params is not None):
                self._rollback(win_rate)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _advance_difficulty(self, delta: float, win_rate: float):
        """Save a healthy checkpoint, then increase difficulty."""
        old = self._difficulty
        self._difficulty = min(1.0, self._difficulty + delta)

        # Save checkpoint BEFORE advancing (rollback safety net)
        ckpt_path = os.path.join(self._log_dir, f"healthy_checkpoint_diff_{old:.2f}.zip")
        self.model.save(ckpt_path)
        self._healthy_params = self.model.get_parameters()
        self._last_healthy_checkpoint = ckpt_path
        if self._difficulty > self._peak_difficulty:
            self._peak_difficulty = self._difficulty

        # Save entry snapshot for the NEW difficulty level
        entry_path = os.path.join(self._log_dir, f"entry_model_diff_{self._difficulty:.2f}.zip")
        self.model.save(entry_path)

        print(f"  >> Checkpoint saved: {ckpt_path}  "
              f"(weights preserved for rollback, consec_good={self._consecutive_good_evals})")
        print(f"  >> Entry snapshot: {entry_path}  "
              f"(initial weights for diff={self._difficulty:.2f})")

        # Reset consecutive counter after successful advance
        self._consecutive_good_evals = 0

    def _rollback(self, win_rate: float):
        """Catastrophic collapse detected — restore model weights and reset difficulty."""
        print(f"\n  !! CLIFF COLLAPSE DETECTED !!  win_rate={win_rate:.0%} < {self.COLLAPSE_WIN_RATE:.0%}")
        print(f"     Restoring model weights from: {self._last_healthy_checkpoint}")
        print(f"     Rolling difficulty back: {self._difficulty:.2f} → {self.MIN_DIFFICULTY:.2f}")

        # Restore model weights from the healthy checkpoint
        self.model.set_parameters(self._healthy_params)

        # Reset environment difficulty to the floor
        self._difficulty = self.MIN_DIFFICULTY
        self._peak_difficulty = self.MIN_DIFFICULTY
        self._last_difficulty_change = self.num_timesteps
        self._consecutive_good_evals = 0

        # Clear the sliding window so the new start isn't contaminated
        # by outcomes from the collapsed policy
        self._recent_outcomes.clear()

        print(f"     Rollback complete. Agent gets a clean second chance at diff={self.MIN_DIFFICULTY:.2f}.\n")


class CurriculumCallback(BaseCallback):
    """Handles 5-stage curriculum advancement with automatic evaluation.

    Enforces a minimum step count per stage to prevent premature advancement
    when the agent achieves high capture rates early (which would otherwise
    skip most of a stage's training, causing catastrophic forgetting).
    """

    MIN_STEPS_PER_STAGE = 40_000   # must train at least this many steps per stage

    def __init__(self, eval_env, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._log_dir = log_dir
        self._best_capture_rate = -1.0          # global best across all stages
        self._stage_best_capture_rate = -1.0     # best within current stage
        self._current_stage = 1.0
        self._stage_start_step = 0
        self._eval_metrics: list = []  # per-eval metrics for CSV

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Don't eval every rollout — only at multiples of EVAL_FREQ
        if self.num_timesteps % EVAL_FREQ > 2048:
            return

        # Evaluate
        successes, min_dists, intercept_times = 0, [], []
        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done = False
            ep_min_dist = 8000.0
            ep_intercept_time = 120.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if "min_dist" in info:
                    ep_min_dist = min(ep_min_dist, info["min_dist"])
            if info.get("reason") == "success":
                successes += 1
                # Get intercept time from the unwrapped SinglePursuitEnv
                base_env = self._eval_env.unwrapped
                ep_intercept_time = base_env._step_counter / 60.0  # CTRL_FREQ
            min_dists.append(ep_min_dist)
            intercept_times.append(ep_intercept_time)

        capture_rate = successes / EVAL_EPISODES
        avg_min_dist = np.mean(min_dists)
        avg_intercept_time = np.mean(intercept_times)

        self.logger.record("eval/capture_rate", capture_rate)
        self.logger.record("eval/avg_min_dist", avg_min_dist)

        print(f"\n  [Eval @ {self.num_timesteps:,} steps] "
              f"stage={self._current_stage:.1f} "
              f"capture_rate={capture_rate:.0%} "
              f"avg_min_dist={avg_min_dist:.0f}m "
              f"avg_intercept={avg_intercept_time:.1f}s")

        # Record metrics for CSV
        self._eval_metrics.append({
            "timesteps": self.num_timesteps,
            "stage": self._current_stage,
            "capture_rate": capture_rate,
            "avg_min_dist": avg_min_dist,
            "avg_intercept_time": avg_intercept_time,
        })

        # ── Per-stage best model ───────────────────────────────────────────
        # Save best model for the CURRENT stage (only when capture_rate > 0).
        # This preserves Stage 1.0's 87% model even after advancing to 1.5.
        if capture_rate > 0 and capture_rate > self._stage_best_capture_rate:
            self._stage_best_capture_rate = capture_rate
            stage_label = f"best_stage_{self._current_stage:.1f}".replace(".", "_")
            stage_path = os.path.join(self._log_dir, stage_label)
            self.model.save(stage_path)
            print(f"  -> New stage-best ({self._current_stage:.1f}) saved: "
                  f"{stage_path}  (capture_rate={capture_rate:.0%})")

        # ── Global best model (across all stages) ──────────────────────────
        if capture_rate > 0 and capture_rate > self._best_capture_rate:
            self._best_capture_rate = capture_rate
            best_path = os.path.join(self._log_dir, "best_model")
            self.model.save(best_path)
            print(f"  -> New global-best saved: {best_path}  "
                  f"(stage={self._current_stage:.1f}, capture_rate={capture_rate:.0%})")

        # ── Stage advancement ──────────────────────────────────────────────
        # Uses stage-best capture rate to decide readiness — if the agent
        # peaked early it can still advance once MIN_STEPS_PER_STAGE is met.
        steps_in_stage = self.num_timesteps - self._stage_start_step
        threshold = (TARGET_CAPTURE_RATE_STAGE_1_2
                     if self._current_stage < 2.0
                     else TARGET_CAPTURE_RATE_STAGE_2_3)
        met_steps = steps_in_stage >= self.MIN_STEPS_PER_STAGE
        met_threshold = self._stage_best_capture_rate >= threshold
        is_last = np.isclose(self._current_stage, CURRICULUM_STAGES[-1])

        if met_steps and met_threshold and not is_last:
            # Advance to next stage (0.5 increment)
            current_idx = next((i for i, s in enumerate(CURRICULUM_STAGES)
                                if np.isclose(s, self._current_stage)), None)
            if current_idx is None:
                print(f"  !! WARNING: stage={self._current_stage} not in CURRICULUM_STAGES, resetting to 1.0")
                self._current_stage = 1.0
                self._eval_env.curriculum_stage = 1.0
                return
            self._current_stage = CURRICULUM_STAGES[current_idx + 1]
            self._stage_start_step = self.num_timesteps
            print(f"  >> Advancing to curriculum stage {self._current_stage:.1f} "
                  f"(stage-best capture rate was {self._stage_best_capture_rate:.0%})")
            self._eval_env.curriculum_stage = self._current_stage
            self._stage_best_capture_rate = -1.0  # reset for new stage


class TacviewEvalCallback(BaseCallback):
    """Generate Tacview + trajectory plots for the best episode during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._last_eval_step = -EVAL_FREQ

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval_step < EVAL_FREQ:
            return
        self._last_eval_step = self.num_timesteps


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def train(seed: int = 0):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"single_pursuit_{timestamp}_s{seed}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    # Environment — agent makes decisions at 2 Hz via ActionRepeatWrapper
    # while FlightController and JSBSim continue at full rate internally.
    base_env = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    env = ResidualExpertWrapper(base_env)
    from src.environment.ablation_wrappers import ActionRepeatWrapper
    env = ActionRepeatWrapper(env, repeat_frames=5)
    env = Monitor(env, log_dir)

    eval_base = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    eval_env = ResidualExpertWrapper(eval_base)
    eval_env = ActionRepeatWrapper(eval_env, repeat_frames=5)

    # Print setup
    print(f"{'='*55}")
    print(f"Single-Agent Pursuit Training  |  JSBSim F-16")
    print(f"  Action space:   {env.action_space}")
    print(f"  Observation:    {env.observation_space.shape}")
    print(f"  Total steps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Monitor:        tensorboard --logdir {log_dir}")
    print(f"{'='*55}\n")

    # PPO model — agent controls d_heading, d_alt, d_speed via FC
    # gSDE (generalised State-Dependent Exploration) produces smooth,
    # state-consistent exploration trajectories instead of per-step white noise,
    # which is critical for 10 Hz control without jitter.
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,             # doubled for BVR — smoother advantage estimates
        batch_size=512,           # doubled to match n_steps
        n_epochs=10,
        gamma=0.998,  # raised for 10Hz — matches 2Hz γ=0.99 effective horizon
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,  # moderate entropy — balanced with long-duration gSDE exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=10,  # 1.0s exploration duration — lets JSBSim aerodynamics respond
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=0.0,    # σ=1.0 — full exploration; ent_coef=0 lets it decay naturally
        ),
    )

    # Train
    try:
        curriculum_cb = CurriculumCallback(eval_env, log_dir)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=curriculum_cb,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")

    # Save final
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    # Save a simple model.zip for evaluate script
    model.save(os.path.join(log_dir, "model"))
    print(f"Simple model saved → {os.path.join(log_dir, 'model')}.zip")

    # Save eval metrics CSV
    import csv
    eval_csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                                "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(curriculum_cb._eval_metrics)
    print(f"Eval metrics saved → {eval_csv_path}")

    # ── Final evaluation with Tacview ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final Evaluation with Tacview (20 episodes)")

    tacview_env = SinglePursuitEnv(curriculum_stage=eval_env.curriculum_stage,
                                   record_tacview=True)
    os.makedirs("results/single_pursuit", exist_ok=True)

    successes = 0
    min_dists = []
    best_ep_reward = -float("inf")
    best_ep_idx = 0
    # Wrap env so RL actions are treated as residuals on top of PN expert
    expert_wrapper = ResidualExpertWrapper(tacview_env)

    for ep in range(EVAL_EPISODES):
        obs, _ = tacview_env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = expert_wrapper.step(action)
            done = terminated or truncated
            total_r += rew
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])

        reason = info.get("reason", "timeout")
        if reason == "success":
            successes += 1
        min_dists.append(ep_min_dist)

        if total_r > best_ep_reward:
            best_ep_reward = total_r
            best_ep_idx = ep

        print(f"  Ep {ep+1:2d}: {reason:15s}  reward={total_r:+7.1f}  "
              f"min_dist={info.get('min_dist', -1):.0f}m")

    # Export Tacview (frames from last episode run)
    tacview_path = os.path.join("results", "single_pursuit",
                                "single_pursuit_engagement.txt.acmi")
    tacview_env.export_tacview(tacview_path)

    capture_rate = successes / EVAL_EPISODES
    print(f"\n  Capture rate: {capture_rate:.0%}")
    print(f"  Avg min dist: {np.mean(min_dists):.0f} ± {np.std(min_dists):.0f}m")
    print(f"  Tacview → {os.path.abspath(tacview_path)}")

    # Generate trajectory plot using matplotlib
    try:
        _plot_best_episode(tacview_env, best_ep_idx)
    except Exception as e:
        print(f"  Plotting skipped: {e}")

    print("\nTraining complete.  Monitor:  tensorboard --logdir", log_dir)


def _plot_best_episode(env: SinglePursuitEnv, ep_idx: int) -> None:
    """Quick 3D trajectory + altitude plot for the best episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = env._tacview_frames
    if not frames:
        return

    a_lat = np.array([f["pursuer"]["lat_deg"] for f in frames])
    a_lon = np.array([f["pursuer"]["lon_deg"] for f in frames])
    a_alt = np.array([f["pursuer"]["alt_m"] for f in frames])
    t_lat = np.array([f["target"]["lat_deg"] for f in frames])
    t_lon = np.array([f["target"]["lon_deg"] for f in frames])
    t_alt = np.array([f["target"]["alt_m"] for f in frames])

    # Convert to approximate meters from reference
    ref_lat, ref_lon = 30.0, 120.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref_lat))

    a_x = (a_lat - ref_lat) * m_per_deg_lat
    a_y = (a_lon - ref_lon) * m_per_deg_lon
    t_x = (t_lat - ref_lat) * m_per_deg_lat
    t_y = (t_lon - ref_lon) * m_per_deg_lon

    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(a_x, a_y, a_alt, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax3d.plot(t_x, t_y, t_alt, "b-", lw=1.5, alpha=0.8, label="Target")
    ax3d.scatter(a_x[0], a_y[0], a_alt[0], color="darkred", s=80, marker="o")
    ax3d.scatter(a_x[-1], a_y[-1], a_alt[-1], color="red", s=80, marker="x")
    ax3d.scatter(t_x[0], t_y[0], t_alt[0], color="darkblue", s=80, marker="o")
    ax3d.scatter(t_x[-1], t_y[-1], t_alt[-1], color="blue", s=80, marker="x")
    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

    ax2d.plot(a_x, a_y, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax2d.plot(t_x, t_y, "b-", lw=1.5, alpha=0.8, label="Target")
    ax2d.scatter(a_x[0], a_y[0], color="darkred", s=80, marker="o")
    ax2d.scatter(a_x[-1], a_y[-1], color="red", s=80, marker="x")
    ax2d.scatter(t_x[0], t_y[0], color="darkblue", s=80, marker="o")
    ax2d.scatter(t_x[-1], t_y[-1], color="blue", s=80, marker="x")
    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title("Top-Down View")
    ax2d.legend()
    ax2d.axis("equal")

    out_path = "results/single_pursuit/single_pursuit_trajectory_best.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    # Altitude profile
    fig2, ax_alt = plt.subplots(figsize=(10, 4))
    times = np.arange(len(frames)) * 0.5
    ax_alt.plot(times, a_alt, "r-", lw=1.5, label="Pursuer")
    ax_alt.plot(times, t_alt, "b-", lw=1.5, label="Target")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude Profile")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)
    alt_path = "results/single_pursuit/single_pursuit_trajectory_best_altitude.png"
    fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Plots saved → {out_path}")


if __name__ == "__main__":
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    # Set seeds for reproducibility
    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    TOTAL_TIMESTEPS = args.steps

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    train(seed=args.seed)


def train_with_config(
    seed: int = 0,
    log_dir: str = "",
    learning_rate: float = 1e-4,
    ent_coef: float = 0.015,  # moderate entropy — balanced with long-duration gSDE
    net_arch_pi: list | None = None,
    n_steps: int = 2048,
    total_timesteps: int = 5_000_000,  # 5M steps for full curriculum exploration
    batch_size: int = 256,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> str:
    """Train with explicit hyperparameters, return path to best model.

    Used by the sweep runner to inject hyperparameter values.
    """
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    if net_arch_pi is None:
        net_arch_pi = [128, 128]

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    base_env = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    env = ResidualExpertWrapper(base_env)
    env = Monitor(env, log_dir)

    eval_base = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    eval_env = ResidualExpertWrapper(eval_base)

    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=True,
        sde_sample_freq=10,  # 1.0s exploration duration — lets JSBSim aerodynamics respond
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=net_arch_pi, vf=net_arch_pi),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=0.0,
        ),
    )

    # Use global config values for eval
    cb = CurriculumCallback(eval_env, log_dir)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    best_path = os.path.join(log_dir, "best_model")
    model.save(best_path)
    model.save(os.path.join(log_dir, "model"))

    # Save eval CSV
    import csv
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                                "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(cb._eval_metrics)

    return best_path + ".zip"
