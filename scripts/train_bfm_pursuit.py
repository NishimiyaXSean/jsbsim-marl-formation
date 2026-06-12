"""Train single-agent pursuit using BFM discrete actions + Stable-Baselines3 PPO.

The RL agent controls one F-16 via 13 discrete Basic Fighter Maneuvers.
A scripted target flies straight.  Curriculum has 3 stages of increasing difficulty.

Key design decisions:
- BFM actions use the full FlightEnvelope + BFMAutopilot pipeline (lambda-g FCS)
- The agent sees the same 19-dim local observation as the MAPPO setup
- Evader uses fixed action 0 (level flight) for objective evaluation
- Stage 1: target in front, slow, straight — learn to close distance
- Stage 2: target weaves gently — learn to track
- Stage 3: target evades — learn pursuit against maneuvering target

Usage:
    conda activate jsbsim_rl
    JSBSIM_DEBUG=0 python scripts/train_bfm_pursuit.py
"""

from __future__ import annotations

import datetime
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.air_combat_env import AirCombatEnv
from src.dynamics.bfm_actions import NUM_PURSUIT_ACTIONS, describe_pursuit_action

# ═══════════════════════════════════════════════════════════════════════════════
#  Training config
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 20
EVAL_FREQ = 10_000
TARGET_CAPTURE_RATE = 0.70


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-agent wrapper: AirCombatEnv → SB3 Gymnasium
# ═══════════════════════════════════════════════════════════════════════════════

class BFMPursuitWrapper(gym.Wrapper):
    """Wrap AirCombatEnv (BFM mode) as single-agent Gymnasium for SB3.

    The RL agent controls the attacker via 13 discrete BFM actions.
    The evader flies straight (BFM action 0 = level flight).
    """

    def __init__(self, env: AirCombatEnv):
        super().__init__(env)
        # SB3 sees: flat 19-dim observation + Discrete(13) action
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(19,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(NUM_PURSUIT_ACTIONS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["attacker_0"]["obs"], info

    def step(self, action: int):
        # Evader always flies straight (BFM action 0) for objective eval
        evader_action = 0
        actions = {"attacker_0": int(action), "evader_0": evader_action}
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return (
            obs.get("attacker_0", {}).get("obs", np.zeros(19, dtype=np.float32)),
            rewards.get("attacker_0", 0.0),
            terminated.get("attacker_0", terminated.get("__all__", False)),
            truncated.get("attacker_0", truncated.get("__all__", False)),
            infos.get("attacker_0", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class BFMEvalCallback(BaseCallback):
    """Evaluate on fixed evader (action 0), track capture rate, advance curriculum."""

    def __init__(self, eval_env: AirCombatEnv, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._log_dir = log_dir
        self._best_capture_rate = -1.0
        self._current_stage = 1

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps % EVAL_FREQ > 2048:
            return

        successes, min_dists, reasons = 0, [], {}
        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done = False
            ep_min_dist = 10000.0
            reason = "timeout"
            while not done:
                # Use the trained attacker policy, evader flies straight
                att_action, _ = self.model.predict(obs["attacker_0"]["obs"], deterministic=True)
                actions = {"attacker_0": int(att_action), "evader_0": 0}
                obs, rewards, terminated, truncated, infos = self._eval_env.step(actions)
                done = terminated["__all__"] or truncated["__all__"]
                if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                    reason = infos["attacker_0"]["reason"]
                # Track min distance from attacker to evader
                dist = np.linalg.norm(
                    self._eval_env.attacker.position_ned - self._eval_env.evader.position_ned
                )
                ep_min_dist = min(ep_min_dist, dist)
            if reason == "success":
                successes += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            min_dists.append(ep_min_dist)

        capture_rate = successes / EVAL_EPISODES
        avg_min_dist = np.mean(min_dists)

        self.logger.record("eval/capture_rate", capture_rate)
        self.logger.record("eval/avg_min_dist", avg_min_dist)

        print(f"\n  [Eval @ {self.num_timesteps:,} steps] "
              f"stage={self._current_stage} "
              f"capture={capture_rate:.0%} "
              f"avg_min_dist={avg_min_dist:.0f}m "
              f"reasons={reasons}")

        if capture_rate > self._best_capture_rate:
            self._best_capture_rate = capture_rate
            best_path = os.path.join(self._log_dir, "best_model")
            self.model.save(best_path)
            print(f"  -> New best model: {best_path}")

        # Curriculum advancement
        if capture_rate >= TARGET_CAPTURE_RATE and self._current_stage < 3:
            self._current_stage += 1
            print(f"  >> Advancing to Stage {self._current_stage}!")
            self._eval_env.set_curriculum_stage(self._current_stage)
            self._best_capture_rate = -1.0


# ═══════════════════════════════════════════════════════════════════════════════

def train():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.abspath(f"./marl_runs/bfm_pursuit_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Training env
    env = AirCombatEnv(gui=False, record_tacview=False, action_mode="pursuit")
    env.set_curriculum_stage(1)
    env = BFMPursuitWrapper(env)
    env = Monitor(env, log_dir)

    # Eval env
    eval_env_raw = AirCombatEnv(gui=False, record_tacview=False, action_mode="pursuit")
    eval_env_raw.set_curriculum_stage(1)

    print(f"{'='*55}")
    print(f"BFM Pursuit Training  |  JSBSim F-16  |  13 BFM Actions")
    print(f"  Action space:   Discrete({NUM_PURSUIT_ACTIONS}) [pursuit subset]")
    print(f"  Observation:    Box(19,)")
    print(f"  Total steps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Log dir:        {log_dir}")
    print(f"{'='*55}")
    for i in range(NUM_PURSUIT_ACTIONS):
        print(f"    {i}: {describe_pursuit_action(i)}")
    print(f"{'='*55}\n")

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
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

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=BFMEvalCallback(eval_env_raw, log_dir),
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Save
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model -> {final_path}.zip")

    # ── Final eval with Tacview ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final Evaluation (20 episodes, Tacview)")

    tacview_env = AirCombatEnv(gui=False, record_tacview=True, action_mode="pursuit")
    tacview_env.set_curriculum_stage(eval_env_raw.curriculum_stage)
    os.makedirs("results/bfm_pursuit", exist_ok=True)

    successes = 0
    for ep in range(EVAL_EPISODES):
        obs, _ = tacview_env.reset()
        done = False
        reason = "timeout"
        while not done:
            att_action, _ = model.predict(obs["attacker_0"]["obs"], deterministic=True)
            actions = {"attacker_0": int(att_action), "evader_0": 0}
            obs, rewards, terminated, truncated, infos = tacview_env.step(actions)
            done = terminated["__all__"] or truncated["__all__"]
            if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                reason = infos["attacker_0"]["reason"]
        if reason == "success":
            successes += 1
        print(f"  Ep {ep+1:2d}: {reason}")

    tacview_path = "results/bfm_pursuit/bfm_pursuit_final.txt.acmi"
    tacview_env.export_tacview(tacview_path)
    print(f"\n  Capture rate: {successes}/{EVAL_EPISODES} = {100*successes/EVAL_EPISODES:.0f}%")
    print(f"  Tacview -> {os.path.abspath(tacview_path)}")
    print(f"\n  Monitor: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    train()
