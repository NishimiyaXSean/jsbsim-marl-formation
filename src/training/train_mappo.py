"""MAPPO training script for 1v1 JSBSim air combat.

Usage:
    python -m src.training.train_mappo
"""

import datetime
import os
import sys

import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.environment.air_combat_env import AirCombatEnv
from src.models.mappo_model import MAPPOModel
from src.training.callbacks import AirCombatCallbacks

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


def env_creator(config: dict):
    return AirCombatEnv(gui=False, record_tacview=False)


def train(
    train_iterations: int = 500,
    eval_interval: int = 10,
    test_episodes: int = 50,
    target_success_rate: float = 0.70,
    checkpoint_freq: int = 20,
    resume_checkpoint: str | None = None,
):
    # ── Setup ──────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_root = os.path.abspath(f"./marl_runs/mappo_run_{timestamp}")
    os.makedirs(f"{project_root}/checkpoints", exist_ok=True)

    ray.init(ignore_reinit_error=True)

    env_name = "air_combat_1v1"
    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model("mappo_ctde_model", MAPPOModel)

    # Probe env for spaces
    temp_env = env_creator({})
    obs_space = temp_env.observation_spaces["attacker_0"]
    act_space = temp_env.action_spaces["attacker_0"]
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    # ── PPO Config ─────────────────────────────────────────────────────
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(
            num_env_runners=2,
            sample_timeout_s=300,
            rollout_fragment_length=256,
        )
        .callbacks(AirCombatCallbacks)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={
                "policy_attacker": (None, obs_space, act_space, {}),
                "policy_evader": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs:
                "policy_attacker" if agent_id == "attacker_0" else "policy_evader",
            policies_to_train=["policy_attacker", "policy_evader"],
        )
        .training(
            model={"custom_model": "mappo_ctde_model"},
            train_batch_size=8192,
            minibatch_size=1024,
            lr=1e-5,
            entropy_coeff=0.01,
            clip_param=0.2,
            vf_clip_param=1000.0,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.2,
        )
    )

    algo = config.build()

    # ── Resume ─────────────────────────────────────────────────────────
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        algo.restore(resume_checkpoint)

    # ── Training loop ──────────────────────────────────────────────────
    current_stage = 1
    best_success_rate = -0.01
    test_env = AirCombatEnv(gui=False, record_tacview=False)
    test_env.set_curriculum_stage(current_stage)

    algo.env_runner_group.foreach_env(lambda env: env.set_curriculum_stage(current_stage))

    print(f"{'='*50}")
    print("MAPPO 1v1 Air Combat Training (JSBSim F-16)")
    print(f"Stage: {current_stage}  |  Iterations: {train_iterations}")
    print(f"{'='*50}\n")

    try:
        for i in range(train_iterations):
            result = algo.train()
            real_iter = result.get("training_iteration", i + 1)

            stats = result.get("env_runners", result)
            policy_rewards = stats.get("policy_reward_mean", {})
            reward_A = policy_rewards.get("policy_attacker", 0.0)
            reward_E = policy_rewards.get("policy_evader", 0.0)

            hist_stats = stats.get("hist_stats", {})
            episodes_this_iter = stats.get("episodes_this_iter", 0)

            def _recent_mean(lst, n):
                if n <= 0 or not lst:
                    return 0.0
                return sum(lst[-n:]) / len(lst[-n:])

            success_rate = _recent_mean(hist_stats.get("rate_success", []), episodes_this_iter)
            crash_rate = _recent_mean(hist_stats.get("rate_crash", []), episodes_this_iter)
            oob_rate = _recent_mean(hist_stats.get("rate_oob", []), episodes_this_iter)
            timeout_rate = _recent_mean(hist_stats.get("rate_timeout", []), episodes_this_iter)

            # Entropy
            learner_info = result.get("info", {}).get("learner", {})
            attacker_learner = learner_info.get("policy_attacker", {})
            learner_stats = attacker_learner.get("learner_stats", attacker_learner)
            entropy = learner_stats.get("entropy", 0.0)

            print(
                f"Iter {real_iter:03d} | "
                f"Rwd A/E: {reward_A:6.1f}/{reward_E:6.1f} | "
                f"Kill:{success_rate*100:5.1f}% Crash:{crash_rate*100:5.1f}% "
                f"OOB:{oob_rate*100:5.1f}% Timeout:{timeout_rate*100:5.1f}% | "
                f"Entropy:{entropy:.4f} | Episodes:{episodes_this_iter}"
            )

            # ── Eval ───────────────────────────────────────────────────
            if (i + 1) % eval_interval == 0:
                print(f"\n--- Stage {current_stage} Eval ({test_episodes} episodes) ---")
                success_count = 0
                for _ in range(test_episodes):
                    obs, _ = test_env.reset()
                    terminated = {"__all__": False}
                    truncated = {"__all__": False}
                    final_reason = "timeout"
                    while not (terminated["__all__"] or truncated["__all__"]):
                        action_A = algo.compute_single_action(
                            obs["attacker_0"], policy_id="policy_attacker", explore=False
                        )
                        actions = {"attacker_0": action_A}
                        if "evader_0" in obs:
                            # Fixed evader target for objective assessment
                            if current_stage == 1:
                                actions["evader_0"] = np.zeros(4, dtype=np.float32)
                            elif current_stage == 2:
                                actions["evader_0"] = np.array([0.5, 0.0, -0.5, 0.0], dtype=np.float32)
                            else:
                                actions["evader_0"] = algo.compute_single_action(
                                    obs["evader_0"], policy_id="policy_evader", explore=False
                                )
                        obs, _, terminated, truncated, infos = test_env.step(actions)
                        if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                            final_reason = infos["attacker_0"]["reason"]
                    if final_reason == "success":
                        success_count += 1

                eval_rate = success_count / test_episodes
                print(f"Eval success rate: {eval_rate*100:.1f}% ({success_count}/{test_episodes})\n")

                # ── Stage advancement ──────────────────────────────────
                if eval_rate >= target_success_rate and current_stage < 3:
                    old_stage = current_stage
                    current_stage += 1
                    print(f">>> Advancing to Stage {current_stage}!")
                    checkpoint_path = os.path.join(
                        project_root, "checkpoints",
                        f"checkpoint_stage_{old_stage}_to_{current_stage}_iter_{real_iter:03d}"
                    )
                    algo.save(checkpoint_path)
                    best_success_rate = -0.01  # Reset baseline for new stage
                    algo.env_runner_group.foreach_env(lambda env: env.set_curriculum_stage(current_stage))
                    test_env.set_curriculum_stage(current_stage)

            # ── Save best ───────────────────────────────────────────────
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_path = os.path.join(project_root, "checkpoints", f"checkpoint_best_iter_{real_iter:03d}")
                algo.save(best_path)
                print(f"  -> New best model saved: {best_path}")

            # ── Periodic save ───────────────────────────────────────────
            if (i + 1) % checkpoint_freq == 0:
                save_path = os.path.join(project_root, "checkpoints", f"checkpoint_{real_iter:06d}")
                algo.save(save_path)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving final checkpoint...")
        algo.save(os.path.join(project_root, "checkpoints", "checkpoint_final"))

    finally:
        ray.shutdown()
        print("Training complete.")


if __name__ == "__main__":
    train()
