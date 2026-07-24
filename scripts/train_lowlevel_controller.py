"""Train LAG-style low-level neural flight controller from scratch.

RL policy learns to directly output control surface deflections
(aileron/elevator/rudder/throttle) from 12-dim body-frame observation.

This is the equivalent of training LAG's BaselineActor on our own JSBSim 1.3.1.
Once trained, the policy weights can be loaded as a NeuralFlightController
for hierarchical tasks.

Usage:
    conda activate marl_env
    python scripts/train_lowlevel_controller.py --iterations 500 --seed 42

Architecture:
    RL Policy (MLP + LSTM) → MultiDiscrete([21,21,21,15])
    → ControlSurfaces → SafetyInterceptor → JSBSim F-16
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
import numpy as np, ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment.base_env import BaseEnv
from src.environment.heading_task import HeadingTrackingTask

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore"); logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

ENV_NAME = "lowlevel_control_v1"


def make_env(env_config: dict | None = None):
    config = env_config or {}
    task = HeadingTrackingTask(config)
    return BaseEnv(task=task, env_config=config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-heading", type=float, default=90.0)
    p.add_argument("--use-lstm", action="store_true", default=True)
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"rllib_lowlevel_{timestamp}_s{args.seed}"
    project_root = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(f"{project_root}/checkpoints", exist_ok=True)

    ray.init(ignore_reinit_error=True, num_cpus=4)
    register_env(ENV_NAME, make_env)

    env_config = {"target_heading": args.target_heading}

    temp_env = make_env(env_config)
    print(f"Obs: {temp_env.observation_space['p0']}")
    print(f"Act: {temp_env.action_space['p0']}  ({21*21*21*15:,} combos)")
    temp_env.close()

    # Model: large MLP + optional LSTM for temporal flight dynamics
    model_cfg = {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "tanh"}
    if args.use_lstm:
        model_cfg["use_lstm"] = True
        model_cfg["lstm_cell_size"] = 128
        model_cfg["max_seq_len"] = 60

    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .training(
            lr=1e-4, gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.02, vf_clip_param=1000.0, grad_clip=1.0,
            train_batch_size=4096, minibatch_size=256, num_epochs=10,
            model=model_cfg,
        )
        .env_runners(num_env_runners=2, num_envs_per_env_runner=2)
        .resources(num_gpus=1)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .debugging(log_level="WARN", seed=args.seed)
    )

    algo = config.build()
    print(f"\nTraining: {run_name}  target_heading={args.target_heading}°  "
          f"lstm={args.use_lstm}  model=512x512x256\n")

    best_reward = -float("inf")
    for i in range(args.iterations):
        result = algo.train()
        reward = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        length = result.get("env_runners", {}).get("episode_len_mean", 0)
        fps = result.get("env_runners", {}).get("num_env_steps_sampled_per_second", 0)
        entropy = result.get("info", {}).get("learner", {}).get("default_policy", {})\
                      .get("learner_stats", {}).get("entropy", float("nan"))
        print(f"[iter {i:4d}] rew={reward:+.4f}  len={length:.0f}  "
              f"ent={entropy:.2f}  fps={fps:.0f}")

        if not np.isnan(reward) and reward > best_reward:
            best_reward = reward
            best_ckpt = algo.save(f"{project_root}/checkpoints/best")

        if (i + 1) % 50 == 0:
            algo.save(f"{project_root}/checkpoints/checkpoint_{i:04d}")

    algo.save(f"{project_root}/checkpoints/checkpoint_final")

    # ── Export policy weights as standard PyTorch .pt file ──────────────
    # RLlib algo.save() produces a directory (rllib_checkpoint.json + algorithm_state.pkl),
    # NOT a torch.load()-compatible state dict. Must explicitly extract.
    import torch as _torch
    policy = algo.get_policy("default_policy")
    export_path = f"{project_root}/checkpoints/best_model.pt"
    _torch.save(policy.model.state_dict(), export_path)
    print(f"\nBest: {best_reward:.4f} → {project_root}/checkpoints/best")
    print(f"Model export: {export_path}  (loadable with torch.load(weights_only=True))")
    ray.shutdown()


if __name__ == "__main__":
    main()
