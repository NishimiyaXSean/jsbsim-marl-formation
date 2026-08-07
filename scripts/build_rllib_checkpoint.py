"""Wrap a plain BC state-dict (.pth) into an RLlib PPO checkpoint.

Loads the weights (encoder.*, action_heads.*) into a ShootMaskModel policy so
the existing visualization / eval / export tools that expect an RLlib
checkpoint can be reused (viz_policy_trajectories, export_acmi_episodes,
eval_shoot_1v1, ...). The value net stays randomly initialized.

Usage:
  python scripts/build_rllib_checkpoint.py \
      --weights data/expert/shoot_bc_asap_distilled.pth \
      --out marl_runs/shoot_bc_asap_distilled/checkpoints/best
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.models.shoot_mask_model import ShootMaskModel

ENV_NAME = "jsbsim_shoot_1v1"


def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="marl_runs/shoot_bc_asap_distilled/checkpoints/best")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location="cpu")
    bc_sd = ck["state_dict"] if "state_dict" in ck else ck

    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")
    register_env(ENV_NAME, lambda c: env_creator(c))
    ModelCatalog.register_custom_model("shoot_mask_model", ShootMaskModel)
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config={
            "difficulty_level": 0.0, "obs_include_closure": True})
        .framework("torch")
        .training(lr=1e-5, train_batch_size=1024, minibatch_size=256,
                  num_epochs=1, model={"custom_model": "shoot_mask_model"})
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .debugging(log_level="ERROR", seed=42)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
    )
    algo = config.build()
    model = algo.get_policy("default_policy").model
    missing = []
    for k, v in bc_sd.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v.to(model.state_dict()[k].device))
        else:
            missing.append(k)
    print(f"[ckpt] loaded {len(bc_sd) - len(missing)} keys, "
          f"missing={sorted(missing)[:6]}, value_net untouched")
    os.makedirs(args.out, exist_ok=True)
    algo.save(os.path.abspath(args.out))
    print(f"[ckpt] saved: {args.out}")
    ray.shutdown()


if __name__ == "__main__":
    main()
