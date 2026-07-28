"""Minimal RLlib PPO training for 1v1 shoot task — end-to-end verification.

Usage:
  python scripts/_train_shoot_1v1.py [--iterations 50] [--checkpoint PATH]
"""

import os, sys, warnings, logging, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCN
from ray.rllib.utils.annotations import override

import torch
import torch.nn as nn

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import (
    SingleCombatShootTask, OBS_DIM, GLOBAL_DIM, N_ACTIONS,
    N_SPEED_DELTA, N_HEADING_DELTA, N_ALT_DELTA, N_FIRE,
)

ENV_NAME = "jsbsim_shoot_1v1_v0"


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom TorchModelV2 for MultiDiscrete + Dict observation
# ═══════════════════════════════════════════════════════════════════════════════

class Shoot1v1Model(TorchModelV2, nn.Module):
    """Simple MLP policy for 1v1 shoot task.

    Observation: Dict{"obs"(24), "global_state"(14), "action_mask"(13)}
    Action: MultiDiscrete([3, 5, 3, 2]) → 4 heads
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._action_dims = [N_SPEED_DELTA, N_HEADING_DELTA, N_ALT_DELTA, N_FIRE]
        self._total_actions = sum(self._action_dims)

        # Encoder: process "obs" flat input
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Value branch (shares encoder features)
        self.value_net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Action heads (one per MultiDiscrete dimension)
        self.action_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in self._action_dims
        ])

        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_dict = input_dict["obs"]
        flat_obs = obs_dict["obs"]  # [B, 24] or [24]
        global_state = obs_dict.get("global_state", None)

        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        feat = self.encoder(flat_obs)  # [B, 128]
        self._features = feat

        # Action logits — concat all heads
        logits = torch.cat([head(feat) for head in self.action_heads], dim=1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_net(self._features).squeeze(1)

    @override(TorchModelV2)
    def get_initial_state(self):
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  Env factory
# ═══════════════════════════════════════════════════════════════════════════════

def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Target evasion difficulty: 0.0=straight, 0.3=gentle, 0.5=aggressive")
    args = parser.parse_args()

    # ── Curriculum config ───────────────────────────────────────────────
    env_config = {"difficulty_level": args.difficulty}

    # ── Register env and model ──────────────────────────────────────────
    register_env(ENV_NAME, lambda c: env_creator(c))
    ModelCatalog.register_custom_model("shoot_1v1_mlp", Shoot1v1Model)

    # ── Project root for Ray workers ────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")
    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")

    # ── Build config ────────────────────────────────────────────────────
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            train_batch_size=2048,
            minibatch_size=128,
            num_epochs=10,
            model={"custom_model": "shoot_1v1_mlp"},
        )
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .resources(num_gpus=0)
        .debugging(log_level="WARN", seed=args.seed)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    # ── Resume if requested ────────────────────────────────────────────
    if args.checkpoint:
        algo.restore(os.path.abspath(args.checkpoint))
        print(f"Resumed from {args.checkpoint}")

    # ── Training loop ───────────────────────────────────────────────────
    best_reward = -float("inf")
    output_dir = args.output or f"marl_runs/shoot_1v1_test_s{args.seed}"
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    print(f"Training 1v1 shoot — {args.iterations} iters, lr={args.lr}")
    print(f"Output: {output_dir}")

    for i in range(args.iterations):
        result = algo.train()
        rew = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        length = result.get("env_runners", {}).get("episode_len_mean", 0)

        if i % 10 == 0 or i < 5:
            print(f"  iter {i:3d}: rew={rew:+.0f}  len={length:.0f}")

        if not np.isnan(rew) and rew > best_reward:
            best_reward = rew
            algo.save(f"{output_dir}/checkpoints/best")

    algo.save(f"{output_dir}/checkpoints/final")
    print(f"\nDone. Best reward: {best_reward:+.0f}")
    print(f"Checkpoints: {output_dir}/checkpoints/")

    ray.shutdown()


if __name__ == "__main__":
    main()
