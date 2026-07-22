"""Render trained SinglePursuitTask to Tacview ACMI.

Usage:
    conda activate marl_env
    python scripts/render_single_pursuit.py --ckpt marl_runs/rllib_pursuit_XXXX_s42/checkpoints/best
"""

from __future__ import annotations

import argparse, os, sys, warnings, logging
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore"); logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

ENV_NAME = "single_pursuit_v1"

def make_env(c): return BaseEnv(task=SinglePursuitTask(c), env_config=c)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", default="results/pursuit_result.acmi")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=1)
    register_env(ENV_NAME, make_env)

    print(f"Loading: {args.ckpt}")
    algo = PPO.from_checkpoint(os.path.abspath(args.ckpt))

    cfg = {"difficulty_level": args.difficulty}
    env = make_env(cfg)
    obs, _ = env.reset(seed=args.seed)
    env.enable_acmi_logging(args.output)
    env.log_acmi_step()

    total_r = 0.0
    for step in range(args.max_steps):
        acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
        obs, rewards, terms, truncs, info = env.step(acts)
        env.log_acmi_step()
        for r in rewards.values(): total_r += r
        if terms.get("__all__") or truncs.get("__all__"):
            print(f"End step {step+1}: {info.get('p0',{}).get('termination_reason','?')}")
            break
    else:
        print(f"Timeout at {args.max_steps} steps")

    print(f"Total reward: {total_r:.0f}  → {args.output}")
    env.close()

if __name__ == "__main__": main()
