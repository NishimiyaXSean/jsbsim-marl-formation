"""PPO P0 gate: RLlib zero-update load consistency for the frozen BC.

Loads the archived BC weights into a real RLlib PPO policy (ShootMaskModel),
performs NO gradient updates, and checks:
  1. per-head logits on a fixed obs batch match the standalone BC model;
  2. closed-loop results on the same 500 seeds reproduce the archived
     paired eval's BC records (kill/lost/launches per seed).

If either check fails, something in the RLlib loading path (preprocessing,
optimizer, exploration, model construction) silently changes the policy.

Usage:
  python scripts/rllib_load_bc_gate.py --seeds 500
"""

from __future__ import annotations

import argparse
import json
import math
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
from scripts.train_shoot_bc import BCShootPolicy, ACTION_DIMS, MASK_NEG

ENV_NAME = "jsbsim_shoot_1v1"
MAX_STEPS = 1500


def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


def decode_action(raw):
    if isinstance(raw, (int, np.integer)):
        flat = int(raw)
    else:
        arr = np.asarray(raw).reshape(-1)
        if arr.size == len(ACTION_DIMS):
            return arr.astype(np.int64)
        flat = int(arr[0])
    out = np.zeros(len(ACTION_DIMS), dtype=np.int64)
    for i, d in enumerate(ACTION_DIMS):
        out[i] = flat % d
    return out


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])


def launch_geometry(env):
    ps = env.pursuers[0]
    tgt = env.targets[0]
    p_pos = ps.aircraft.position_ned
    t_pos = tgt.aircraft.position_ned
    los = t_pos - p_pos
    dist = float(np.linalg.norm(los))
    los_dir = los / max(dist, 1e-6)
    p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
    t_fwd = compute_forward_vector(tgt.aircraft.rpy_rad)
    ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    return {"range_m": dist, "ata_deg": ata_deg, "closure_mps": closure}


def classify_launch(g):
    if g["closure_mps"] < 0 and g["ata_deg"] < 10.0 and 2000.0 < g["range_m"] < 4000.0:
        return "premium"
    if g["closure_mps"] < 0:
        return "good"
    return "bad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--baseline", default="results/shoot_eval/paired_bc_vs_expert_500.json")
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="marl_runs/shoot_bc_gate_s42/checkpoints/best")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location="cpu")
    bc_sd = {k: v for k, v in ck["state_dict"].items()}

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
    policy = algo.get_policy("default_policy")
    model = policy.model

    # ── load BC weights into the policy (encoder + action heads only) ──────
    missing = set()
    for k, v in bc_sd.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)
        else:
            missing.add(k)
    value_loaded = [k for k in model.state_dict() if k.startswith("value_net")]
    print(f"[gate] loaded {len(bc_sd) - len(missing)} policy keys; "
          f"value_net untouched ({len(value_loaded)} keys), missing={sorted(missing)[:5]}")
    model.to(device)

    # ── 1. per-head logits match on a fixed obs batch ──────────────────────
    bc_model = BCShootPolicy().to(device)
    bc_model.load_state_dict(bc_sd)
    bc_model.eval()
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, (128, 41)).astype(np.float32)
    x[:, 30:] = rng.integers(0, 2, (128, 11)).astype(np.float32)
    with torch.no_grad():
        bc_l = torch.cat(bc_model(torch.tensor(x[:, :30], device=device)), dim=1)
        mb = torch.tensor(x[:, 30:], device=device)
        off = 0
        for d in ACTION_DIMS:
            bc_l[:, off:off + d] = bc_l[:, off:off + d] \
                + (1.0 - mb[:, off:off + d]) * MASK_NEG
            off += d
        rl_l, _ = model({"obs": torch.tensor(x, device=device)}, [], None)
    logit_diff = float((bc_l - rl_l).abs().max())
    logit_ok = logit_diff < 1e-4
    print(f"[gate] per-head logits: max diff={logit_diff:.2e} -> "
          f"{'PASS' if logit_ok else 'FAIL'}")

    # ── 2. zero-update closed-loop on the same seeds ───────────────────────
    baseline = json.load(open(args.baseline))["per_seed"]
    matches = {"kill": 0, "lost": 0, "launches": 0}
    totals = {"kill": 0, "lost": 0, "launches": 0, "hits": 0}
    kills = lost = 0
    n = 0
    for s in range(args.start_seed, args.start_seed + args.seeds):
        env = BaseEnv(task=SingleCombatShootTask(
            {"difficulty_level": 0.0, "obs_include_closure": True}))
        obs, _ = env.reset(seed=s)
        reason = "timeout"
        launches = hits = 0
        quality = {"bad": 0, "good": 0, "premium": 0}
        for step in range(MAX_STEPS):
            act = decode_action(policy.compute_single_action(
                obs["p0"], explore=False)[0])
            obs, rews, terms, truncs, info = env.step({"p0": act})
            hits += env.task._hit_this_step.get("p0", 0)
            if env.task._has_launched_this_step.get("p0", False):
                launches += 1
                quality[classify_launch(launch_geometry(env))] += 1
            if terms.get("__all__") or truncs.get("__all__"):
                reason = info.get("p0", {}).get("termination_reason", "unknown")
                break
        env.close()
        key = str(s)
        if key in baseline:
            b = baseline[key]["bc"]
            matches["kill"] += int(reason == "target_killed") == int(b["kill"])
            matches["lost"] += int(reason == "lost_target") == int(b["lost"])
            matches["launches"] += launches == int(b["launches"])
        kills += int(reason == "target_killed")
        lost += int(reason == "lost_target")
        totals["launches"] += launches
        totals["hits"] += hits
        n += 1
        if (s + 1 - args.start_seed) % 100 == 0:
            print(f"  seed {s+1-args.start_seed}/{args.seeds} done")

    match_rate = {k: v / max(n, 1) for k, v in matches.items()}
    print("=" * 66)
    print(f"[gate] zero-update 500-seed reproduction")
    print(f"  kills={kills} ({kills/max(n,1)*100:.1f}%)  lost={lost} "
          f"({lost/max(n,1)*100:.1f}%)  launches={totals['launches']/max(n,1):.2f}/ep  "
          f"hits={totals['hits']/max(totals['launches'],1)*100:.1f}%")
    print(f"  per-seed match (vs archived BC): kill={match_rate['kill']*100:.1f}%  "
          f"lost={match_rate['lost']*100:.1f}%  launches={match_rate['launches']*100:.1f}%")
    reproduce_ok = (match_rate["kill"] == 1.0 and match_rate["lost"] == 1.0
                    and match_rate["launches"] == 1.0)
    print(f"[gate] VERDICT: "
          f"{'PASS' if (logit_ok and reproduce_ok) else 'FAIL'} "
          f"(logits {'PASS' if logit_ok else 'FAIL'}, "
          f"reproduce {'PASS' if reproduce_ok else 'FAIL'})")

    os.makedirs(args.checkpoint, exist_ok=True)
    algo.save(os.path.abspath(args.checkpoint))
    print(f"[gate] saved RLlib checkpoint: {args.checkpoint}")
    ray.shutdown()


if __name__ == "__main__":
    main()

