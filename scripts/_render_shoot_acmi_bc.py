"""Render 1v1 missile shoot episodes with the BC+ASAP-distilled policy and
emit Tacview/ACMI files so they can be opened in Tacview for visual review.

Why a new script:
  * scripts/_render_shoot_acmi.py exists but its main() hardcodes PPO v3
    checkpoint paths that are not in this checkout, and it loads via
    PPO.from_checkpoint() which cannot consume our BC distilled .pth.
  * scripts/eval_bc_1v1.py does not call enable_acmi_logging / log_acmi_step.

This script imports the same policy_action and BCShootPolicy as the BC eval
and runs a small set of episodes across two difficulties, writing one .acmi
per episode plus a JSON manifest.

Usage:
  python scripts/_render_shoot_acmi_bc.py \
      --weights data/expert/shoot_bc_asap_distilled.pth \
      --outdir results/shoot_acmi
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy  # noqa: E402
from scripts.eval_bc_1v1 import policy_action  # noqa: E402

MAX_STEPS = 1500


def _fwd(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])


def run_episode(model, device, seed, difficulty, acmi_path):
    env = BaseEnv(task=SingleCombatShootTask({
        "difficulty_level": difficulty,
        "obs_include_closure": True,
    }))
    obs, _ = env.reset(seed=seed)
    p0 = env.pursuers[0]
    t0 = env.targets[0]

    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()

    init_dist = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
    fires = 0
    min_dist = init_dist
    wez_first = None
    fire_first = None
    final_dist = init_dist
    total_steps = 0
    reason = "timeout"

    for step in range(MAX_STEPS):
        act = policy_action(model, obs["p0"], device)
        if act[3] == 1:
            fires += 1
        obs, _rews, terms, truncs, info = env.step({"p0": act})
        env.log_acmi_step()
        d = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
        if d < min_dist:
            min_dist = d
        final_dist = d
        if wez_first is None and env.task._is_valid_launch_envelope(p0, t0):
            wez_first = step
        if env.task._has_launched_this_step.get("p0", False) and fire_first is None:
            fire_first = step
        total_steps = step + 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break

    env.close_acmi()
    env.close()
    return {
        "acmi": acmi_path,
        "seed": seed,
        "difficulty": difficulty,
        "steps": total_steps,
        "fires": fires,
        "init_dist_m": init_dist,
        "min_dist_m": min_dist,
        "final_dist_m": final_dist,
        "reason": reason,
        "wez_first_step": wez_first,
        "fire_first_step": fire_first,
        "wez_to_fire_latency": (fire_first - wez_first) if (wez_first is not None and fire_first is not None) else None,
        "kill": reason == "target_killed",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    ap.add_argument("--outdir", default="results/shoot_acmi")
    ap.add_argument("--seeds", type=str, default="42,43,44",
                    help="comma-separated seeds for ID-difficulty episodes")
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[render-bc] weights={args.weights} device={device} "
          f"meta.best_epoch={ck.get('meta', {}).get('best_epoch')}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    manifest = []
    for s in seeds:
        tag = f"bc_s{s:02d}_d{int(args.difficulty * 10):02d}"
        acmi = os.path.join(args.outdir, f"shoot_{tag}.acmi")
        t0 = time.time()
        info = run_episode(model, device, s, args.difficulty, acmi)
        info["wall_s"] = round(time.time() - t0, 2)
        sz_kb = round(os.path.getsize(acmi) / 1024.0, 1) if os.path.exists(acmi) else 0
        info["acmi_kb"] = sz_kb
        manifest.append(info)
        print(
            f"  [{tag}] steps={info['steps']:4d} fires={info['fires']:2d} "
            f"min_d={info['min_dist_m']:6.0f}m reason={info['reason']} "
            f"kill={info['kill']} acmi={sz_kb}KB ({info['wall_s']}s)"
        )

    out = os.path.join(args.outdir, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "difficulty": args.difficulty,
                   "episodes": manifest}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()