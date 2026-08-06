"""Closed-loop evaluation of the BC policy on the 1v1 shoot task.

Drives the environment with the trained BCShootPolicy (deterministic argmax
on masked logits) and reports the same metrics as eval_shoot_1v1.py plus the
approach-quality metrics from the BC plan (time_to_WEZ, ATA p90,
closure>0 ratio, premium/good/bad launch shares).

Usage:
  python scripts/eval_bc_1v1.py --episodes 20
  python scripts/eval_bc_1v1.py --episodes 100 --out results/shoot_eval/eval_bc_round1.json
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

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.models.shoot_mask_model import ShootMaskModel
from scripts.train_shoot_bc import BCShootPolicy, ACTION_DIMS, MASK_NEG

MAX_STEPS = 1500


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
    aa_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    return {"range_m": dist, "ata_deg": ata_deg, "aa_deg": aa_deg, "closure_mps": closure}


def classify_launch(g):
    if g["closure_mps"] < 0 and g["ata_deg"] < 10.0 and 2000.0 < g["range_m"] < 4000.0:
        return "premium"
    if g["closure_mps"] < 0:
        return "good"
    return "bad"


def summarize(vals):
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(len(arr)), "mean": float(np.mean(arr)),
            "median": float(np.median(arr)), "min": float(np.min(arr)),
            "max": float(np.max(arr)), "std": float(np.std(arr))}


def policy_action(model, obs, device):
    """Deterministic masked-argmax action from a BCShootPolicy."""
    x = torch.tensor(obs[:30], dtype=torch.float32, device=device).unsqueeze(0)
    m = obs[30:]
    with torch.no_grad():
        logits = model(x)
    act = np.zeros(4, dtype=np.int64)
    off = 0
    for i, d in enumerate(ACTION_DIMS):
        lg = logits[i][0] + (1.0 - torch.tensor(m[off:off + d], dtype=torch.float32,
                                                device=device)) * MASK_NEG
        act[i] = int(lg.argmax().item())
        off += d
    return act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/expert/shoot_bc_weights.pth")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--out", default="results/shoot_eval/eval_bc_round1.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[eval] BC weights: {args.weights} (best_epoch={ck.get('meta', {}).get('best_epoch')}) "
          f"device={device}")

    n_ep = 0
    term_reasons = {}
    launches_total = hits_total = kills = 0
    launch_geo = {"range_m": [], "ata_deg": [], "aa_deg": [], "closure_mps": []}
    quality = {"bad": 0, "good": 0, "premium": 0}
    wez_firsts, fire_firsts, wez_fire_latency = [], [], []
    ata_p90s, closure_pos_ratios, ep_steps = [], [], []
    lost_after_wez = 0
    wez_reached = 0

    for ep in range(args.episodes):
        env = BaseEnv(task=SingleCombatShootTask({
            "difficulty_level": args.difficulty,
            "obs_include_closure": True,
        }))
        obs, _ = env.reset(seed=args.seed + ep)
        reason = "timeout"
        wez_first = fire_first = None
        ep_launches = ep_hits = 0
        ata_hist, closure_hist = [], []
        for step in range(MAX_STEPS):
            act = policy_action(model, obs["p0"], device)
            obs, rews, terms, truncs, info = env.step({"p0": act})
            ep_hits += env.task._hit_this_step.get("p0", 0)
            g = launch_geometry(env)
            ata_hist.append(g["ata_deg"])
            closure_hist.append(g["closure_mps"])
            if wez_first is None and env.task._is_valid_launch_envelope(
                    env.pursuers[0], env.targets[0]):
                wez_first = step
            if env.task._has_launched_this_step.get("p0", False):
                if fire_first is None:
                    fire_first = step
                for k in launch_geo:
                    launch_geo[k].append(g[k])
                quality[classify_launch(g)] += 1
                ep_launches += 1
            if terms.get("__all__") or truncs.get("__all__"):
                reason = info.get("p0", {}).get("termination_reason", "unknown")
                break
            if not np.isfinite(obs["p0"]).all():
                reason = "jsbsim_nan"
                break
        launches_total += ep_launches
        hits_total += ep_hits
        if reason == "target_killed":
            kills += 1
        term_reasons[reason] = term_reasons.get(reason, 0) + 1
        if wez_first is not None:
            wez_reached += 1
            wez_firsts.append(wez_first)
            if reason == "lost_target":
                lost_after_wez += 1
            if fire_first is not None:
                wez_fire_latency.append(fire_first - wez_first)
        if fire_first is not None:
            fire_firsts.append(fire_first)
        ata_p90s.append(float(np.percentile(ata_hist, 90)) if ata_hist else 0.0)
        closure_pos_ratios.append(
            float(np.mean([c > 0 for c in closure_hist])) if closure_hist else 0.0)
        ep_steps.append(step + 1)
        n_ep += 1
        env.close()
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}/{args.episodes} done")

    n = max(n_ep, 1)
    out = {
        "weights": args.weights,
        "episodes": n_ep,
        "difficulty": args.difficulty,
        "termination_reasons": term_reasons,
        "lost_target_rate": term_reasons.get("lost_target", 0) / n,
        "kill_rate": kills / n,
        "launches_per_episode": launches_total / n,
        "hits": hits_total,
        "hit_rate": hits_total / max(launches_total, 1),
        "launch_geometry": {k: summarize(v) for k, v in launch_geo.items()},
        "launch_quality": quality,
        "wez_reach_rate": wez_reached / n,
        "time_to_wez_steps": summarize(wez_firsts),
        "first_fire_steps": summarize(fire_firsts),
        "wez_to_fire_latency_steps": summarize(wez_fire_latency),
        "ata_p90": float(np.mean(ata_p90s)),
        "closure_positive_ratio": float(np.mean(closure_pos_ratios)),
        "lost_after_wez": lost_after_wez,
        "mean_steps": float(np.mean(ep_steps)),
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 66)
    print(f"BC CLOSED-LOOP EVAL — {n_ep} episodes, difficulty={args.difficulty}")
    print(f"  lost_target: {term_reasons.get('lost_target', 0)}/{n_ep} "
          f"({term_reasons.get('lost_target', 0)/n*100:.1f}%)   reasons: {term_reasons}")
    print(f"  WEZ reach: {wez_reached}/{n_ep} ({wez_reached/n*100:.1f}%)  "
          f"lost-after-WEZ: {lost_after_wez}")
    print(f"  launches: {launches_total} ({launches_total/n:.2f}/ep)  "
          f"hits: {hits_total} ({hits_total/max(launches_total,1):.2f}/launch)  "
          f"kills: {kills} ({kills/n*100:.1f}%)")
    q = quality
    qn = sum(q.values())
    print(f"  launch quality: premium={q['premium']} good={q['good']} bad={q['bad']} "
          f"(premium+good {100*(q['premium']+q['good'])/max(qn,1):.0f}%)")
    if wez_firsts:
        print(f"  time_to_WEZ: median={np.median(wez_firsts)*0.2:.1f}s  "
              f"first_fire median={np.median(fire_firsts)*0.2:.1f}s" if fire_firsts
              else f"  time_to_WEZ: median={np.median(wez_firsts)*0.2:.1f}s")
    print(f"  ATA p90 mean={out['ata_p90']:.1f} deg  "
          f"closure>0 ratio={out['closure_positive_ratio']*100:.1f}%  "
          f"mean_steps={out['mean_steps']:.0f}")
    print(f"\n  vs v19 baseline (d0.0): lost 32% / 0.48 launches-ep / 10% kills")
    print(f"  vs discrete rule (100ep): lost 0% / 2.68 launches-ep / 23% kills / 100% hits")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
