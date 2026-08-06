"""Paired-seed closed-loop evaluation: BC policy vs discrete rule expert.

For each seed s the SAME initial geometry (env.reset(seed=s)) is played by
the discrete expert and by the BC policy, so episode outcomes are paired.
Reports the review-requested metrics:
  first-launch hit rate, kill rate given first hit, missiles per kill,
  damage per launch, first-fire time, WEZ->first-fire latency,
  per-seed BC-expert deltas (kill / lost / launches).

Usage:
  python scripts/eval_paired_bc_vs_expert.py --seeds 500
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
from scripts.generate_shoot_rule_expert import (
    hdg_label, spd_label, fire_desired, FIRE_IDX)
from scripts.eval_bc_1v1 import policy_action
from scripts.train_shoot_bc import BCShootPolicy

MAX_STEPS = 1500
CMD_SPEED = 280.0


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


def expert_action(env, obs):
    mask = env.task.get_action_mask(env, "p0")
    allowed = mask[FIRE_IDX] == 1.0
    desired = fire_desired(obs)
    fire = 1 if (allowed and desired) else 0
    return np.array([spd_label(obs, CMD_SPEED), hdg_label(obs), 0, fire],
                    dtype=np.int64)


def run_one_episode(policy, model, device, seed, difficulty=0.0,
                    max_steps=MAX_STEPS, **task_cfg):
    """Play one seeded episode with the given policy; return per-episode stats."""
    cfg = {"difficulty_level": difficulty, "obs_include_closure": True}
    cfg.update(task_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    wez_first = fire_first = None
    launches = hits = 0
    quality = {"bad": 0, "good": 0, "premium": 0}
    ata_hist, closure_hist = [], []
    for step in range(max_steps):
        if policy == "expert":
            act = expert_action(env, obs["p0"])
        else:
            act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits += env.task._hit_this_step.get("p0", 0)
        g = launch_geometry(env)
        ata_hist.append(g["ata_deg"])
        closure_hist.append(g["closure_mps"])
        if wez_first is None and env.task._is_valid_launch_envelope(
                env.pursuers[0], env.targets[0]):
            wez_first = step
        if env.task._has_launched_this_step.get("p0", False):
            if fire_first is None:
                fire_first = step
            quality[classify_launch(g)] += 1
            launches += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break
    first_missile_success = False
    if env.pursuers[0].launch_missiles:
        m = env.pursuers[0].launch_missiles[0]
        first_missile_success = bool(m.is_done and m.is_success)
    env.close()
    return {
        "seed": int(seed),
        "policy": policy,
        "reason": reason,
        "steps": step + 1,
        "kill": reason == "target_killed",
        "lost": reason == "lost_target",
        "launches": launches,
        "hits": hits,
        "first_fire": fire_first,
        "wez_first": wez_first,
        "quality": quality,
        "first_missile_success": first_missile_success,
        "ata_p90": float(np.percentile(ata_hist, 90)) if ata_hist else 0.0,
        "closure_pos_ratio": float(np.mean([c > 0 for c in closure_hist]))
        if closure_hist else 0.0,
    }


def aggregate(records):
    n = len(records)
    launches = sum(r["launches"] for r in records)
    hits = sum(r["hits"] for r in records)
    kills = sum(r["kill"] for r in records)
    lost = sum(r["lost"] for r in records)
    first_hit = sum(r["first_missile_success"] for r in records)
    kill_eps = [r for r in records if r["kill"]]
    wez = [r["wez_first"] for r in records if r["wez_first"] is not None]
    ff = [r["first_fire"] for r in records if r["first_fire"] is not None]
    lat = [r["first_fire"] - r["wez_first"] for r in records
           if r["first_fire"] is not None and r["wez_first"] is not None]
    return {
        "n": n,
        "lost_rate": lost / max(n, 1),
        "kill_rate": kills / max(n, 1),
        "launches_per_episode": launches / max(n, 1),
        "hit_rate": hits / max(launches, 1),
        "wez_reach_rate": len(wez) / max(n, 1),
        "first_fire_median_steps": float(np.median(ff)) if ff else None,
        "wez_to_first_fire_median_steps": float(np.median(lat)) if lat else None,
        "first_missile_hit_rate": first_hit / max(n, 1),
        "kill_given_first_hit": sum(r["kill"] for r in records
                                    if r["first_missile_success"]) / max(first_hit, 1),
        "missiles_per_kill": (sum(r["launches"] for r in kill_eps)
                              / max(len(kill_eps), 1)) if kill_eps else None,
        "damage_per_launch": hits / max(launches, 1),
        "ata_p90_mean": float(np.mean([r["ata_p90"] for r in records])),
        "closure_pos_ratio_mean": float(np.mean([r["closure_pos_ratio"]
                                                 for r in records])),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "quality": {
            "bad": sum(r["quality"]["bad"] for r in records),
            "good": sum(r["quality"]["good"] for r in records),
            "premium": sum(r["quality"]["premium"] for r in records),
        },
        "reasons": {k: sum(1 for r in records if r["reason"] == k)
                    for k in sorted({r["reason"] for r in records})},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out", default="results/shoot_eval/paired_bc_vs_expert_500.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    records = []
    for s in range(args.start_seed, args.start_seed + args.seeds):
        rec_exp = run_one_episode("expert", None, device, s,
                                  difficulty=args.difficulty)
        rec_bc = run_one_episode("bc", model, device, s,
                                 difficulty=args.difficulty)
        records.append(rec_exp)
        records.append(rec_bc)
        if (s + 1 - args.start_seed) % 50 == 0:
            print(f"  seed {s+1-args.start_seed}/{args.seeds} done")

    exp = aggregate([r for r in records if r["policy"] == "expert"])
    bc = aggregate([r for r in records if r["policy"] == "bc"])
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed"], {})[r["policy"]] = r
    d_kill = [by_seed[s]["bc"]["kill"] - by_seed[s]["expert"]["kill"]
              for s in by_seed]
    d_lost = [by_seed[s]["bc"]["lost"] - by_seed[s]["expert"]["lost"]
              for s in by_seed]
    d_launch = [by_seed[s]["bc"]["launches"] - by_seed[s]["expert"]["launches"]
                for s in by_seed]
    paired = {
        "n_seeds": len(by_seed),
        "delta_kill_mean": float(np.mean(d_kill)),
        "delta_kill_win_lose_tie": [
            int(sum(1 for x in d_kill if x > 0)),
            int(sum(1 for x in d_kill if x < 0)),
            int(sum(1 for x in d_kill if x == 0))],
        "delta_lost_mean": float(np.mean(d_lost)),
        "delta_lost_win_lose_tie": [
            int(sum(1 for x in d_lost if x < 0)),
            int(sum(1 for x in d_lost if x > 0)),
            int(sum(1 for x in d_lost if x == 0))],
        "delta_launches_mean": float(np.mean(d_launch)),
    }
    out = {
        "weights": args.weights,
        "difficulty": args.difficulty,
        "seeds": [args.start_seed, args.start_seed + args.seeds - 1],
        "expert": exp,
        "bc": bc,
        "paired": paired,
        "per_seed": {str(s): {"expert": by_seed[s]["expert"],
                              "bc": by_seed[s]["bc"]} for s in sorted(by_seed)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 66)
    print(f"PAIRED EVAL — {args.seeds} seeds, difficulty={args.difficulty}")
    for name, a in (("expert", exp), ("bc", bc)):
        print(f"\n[{name}] lost={a['lost_rate']*100:.1f}%  kills={a['kill_rate']*100:.1f}%  "
              f"launches={a['launches_per_episode']:.2f}/ep  "
              f"hits={a['hit_rate']*100:.1f}%  WEZ={a['wez_reach_rate']*100:.0f}%")
        print(f"   first-fire median={a['first_fire_median_steps']} steps  "
              f"WEZ->fire median={a['wez_to_first_fire_median_steps']} steps  "
              f"first-missile hit={a['first_missile_hit_rate']*100:.1f}%  "
              f"kill|first-hit={a['kill_given_first_hit']*100:.1f}%")
        print(f"   missiles/kill={a['missiles_per_kill']}  "
              f"damage/launch={a['damage_per_launch']:.2f}  "
              f"quality={a['quality']}")
    print(f"\n[paired] delta_kill mean={paired['delta_kill_mean']:+.3f} "
          f"(win/lose/tie={paired['delta_kill_win_lose_tie']})")
    print(f"         delta_lost mean={paired['delta_lost_mean']:+.3f} "
          f"(win/lose/tie={paired['delta_lost_win_lose_tie']})")
    print(f"         delta_launches mean={paired['delta_launches_mean']:+.3f}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
