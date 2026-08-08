"""Closed-loop evaluation of a geometry anti-overshoot speed rule.

Rule (four variables, stateless from the 41-dim obs):
    range < R AND delta_speed > DV AND closure < -C AND ATA < A

Execution with hysteresis:
  enter : strict rule holds
  exit  : pursuer speed <= 245 OR range >= 1.5*R OR closure >= -max(10, 0.5*C)
  while active: speed_action = -20 if airspeed > 240 else 0 (hold)
  else        : speed_action = frozen distilled policy
  heading     : frozen distilled policy
  fire        : ASAP (mask)

Metrics: rescue rate (failure seeds), success retention, overall kill, lost,
bad, 3-hit timeout, 4th-window rate, fire3->fire4 gap, min range near 3rd,
ATA>90 ratio, time_to_kill p50/p90, override duty cycle.

Usage:
  python scripts/rule_override_eval.py --rule "3600,90,100,12" \
      --blind-id-seeds 100 --blind-d2-seeds 50
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
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch

MAX_STEPS = 1500


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def state_features(env):
    ps = env.pursuers[0]
    tgt = env.targets[0]
    p_pos = ps.aircraft.position_ned
    t_pos = tgt.aircraft.position_ned
    los = t_pos - p_pos
    dist = float(np.linalg.norm(los))
    los_dir = los / max(dist, 1e-6)
    p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
    t_fwd = compute_forward_vector(tgt.aircraft.rpy_rad)
    ata = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned,
                           los_dir))
    p_spd = float(np.linalg.norm(ps.aircraft.velocity_ned))
    t_spd = float(np.linalg.norm(tgt.aircraft.velocity_ned))
    return {"range": dist, "delta_speed": p_spd - t_spd, "ata": ata,
            "closure": closure, "p_spd": p_spd}


def run_episode(model, device, seed, cell_cfg, rule):
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(cell_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    R, DV, C, A = rule
    in_mode = False
    mode_steps = 0
    reason = "timeout"
    launches = hits = 0
    fire3 = None
    fire4win = None
    fire3_t = None
    min_range_fire3 = float("inf")
    ata_gt90 = 0
    fire3_win_steps = 0
    bad = 0
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        f = state_features(env)
        base = policy_action(model, obs["p0"], device)
        act = base.copy()
        strict = (f["range"] < R and f["delta_speed"] > DV
                  and f["closure"] < -C and f["ata"] < A)
        if not in_mode and strict:
            in_mode = True
        if in_mode:
            mode_steps += 1
            if (f["p_spd"] <= 245.0 or f["range"] >= 1.5 * R
                    or f["closure"] >= -max(10.0, 0.5 * C)):
                in_mode = False
            else:
                act[0] = 0 if f["p_spd"] > 240.0 else 1
        act[3] = 1 if mask[10] == 1.0 else 0
        if fire3_t is not None and step >= fire3_t:
            fire3_win_steps += 1
            min_range_fire3 = min(min_range_fire3, f["range"])
            if f["ata"] > 90.0:
                ata_gt90 += 1
            if mask[10] == 1.0 and fire4win is None:
                fire4win = step
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches == 3:
                fire3_t = step
            if classify_launch(launch_geometry(env)) == "bad":
                bad += 1
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {
        "seed": seed, "kill": reason == "target_killed",
        "lost": reason == "lost_target", "reason": reason,
        "launches": launches, "hits": hits, "bad": bad,
        "steps": step + 1,
        "3hit_timeout": reason == "timeout" and hits == 3,
        "fire4win_gap": (fire4win - fire3_t) if fire4win is not None
        and fire3_t is not None else None,
        "min_range_fire3": min_range_fire3 if fire3_t is not None else None,
        "ata_gt90_ratio": ata_gt90 / max(fire3_win_steps, 1),
        "duty_cycle": mode_steps / max(step + 1, 1),
    }


def aggregate(recs):
    n = len(recs)
    kills = sum(r["kill"] for r in recs)
    ttk = [r["steps"] * 0.2 for r in recs if r["kill"]]
    gaps = [r["fire4win_gap"] for r in recs if r["fire4win_gap"] is not None]
    mrs = [r["min_range_fire3"] for r in recs if r["min_range_fire3"] is not None]
    return {
        "n": n,
        "kill_rate": kills / max(n, 1),
        "lost": sum(r["lost"] for r in recs),
        "bad": sum(r["bad"] for r in recs),
        "3hit_timeout": sum(r["3hit_timeout"] for r in recs),
        "4th_window_rate": sum(1 for r in recs
                               if r["fire4win_gap"] is not None) / max(n, 1),
        "fire3_to_fire4_median": float(np.median(gaps)) if gaps else None,
        "min_range_fire3_median": float(np.median(mrs)) if mrs else None,
        "ata_gt90_ratio_mean": float(np.mean([r["ata_gt90_ratio"]
                                              for r in recs])),
        "ttk_p50": float(np.median(ttk)) if ttk else None,
        "ttk_p90": float(np.percentile(ttk, 90)) if ttk else None,
        "duty_cycle_mean": float(np.mean([r["duty_cycle"] for r in recs])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=str, required=True,
                        help='comma "R,DV,C,A"')
    parser.add_argument("--fail-seeds", default="",
                        help="comma list of 3-hit timeout seeds")
    parser.add_argument("--success-seeds", default="",
                        help="comma list of success seeds (control)")
    parser.add_argument("--blind-id-seeds", type=int, default=100)
    parser.add_argument("--blind-d2-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="results/shoot_eval/rule_override_eval.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rule = tuple(float(x) for x in args.rule.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    groups = {}
    if args.fail_seeds:
        groups["fail_rescue"] = ([int(x) for x in args.fail_seeds.split(",")], {})
    if args.success_seeds:
        groups["success_control"] = ([int(x) for x in args.success_seeds.split(",")], {})
    groups["blind_id"] = (list(range(args.base_seed,
                                     args.base_seed + args.blind_id_seeds)), {})
    groups["blind_d2"] = (list(range(args.base_seed + 1000,
                                     args.base_seed + 1000 + args.blind_d2_seeds)),
                          {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0})

    out = {"rule": rule}
    print(f"[rule-override] rule R={rule[0]:.0f} DV={rule[1]:.0f} "
          f"C={rule[2]:.0f} A={rule[3]:.0f}")
    for name, (seeds, cfg) in groups.items():
        recs = [run_episode(model, device, s, cfg, rule) for s in seeds]
        agg = aggregate(recs)
        out[name] = agg
        print(f"  [{name:<16s}] n={agg['n']:3d} kill={agg['kill_rate']*100:5.1f}% "
              f"lost={agg['lost']} bad={agg['bad']} "
              f"3hitT={agg['3hit_timeout']:3d} 4th={agg['4th_window_rate']*100:4.1f}% "
              f"ttk_p50={agg['ttk_p50']} duty={agg['duty_cycle_mean']*100:.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

