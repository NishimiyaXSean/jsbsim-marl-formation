"""Counterfactual oracle at t_hit3 for 3-hit timeout episodes.

Replays the ORIGINAL distilled actions up to the 3rd hit (t_hit3), then
switches to one of:
  original
  speed_target_240 / speed_target_220 / speed_hold   (speed-first)
  hdg_tight                                           (heading-first)
  hdg_tight + speed_target_240                        (joint)

Success: does the 4th legal window / 4th launch appear after t_hit3?
This decides whether to unfreeze speed only, heading only, or both.

Usage:
  python scripts/hit3_counterfactual.py --seeds 6,23,26,...
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
from src.environment.singlecombat_shoot_task import (
    MIN_ATTACK_DISTANCE, MAX_ATTACK_ANGLE, MIN_ATTACK_INTERVAL)
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action
from scripts.generate_shoot_rule_expert import wrap180

MAX_STEPS = 1500
DELTA_HDG = [-10.0, -5.0, 0.0, 5.0, 10.0]


def hdg_tight(obs, deadband=1.25):
    err = wrap180(float(obs[21]) * 180.0)
    if abs(err) < deadband:
        return 2
    best_i, best_c = 2, float("inf")
    for i, d in enumerate(DELTA_HDG):
        next_err = abs(wrap180(float(obs[21]) * 180.0 - d))
        cost = next_err + 0.05 * abs(d)
        if cost < best_c:
            best_c, best_i = cost, i
    return best_i


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def geometry(env):
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
    aa = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    dyn_max = 3000.0 + 5000.0 * (aa / 180.0)
    return {"dist": dist, "ata": ata, "closure": closure, "dyn_max": dyn_max}


def run_variant(model, device, seed, variant, t_hit3, prefix_acts,
                t_fire3=None, from_step="hit3"):
    """Replay prefix_acts up to t_hit3 (or t_fire3), then apply override."""
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    launches = hits = 0
    t4win = None
    static_ok_after_cd = False
    last_fire = None
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        t_switch = t_fire3 if (from_step == "fire3" and t_fire3 is not None) \
            else t_hit3
        if step < t_switch:
            act = prefix_acts[step].copy()
        else:
            base = policy_action(model, obs["p0"], device)
            act = base.copy()
            if variant.startswith("speed_target_"):
                target = float(variant.split("_")[-1])
                cmd = float(env.pursuers[0]._cmd_speed)
                act[0] = 2 if cmd < target - 5.0 else (0 if cmd > target + 5.0 else 1)
            elif variant == "speed_hold":
                act[0] = 1
            if variant.startswith("hdg_tight"):
                act[1] = hdg_tight(obs["p0"])
            act[3] = 1 if mask[10] == 1.0 else 0
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            last_fire = env._step_counter
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if step >= t_switch and mask[10] == 1.0 and t4win is None:
            t4win = step
        if step >= t_switch:
            g = geometry(env)
            cd_clear = last_fire is None or env._step_counter - last_fire >= MIN_ATTACK_INTERVAL
            if (g["ata"] < MAX_ATTACK_ANGLE and g["closure"] < 0.0
                    and MIN_ATTACK_DISTANCE < g["dist"] < g["dyn_max"]
                    and cd_clear):
                static_ok_after_cd = True
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {
        "seed": seed, "variant": variant,
        "4th_launch": launches >= 4,
        "kill": reason == "target_killed",
        "reason": reason,
        "t_4th_window": t4win,
        "t_4th_window_gap": (t4win - t_switch) if t4win is not None else None,
        "static_ok_after_cd": static_ok_after_cd,
        "hits": hits, "launches": launches,
    }


def find_hit3(model, device, seed):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    acts = []
    hits = 0
    launches = 0
    t_hit3 = None
    t_fire3 = None
    reason = "timeout"
    for step in range(MAX_STEPS):
        act = policy_action(model, obs["p0"], device)
        acts.append(act.copy())
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches == 3:
                t_fire3 = step
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
            if hits == 3:
                t_hit3 = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return t_hit3, acts, reason, t_fire3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True,
                        help="comma-separated 3-hit timeout seeds")
    parser.add_argument("--from", dest="from_step", default="hit3",
                        choices=["hit3", "fire3"],
                        help="intervene from the 3rd hit or the 3rd launch")
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="results/shoot_eval/hit3_counterfactual.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    variants = ["original", "speed_target_240", "speed_target_220",
                "speed_target_200", "speed_hold", "hdg_tight",
                "hdg_tight+speed_target_220", "hdg_tight+speed_target_200"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    results = {}
    for s in seeds:
        t_hit3, acts, reason, t_fire3 = find_hit3(model, device, s)
        print(f"seed {s}: hit3={t_hit3} fire3={t_fire3} reason={reason}")
        if t_hit3 is None:
            continue
        row = {}
        for v in variants:
            r = run_variant(model, device, s, v, t_hit3, acts,
                            t_fire3=t_fire3, from_step=args.from_step)
            row[v] = r
            ok = "OK " if (r["4th_launch"] or r["kill"]) else "-- "
            print(f"   {v:<26s} {ok} 4th={r['4th_launch']} "
                  f"kill={r['kill']} t4gap={r['t_4th_window_gap']} "
                  f"static_cd={r['static_ok_after_cd']} reason={r['reason']}")
        results[s] = row

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

