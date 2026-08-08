"""Anti-overshoot counterfactual sweep (failure + matched-success control).

For each seed, replay the ORIGINAL distilled actions up to an intervention
step, then override the SPEED toward a target (heading stays distilled, fire
stays ASAP). Grid:
  starts: fire2 / hit2 / fire3 / hit3 / fire3-20 / fire3-40 / fire3-80
  speeds: 260 / 240 / 220 / 200 / 180

Failure seeds (3-hit timeout) measure P(kill | start, speed), 4th-window
recovery, new lost, episode length. Matched success seeds (original 4-hit
kill) measure degradation: kill retention and time_to_kill p50/p90.

Usage:
  python scripts/anti_overshoot_sweep.py --fail-seeds 20 --success-seeds 12
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

MAX_STEPS = 1500
STARTS = ["fire2", "hit2", "fire3", "hit3", "fire3-20", "fire3-40", "fire3-80"]
SPEEDS = [260.0, 240.0, 220.0, 200.0, 180.0]


def find_events(model, device, seed):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    acts, launches, hits = [], 0, 0
    events = {}
    reason = "timeout"
    for step in range(MAX_STEPS):
        act = policy_action(model, obs["p0"], device)
        acts.append(act.copy())
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches <= 3:
                events[f"fire{launches}"] = step
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
            if hits <= 3:
                events[f"hit{hits}"] = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return acts, events, reason, launches, hits, step + 1


def switch_step(events, start):
    if start == "fire3-20":
        return max(0, events["fire3"] - 20)
    if start == "fire3-40":
        return max(0, events["fire3"] - 40)
    if start == "fire3-80":
        return max(0, events["fire3"] - 80)
    return events[start]


def run_variant(model, device, seed, acts, events, start, target):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    t_switch = switch_step(events, start)
    reason = "timeout"
    launches = hits = 0
    t4win = None
    min_range_after = float("inf")
    ata_max_after = 0.0
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        if step < t_switch:
            act = acts[step].copy()
        else:
            base = policy_action(model, obs["p0"], device)
            act = base.copy()
            cmd = float(env.pursuers[0]._cmd_speed)
            act[0] = 2 if cmd < target - 5.0 else (0 if cmd > target + 5.0 else 1)
            act[3] = 1 if mask[10] == 1.0 else 0
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if step >= t_switch:
            ps, ts = env.pursuers[0], env.targets[0]
            d = float(np.linalg.norm(ps.aircraft.position_ned
                                     - ts.aircraft.position_ned))
            min_range_after = min(min_range_after, d)
            los = ts.aircraft.position_ned - ps.aircraft.position_ned
            los_dir = los / max(d, 1e-6)
            p_fwd = np.array([
                np.cos(ps.aircraft.rpy_rad[1]) * np.cos(ps.aircraft.rpy_rad[2]),
                np.cos(ps.aircraft.rpy_rad[1]) * np.sin(ps.aircraft.rpy_rad[2]),
                np.sin(ps.aircraft.rpy_rad[1])])
            ata = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
            ata_max_after = max(ata_max_after, ata)
            if mask[10] == 1.0 and t4win is None:
                t4win = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {"kill": reason == "target_killed", "lost": reason == "lost_target",
            "reason": reason, "4th_launch": launches >= 4, "hits": hits,
            "launches": launches, "t4win_gap": (t4win - t_switch)
            if t4win is not None else None, "steps": step + 1,
            "min_range_after": min_range_after, "ata_max_after": ata_max_after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-seeds", type=int, default=20,
                        help="number of 3-hit timeout failure seeds")
    parser.add_argument("--success-seeds", type=int, default=12,
                        help="number of 4-hit kill success seeds (control)")
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="results/shoot_eval/anti_overshoot_sweep.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    fail_seeds, succ_seeds = [], []
    seed = 0
    while (len(fail_seeds) < args.fail_seeds or
           len(succ_seeds) < args.success_seeds) and seed < 1200:
        acts, events, reason, launches, hits, steps = find_events(model, device, seed)
        if reason == "timeout" and hits == 3 and len(fail_seeds) < args.fail_seeds:
            fail_seeds.append({"seed": seed, "acts": acts, "events": events,
                               "steps": steps})
        elif reason == "target_killed" and hits == 4 and len(succ_seeds) < args.success_seeds:
            succ_seeds.append({"seed": seed, "acts": acts, "events": events,
                               "steps": steps})
        seed += 1
    print(f"[sweep] fail_seeds={len(fail_seeds)} success_seeds={len(succ_seeds)} "
          f"(scanned {seed})")

    out = {"starts": STARTS, "speeds": SPEEDS, "fail": {}, "success": {}}
    for grp, recs in (("fail", fail_seeds), ("success", succ_seeds)):
        for r in recs:
            s = r["seed"]
            out[grp][s] = {"original": {"reason": None, "steps": r["steps"]},
                           "variants": {}}
            for start in STARTS:
                if start not in r["events"]:
                    continue
                for spd in SPEEDS:
                    v = run_variant(model, device, s, r["acts"], r["events"],
                                    start, spd)
                    out[grp][s]["variants"][f"{start}@{spd}"] = v
        print(f"[sweep] {grp} group done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
