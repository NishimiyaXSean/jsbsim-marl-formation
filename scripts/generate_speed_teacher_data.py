"""S3: generate the speed-head distillation dataset.

Composition:
  * 65% original ASAP-distilled replay (labels = original speed actions)
  * 25% B2 teacher CLOSED-LOOP rollout steps inside the anti-overshoot mode
    (the full chain: trigger -> -20 -> hold near 240 -> exit), so the student
    sees the states reached AFTER decelerating, not just relabeled old states
  * 10% hard negatives / trigger-boundary samples (near-threshold states,
    wide-zone-but-slow, post-exit, near-trigger successes) labeled with the
    ORIGINAL speed action (i.e., "do not decelerate here")

B2 teacher (frozen, stateless):
  decel when (R<4000 & DV>90 & closure<-100 & ATA<10)
        or  (R<4800 & DV>63 & closure<-70 & ATA<15 & p_spd>245)
  while active: speed = -20 if p_spd>240 else 0

Usage:
  python scripts/generate_speed_teacher_data.py --replay-episodes 300 \
      --teacher-episodes 200 --out data/expert/speed_teacher_data.npz
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

# B2 frozen thresholds (never retune after S0)
R_S, DV_S, C_S, A_S = 4000.0, 90.0, 100.0, 10.0
R_W, DV_W, C_W, A_W = 4800.0, 63.0, 70.0, 15.0


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


def b2_trigger(f):
    strict = (f["range"] < R_S and f["delta_speed"] > DV_S
              and f["closure"] < -C_S and f["ata"] < A_S)
    wide = (f["range"] < R_W and f["delta_speed"] > DV_W
            and f["closure"] < -C_W and f["ata"] < A_W)
    return strict or (wide and f["p_spd"] > 245.0)


def b2_speed_action(f):
    return 0 if f["p_spd"] > 240.0 else 1


def is_boundary(f):
    """Hard-negative candidates: near thresholds but NOT a b2 trigger."""
    b = (0.9 * R_S < f["range"] < 1.1 * R_W
         or 0.8 * DV_S < f["delta_speed"] < 1.1 * DV_W
         or -1.1 * C_S < f["closure"] < -0.8 * C_S
         or 0.8 * A_S < f["ata"] < 1.1 * A_W)
    wide = (f["range"] < R_W and f["delta_speed"] > DV_W
            and f["closure"] < -C_W and f["ata"] < A_W)
    return b or (wide and f["p_spd"] <= 245.0)


def run_episode(model, device, seed, mode):
    """mode: 'replay' (original distilled) or 'teacher' (B2 override)."""
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    samples = []
    in_mode = False
    steps_since_exit = 999
    reason = "timeout"
    for step in range(MAX_STEPS):
        f = state_features(env)
        orig = policy_action(model, obs["p0"], device)
        act = orig.copy()
        trig = b2_trigger(f)
        if mode == "teacher":
            if trig:
                in_mode = True
            if in_mode:
                if f["p_spd"] <= 245.0:
                    in_mode = False
                    steps_since_exit = 0
                else:
                    act[0] = b2_speed_action(f)
            if not in_mode:
                steps_since_exit += 1
        else:
            in_mode = False
        teacher_spd = int(act[0])
        hard_neg = (mode == "teacher" and not in_mode
                    and (is_boundary(f) or steps_since_exit <= 30))
        samples.append({
            "obs": obs["p0"].astype(np.float32).copy(),
            "orig_spd": int(orig[0]),
            "teacher_spd": teacher_spd,
            "b2_region": int(trig or in_mode),
            "hard_neg": int(hard_neg),
        })
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return samples, reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-episodes", type=int, default=300)
    parser.add_argument("--teacher-episodes", type=int, default=200)
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="data/expert/speed_teacher_data.npz")
    parser.add_argument("--stats", default="results/shoot_eval/speed_teacher_stats.json")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale down target dataset sizes (smoke)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    replay, teacher = [], []
    seed = 0
    while len(replay) < args.replay_episodes and seed < 2000:
        s, _ = run_episode(model, device, seed, "replay")
        replay.append(s)
        seed += 1
    print(f"[s3] replay episodes={len(replay)}")
    seed = 0
    while len(teacher) < args.teacher_episodes and seed < 2000:
        s, _ = run_episode(model, device, 30000 + seed, "teacher")
        teacher.append(s)
        seed += 1
    print(f"[s3] teacher episodes={len(teacher)}")

    # assemble: replay 65%, teacher-mode 25%, hard-neg 10%
    # keep ONLY non-B2-region replay states: B2-region labels must come from
    # the teacher alone, otherwise the student gets conflicting labels
    # ("orig no-decel" vs "teacher decel") in the same region.
    all_replay = [x for ep in replay for x in ep if x["b2_region"] == 0]
    teacher_mode = [x for ep in teacher for x in ep if x["b2_region"] == 1]
    hard = [x for ep in teacher for x in ep
            if x["hard_neg"] == 1 and x["b2_region"] == 0]
    # fixed 65/25/10 target sizes (scale only for smoke tests)
    n_replay_t = int(200000 * args.scale)
    n_teacher_t = int(0.25 / 0.65 * n_replay_t)
    n_hard_t = int(0.10 / 0.65 * n_replay_t)
    rng = np.random.default_rng(20260809)
    replay_sel = rng.choice(all_replay, n_replay_t,
                            replace=len(all_replay) < n_replay_t) \
        if all_replay else np.array([])
    teacher_sel = rng.choice(teacher_mode, n_teacher_t,
                             replace=len(teacher_mode) < n_teacher_t) \
        if teacher_mode else np.array([])
    hard_sel = rng.choice(hard, n_hard_t,
                          replace=len(hard) < n_hard_t) if hard else np.array([])
    print(f"[s3] replay_sel={len(replay_sel)} teacher_sel={len(teacher_sel)} "
          f"hard_sel={len(hard_sel)} (pools: {len(all_replay)}/{len(teacher_mode)}/"
          f"{len(hard)})")

    sel = list(replay_sel) + list(teacher_sel) + list(hard_sel)
    obs = np.stack([s["obs"] for s in sel])
    speed = np.array([s["teacher_spd"] for s in sel], dtype=np.int64)
    orig = np.array([s["orig_spd"] for s in sel], dtype=np.int64)
    b2_reg = np.array([s["b2_region"] for s in sel], dtype=np.int64)
    hard_f = np.array([s["hard_neg"] for s in sel], dtype=np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, obs=obs, speed=speed, orig_speed=orig,
                        b2_region=b2_reg, hard_neg=hard_f)
    print(f"[s3] saved {args.out}: n={len(sel)} b2_region="
          f"{b2_reg.mean()*100:.1f}% hard={hard_f.mean()*100:.1f}%")
    stats = {
        "n": len(sel),
        "b2_region_frac": float(b2_reg.mean()),
        "hard_frac": float(hard_f.mean()),
        "speed_hist": np.bincount(speed, minlength=3).tolist(),
        "replay_episodes": len(replay),
        "teacher_episodes": len(teacher),
    }
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()


