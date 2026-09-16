"""L5.3 + L4.2: Fire hesi tancy diagnostic.

Quantifies how often the BC policy's fire head says "don't fire"
on steps where the env mask allows fire. Compares to the rule expert
on the same scenarios.

Decision table (for paper direction):
  bc_hesi_rate < 1%          -> STORY_COLLAPSE: no BC pathology
  bc_hesi_rate 5-15% + expert_fire >= 95% -> ANGLE_A_SOLID: BC pathology
  bc_hesi_rate 5-15% + expert_fire < 95% -> RE_FRAME: expert also hesitates

Usage:
  python scripts/diag_fire_hesitancy.py --smoke       # 20 ep, quick
  python scripts/diag_fire_hesitancy.py --episodes 200  # full
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action

MAX_STEPS = 1500


def get_fire_head_pred(bc_model, obs_30, device):
    """Get fire head argmax prediction from BC policy."""
    obs_t = torch.tensor(obs_30, device=device).unsqueeze(0)
    with torch.no_grad():
        feat = bc_model.encoder(obs_t)
        fire_logits = bc_model.action_heads[3](feat)
    return int(fire_logits.argmax(1).item())


def rollout_bc(bc_model, device, episodes, base_seed=20000):
    """Roll out BC; track fire head predictions on env-allowed steps."""
    allowed_count = 0
    fire_pred_count = 0  # BC fire head predicted "fire=1" on allowed steps
    ep_lengths = []

    for ep in range(episodes):
        cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
        env = BaseEnv(task=SingleCombatShootTask(cfg))
        obs, _ = env.reset(seed=base_seed + ep)

        for step in range(MAX_STEPS):
            mask = env.task.get_action_mask(env, "p0")
            fire_allowed = mask[10] == 1.0

            if fire_allowed:
                allowed_count += 1
                fire_pred = get_fire_head_pred(bc_model, obs["p0"][:30], device)
                if fire_pred == 1:
                    fire_pred_count += 1

            act = policy_action(bc_model, obs["p0"], device)
            obs, rews, terms, truncs, info = env.step({"p0": act})

            if terms.get("__all__") or truncs.get("__all__"):
                ep_lengths.append(step + 1)
                break
            if not np.isfinite(obs["p0"]).all():
                ep_lengths.append(step + 1)
                break
        else:
            ep_lengths.append(MAX_STEPS)

        env.close()
        if (ep + 1) % 20 == 0:
            print(f"  rollout {ep+1}/{episodes}", flush=True)

    return allowed_count, fire_pred_count, ep_lengths


def expert_fire_stats(npz_path):
    """Compute fire rate from expert npz using action[:, 3] (authoritative).
    Cross-validates with fire_desired for sanity.

    Note: npz has both `action[:, 3]` (what expert did, mask-enforced to 0 on disallowed)
    and `fire_desired` (what expert wanted). On allowed steps they match.
    """
    data = np.load(npz_path)
    fire_allowed = data["fire_allowed"]
    action_fire = data["action"][:, 3]
    fire_desired = data["fire_desired"]

    allowed_mask = fire_allowed == 1.0
    n_allowed = int(allowed_mask.sum())
    n_fire = int(action_fire[allowed_mask].sum())
    n_fire_desired = int(fire_desired[allowed_mask].sum())
    n_disallowed = int((~allowed_mask).sum())
    n_fire_desired_disallowed = int(fire_desired[~allowed_mask].sum())

    return {
        "n_allowed": n_allowed,
        "n_fire": n_fire,
        "fire_rate_on_allowed": n_fire / max(n_allowed, 1),
        "n_fire_desired_on_allowed": n_fire_desired,
        "n_disallowed": n_disallowed,
        "n_fire_desired_on_disallowed": n_fire_desired_disallowed,
        "fire_desired_disallowed_rate": n_fire_desired_disallowed / max(n_disallowed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--expert-npz", default="data/expert/shoot_rule_expert.npz")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--smoke", action="store_true",
                        help="quick check: 20 episodes")
    parser.add_argument("--out", default="results/health_check/fire_hesitancy.json")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[diag] device = {device}")
    print(f"[diag] loading BC from {args.bc_weights}")
    if not os.path.exists(args.bc_weights):
        print(f"[diag] FATAL: BC weights not found at {args.bc_weights}")
        sys.exit(1)
    bc = BCShootPolicy().to(device)
    bc.load_state_dict(torch.load(args.bc_weights, map_location="cpu")["state_dict"])
    bc.eval()

    print(f"[diag] loading expert from {args.expert_npz}")
    if not os.path.exists(args.expert_npz):
        print(f"[diag] FATAL: expert npz not found at {args.expert_npz}")
        sys.exit(1)

    print(f"[diag] rolling out BC for {args.episodes} episodes...")
    n_allowed, n_fire, ep_lengths = rollout_bc(bc, device, args.episodes)
    bc_hesi_rate = 1 - (n_fire / max(n_allowed, 1))
    bc_fire_rate_on_allowed = n_fire / max(n_allowed, 1)

    expert_stats = expert_fire_stats(args.expert_npz)
    expert_fire = expert_stats["fire_rate_on_allowed"]

    # Decision logic (revised: BC faithfully imitates expert, both hesitant)
    if bc_hesi_rate < 0.01 and expert_fire >= 0.95:
        verdict = "STORY_COLLAPSE: BC + expert both ASAP-perfect (no pathology)"
        action = "Switch topic entirely (Angle B)"
    elif bc_hesi_rate >= 0.5 and expert_fire < 0.20:
        verdict = "EXPERT_PATHOLOGY: BC inherits expert hesi tancy; ASAP recovers it"
        action = "ANGLE A REVISED — frame as 'expert suboptimality' not 'BC pathology'"
    elif 0.05 <= bc_hesi_rate <= 0.20 and expert_fire >= 0.95:
        verdict = "BC_PATHOLOGY: BC hesitant despite ASAP-perfect expert"
        action = "Original Angle A solid (BC pathology story)"
    elif 0.05 <= bc_hesi_rate <= 0.20 and expert_fire < 0.95:
        verdict = "MIXED: both BC and expert hesitant at moderate level"
        action = "Frame carefully; verify with multi-seed"
    else:
        verdict = f"INVESTIGATE: bc_hesi_rate={bc_hesi_rate:.3f}, expert_fire={expert_fire:.3f}"
        action = "Inspect numbers manually"

    print("=" * 60)
    print(f"[diag] BC rollout stats (action fire head predictions):")
    print(f"  episodes rolled:      {args.episodes}")
    print(f"  mean ep length:       {np.mean(ep_lengths):.1f} steps")
    print(f"  allowed-fire steps:   {n_allowed}")
    print(f"  BC fire head says fire=1: {n_fire} ({bc_fire_rate_on_allowed*100:.2f}%)")
    print(f"  BC fire hesi tancy rate:  {bc_hesi_rate*100:.2f}%")
    print(f"[diag] Expert stats (from npz action[:, 3]):")
    print(f"  allowed-fire steps:   {expert_stats['n_allowed']}")
    print(f"  expert fires (action): {expert_stats['n_fire']} ({expert_fire*100:.2f}%)")
    print(f"  expert fires_desired on allowed: {expert_stats['n_fire_desired_on_allowed']}")
    print(f"  expert fires_desired on disallowed: {expert_stats['n_fire_desired_on_disallowed']} / {expert_stats['n_disallowed']} ({expert_stats['fire_desired_disallowed_rate']*100:.2f}%)")
    print(f"  (fire_desired on disallowed shows what expert WANTED to fire but mask disallowed)")
    print("=" * 60)
    print(f"[diag] VERDICT: {verdict}")
    print(f"[diag] NEXT:    {action}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "bc_hesi_rate": bc_hesi_rate,
            "bc_fire_rate_on_allowed": bc_fire_rate_on_allowed,
            "bc_allowed_steps": n_allowed,
            "bc_fire_steps": n_fire,
            "bc_episodes": args.episodes,
            "bc_mean_ep_len": float(np.mean(ep_lengths)),
            "expert": expert_stats,
            "verdict": verdict,
            "action": action,
        }, f, indent=2)
    print(f"[diag] saved to {args.out}")


if __name__ == "__main__":
    main()