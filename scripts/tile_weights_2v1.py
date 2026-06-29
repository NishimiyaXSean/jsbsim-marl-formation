"""Tile Phase 3.6 single-pursuer weights onto 2v1 FormationEnv policy.

Exploits symmetry: both pursuers share identical physics and observation
formats.  The 33-dim per-pursuer observation is tiled across the 66-dim
input, and the 2-dim per-pursuer action output is duplicated across the
4-dim output.

Usage:
    python scripts/tile_weights_2v1.py
    python scripts/tile_weights_2v1.py --source marl_runs/phase2_continuous_0629_1447_s42_bc/phase2_final.zip
"""

from __future__ import annotations

import argparse, os, sys, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO

from src.environment.formation_env import FormationEnv

OBS_PER_PURSUER = 33     # 27 base + 6 mate
ACT_PER_PURSUER = 2       # turn, speed
NUM_PURSUERS = 2


def tile_weights(source_model: PPO, target_model: PPO):
    """Tile single-pursuer weights onto multi-pursuer policy via symmetry."""
    src_state = source_model.policy.state_dict()
    tgt_state = target_model.policy.state_dict()

    tiled_count = 0
    random_count = 0

    for key in tgt_state:
        src_key = key  # same architecture, different dimensions

        if key not in src_state:
            print(f"  [SKIP] {key} — not in source (random init)")
            random_count += 1
            continue

        src_val = src_state[src_key]
        tgt_val = tgt_state[key]

        if src_val.shape == tgt_val.shape:
            # Same shape — direct copy (e.g. log_std)
            tgt_state[key] = src_val.clone()
            tiled_count += 1
            continue

        # ── First layer: tile input weights across pursuer feature blocks
        if 'mlp_extractor.policy_net' in key and src_val.dim() == 2:
            if src_val.shape[1] == OBS_PER_PURSUER and tgt_val.shape[1] == OBS_PER_PURSUER * NUM_PURSUERS:
                # Weight matrix: tile columns (input features)
                # Phase 3.6: [256, 33] -> tiled: [256, 66]
                tiled = torch.cat([src_val, src_val], dim=1)  # duplicate for 2 pursuers
                tgt_state[key] = tiled
                tiled_count += 1
                continue
            elif src_val.shape[1] == OBS_PER_PURSUER + 4 and tgt_val.shape[1] == OBS_PER_PURSUER * NUM_PURSUERS + 4:
                # shared_net case
                tiled = torch.cat([src_val[:, :OBS_PER_PURSUER], src_val[:, :OBS_PER_PURSUER]], dim=1)
                tgt_state[key] = tiled
                tiled_count += 1
                continue

        # ── First layer weight for value_net
        if 'mlp_extractor.value_net' in key and src_val.dim() == 2:
            if src_val.shape[1] == OBS_PER_PURSUER and tgt_val.shape[1] == OBS_PER_PURSUER * NUM_PURSUERS:
                # Don't tile value net — random init for new Critic
                print(f"  [RANDOM] {key} — Critic trains from scratch")
                random_count += 1
                continue

        # ── Action output layer: tile output neurons
        if 'action_net' in key and src_val.dim() == 2:
            if src_val.shape[0] == ACT_PER_PURSUER and tgt_val.shape[0] == ACT_PER_PURSUER * NUM_PURSUERS:
                # [2, 256] -> [4, 256]: duplicate rows
                tgt_state[key] = torch.cat([src_val, src_val], dim=0)
                tiled_count += 1
                continue
            if src_val.shape[0] == ACT_PER_PURSUER:
                tgt_state[key][:ACT_PER_PURSUER] = src_val
                tgt_state[key][ACT_PER_PURSUER:] = src_val
                tiled_count += 1
                continue

        # ── Bias vectors
        if 'action_net' in key and src_val.dim() == 1:
            if len(src_val) == ACT_PER_PURSUER and len(tgt_val) == ACT_PER_PURSUER * NUM_PURSUERS:
                tgt_state[key] = torch.cat([src_val, src_val], dim=0)
                tiled_count += 1
                continue

        # ── Everything else (same-dimension intermediate layers) ─
        if src_val.shape == tgt_val.shape:
            tgt_state[key] = src_val.clone()
            tiled_count += 1
            continue

        print(f"  [RANDOM] {key} — shape mismatch {list(src_val.shape)} -> {list(tgt_val.shape)}")
        random_count += 1

    target_model.policy.load_state_dict(tgt_state)
    print(f"Tiled: {tiled_count} params  |  Random init: {random_count} params")
    return tiled_count, random_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default="marl_runs/phase2_continuous_0629_1447_s42_bc/phase2_final.zip")
    parser.add_argument("--output", type=str, default="./data/expert/tiled_2v1_phase36.zip")
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    print(f"Loading Phase 3.6 source: {args.source}")
    src_model = PPO.load(args.source, device="cpu")

    # Build 2v1 target
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=0.0)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                         activation_fn=torch.nn.Tanh)
    tgt_model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    verbose=0, device="cpu")

    print("Tiling weights...")
    tiled, random = tile_weights(src_model, tgt_model)

    tgt_model.save(args.output)
    print(f"Saved: {args.output}")
    print(f"Actor: tiled from Phase 3.6 (symmetry)")
    print(f"Critic: random init (will train from scratch)")


if __name__ == "__main__":
    main()
