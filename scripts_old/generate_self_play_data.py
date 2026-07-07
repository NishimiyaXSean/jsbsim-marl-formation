"""Self-distillation: use the trained RL model to generate expert trajectories.

Runs the Phase 3v3 checkpoint at high difficulty (diff 0.50–0.65),
collects only successful episodes, and saves them as a BC dataset.
This is DAgger-style expert iteration — the model teaches itself
how to handle OOD initial conditions.

Usage:
    python scripts/generate_self_play_data.py
    python scripts/generate_self_play_data.py --episodes 200 --diff-min 0.55 --diff-max 0.65
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from src.environment.continuous_pursuit_env import ContinuousPursuitEnv
from src.environment.ablation_wrappers import LeadPursuitRewardWrapper


def main():
    parser = argparse.ArgumentParser(description="Self-distillation data generator")
    parser.add_argument("--model", type=str,
                        default="marl_runs/phase2_continuous_0627_2326_s42_bc/phase2_final.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--diff-min", type=float, default=0.50)
    parser.add_argument("--diff-max", type=float, default=0.65)
    parser.add_argument("--output-dir", type=str, default="./data/expert_selfplay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device="cpu")

    rng = np.random.default_rng(args.seed)

    all_obs, all_actions = [], []
    success_count = 0
    attempt = 0
    term_counts = {}

    print(f"Self-distillation: {args.episodes} successes, diff [{args.diff_min:.2f}, {args.diff_max:.2f}]")

    while success_count < args.episodes:
        attempt += 1
        difficulty = float(rng.uniform(args.diff_min, args.diff_max))

        env = ContinuousPursuitEnv(
            lock_altitude=True, difficulty_level=difficulty, record_tacview=False)
        env = LeadPursuitRewardWrapper(env)

        obs, _ = env.reset()
        ep_obs, ep_actions = [], []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

        reason = info.get("reason", "unknown")
        term_counts[reason] = term_counts.get(reason, 0) + 1

        if reason == "success":
            success_count += 1
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)

            if success_count % 25 == 0 or success_count <= 5:
                print(f"  [{success_count}/{args.episodes}] success  diff={difficulty:.2f}  "
                      f"steps={len(ep_obs)}  cumul={len(all_obs)}")

        if attempt % 200 == 0:
            total = sum(term_counts.values())
            sr = term_counts.get("success", 0) / total * 100 if total > 0 else 0
            print(f"  [{attempt} attempts]  successes={success_count}/{args.episodes}  "
                  f"SR={sr:.1f}%  terms={dict(term_counts)}")

    # Save
    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.float32)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    fname = f"selfplay_{success_count}ep_{obs_array.shape[0]}steps_{timestamp}.npz"
    fpath = output_dir / fname

    np.savez_compressed(fpath,
        observations=obs_array, actions=act_array,
        episode_count=success_count, total_steps=obs_array.shape[0],
        difficulty_min=args.diff_min, difficulty_max=args.diff_max,
        source_model=args.model)

    total = sum(term_counts.values())
    print(f"\nSaved: {fpath}")
    print(f"  Episodes: {success_count}  Steps: {obs_array.shape[0]:,}")
    print(f"  Obs: {obs_array.shape}  Act: {act_array.shape}")
    print(f"  Success rate: {success_count}/{attempt} ({success_count/attempt*100:.1f}%)")
    print(f"  Terms: {dict(term_counts)}")
    print("Done.")


if __name__ == "__main__":
    main()
