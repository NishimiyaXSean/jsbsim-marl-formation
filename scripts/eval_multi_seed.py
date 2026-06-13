"""Multi-seed evaluation with Wilson CI for single-pursuit models."""

from __future__ import annotations
import argparse, os, sys, csv, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from src.environment.single_pursuit_env import SinglePursuitEnv
from scripts.train_single_pursuit import ResidualExpertWrapper


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def evaluate_seed(model_path: str, stage: float, episodes: int, record_tacview: bool = False):
    """Evaluate one model at a given curriculum stage."""
    base_env = SinglePursuitEnv(curriculum_stage=stage, record_tacview=record_tacview)
    env = ResidualExpertWrapper(base_env)
    model = PPO.load(model_path)

    successes = 0
    min_dists = []
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += rew
            if "min_dist" in info and info["min_dist"] > 0:
                ep_min_dist = min(ep_min_dist, info["min_dist"])

        if info.get("reason") == "success":
            successes += 1
        min_dists.append(ep_min_dist)
        rewards.append(total_r)

    return {
        "successes": successes,
        "capture_rate": successes / episodes,
        "avg_min_dist": np.mean(min_dists),
        "std_min_dist": np.std(min_dists),
        "avg_reward": np.mean(rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs to evaluate")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stage", type=float, default=None,
                        help="Curriculum stage (default: use final stage from training)")
    parser.add_argument("--csv", type=str, default="", help="Export CSV path")
    args = parser.parse_args()

    all_results = []
    for run_dir in args.runs:
        model_path = os.path.join(run_dir, "best_model.zip")
        if not os.path.exists(model_path):
            print(f"  SKIP {run_dir}: no best_model.zip")
            continue

        # Determine stage: use the final stage from training eval_metrics, or arg
        eval_csv = os.path.join(run_dir, "eval_metrics.csv")
        stage = args.stage
        if stage is None and os.path.exists(eval_csv):
            with open(eval_csv) as f:
                rows = list(csv.DictReader(f))
                stage = float(rows[-1]["stage"]) if rows else 1.0

        print(f"\n{'='*55}")
        print(f"Evaluating: {os.path.basename(run_dir)}")
        print(f"  Model:     {model_path}")
        print(f"  Stage:     {stage}")
        print(f"  Episodes:  {args.episodes}")

        result = evaluate_seed(model_path, stage, args.episodes, record_tacview=False)
        result["run"] = os.path.basename(run_dir)
        result["stage"] = stage
        all_results.append(result)

        p, lo, hi = wilson_ci(result["successes"], args.episodes)
        print(f"  Capture:   {result['capture_rate']:.1%}  (Wilson 95% CI: [{lo:.1%}, {hi:.1%}])")
        print(f"  Min dist:  {result['avg_min_dist']:.0f} ± {result['std_min_dist']:.0f} m")
        print(f"  Avg rew:   {result['avg_reward']:.0f}")

    # Summary
    if all_results:
        rates = [r["capture_rate"] for r in all_results]
        print(f"\n{'='*55}")
        print(f"SUMMARY: {len(all_results)} seeds")
        print(f"  Capture rates: {[f'{r:.1%}' for r in rates]}")
        print(f"  Mean:  {np.mean(rates):.1%} ± {np.std(rates):.1%}")

        # Overall Wilson CI (pooled)
        total_success = sum(r["successes"] for r in all_results)
        total_ep = args.episodes * len(all_results)
        p, lo, hi = wilson_ci(total_success, total_ep)
        print(f"  Pooled Wilson 95% CI: [{lo:.1%}, {hi:.1%}]")

        if args.csv:
            with open(args.csv, "w", newline="") as f:
                fields = ["run", "stage", "capture_rate", "avg_min_dist", "avg_reward",
                          "wilson_lo", "wilson_hi"]
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for r in all_results:
                    p, lo, hi = wilson_ci(r["successes"], args.episodes)
                    r.update(wilson_lo=lo, wilson_hi=hi)
                    writer.writerow(r)
            print(f"  CSV -> {args.csv}")


if __name__ == "__main__":
    main()
