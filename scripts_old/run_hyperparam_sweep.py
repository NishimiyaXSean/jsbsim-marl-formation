"""Orthogonal hyperparameter sweep for single-pursuit PPO training.

Runs a fractional factorial design (8 configs) x 1 seed (500k steps),
selects top-2 configs, then runs them with 5 seeds each.

Usage:
    conda activate jsbsim_rl
    JSBSIM_DEBUG=0 python scripts/run_hyperparam_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import warnings
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_single_pursuit import train_with_config


# =============================================================================
#  Sweep design - 4 params, 2 levels each, fractional factorial = 8 configs
# =============================================================================

SWEEP_PARAMS = {
    "lr": [1e-4, 3e-4],
    "ent_coef": [0.005, 0.01],
    "net_arch_pi": [[128, 128], [256, 128]],
    "n_steps": [2048, 4096],
}

FIXED_CONFIG = {
    "total_timesteps": 500_000,
    "batch_size": 256,
    "gamma": 0.99,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def generate_configs() -> List[Dict]:
    """Generate 8 fractional factorial configs (even-index parity rows of full factorial)."""
    all_levels = [
        list(enumerate(SWEEP_PARAMS["lr"])),
        list(enumerate(SWEEP_PARAMS["ent_coef"])),
        list(enumerate(SWEEP_PARAMS["net_arch_pi"])),
        list(enumerate(SWEEP_PARAMS["n_steps"])),
    ]
    full = list(product(*all_levels))
    configs = []
    for combo in full:
        indices = [c[0] for c in combo]
        # Fractional factorial: sum of indices even
        if sum(indices) % 2 == 0:
            configs.append({
                "lr": combo[0][1],
                "ent_coef": combo[1][1],
                "net_arch_pi": combo[2][1],
                "n_steps": combo[3][1],
            })
    return configs


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


def evaluate_config(model_path: str, stage: float = 3.0, n_episodes: int = 50) -> Dict:
    """Evaluate a trained model on a given curriculum stage."""
    from src.environment.single_pursuit_env import SinglePursuitEnv
    from scripts.train_single_pursuit import ResidualExpertWrapper
    from stable_baselines3 import PPO

    env = SinglePursuitEnv(curriculum_stage=stage, record_tacview=False)
    wrapper = ResidualExpertWrapper(env)
    model = PPO.load(model_path)

    successes = 0
    min_dists = []
    intercept_times = []

    for _ in range(n_episodes):
        obs, _ = wrapper.reset()
        done = False
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])
        if info.get("reason") == "success":
            successes += 1
            intercept_times.append(env._step_counter / 60.0)
        else:
            intercept_times.append(120.0)
        min_dists.append(ep_min_dist)

    p, lo, hi = wilson_ci(successes, n_episodes)
    return {
        "capture_rate": p,
        "ci_low": lo,
        "ci_high": hi,
        "avg_min_dist": float(np.mean(min_dists)),
        "std_min_dist": float(np.std(min_dists)),
        "avg_intercept_time": float(np.mean(intercept_times)),
        "n_episodes": n_episodes,
        "successes": successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for single-pursuit PPO")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without training")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(f"results/sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = generate_configs()
    print(f"{'='*60}")
    print(f"Hyperparameter Sweep - {len(configs)} configs")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"  Config {i+1}: {cfg}")
        return

    # ==================== Phase 1: Run all 8 configs with seed=0 ====================
    phase1_results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'─'*50}")
        print(f"Phase 1 - Config {i+1}/{len(configs)}: {cfg}")
        print(f"{'─'*50}")

        run_name = f"sweep_cfg{i+1}_s0"
        log_dir = output_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_path = train_with_config(
                seed=0,
                log_dir=str(log_dir),
                learning_rate=cfg["lr"],
                ent_coef=cfg["ent_coef"],
                net_arch_pi=cfg["net_arch_pi"],
                n_steps=cfg["n_steps"],
                **FIXED_CONFIG,
            )

            # Evaluate on Stage 3
            result = evaluate_config(model_path, stage=3.0, n_episodes=50)
            result["config"] = cfg
            result["config_id"] = f"cfg{i+1}"
            phase1_results.append(result)
            print(f"  Result: capture_rate={result['capture_rate']:.2%} "
                  f"[{result['ci_low']:.2%}, {result['ci_high']:.2%}] "
                  f"min_dist={result['avg_min_dist']:.0f}±{result['std_min_dist']:.0f}m")
        except Exception as e:
            print(f"  ERROR: {e}")
            phase1_results.append({
                "capture_rate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "avg_min_dist": 10000.0, "std_min_dist": 0.0,
                "avg_intercept_time": 120.0, "n_episodes": 50, "successes": 0,
                "config": cfg, "config_id": f"cfg{i+1}",
            })

    # ==================== Phase 2: Rank, pick top-2 ====================
    phase1_results.sort(key=lambda r: r["capture_rate"], reverse=True)
    top2 = phase1_results[:2]
    print(f"\n{'='*60}")
    print("Top-2 Configs:")
    for r in top2:
        print(f"  {r['config_id']}: capture_rate={r['capture_rate']:.2%}  {r['config']}")

    # ==================== Phase 3: Run top-2 with 5 seeds each ====================
    all_results = []
    for rank, result in enumerate(top2):
        cfg = result["config"]
        cfg_label = f"top{rank+1}"
        print(f"\n{'─'*50}")
        print(f"Phase 3 - {cfg_label}: {cfg} x 5 seeds")
        print(f"{'─'*50}")

        seed_results = []
        for seed in range(5):
            run_name = f"sweep_{cfg_label}_s{seed}"
            log_dir = output_dir / run_name
            log_dir.mkdir(parents=True, exist_ok=True)

            try:
                model_path = train_with_config(
                    seed=seed,
                    log_dir=str(log_dir),
                    learning_rate=cfg["lr"],
                    ent_coef=cfg["ent_coef"],
                    net_arch_pi=cfg["net_arch_pi"],
                    n_steps=cfg["n_steps"],
                    **FIXED_CONFIG,
                )

                eval_result = evaluate_config(model_path, stage=3.0, n_episodes=50)
                eval_result["seed"] = seed
                eval_result["config_id"] = cfg_label
                seed_results.append(eval_result)
                print(f"    seed={seed}: capture_rate={eval_result['capture_rate']:.2%} "
                      f"min_dist={eval_result['avg_min_dist']:.0f}±{eval_result['std_min_dist']:.0f}m")
            except Exception as e:
                print(f"    seed={seed}: ERROR: {e}")
                seed_results.append({
                    "capture_rate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                    "avg_min_dist": 10000.0, "std_min_dist": 0.0,
                    "avg_intercept_time": 120.0, "seed": seed,
                    "config_id": cfg_label,
                })

        cr_values = [r["capture_rate"] for r in seed_results]
        all_results.append({
            "config_id": cfg_label,
            "config": cfg,
            "mean_capture_rate": float(np.mean(cr_values)),
            "std_capture_rate": float(np.std(cr_values)),
            "mean_min_dist": float(np.mean([r["avg_min_dist"] for r in seed_results])),
            "mean_intercept_time": float(np.mean([r["avg_intercept_time"] for r in seed_results])),
            "seed_details": seed_results,
        })

    # ==================== Write report CSV ====================
    report_path = output_dir / "report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "seed", "capture_rate", "ci_low", "ci_high",
                          "avg_min_dist", "std_min_dist", "avg_intercept_time"])
        for top_result in all_results:
            for sd in top_result["seed_details"]:
                writer.writerow([
                    sd["config_id"], sd.get("seed", ""),
                    f"{sd['capture_rate']:.4f}",
                    f"{sd['ci_low']:.4f}", f"{sd['ci_high']:.4f}",
                    f"{sd['avg_min_dist']:.1f}", f"{sd['std_min_dist']:.1f}",
                    f"{sd['avg_intercept_time']:.1f}",
                ])

    # ==================== Write summary ====================
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Hyperparameter Sweep Summary - {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Phase 1 Results (8 configs x seed=0):\n")
        for r in phase1_results:
            f.write(f"  {r['config_id']}: capture_rate={r['capture_rate']:.2%} "
                    f"[{r['ci_low']:.2%}, {r['ci_high']:.2%}]  "
                    f"min_dist={r['avg_min_dist']:.0f}m  {r['config']}\n")

        f.write(f"\nPhase 3 Results (top-2 x 5 seeds):\n")
        for top in all_results:
            f.write(f"  {top['config_id']}: mean_capture_rate={top['mean_capture_rate']:.2%} "
                    f"±{top['std_capture_rate']:.2%}  "
                    f"mean_min_dist={top['mean_min_dist']:.0f}m  "
                    f"mean_intercept={top['mean_intercept_time']:.1f}s  "
                    f"config={top['config']}\n")
            f.write(f"    Per-seed: " +
                    ", ".join(f"s{sd['seed']}={sd['capture_rate']:.2%}"
                              for sd in top["seed_details"]) + "\n")

    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    main()
