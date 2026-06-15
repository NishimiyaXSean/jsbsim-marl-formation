"""Ablation study orchestrator for single-pursuit training.

Runs 4 configurations x 3 seeds = 12 training jobs at 200K timesteps each,
then produces a summary CSV comparing Stage 1 -> 1.5 transfer performance.

Usage:
    conda activate jsbsim_rl
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --seeds 0 1 2 3 4
    python scripts/run_ablation_study.py --steps 100000
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import (
    CubicActionWrapper,
    LeadPursuitRewardWrapper,
)
from scripts.train_single_pursuit import (
    CurriculumCallback,
    CURRICULUM_STAGES,
    EVAL_EPISODES,
    EVAL_FREQ,
    TARGET_CAPTURE_RATE_STAGE_1_2,
    ResidualExpertWrapper,
)

# ==============================================================================
#  Ablation configuration
# ==============================================================================

ABLATIONS = [
    {"name": "baseline",          "label": "BL",   "wrappers": []},
    {"name": "lead_pursuit",      "label": "RW",   "wrappers": [LeadPursuitRewardWrapper]},
    {"name": "cubic_action",      "label": "CA",   "wrappers": [CubicActionWrapper]},
    {"name": "cubic+lead",        "label": "CARW", "wrappers": [CubicActionWrapper, LeadPursuitRewardWrapper]},
]

STAGES_FOR_ABLATION = [1.0, 1.5]  # Only Stage 1.0 and 1.5

# PPO hyperparameters -- identical across all variants
PPO_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,          # entropy bonus to prevent policy collapse
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
        ortho_init=True,
        log_std_init=0.0,
    ),
)


def build_env(ablation_config: dict, record_tacview: bool = False):
    """Build the full env chain: SinglePursuitEnv -> wrappers... -> ResidualExpertWrapper."""
    base = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=record_tacview)
    for wrapper_cls in ablation_config.get("wrappers", []):
        base = wrapper_cls(base)
    wrapped = ResidualExpertWrapper(base)
    return wrapped


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def run_one(ablation_config: dict, seed: int, total_steps: int, output_dir: str):
    """Run one training job. Returns path to eval_metrics.csv."""
    label = ablation_config["label"]
    name = ablation_config["name"]
    run_name = f"{label}_s{seed}"
    log_dir = os.path.join(output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{label}] {name}  |  seed={seed}  |  steps={total_steps:,}")
    print(f"  Log: {log_dir}")
    print(f"{'='*60}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build envs
    train_env = build_env(ablation_config, record_tacview=False)
    train_env = Monitor(train_env, log_dir)
    eval_env = build_env(ablation_config, record_tacview=False)

    # PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir, **PPO_CONFIG)

    # Curriculum callback
    curriculum_cb = CurriculumCallback(eval_env, log_dir)
    # Override CURRICULUM_STAGES for ablation (2-stage only)
    import scripts.train_single_pursuit as train_mod
    original_stages = train_mod.CURRICULUM_STAGES
    train_mod.CURRICULUM_STAGES = STAGES_FOR_ABLATION
    try:
        model.learn(total_timesteps=total_steps, callback=curriculum_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("\n  Interrupted -- saving checkpoint...")
    finally:
        train_mod.CURRICULUM_STAGES = original_stages

    # Save final model (don't overwrite best_model saved by CurriculumCallback)
    model.save(os.path.join(log_dir, "model"))
    model.save(os.path.join(log_dir, "final_model"))

    # Save eval metrics CSV
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                               "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(curriculum_cb._eval_metrics)

    print(f"  Done -> {csv_path}")
    return csv_path


def summarize(output_dir: str):
    """Read all eval_metrics.csv files and produce a comparison summary."""
    rows = []
    for ablation in ABLATIONS:
        label = ablation["label"]
        name = ablation["name"]
        for seed in range(10):  # scan for existing seed dirs
            run_dir = os.path.join(output_dir, f"{label}_s{seed}")
            csv_path = os.path.join(run_dir, "eval_metrics.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                metrics = list(reader)
            if not metrics:
                continue

            # Peak Stage 1 capture rate
            stage1_rates = [float(r["capture_rate"]) for r in metrics
                           if float(r["stage"]) == 1.0]
            peak_s1 = max(stage1_rates) if stage1_rates else 0.0

            # Stage 1.5 transfer: use best Stage 1.5 capture rate
            stage15_metrics = [r for r in metrics if float(r["stage"]) == 1.5]
            s15_rate = 0.0
            s15_avg_dist = 0.0
            s15_total_evals = len(stage15_metrics)
            if stage15_metrics:
                s15_rates = [float(r["capture_rate"]) for r in stage15_metrics]
                s15_rate = max(s15_rates)
                s15_dists = [float(r["avg_min_dist"]) for r in stage15_metrics]
                s15_avg_dist = np.mean(s15_dists)

            # Time-to-advance (first timestep where stage >= 1.5)
            advance_step = int(float(metrics[0]["timesteps"]))
            for r in metrics:
                if float(r["stage"]) >= 1.5:
                    advance_step = int(float(r["timesteps"]))
                    break

            rows.append({
                "label": label,
                "name": name,
                "seed": seed,
                "peak_stage1": peak_s1,
                "stage15_best": s15_rate,
                "stage15_avg_dist": s15_avg_dist,
                "stage15_evals": s15_total_evals,
                "advance_step": advance_step,
            })

    if not rows:
        print("  No results found.")
        return

    # Save detailed CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    fieldnames = ["label", "name", "seed", "peak_stage1", "stage15_best",
                  "stage15_avg_dist", "stage15_evals", "advance_step"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary CSV -> {summary_path}")

    # Print ranked table by Stage 1.5 capture rate
    print(f"\n{'='*80}")
    print("ABLATION RESULTS -- Ranked by Stage 1.5 capture rate")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Var':<6} {'Name':<16} {'Seeds':<8} {'Peak S1':<10} {'Best S1.5':<12} {'95% CI':<20} {'Adv@':<10}")
    print("-" * 80)

    # Aggregate per variant
    variants = {}
    for r in rows:
        v = r["label"]
        if v not in variants:
            variants[v] = {"name": r["name"], "rows": []}
        variants[v]["rows"].append(r)

    # Sort by mean Stage 1.5 best
    ranked = sorted(variants.items(),
                    key=lambda kv: np.mean([r["stage15_best"] for r in kv[1]["rows"]]),
                    reverse=True)

    for rank, (label, vdata) in enumerate(ranked, 1):
        vrows = vdata["rows"]
        name = vdata["name"]
        n_seeds = len(vrows)
        peak_s1 = np.mean([r["peak_stage1"] for r in vrows])
        best_s15 = np.mean([r["stage15_best"] for r in vrows])
        total_successes = sum(int(r["stage15_best"] * EVAL_EPISODES) for r in vrows)
        p, lo, hi = wilson_ci(total_successes, n_seeds * EVAL_EPISODES)
        avg_adv = np.mean([r["advance_step"] for r in vrows])

        print(f"{rank:<6} {label:<6} {name:<16} {n_seeds:<8} "
              f"{peak_s1:<10.1%} {best_s15:<12.1%} "
              f"[{lo:.1%}, {hi:.1%}]  "
              f"{avg_adv:>8.0f}")

    print("-" * 80)
    print(f"  Winner: {ranked[0][0]} ({ranked[0][1]['name']})")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for single-pursuit training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                       help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--steps", type=int, default=200_000,
                       help="Total timesteps per run (default: 200000)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, just regenerate summary from existing CSVs")
    parser.add_argument("--ablation", type=str, nargs="+",
                       choices=["BL", "RW", "CA", "CARW"],
                       help="Run only specific ablations (e.g. --ablation BL CARW)")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.abspath(f"./marl_runs/ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ABLATION STUDY: Single-Pursuit Training Optimizations")
    print(f"  Variants:    {len(ABLATIONS)}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Total runs:  {len(ABLATIONS) * len(args.seeds)}")
    print(f"  Steps/run:   {args.steps:,}")
    print(f"  Stages:      {STAGES_FOR_ABLATION}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # Filter ablations if --ablation specified
    active = ABLATIONS
    if args.ablation:
        active = [a for a in ABLATIONS if a["label"] in args.ablation]
        print(f"  Running only: {[a['label'] for a in active]}")

    if not args.skip_training:
        for ablation_config in active:
            for seed in args.seeds:
                run_one(ablation_config, seed, args.steps, output_dir)

    # Generate summary
    print(f"\n{'='*60}")
    print("Generating summary...")
    summarize(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    main()
