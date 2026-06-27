"""PN-guidance expert data generator for behavioural cloning (Phase 2 BC).

Runs Proportional Navigation (PN) guidance as an expert policy in
ContinuousPursuitEnv, recording (obs, action) pairs from successful
intercepts.  The resulting .npz dataset is fed to BC pretraining
before PPO fine-tuning.

PN → Action mapping
-------------------
  turn action  = K_p × heading_error / MAX_TURN_RATE_DPS    (clipped ±1)
  speed action = speed_schedule(current_dist)                (ramped)

This replaces the old discrete-ATA heuristic (heuristic_pursuit_test.py)
with continuous Box(2) actions suitable for ContinuousPursuitEnv.

Usage:
    python scripts/generate_pn_expert_data.py
    python scripts/generate_pn_expert_data.py --episodes 1000 --difficulty 0.3
    python scripts/generate_pn_expert_data.py --episodes 500 --difficulty-min 0.0 --difficulty-max 0.5
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.environment.continuous_pursuit_env import ContinuousPursuitEnv, MAX_TURN_RATE_DPS
from src.utils.pn_guidance import compute_pn_heading

# ── Expert hyperparameters (validated 2026-06-27: 100% success @ diff=0) ─
NAV_CONSTANT = 3.0         # PN navigation constant (3 = moderate lead)
KP_HEADING = 0.7           # P-controller gain: 10° error → ~50% turn
PN_DT = 0.5                # PN computation period (matches 2 Hz cruise rate)


def speed_schedule(current_dist: float) -> float:
    """Speed action ∈ [-1, 1] → [150, 350] m/s.

    High-speed cruise (350 m/s) for rapid closure at high difficulty.
    Energy-preserving ramp-down through terminal phase.
    """
    if current_dist > 1500.0:
        return 1.0                     # 350 m/s — full speed to close range
    elif current_dist > 800.0:
        return 0.7                     # 320 m/s — fast cruise
    elif current_dist > 500.0:
        frac = (current_dist - 500.0) / 300.0
        return 0.2 + 0.5 * frac        # 270→320 m/s
    elif current_dist > 200.0:
        frac = (current_dist - 200.0) / 300.0
        return -0.2 + 0.4 * frac       # 230→270 m/s  — terminal
    else:
        return -0.3                    # ~220 m/s — coast to kill


def compute_expert_action(
    env: ContinuousPursuitEnv,
    nav_constant: float = NAV_CONSTANT,
    kp_heading: float = KP_HEADING,
) -> np.ndarray:
    """Compute Box(2) expert action with energy-aware turn limiting.

    Returns
    -------
    action : np.ndarray  shape (2,)  [turn_factor, speed_factor]
    """
    # ── Raw state ────────────────────────────────────────────────────
    p_pos = env.pursuer.position_ned.copy()
    p_vel = env.pursuer.velocity_ned.copy()
    t_pos = env.target_ac.position_ned.copy()
    t_vel = env.target_ac.velocity_ned.copy()
    current_hdg = float(env.pursuer.state["yaw_deg"])
    current_dist = float(np.linalg.norm(p_pos - t_pos))

    # ── PN desired heading ───────────────────────────────────────────
    desired_hdg = compute_pn_heading(
        pursuer_ned=p_pos,
        pursuer_vel=p_vel,
        target_ned=t_pos,
        target_vel=t_vel,
        current_heading_deg=current_hdg,
        dt=PN_DT,
        nav_constant=nav_constant,
        max_turn_rate_dps=MAX_TURN_RATE_DPS,
    )

    # ── P-controller: heading error → turn action ─────────────────────
    hdg_error = (desired_hdg - current_hdg + 180.0) % 360.0 - 180.0
    turn_action = float(np.clip(kp_heading * hdg_error / MAX_TURN_RATE_DPS,
                                -1.0, 1.0))

    # ── Speed schedule ────────────────────────────────────────────────
    speed_action = speed_schedule(current_dist)

    return np.array([turn_action, speed_action], dtype=np.float32)


def run_expert_episode(
    env: ContinuousPursuitEnv,
    nav_constant: float = NAV_CONSTANT,
    kp_heading: float = KP_HEADING,
    max_steps: int = 600,    # 600 × 0.5 s = 300 s max (well above 60 s limit)
) -> tuple[list, list, dict]:
    """Run one episode.  Returns (obs_list, action_list, final_info).

    obs_list   — list of (25,)  float32 arrays
    action_list — list of (2,)  float32 arrays
    final_info  — env info dict from the terminal step
    """
    obs_list = []
    action_list = []

    obs, _ = env.reset()

    for _ in range(max_steps):
        action = compute_expert_action(env, nav_constant, kp_heading)

        obs_list.append(obs.copy())
        action_list.append(action.copy())

        obs, _reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return obs_list, action_list, info


def countdown_print(msg: str) -> None:
    """Print with flush so long-running scripts show progress."""
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PN-guidance expert trajectories for BC pretraining")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Total episodes to attempt (only successes kept)")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Fixed difficulty (overrides --difficulty-min/max)")
    parser.add_argument("--difficulty-min", type=float, default=0.20,
                        help="Minimum difficulty for uniform sampling")
    parser.add_argument("--difficulty-max", type=float, default=0.50,
                        help="Maximum difficulty for uniform sampling")
    parser.add_argument("--nav-constant", type=float, default=NAV_CONSTANT,
                        help="PN navigation constant")
    parser.add_argument("--kp", type=float, default=KP_HEADING,
                        help="P-controller gain for heading")
    parser.add_argument("--output-dir", type=str, default="./data/expert",
                        help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Difficulty sampling ───────────────────────────────────────────
    if args.difficulty is not None:
        diff_range = (args.difficulty, args.difficulty)
    else:
        diff_range = (args.difficulty_min, args.difficulty_max)

    rng = np.random.default_rng(args.seed)

    # ── Stats ─────────────────────────────────────────────────────────
    all_obs = []
    all_actions = []
    success_count = 0
    attempt = 0
    term_counts: dict = {}

    countdown_print(f"PN Expert Data Generator")
    countdown_print(f"  Episodes to attempt: {args.episodes}")
    countdown_print(f"  Difficulty range:    [{diff_range[0]:.2f}, {diff_range[1]:.2f}]")
    countdown_print(f"  Nav constant:        {args.nav_constant}")
    countdown_print(f"  Kp heading:          {args.kp}")
    countdown_print(f"  Output:              {output_dir}")

    while success_count < args.episodes:
        attempt += 1
        difficulty = float(rng.uniform(*diff_range))

        env = ContinuousPursuitEnv(
            lock_altitude=True,
            difficulty_level=difficulty,
            record_tacview=False,
        )

        try:
            obs_list, action_list, info = run_expert_episode(
                env, nav_constant=args.nav_constant, kp_heading=args.kp)
        except Exception as exc:
            countdown_print(f"  [{attempt}] Exception: {exc} — skipping")
            continue

        reason = info.get("reason", "unknown")
        term_counts[reason] = term_counts.get(reason, 0) + 1

        if reason == "success":
            success_count += 1
            all_obs.extend(obs_list)
            all_actions.extend(action_list)

            if success_count % 50 == 0 or success_count <= 10:
                dist = info.get("end_dist", 0)
                steps = len(obs_list)
                countdown_print(
                    f"  [{success_count}/{args.episodes}] success  "
                    f"diff={difficulty:.2f}  steps={steps}  "
                    f"min_dist={info.get('min_dist', 0):.0f}m  "
                    f"cumul_steps={len(all_obs)}")
        else:
            if attempt % 100 == 0:
                countdown_print(
                    f"  [{attempt} attempts]  successes={success_count}/{args.episodes}  "
                    f"last_reason={reason}  "
                    f"terms={dict(term_counts)}")

    # ── Save dataset ──────────────────────────────────────────────────
    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.float32)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    filename = f"pn_expert_{success_count}ep_{obs_array.shape[0]}steps_{timestamp}.npz"
    filepath = output_dir / filename

    np.savez_compressed(
        filepath,
        observations=obs_array,
        actions=act_array,
        episode_count=success_count,
        total_steps=obs_array.shape[0],
        difficulty_min=diff_range[0],
        difficulty_max=diff_range[1],
        nav_constant=args.nav_constant,
        kp_heading=args.kp,
    )

    countdown_print(f"\nDataset saved: {filepath}")
    countdown_print(f"  Episodes:     {success_count}")
    countdown_print(f"  Total steps:  {obs_array.shape[0]:,}")
    countdown_print(f"  Obs shape:    {obs_array.shape}")
    countdown_print(f"  Action shape: {act_array.shape}")
    countdown_print(f"  Termination distribution: {dict(term_counts)}")
    countdown_print(f"  Success rate: {success_count / attempt:.1%} ({success_count}/{attempt})")
    countdown_print("Done.")


if __name__ == "__main__":
    main()
