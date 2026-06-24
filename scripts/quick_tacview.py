"""Quick Tacview + 3D trajectory plot from a trained model.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/quick_tacview.py marl_runs/v10_5_0624_0045/s0/best_model
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/quick_tacview.py marl_runs/v10_5_0624_0045/s0/best_model_diff_0.20
"""

import os, sys, warnings, logging, argparse
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import (
    ActionRepeatWrapper, BlendedActionWrapper, LeadPursuitRewardWrapper,
)
from scripts.train_single_pursuit import ResidualExpertWrapper


def build_env(difficulty: float = 0.15):
    """Match the training env chain exactly."""
    base = SinglePursuitEnv(difficulty_level=difficulty, record_tacview=True)
    base = BlendedActionWrapper(base, alpha=0.02)
    base = LeadPursuitRewardWrapper(base)
    wrapped = ResidualExpertWrapper(base)
    wrapped = ActionRepeatWrapper(wrapped, repeat_frames=5)
    return wrapped


def plot_trajectory(env: SinglePursuitEnv, out_dir: str, tag: str):
    """Generate 3D trajectory + altitude profile plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = env._tacview_frames
    if not frames:
        print("  No frames to plot!")
        return

    a_lat = np.array([f["pursuer"]["lat_deg"] for f in frames])
    a_lon = np.array([f["pursuer"]["lon_deg"] for f in frames])
    a_alt = np.array([f["pursuer"]["alt_m"] for f in frames])
    t_lat = np.array([f["target"]["lat_deg"] for f in frames])
    t_lon = np.array([f["target"]["lon_deg"] for f in frames])
    t_alt = np.array([f["target"]["alt_m"] for f in frames])

    ref_lat, ref_lon = 30.0, 120.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref_lat))
    a_x = (a_lat - ref_lat) * m_per_deg_lat
    a_y = (a_lon - ref_lon) * m_per_deg_lon
    t_x = (t_lat - ref_lat) * m_per_deg_lat
    t_y = (t_lon - ref_lon) * m_per_deg_lon

    # 3D trajectory
    fig = plt.figure(figsize=(16, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(a_x, a_y, a_alt, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax3d.plot(t_x, t_y, t_alt, "b-", lw=1.5, alpha=0.8, label="Target")
    ax3d.scatter(a_x[0], a_y[0], a_alt[0], color="darkred", s=80, marker="o", label="P start")
    ax3d.scatter(a_x[-1], a_y[-1], a_alt[-1], color="red", s=80, marker="x", label="P end")
    ax3d.scatter(t_x[0], t_y[0], t_alt[0], color="darkblue", s=80, marker="o", label="T start")
    ax3d.scatter(t_x[-1], t_y[-1], t_alt[-1], color="blue", s=80, marker="x", label="T end")
    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title(f"3D Trajectory — {tag}")
    ax3d.legend()

    # Top-down
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.plot(a_x, a_y, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax2d.plot(t_x, t_y, "b-", lw=1.5, alpha=0.8, label="Target")
    ax2d.scatter(a_x[0], a_y[0], color="darkred", s=80, marker="o")
    ax2d.scatter(a_x[-1], a_y[-1], color="red", s=80, marker="x")
    ax2d.scatter(t_x[0], t_y[0], color="darkblue", s=80, marker="o")
    ax2d.scatter(t_x[-1], t_y[-1], color="blue", s=80, marker="x")
    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title("Top-Down View")
    ax2d.legend()
    ax2d.axis("equal")

    traj_path = os.path.join(out_dir, f"{tag}_trajectory.png")
    fig.savefig(traj_path, dpi=150, bbox_inches="tight")
    print(f"  Trajectory plot -> {traj_path}")

    # Altitude profile
    fig2, ax_alt = plt.subplots(figsize=(10, 4))
    times = np.arange(len(frames)) * 0.5
    ax_alt.plot(times, a_alt, "r-", lw=1.5, label="Pursuer")
    ax_alt.plot(times, t_alt, "b-", lw=1.5, label="Target")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title(f"Altitude Profile — {tag}")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)
    alt_path = os.path.join(out_dir, f"{tag}_altitude.png")
    fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Altitude plot -> {alt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .zip (without .zip extension)")
    parser.add_argument("--difficulty", type=float, default=0.15, help="Difficulty for eval")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model_path
    tag = os.path.basename(args.model_path)
    out_dir = os.path.join("results", "v10_5_tacview", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {model_path}")
    model = PPO.load(model_path)

    env = build_env(difficulty=args.difficulty)

    successes = 0
    min_dists = []
    best_reward = -float("inf")
    best_ep_frames = None

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += rew
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])

        reason = info.get("reason", "timeout")
        if reason == "success":
            successes += 1
        min_dists.append(ep_min_dist)
        print(f"  Ep {ep+1:2d}: {reason:15s}  reward={total_r:+7.1f}  min_dist={ep_min_dist:.0f}m")

        if total_r > best_reward:
            best_reward = total_r
            best_ep_frames = list(env.unwrapped._tacview_frames)

    capture_rate = successes / args.episodes
    print(f"\n  Capture rate: {capture_rate:.0%}  ({successes}/{args.episodes})")
    print(f"  Avg min dist: {np.mean(min_dists):.0f} +/- {np.std(min_dists):.0f}m")

    # Tacview of best episode
    if best_ep_frames is not None:
        env.unwrapped._tacview_frames = best_ep_frames
        tacview_path = os.path.join(out_dir, f"{tag}_diff_{args.difficulty:.2f}.txt.acmi")
        env.unwrapped.export_tacview(tacview_path)
        print(f"  Tacview -> {tacview_path}")

    # Plots of best episode
    if best_ep_frames is not None:
        env.unwrapped._tacview_frames = best_ep_frames
        plot_trajectory(env.unwrapped, out_dir, f"{tag}_diff_{args.difficulty:.2f}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
