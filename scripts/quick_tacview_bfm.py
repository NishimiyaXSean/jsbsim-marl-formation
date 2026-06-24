"""Quick Tacview + 3D trajectory for BFM discrete pursuit models.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/quick_tacview_bfm.py marl_runs/bfm_pursuit_0624_1622_s0/best_model_diff_0.15
"""

import os, sys, warnings, logging, argparse
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from src.environment.bfm_pursuit_env import BFMPursuitEnv
from src.environment.ablation_wrappers import BlendedActionWrapper, LeadPursuitRewardWrapper
from src.dynamics.bfm_actions import describe_pursuit_action


def build_env(difficulty: float = 0.15):
    base = BFMPursuitEnv(difficulty_level=difficulty, record_tacview=True)
    base = BlendedActionWrapper(base, alpha=0.02)
    base = LeadPursuitRewardWrapper(base)
    return base


def plot_trajectory(env, out_dir: str, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = env.unwrapped._tacview_frames
    if not frames:
        print("  No frames to plot")
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
    fig.savefig(os.path.join(out_dir, f"{tag}_trajectory.png"), dpi=150, bbox_inches="tight")
    print(f"  Trajectory -> {os.path.join(out_dir, f'{tag}_trajectory.png')}")

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
    fig2.savefig(os.path.join(out_dir, f"{tag}_altitude.png"), dpi=150, bbox_inches="tight")
    print(f"  Altitude -> {os.path.join(out_dir, f'{tag}_altitude.png')}")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model (without .zip)")
    parser.add_argument("--difficulty", type=float, default=0.15)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path
    tag = os.path.basename(args.model_path)
    out_dir = os.path.join("results", "bfm_tacview", tag)
    os.makedirs(out_dir, exist_ok=True)

    model = PPO.load(model_path, device="cpu")
    env = build_env(difficulty=args.difficulty)

    successes = 0
    min_dists = []
    best_reward = -float("inf")
    best_frames = None
    best_actions = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        actions = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_r += rew
            actions.append(int(action))
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])

        reason = info.get("reason", "timeout")
        if reason == "success":
            successes += 1
        min_dists.append(ep_min_dist)
        print(f"  Ep {ep+1}: {reason:15s}  reward={total_r:+7.1f}  min_dist={ep_min_dist:.0f}m"
              f"  actions={[describe_pursuit_action(a) for a in actions[:10]]}...")

        if total_r > best_reward:
            best_reward = total_r
            best_frames = list(env.unwrapped._tacview_frames)
            best_actions = actions

    capture_rate = successes / args.episodes
    print(f"\n  Capture rate: {capture_rate:.0%}")
    print(f"  Avg min dist: {np.mean(min_dists):.0f} +/- {np.std(min_dists):.0f}m")
    if best_actions:
        print(f"  Best ep actions: {[describe_pursuit_action(a) for a in best_actions[:20]]}...")

    if best_frames:
        env.unwrapped._tacview_frames = best_frames
        tacview_path = os.path.join(out_dir, f"{tag}_diff_{args.difficulty:.2f}.txt.acmi")
        env.unwrapped.export_tacview(tacview_path)
        print(f"  Tacview -> {tacview_path}")
        plot_trajectory(env.unwrapped, out_dir, f"{tag}_diff_{args.difficulty:.2f}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
