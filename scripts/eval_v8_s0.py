"""Generate Tacview + trajectory plots for V8 s0 best_model and final_model."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings, logging
warnings.filterwarnings("ignore")
for mod in ["jsbsim", "gymnasium"]:
    logging.getLogger(mod).setLevel(logging.CRITICAL)

from stable_baselines3 import PPO
from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import CubicActionWrapper, LeadPursuitRewardWrapper
from scripts.train_single_pursuit import ResidualExpertWrapper
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path("results/v8_s0_eval")
os.makedirs(base_dir, exist_ok=True)
os.makedirs("data/tacview", exist_ok=True)

models = {
    "best_model": "marl_runs/ablation_0623_1058/CARW_s0/best_model.zip",
    "final_model": "marl_runs/ablation_0623_1058/CARW_s0/final_model.zip",
}

DIFFICULTY = 0.15
N_EPISODES = 30

for model_name, model_path in models.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} (difficulty={DIFFICULTY})")
    print(f"{'='*60}")

    model = PPO.load(model_path)

    episodes = []
    for ep in range(N_EPISODES):
        env = SinglePursuitEnv(difficulty_level=DIFFICULTY, record_tacview=False)
        env = CubicActionWrapper(env)
        env = LeadPursuitRewardWrapper(env)
        env = ResidualExpertWrapper(env)

        obs, _ = env.reset()
        done = False
        total_r = 0.0
        reason = "timeout"
        min_dist = 8000.0
        traj_p, traj_t, times = [], [], []

        while not done:
            t = env.unwrapped._step_counter / 60.0
            traj_p.append(env.unwrapped.pursuer.position_ned.copy())
            traj_t.append(env.unwrapped.target_ac.position_ned.copy())
            times.append(t)

            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += rew
            if "reason" in info: reason = info["reason"]
            if "min_dist" in info: min_dist = min(min_dist, info["min_dist"])

        episodes.append({
            "p": np.array(traj_p),
            "t": np.array(traj_t),
            "times": np.array(times),
            "total_r": total_r,
            "reason": reason,
            "min_dist": min_dist,
        })
        print(f"  Ep {ep+1:2d}: {reason:15s}  reward={total_r:+8.0f}  min_dist={min_dist:6.0f}m  steps={len(traj_p)}")

    reasons = [e["reason"] for e in episodes]
    n_success = sum(1 for r in reasons if r == "success")
    n_stall = sum(1 for r in reasons if r == "stall")
    min_dists = [e["min_dist"] for e in episodes]
    rewards = [e["total_r"] for e in episodes]

    print(f"\n  Summary: {n_success}/{N_EPISODES} success ({n_success/N_EPISODES:.0%})")
    print(f"  Avg min_dist: {np.mean(min_dists):.0f} +- {np.std(min_dists):.0f}m")
    print(f"  Avg reward: {np.mean(rewards):.0f} +- {np.std(rewards):.0f}")
    print(f"  Terminations: success={n_success}, stall={n_stall}")

    # Tacview for BEST episode
    best_ep = max(range(len(episodes)), key=lambda i: episodes[i]["total_r"])
    best = episodes[best_ep]

    env_tac = SinglePursuitEnv(difficulty_level=DIFFICULTY, record_tacview=True)
    env_tac = CubicActionWrapper(env_tac)
    env_tac = LeadPursuitRewardWrapper(env_tac)
    env_tac = ResidualExpertWrapper(env_tac)

    obs, _ = env_tac.reset()
    done, step = False, 0
    max_steps = len(best["p"])
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env_tac.step(action)
        done = terminated or truncated
        step += 1

    tac_path = base_dir / f"{model_name}_best_engagement.txt.acmi"
    env_tac.unwrapped.export_tacview(str(tac_path))
    print(f"\n  Tacview exported: {tac_path}")

    # Tacview for WORST stall episode
    stall_eps = [i for i, e in enumerate(episodes) if e["reason"] == "stall"]
    if stall_eps:
        worst_stall = max(stall_eps, key=lambda i: episodes[i]["min_dist"])
        worst = episodes[worst_stall]

        env_tac2 = SinglePursuitEnv(difficulty_level=DIFFICULTY, record_tacview=True)
        env_tac2 = CubicActionWrapper(env_tac2)
        env_tac2 = LeadPursuitRewardWrapper(env_tac2)
        env_tac2 = ResidualExpertWrapper(env_tac2)

        obs, _ = env_tac2.reset()
        done, step = False, 0
        max_steps = len(worst["p"])
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env_tac2.step(action)
            done = terminated or truncated
            step += 1

        tac_path2 = base_dir / f"{model_name}_worst_stall_engagement.txt.acmi"
        env_tac2.unwrapped.export_tacview(str(tac_path2))
        print(f"  Tacview (worst stall) exported: {tac_path2}")

# Trajectory plots
print(f"\n{'='*60}")
print("Generating trajectory plots...")
print(f"{'='*60}")

for model_name, episodes in [("best_model", [])]:
    pass  # filled below

all_eps = {}
for model_name, model_path in models.items():
    # Re-evaluate for plots (collect trajectories)
    model = PPO.load(model_path)
    episodes = []
    for ep in range(N_EPISODES):
        env = SinglePursuitEnv(difficulty_level=DIFFICULTY, record_tacview=False)
        env = CubicActionWrapper(env)
        env = LeadPursuitRewardWrapper(env)
        env = ResidualExpertWrapper(env)

        obs, _ = env.reset()
        done = False
        total_r = 0.0
        reason = "timeout"
        min_dist = 8000.0
        traj_p, traj_t = [], []

        while not done:
            traj_p.append(env.unwrapped.pursuer.position_ned.copy())
            traj_t.append(env.unwrapped.target_ac.position_ned.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += rew
            if "reason" in info: reason = info["reason"]
            if "min_dist" in info: min_dist = min(min_dist, info["min_dist"])

        episodes.append({
            "p": np.array(traj_p),
            "t": np.array(traj_t),
            "total_r": total_r,
            "reason": reason,
            "min_dist": min_dist,
        })
    all_eps[model_name] = episodes

for model_name, episodes in all_eps.items():
    n_success = sum(1 for e in episodes if e["reason"] == "success")
    min_dists = [e["min_dist"] for e in episodes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    # All trajectories overlaid
    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(18, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    for i, ep in enumerate(episodes):
        p, t = ep["p"], ep["t"]
        label = f"#{i+1}" if i < 10 else None
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[i], lw=0.6, alpha=0.5, label=label)
        ax2d.plot(p[:, 0], p[:, 1], color=colors[i], lw=0.6, alpha=0.5, label=label)
        if ep["reason"] == "success":
            ax3d.scatter(*p[-1], color=colors[i], s=30, marker="*", zorder=5)
            ax2d.scatter(*p[-1, :2], color=colors[i], s=30, marker="*", zorder=5)

    ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title(f"{model_name} | {N_EPISODES} Episodes (diff={DIFFICULTY}) | "
                   f"{n_success}/{N_EPISODES} success, avg min_dist={np.mean(min_dists):.0f}m")
    ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)")
    ax2d.set_title("Top-Down View")
    ax2d.axis("equal")
    fig.tight_layout()
    fig.savefig(base_dir / f"{model_name}_all_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {model_name}_all_trajectories.png saved")

    # Distance profiles
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ep in enumerate(episodes):
        p, t = ep["p"], ep["t"]
        min_len = min(len(p), len(t))
        dists = np.linalg.norm(p[:min_len] - t[:min_len], axis=1)
        times_d = np.arange(min_len) * 0.1
        ax.plot(times_d, dists, color=colors[i], lw=0.8, alpha=0.7)
    ax.axhline(y=200, color="green", ls="--", alpha=0.5, label="Capture (200m)")
    ax.axhline(y=800, color="orange", ls="--", alpha=0.5, label="Zone-of-Death HI")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    ax.set_title(f"{model_name} | Distance Profiles ({N_EPISODES} eps, diff={DIFFICULTY})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(base_dir / f"{model_name}_distance_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {model_name}_distance_profiles.png saved")

    # Altitude profiles
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ep in enumerate(episodes):
        p = ep["p"]
        times_p = np.arange(len(p)) * 0.1
        ax.plot(times_p, p[:, 2], color=colors[i], lw=0.8, alpha=0.7)
    ax.axhline(y=10, color="red", ls="--", alpha=0.5, label="Ground (10m)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
    ax.set_title(f"{model_name} | Altitude Profiles ({N_EPISODES} eps, diff={DIFFICULTY})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(base_dir / f"{model_name}_altitude_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {model_name}_altitude_profiles.png saved")

print(f"\nAll outputs saved to: {base_dir}/")
for f in sorted(base_dir.glob("*")):
    size = os.path.getsize(f)
    print(f"  {f.name} ({size:,} bytes)")
