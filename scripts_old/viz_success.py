"""Visualize successful pursuit episodes from the best trained model."""
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

os.makedirs("results/v4_final", exist_ok=True)

# Load CARW_s1 model (best post-collapse recovery: 88% non-zero)
model = PPO.load("marl_runs/ablation_0617_1649/CARW_s1/model.zip")

# Build env ONCE, reuse across episodes (matching training callback pattern)
base = SinglePursuitEnv(difficulty_level=0.0, record_tacview=False)
base = CubicActionWrapper(base)
base = LeadPursuitRewardWrapper(base)
eval_env = ResidualExpertWrapper(base)
eval_env.difficulty_level = 0.15

success_trajs = []
total_ep = 0
while len(success_trajs) < 5 and total_ep < 200:
    obs, _ = eval_env.reset()
    done = False; total_r = 0.0
    traj = {"p": [], "t": [], "times": []}; reason = "timeout"; md = 8000.0

    while not done:
        t = eval_env.unwrapped._step_counter / 60.0
        traj["p"].append(eval_env.unwrapped.pursuer.position_ned.copy())
        traj["t"].append(eval_env.unwrapped.target_ac.position_ned.copy())
        traj["times"].append(t)
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated; total_r += rew
        if "reason" in info: reason = info["reason"]
        if "min_dist" in info: md = min(md, info["min_dist"])

    total_ep += 1
    if reason == "success":
        traj["p"] = np.array(traj["p"]); traj["t"] = np.array(traj["t"])
        traj["times"] = np.array(traj["times"])
        traj["total_r"] = total_r; traj["md"] = md
        success_trajs.append(traj)
        print(f"SUCCESS #{len(success_trajs)}: ep={total_ep} reward={total_r:+.0f} min_dist={md:.0f}m steps={len(traj['p'])}")

print(f"\nCollected {len(success_trajs)} successes in {total_ep} episodes ({len(success_trajs)/total_ep:.1%})")

# Generate Tacview for each success
print("\nGenerating Tacview files...")
for i, traj in enumerate(success_trajs):
    env_tac = SinglePursuitEnv(difficulty_level=0.0, record_tacview=True)
    env_tac = CubicActionWrapper(env_tac)
    env_tac = LeadPursuitRewardWrapper(env_tac)
    env_tac = ResidualExpertWrapper(env_tac)
    env_tac.difficulty_level = 0.15

    obs, _ = env_tac.reset()
    done = False; step = 0; max_steps = len(traj["p"])
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env_tac.step(action)
        done = terminated or truncated; step += 1

    tacview_path = f"results/v4_final/success_{i+1:02d}.txt.acmi"
    env_tac.unwrapped.export_tacview(tacview_path)
    print(f"  #{i+1}: {tacview_path}")

# ── 3D Trajectory Plots ────────────────────────────────────────────────────
print("\nGenerating trajectory plots...")
colors = plt.cm.Set1(np.linspace(0, 1, len(success_trajs)))

# Best individual
best = max(success_trajs, key=lambda t: t["total_r"])
p, t = best["p"], best["t"]
fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")

ax3d.plot(p[:, 0], p[:, 1], p[:, 2], "r-", lw=1.5, alpha=0.9, label="Pursuer (F-16)")
ax3d.plot(t[:, 0], t[:, 1], t[:, 2], "b-", lw=1.5, alpha=0.9, label="Target")
ax3d.scatter(*p[0], c="darkred", s=120, marker="o", label="P start", zorder=5)
ax3d.scatter(*p[-1], c="red", s=120, marker="*", label="P end (capture!)", zorder=5)
ax3d.scatter(*t[0], c="darkblue", s=100, marker="o", label="T start", zorder=5)
ax3d.scatter(*t[-1], c="blue", s=100, marker="x", label="T end", zorder=5)
ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")
ax3d.set_title(f"Best Success: {len(p)*0.1:.1f}s, min_dist={best['md']:.0f}m, reward={best['total_r']:+.0f}")
ax3d.legend(fontsize=8, loc="upper left")

ax2d.plot(p[:, 0], p[:, 1], "r-", lw=1.5, alpha=0.9, label="Pursuer")
ax2d.plot(t[:, 0], t[:, 1], "b-", lw=1.5, alpha=0.9, label="Target")
ax2d.scatter(*p[0, :2], c="darkred", s=120, marker="o")
ax2d.scatter(*p[-1, :2], c="red", s=120, marker="*")
ax2d.scatter(*t[0, :2], c="darkblue", s=100, marker="o")
ax2d.scatter(*t[-1, :2], c="blue", s=100, marker="x")
d0 = np.linalg.norm(p[0] - t[0]); d1 = np.linalg.norm(p[-1] - t[-1])
ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)")
ax2d.set_title(f"Top-Down: start={d0:.0f}m -> end={d1:.0f}m"); ax2d.axis("equal"); ax2d.legend(fontsize=8)
fig.savefig("results/v4_final/trajectory_best_success.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  Best trajectory: results/v4_final/trajectory_best_success.png")

# All successes overlay
fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
for i, traj in enumerate(success_trajs):
    p, t = traj["p"], traj["t"]
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[i], lw=1.2, alpha=0.8, label=f"#{i+1} ({len(p)*0.1:.0f}s)")
    ax3d.scatter(*p[-1], color=colors[i], s=60, marker="*")
    ax2d.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.2, alpha=0.8, label=f"#{i+1}")
    ax2d.scatter(*p[-1, :2], color=colors[i], s=60, marker="*")
ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")
ax3d.set_title(f"{len(success_trajs)} Success Trajectories Overlaid"); ax3d.legend(fontsize=7)
ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)")
ax2d.set_title("Top-Down Overlay"); ax2d.axis("equal"); ax2d.legend(fontsize=7)
fig.savefig("results/v4_final/trajectory_all_successes.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  All successes: results/v4_final/trajectory_all_successes.png")

# Altitude profiles
fig, ax = plt.subplots(figsize=(12, 5))
for i, traj in enumerate(success_trajs):
    p, t = traj["p"], traj["t"]
    times_p = np.arange(len(p)) * 0.1
    times_t = np.arange(len(t)) * 0.1
    ax.plot(times_p, p[:, 2], color=colors[i], lw=1.2, label=f"#{i+1} Pursuer")
    ax.plot(times_t, t[:, 2], "--", color=colors[i], lw=0.8, alpha=0.5, label=f"#{i+1} Target")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
ax.set_title("Altitude Profiles — All Success Episodes"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
fig.savefig("results/v4_final/altitude_profiles.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  Altitude profiles: results/v4_final/altitude_profiles.png")

# Distance vs time
fig, ax = plt.subplots(figsize=(12, 5))
for i, traj in enumerate(success_trajs):
    p, t = traj["p"], traj["t"]
    min_len = min(len(p), len(t))
    dists = np.linalg.norm(p[:min_len] - t[:min_len], axis=1)
    times = np.arange(min_len) * 0.1
    ax.plot(times, dists, color=colors[i], lw=1.5, label=f"#{i+1}")
ax.axhline(y=200, color="green", ls="--", alpha=0.5, label="Capture threshold (200m)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
ax.set_title("Distance-to-Target — All Success Episodes"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.savefig("results/v4_final/distance_profiles.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  Distance profiles: results/v4_final/distance_profiles.png")

print(f"\nDone! Outputs in: results/v4_final/")
import glob
for f in sorted(glob.glob("results/v4_final/*")):
    size = os.path.getsize(f)
    print(f"  {os.path.basename(f)} ({size:,} bytes)")
