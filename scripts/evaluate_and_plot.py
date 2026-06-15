"""Evaluate best model: Tacview + trajectory + guidance metrics vs time."""
import sys, os, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3 import PPO
from src.environment.single_pursuit_env import SinglePursuitEnv, MAX_VEL
from src.environment.ablation_wrappers import CubicActionWrapper, LeadPursuitRewardWrapper
from scripts.train_single_pursuit import ResidualExpertWrapper
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH = "marl_runs/ablation_0615_1954/CARW_s0/best_model"
OUT_DIR = "results/eval_v7"
NUM_TRIES = 20
os.makedirs(OUT_DIR, exist_ok=True)

# ── Build env ─────────────────────────────────────────────────────────────
base = SinglePursuitEnv(difficulty_level=0.02, record_tacview=True)
base = CubicActionWrapper(base)
base = LeadPursuitRewardWrapper(base)
env = ResidualExpertWrapper(base)

model = PPO.load(MODEL_PATH)
print(f"Model: {MODEL_PATH}")
print(f"Running {NUM_TRIES} episodes at difficulty=0.02 ...")

# ── Run episodes, track per-step metrics ──────────────────────────────────
best_metrics = None
best_reason = ""
best_min_dist = 9999.0

for ep in range(NUM_TRIES):
    obs, _ = env.reset()
    done = False
    # Per-step tracking
    times, vc_list, los_rate_list, ata_list, dist_list, speed_list = [], [], [], [], [], []
    prev_dist = float(np.linalg.norm(
        env.unwrapped.pursuer.position_ned - env.unwrapped.target_ac.position_ned))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t = env.unwrapped._step_counter / 60.0  # seconds
        a_pos = env.unwrapped.pursuer.position_ned
        t_pos = env.unwrapped.target_ac.position_ned
        a_vel = env.unwrapped.pursuer.velocity_ned
        t_vel = env.unwrapped.target_ac.velocity_ned
        a_rpy = env.unwrapped.pursuer.rpy_rad
        t_rpy = env.unwrapped.target_ac.rpy_rad

        current_dist = float(np.linalg.norm(a_pos - t_pos))
        vc = (prev_dist - current_dist) / 0.1  # m/s closure rate

        # LOS rate
        los_vec = t_pos - a_pos
        los_dist = max(float(np.linalg.norm(los_vec)), 1.0)
        los_dir = los_vec / los_dist
        rel_vel = t_vel - a_vel
        rel_vel_parallel = float(np.dot(rel_vel, los_dir)) * los_dir
        rel_vel_perp = rel_vel - rel_vel_parallel
        los_rate = float(np.linalg.norm(rel_vel_perp)) / los_dist

        # ATA
        a_forward = compute_forward_vector(a_rpy)
        _, los_dir2, _ = compute_los(a_pos, t_pos)
        cos_ata = float(np.clip(np.dot(a_forward, los_dir2), -1.0, 1.0))

        aspd = float(env.unwrapped.pursuer.state["airspeed_mps"])

        times.append(t)
        vc_list.append(vc)
        los_rate_list.append(los_rate)
        ata_list.append(np.arccos(np.clip(cos_ata, -1, 1)) * 180 / np.pi)  # deg
        dist_list.append(current_dist)
        speed_list.append(aspd)

        prev_dist = current_dist

    reason = info.get("reason", "timeout")
    ep_min_dist = float(np.min(dist_list)) if dist_list else 9999.0

    status = "OK" if reason == "success" else "--"
    print(f"  Ep {ep+1:2d}: {status} {reason:12s}  min_dist={ep_min_dist:.0f}m  "
          f"t={times[-1]:.1f}s  steps={len(times)}")

    if reason == "success" or (best_min_dist > ep_min_dist and reason != "success"):
        if reason == "success" or ep_min_dist < best_min_dist:
            best_min_dist = ep_min_dist
            best_reason = reason
            best_metrics = {
                "times": np.array(times),
                "vc": np.array(vc_list),
                "los_rate": np.array(los_rate_list),
                "ata": np.array(ata_list),
                "dist": np.array(dist_list),
                "speed": np.array(speed_list),
            }
            # Save tacview frames for best episode
            best_frames = list(env.unwrapped._tacview_frames)

if best_metrics is None:
    print("No episodes completed!")
    exit(1)

m = best_metrics
print(f"\nBest episode: reason={best_reason} min_dist={best_min_dist:.0f}m "
      f"duration={m['times'][-1]:.1f}s")

# ── Export Tacview ─────────────────────────────────────────────────────────
env.unwrapped._tacview_frames = best_frames
tacview_path = os.path.join(OUT_DIR, "engagement.txt.acmi")
env.unwrapped.export_tacview(tacview_path)
print(f"Tacview → {os.path.abspath(tacview_path)}")

# ── Guidance Metrics vs Time ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. Distance & Speed
ax = axes[0, 0]
ax.plot(m["times"], m["dist"], "b-", lw=1.5, label="Distance (m)")
ax.set_ylabel("Distance (m)", color="b")
ax2 = ax.twinx()
ax2.plot(m["times"], m["speed"], "r-", lw=1.2, alpha=0.7, label="Speed (m/s)")
ax2.set_ylabel("Speed (m/s)", color="r")
ax2.axhline(y=100, color="orange", ls="--", lw=0.8, alpha=0.5, label="Low-speed threshold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax.set_title(f"Distance & Speed  |  {best_reason}  min_dist={best_min_dist:.0f}m")
ax.grid(True, alpha=0.3)

# 2. Closure Rate Vc
ax = axes[0, 1]
ax.plot(m["times"], m["vc"], "g-", lw=1.5)
ax.axhline(y=2.0, color="orange", ls="--", lw=0.8, label="Anti-stall threshold (2 m/s)")
ax.axhline(y=0, color="red", ls=":", lw=0.5)
ax.set_ylabel("Closure Rate (m/s)")
ax.set_xlabel("Time (s)")
ax.set_title("Closure Rate Vc")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. LOS Rate λ̇
ax = axes[1, 0]
ax.plot(m["times"], m["los_rate"], "purple", lw=1.5)
ax.axhline(y=0, color="red", ls=":", lw=0.5)
ax.set_ylabel("LOS Rate λ̇ (rad/s)")
ax.set_xlabel("Time (s)")
ax.set_title("Line-of-Sight Rate (λ̇ → 0 = collision course)")
ax.grid(True, alpha=0.3)

# 4. ATA
ax = axes[1, 1]
ax.plot(m["times"], m["ata"], "b-", lw=1.5)
ax.axhline(y=18.2, color="green", ls="--", lw=0.8, label="cos(ATA)=0.95 threshold")
ax.axhline(y=0, color="red", ls=":", lw=0.5)
ax.set_ylabel("ATA (deg)")
ax.set_xlabel("Time (s)")
ax.set_title("Antenna Train Angle (ATA)")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Vc vs ATA scatter (velocity shaping evidence)
ax = axes[2, 0]
ata_thresh_mask = m["ata"] < 18.2  # cos(ATA) > 0.95
ax.scatter(m["ata"][~ata_thresh_mask], m["vc"][~ata_thresh_mask],
           c="gray", s=10, alpha=0.3, label="ATA > 18.2°")
ax.scatter(m["ata"][ata_thresh_mask], m["vc"][ata_thresh_mask],
           c="red", s=25, alpha=0.6, label="ATA < 18.2° (well-aligned)")
ax.axhline(y=50, color="green", ls="--", lw=0.8, alpha=0.5, label="High Vc")
ax.axvline(x=18.2, color="green", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("ATA (deg)")
ax.set_ylabel("Closure Rate Vc (m/s)")
ax.set_title("Velocity Shaping Evidence: Vc vs ATA\n(expect high Vc when well-aligned)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# 6. LOS Rate vs Time (terminal zoom)
ax = axes[2, 1]
# Last 30% of episode
t_terminal = int(len(m["times"]) * 0.7)
if t_terminal > 0:
    ax.plot(m["times"][t_terminal:], m["los_rate"][t_terminal:], "purple", lw=1.5)
    ax.axhline(y=0, color="red", ls=":", lw=0.5)
    ax.set_ylabel("LOS Rate λ̇ (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Terminal Phase LOS Rate (t > {m['times'][t_terminal]:.1f}s)")
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "Episode too short", ha="center")

fig.tight_layout()
metrics_path = os.path.join(OUT_DIR, "guidance_metrics.png")
fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
print(f"Guidance metrics → {os.path.abspath(metrics_path)}")
plt.close("all")

# ── 3D Trajectory ──────────────────────────────────────────────────────────
frames = best_frames
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

fig = plt.figure(figsize=(18, 8))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.plot(a_x, a_y, a_alt, "r-", lw=1.5, alpha=0.8, label="Pursuer (F-16)")
ax3d.plot(t_x, t_y, t_alt, "b-", lw=1.5, alpha=0.8, label="Target")
ax3d.scatter(a_x[0], a_y[0], a_alt[0], color="darkred", s=100, marker="o")
ax3d.scatter(a_x[-1], a_y[-1], a_alt[-1], color="red", s=100, marker="x")
ax3d.scatter(t_x[0], t_y[0], t_alt[0], color="darkblue", s=100, marker="o")
ax3d.scatter(t_x[-1], t_y[-1], t_alt[-1], color="blue", s=100, marker="x")
ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")
ax3d.set_title(f"3D Trajectory  |  {best_reason}  min_dist={best_min_dist:.0f}m")
ax3d.legend()

ax2d = fig.add_subplot(1, 2, 2)
ax2d.plot(a_x, a_y, "r-", lw=1.5, alpha=0.8, label="Pursuer")
ax2d.plot(t_x, t_y, "b-", lw=1.5, alpha=0.8, label="Target")
ax2d.scatter(a_x[0], a_y[0], color="darkred", s=100, marker="o")
ax2d.scatter(a_x[-1], a_y[-1], color="red", s=100, marker="x")
ax2d.scatter(t_x[0], t_y[0], color="darkblue", s=100, marker="o")
ax2d.scatter(t_x[-1], t_y[-1], color="blue", s=100, marker="x")
ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)")
ax2d.set_title(f"Top-Down  |  {best_reason}  min_dist={best_min_dist:.0f}m")
ax2d.legend(); ax2d.axis("equal")
traj_path = os.path.join(OUT_DIR, "trajectory_3d.png")
fig.savefig(traj_path, dpi=150, bbox_inches="tight")
print(f"Trajectory → {os.path.abspath(traj_path)}")
plt.close("all")

# ── Altitude Profile ───────────────────────────────────────────────────────
fig2, ax_alt = plt.subplots(figsize=(12, 4))
tac_times = np.arange(len(frames)) * 0.1
ax_alt.plot(tac_times, a_alt, "r-", lw=1.5, label="Pursuer")
ax_alt.plot(tac_times, t_alt, "b-", lw=1.5, label="Target")
ax_alt.set_xlabel("Time (s)"); ax_alt.set_ylabel("Altitude (m)")
ax_alt.set_title(f"Altitude Profile  |  {best_reason}")
ax_alt.legend(); ax_alt.grid(True, alpha=0.3)
alt_path = os.path.join(OUT_DIR, "altitude.png")
fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
print(f"Altitude → {os.path.abspath(alt_path)}")
plt.close("all")

print("\nAll plots generated successfully!")
