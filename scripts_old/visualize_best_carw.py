"""Visualise best CARW model: Tacview + 3D trajectory for one episode."""
import sys, os, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

import numpy as np
from stable_baselines3 import PPO

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import CubicActionWrapper, LeadPursuitRewardWrapper
from scripts.train_single_pursuit import ResidualExpertWrapper

MODEL_PATH = "marl_runs/ablation_0615_1807/CARW_s0/best_model"
OUT_DIR = "results/carw_10hz"
NUM_TRIES = 10  # try multiple episodes to find a capture
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Build env with Tacview recording ────────────────────────────────────
base = SinglePursuitEnv(curriculum_stage=1.5, record_tacview=True)
base = CubicActionWrapper(base)
base = LeadPursuitRewardWrapper(base)
env = ResidualExpertWrapper(base)

# ── Load model ──────────────────────────────────────────────────────────
model = PPO.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# ── Run multiple episodes, pick the best (lowest min_dist) ──────────────
best_frames = None
best_reason = ""
best_min_dist = 9999.0
best_reward = -float("inf")
best_step = 0

for ep in range(NUM_TRIES):
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    step = 0
    ep_min_dist = 8000.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_r += rew
        step += 1
        if "min_dist" in info:
            ep_min_dist = min(ep_min_dist, info["min_dist"])
    reason = info.get("reason", "timeout")
    print(f"  Ep {ep+1:2d}: {step:4d} steps ({step*0.1:5.1f}s) | "
          f"reward={total_r:+.0f} | {reason:15s} | min_dist={ep_min_dist:.0f}m")
    if ep_min_dist < best_min_dist:
        best_min_dist = ep_min_dist
        best_reason = reason
        best_reward = total_r
        best_step = step
        # Clone the tacview frames before next reset
        best_frames = list(env.unwrapped._tacview_frames)

if best_frames is None:
    print("No frames captured!")
    exit(1)

print(f"\nBest episode: {best_step} steps | reward={best_reward:.0f} | "
      f"reason={best_reason} | min_dist={best_min_dist:.0f}m")

# ── Export Tacview from best episode ────────────────────────────────────
# Re-run the best episode scenario... but we already have the frames.
# We need to rebuild frames with the right data. Let's just use what we have.
# Actually, export_tacview uses self._tacview_frames. We need to set them.
env.unwrapped._tacview_frames = best_frames
tacview_path = os.path.join(OUT_DIR, "carw_10hz_engagement.txt.acmi")
env.unwrapped.export_tacview(tacview_path)
print(f"Tacview → {os.path.abspath(tacview_path)}")

# ── 3D Trajectory Plot ─────────────────────────────────────────────────
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

# 3D view
fig = plt.figure(figsize=(18, 8))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.plot(a_x, a_y, a_alt, "r-", lw=1.5, alpha=0.8, label="Pursuer (F-16)")
ax3d.plot(t_x, t_y, t_alt, "b-", lw=1.5, alpha=0.8, label="Target")
ax3d.scatter(a_x[0], a_y[0], a_alt[0], color="darkred", s=100, marker="o", label="Pursuer start")
ax3d.scatter(a_x[-1], a_y[-1], a_alt[-1], color="red", s=100, marker="x", label="Pursuer end")
ax3d.scatter(t_x[0], t_y[0], t_alt[0], color="darkblue", s=100, marker="o", label="Target start")
ax3d.scatter(t_x[-1], t_y[-1], t_alt[-1], color="blue", s=100, marker="x", label="Target end")
ax3d.set_xlabel("North (m)")
ax3d.set_ylabel("East (m)")
ax3d.set_zlabel("Altitude (m)")
ax3d.set_title(f"CARW @ 10Hz — 3D Trajectory\nreason={reason}  min_dist={ep_min_dist:.0f}m")
ax3d.legend(loc="upper left")

# Top-down view
ax2d = fig.add_subplot(1, 2, 2)
ax2d.plot(a_x, a_y, "r-", lw=1.5, alpha=0.8, label="Pursuer (F-16)")
ax2d.plot(t_x, t_y, "b-", lw=1.5, alpha=0.8, label="Target")
ax2d.scatter(a_x[0], a_y[0], color="darkred", s=100, marker="o")
ax2d.scatter(a_x[-1], a_y[-1], color="red", s=100, marker="x")
ax2d.scatter(t_x[0], t_y[0], color="darkblue", s=100, marker="o")
ax2d.scatter(t_x[-1], t_y[-1], color="blue", s=100, marker="x")
ax2d.set_xlabel("North (m)")
ax2d.set_ylabel("East (m)")
ax2d.set_title(f"Top-Down View  |  {reason}  min_dist={ep_min_dist:.0f}m")
ax2d.legend()
ax2d.axis("equal")

traj_path = os.path.join(OUT_DIR, "carw_10hz_trajectory_3d.png")
fig.savefig(traj_path, dpi=150, bbox_inches="tight")
print(f"3D plot → {os.path.abspath(traj_path)}")

# Altitude profile
fig2, ax_alt = plt.subplots(figsize=(12, 4))
times = np.arange(len(frames)) * 0.1  # 10Hz
ax_alt.plot(times, a_alt, "r-", lw=1.5, label="Pursuer")
ax_alt.plot(times, t_alt, "b-", lw=1.5, label="Target")
ax_alt.set_xlabel("Time (s)")
ax_alt.set_ylabel("Altitude (m)")
ax_alt.set_title(f"Altitude Profile  |  CARW @ 10Hz  |  {reason}")
ax_alt.legend()
ax_alt.grid(True, alpha=0.3)
alt_path = os.path.join(OUT_DIR, "carw_10hz_altitude.png")
fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
print(f"Altitude plot → {os.path.abspath(alt_path)}")
plt.close("all")
print("\nDone!")
