"""Visualize bottom-level controller performance.

Part A: turn response — command heading 0->90 deg, plot yaw/roll/turn-rate.
Part B: open-loop chase tracking — rule-based lead pursuit, 3D + top-down
        trajectories with the exact RL-used controller chain.
Part C: training reward curves from run logs (v14 / v18 / v19).
Part D (optional): trained-policy episode-reward histogram from an eval JSON.

Usage:
  python scripts/viz_controller_performance.py [--eval-json results/shoot_eval/eval_*.json]
"""
import os, sys, warnings, math, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.dynamics.aircraft import Aircraft
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.controller_base import FlightTarget
from src.dynamics.pid_controller import PIDFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor
from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.environment.formation_task import PHYSICS_DT

OUT = 'results/ctrl_viz'
os.makedirs(OUT, exist_ok=True)
DT = 1.0 / 60.0


def bearing_deg(p, t):
    return float((math.degrees(math.atan2(t[1] - p[1], t[0] - p[0])) + 360.0) % 360.0)


# ============================ Part A: turn response ============================
def turn_performance():
    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=9842.5, heading_deg=0.0,
             speed_kts=int(250.0 / 0.5144), trim=False)
    try:
        ac.fdm["propulsion/engine[0]/set-running"] = 1
    except KeyError:
        pass
    fc = FlightController()
    fc.reset()
    for _ in range(180):  # 3s level warmup
        s = ac.state
        thr, elev, ail, rud = fc.compute(s, FlightControlTargets(
            heading_deg=0.0, altitude_m=3000.0, speed_mps=250.0), DT)
        ac.set_controls(thr, elev, ail, rud)
        ac.run()
    ts, yaws, rolls, rates, cmds = [], [], [], [], []
    for i in range(360):  # 6s commanded 90 deg turn
        s = ac.state
        thr, elev, ail, rud = fc.compute(s, FlightControlTargets(
            heading_deg=90.0, altitude_m=3000.0, speed_mps=250.0), DT)
        ac.set_controls(thr, elev, ail, rud)
        ac.run()
        ts.append(i * DT)
        yaws.append(s["yaw_deg"])
        rolls.append(s["roll_deg"])
        cmds.append(90.0)
        if len(yaws) > 1:
            rates.append((yaws[-1] - yaws[-2]) / DT)
        else:
            rates.append(0.0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(ts, cmds, 'k--', lw=1.5, label='command (90 deg)')
    axes[0].plot(ts, yaws, 'b-', lw=2, label='actual heading')
    axes[0].set_ylabel('Heading (deg)')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Part A - Heading hold response: 0 -> 90 deg (250 m/s)')
    axes[1].plot(ts, rolls, 'r-', lw=2, label='bank angle')
    axes[1].plot(ts, rates, 'g-', lw=1, alpha=0.7, label='turn rate (deg/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('deg / deg/s')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'turn_response.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    settle = next((i for i, y in enumerate(yaws) if abs(y - 90.0) < 2.0), None)
    settle_str = f"{settle / 60:.1f}" if settle is not None else ">6"
    print(f'[Part A] 90-deg turn: settle_time={settle_str}s '
          f'max_bank={max(abs(r) for r in rolls):.0f} deg '
          f'max_turn_rate={max(abs(r) for r in rates):.1f} deg/s')
    return p


# ====================== Part B: open-loop chase tracking =======================
def chase_rule(p0, t0, cmd_speed):
    p_pos = p0.aircraft.position_ned
    t_pos = t0.aircraft.position_ned
    t_vel = t0.aircraft.velocity_ned
    dist = float(np.linalg.norm(p_pos - t_pos))
    lead_time = float(np.clip(dist / 1000.0 - 0.5, 0.0, 5.0))
    aim = t_pos + t_vel * lead_time
    hdg = bearing_deg(p_pos, aim)
    s = p0.aircraft.state
    hdg_err = abs((hdg - s["yaw_deg"] + 180.0) % 360.0 - 180.0)
    speed = cmd_speed if hdg_err < 15.0 else max(200.0, cmd_speed - 60.0)
    return hdg, speed


def run_chase(cmd_speed, seed=42):
    env = BaseEnv(task=SingleCombatShootTask({"difficulty_level": 0.0}))
    env.reset(seed=seed)
    p0, t0 = env.pursuers[0], env.targets[0]
    p0.controller = SafetyInterceptor(PIDFlightController())
    # fixed tail-chase geometry (same as the open-loop test)
    t_hdg, t_alt = 0.0, 3000.0
    t0.aircraft.reset(lat_deg=30.02, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=t_hdg, speed_kts=int(200.0 / 0.5144), trim=False)
    t0.aircraft.position_ned = np.array([2224.0, 0.0, t_alt])
    t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt
    p_hdg = 30.0
    p_lat = 30.0 + (2224.0 - 4000.0) / 111320.0
    p0.aircraft.reset(lat_deg=p_lat, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=p_hdg, speed_kts=int(cmd_speed / 0.5144), trim=False)
    p0.aircraft.position_ned = np.array([2224.0 - 4000.0, 0.0, t_alt])
    p0.ref_hdg, p0.ref_alt_m = p_hdg, t_alt
    p0._cmd_speed = cmd_speed
    for _ in range(60):  # 1s warmup through the RL-used controller chain
        s = p0.aircraft.state
        target = FlightTarget(heading_deg=p_hdg, altitude_m=t_alt, speed_mps=cmd_speed)
        surf = p0.controller.predict(s, target, PHYSICS_DT)
        p0.aircraft.set_controls(float(np.clip(surf.throttle, 0, 1)),
                                 float(np.clip(surf.elevator, -1, 1)),
                                 float(np.clip(surf.aileron, -1, 1)),
                                 float(np.clip(surf.rudder, -1, 1)))
        ts = t0.aircraft.state
        tgt = FlightControlTargets(heading_deg=t_hdg, altitude_m=t_alt, speed_mps=200.0)
        thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
        t0.aircraft.set_controls(thr, elev, ail, rud)
        p0.aircraft.run()
        p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
        t0.aircraft.run()
        t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]

    p_traj, t_traj, dists, hdgs, banks, spds = [], [], [], [], [], []
    for step in range(1500):
        p_pos = p0.aircraft.position_ned.copy()
        t_pos = t0.aircraft.position_ned.copy()
        dist = float(np.linalg.norm(p_pos - t_pos))
        p_traj.append(p_pos)
        t_traj.append(t_pos)
        dists.append(dist)
        s = p0.aircraft.state
        hdgs.append(s["yaw_deg"])
        banks.append(abs(s["roll_deg"]))
        spds.append(s["airspeed_mps"])
        if dist < 300.0:
            reason = "caught"
            break
        hdg_cmd, spd_cmd = chase_rule(p0, t0, cmd_speed)
        target = FlightTarget(heading_deg=hdg_cmd, altitude_m=float(t0.aircraft.state["alt_m"]),
                              speed_mps=spd_cmd)
        for _ in range(12):
            s = p0.aircraft.state
            surf = p0.controller.predict(s, target, PHYSICS_DT)
            p0.aircraft.set_controls(float(np.clip(surf.throttle, 0, 1)),
                                     float(np.clip(surf.elevator, -1, 1)),
                                     float(np.clip(surf.aileron, -1, 1)),
                                     float(np.clip(surf.rudder, -1, 1)))
            ts = t0.aircraft.state
            tgt = FlightControlTargets(heading_deg=t_hdg, altitude_m=t_alt, speed_mps=200.0)
            thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
            t0.aircraft.set_controls(thr, elev, ail, rud)
            p0.aircraft.run()
            p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
            p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
            t0.aircraft.run()
            t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
            t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]
        if step >= 1499:
            reason = "timeout"
    env.close()
    return {
        "speed": cmd_speed, "reason": reason, "steps": len(p_traj),
        "min_dist": float(min(dists)), "p_traj": np.array(p_traj),
        "t_traj": np.array(t_traj), "dists": dists,
        "hdgs": hdgs, "banks": banks, "spds": spds,
    }


def chase_plots(runs):
    fig = plt.figure(figsize=(15, 6))
    colors = {250: '#1f77b4', 280: '#d62728'}
    ax = fig.add_subplot(121, projection='3d')
    for r in runs:
        c = colors.get(r["speed"], '#333')
        p = r["p_traj"] - r["t_traj"][0]
        t = r["t_traj"] - r["t_traj"][0]
        ax.plot(p[:, 1], p[:, 0], p[:, 2], color=c, lw=1.8,
                label=f'pursuer {r["speed"]} m/s ({r["reason"]}, min {r["min_dist"]:.0f}m)')
        ax.plot(t[:, 1], t[:, 0], t[:, 2], 'k--', lw=1.2, alpha=0.6)
        mi = int(np.argmin(r["dists"]))
        ax.scatter([p[mi, 1]], [p[mi, 0]], [p[mi, 2]], color=c, marker='*', s=160,
                   edgecolors='white', zorder=5)
    ax.scatter([0], [0], [0], color='green', marker='o', s=80, label='target start')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_zlabel('Alt (m)')
    ax.set_title('Part B - Open-loop chase 3D (rule-based, controller chain)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)

    ax2 = fig.add_subplot(122)
    for r in runs:
        c = colors.get(r["speed"], '#333')
        ax2.plot(np.arange(len(r["dists"])) * 0.2, r["dists"], color=c, lw=1.5,
                 label=f'{r["speed"]} m/s')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Range to target (m)')
    ax2.set_title('Range vs time (capture = <300 m)')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'openloop_chase_3d.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[Part B] chase: ' + ' | '.join(
        f'{r["speed"]}m/s {r["reason"]} min={r["min_dist"]:.0f}m' for r in runs))
    return p


# ========================== Part C: training curves ============================
def reward_curves(smooth=3):
    """Raw per-iteration rewards with minimal smoothing (default window 3).

    The raw points are the primary curve (faithful to the collected data);
    a light moving average is overlaid only to help read the trend.
    """
    logs = [
        ('v14 (controller fix)', 'marl_runs/shoot_v14_fullctrl_s42/train.log'),
        ('v18 (steering aids)', 'marl_runs/shoot_v18_steer_s42/train.log'),
        ('v19 (GPU+batch)', 'marl_runs/shoot_v19_gpu_s42/train.log'),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, path in logs:
        if not os.path.exists(path):
            continue
        rews = []
        for line in open(path, encoding='utf-8', errors='replace'):
            if 'iter' in line and 'rew=' in line:
                parts = line.split('rew=')
                if len(parts) > 1:
                    try:
                        rews.append(float(parts[1].split()[0]))
                    except ValueError:
                        pass
        if not rews:
            continue
        x = np.arange(len(rews))
        y = np.array(rews)
        # RAW data points — the primary, faithful curve.
        ax.plot(x, y, lw=0.8, alpha=0.55,
                label=f'{label} (raw {len(rews)} pts)')
        # Light smoothing overlay only when requested (default window 3).
        k = max(1, min(smooth, len(y)))
        if k > 1:
            kernel = np.ones(k) / k
            ysm = np.convolve(y, kernel, mode='same')
            ax.plot(x, ysm, lw=1.8, alpha=0.9,
                    label=f'{label} (smooth-{k})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Episode reward mean')
    ax.set_title('Part C - Training reward curves (raw per-iteration data)')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'training_rewards.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[Part C] training curves saved (raw + smooth-{smooth})')
    return p


# ==================== Part D: eval reward distribution ====================
def reward_hist(eval_json=None):
    if not eval_json:
        cands = sorted(glob.glob('results/shoot_eval/eval_*.json'))
        eval_json = cands[-1] if cands else None
    if not eval_json or not os.path.exists(eval_json):
        print('[Part D] no eval JSON yet — skipped')
        return None
    data = json.load(open(eval_json))
    fig, ax = plt.subplots(figsize=(9, 5))
    for diff, res in data.get('results', {}).items():
        if 'episode_rewards' in res:
            ax.hist(res['episode_rewards'], bins=30, alpha=0.6,
                    label=f"difficulty={diff} (mean {res['mean_reward']:+.0f})")
    ax.set_xlabel('Episode reward'); ax.set_ylabel('Count')
    ax.set_title(f'Part D - Eval reward distribution ({os.path.basename(eval_json)})')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'eval_reward_dist.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[Part D] reward histogram from {eval_json}')
    return p


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-json', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=3,
                        help='moving-average window for the training curve overlay (1 = raw only)')
    args = parser.parse_args()
    a = turn_performance()
    runs = [run_chase(s) for s in [250, 280]]
    b = chase_plots(runs)
    c = reward_curves(smooth=args.smooth)
    d = reward_hist(args.eval_json)
    print(f'\nSaved to {OUT}:')
    for f in [a, b, c, d]:
        if f:
            print('  -', f)