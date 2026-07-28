"""Generate ACMI + top-down trajectory for v5 WEZ-trained checkpoints."""
import os, sys, warnings, logging
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, ray, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask

ENV = "jsbsim_shoot_1v1_v1"
OUTDIR = "results/shoot_training"
COLORS = {'p0': '#377eb8', 't0': '#e41a1c', 'missile': '#ff7f00'}

def render_and_plot(tag, ckpt, difficulty, seed):
    register_env(ENV, lambda c: BaseEnv(task=SingleCombatShootTask(c)))
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level="ERROR")
    algo = PPO.from_checkpoint(os.path.abspath(ckpt))
    policy = algo.get_policy("default_policy")
    env = BaseEnv(task=SingleCombatShootTask({"difficulty_level": difficulty}))
    obs, _ = env.reset(seed=seed)
    p0 = env.pursuers[0]; t0 = env.targets[0]

    # Log initial state
    s_p0 = p0.aircraft.state; s_t0 = t0.aircraft.state
    print(f"[{tag}] P0=({s_p0['lat_deg']:.4f},{s_p0['lon_deg']:.4f}) "
          f"T0=({s_t0['lat_deg']:.4f},{s_t0['lon_deg']:.4f}) "
          f"dist={float(np.linalg.norm(p0.aircraft.position_ned-t0.aircraft.position_ned)):.0f}m")

    acmi_path = os.path.join(OUTDIR, f"wez_{tag}_s{seed}.acmi")
    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()

    # Trajectory capture
    p0_traj = [p0.aircraft.position_ned.copy()]
    t0_traj = [t0.aircraft.position_ned.copy()]
    missile_trajs = {}
    fires_attempted = 0; launches = 0
    totals = {'prog': 0, 'ata': 0, 'alt': 0, 'event': 0}

    for step in range(300):
        a = policy.compute_single_action(obs['p0'], explore=False)[0]
        if isinstance(a, (int, np.integer)):
            a = np.array([a%3, (a//3)%5, (a//15)%3, (a//45)%2], dtype=np.int64)
        else:
            a = np.asarray(a, dtype=np.int64).flatten()
        if a[3] == 1: fires_attempted += 1
        prev_count = len(env._tempsims)
        obs, rews, terms, truncs, info = env.step({'p0': a})
        env.log_acmi_step()
        if len(env._tempsims) > prev_count: launches += 1

        p0_traj.append(p0.aircraft.position_ned.copy())
        t0_traj.append(t0.aircraft.position_ned.copy())
        for uid, sim in env._tempsims.items():
            if sim.is_alive:
                if uid not in missile_trajs:
                    missile_trajs[uid] = []
                missile_trajs[uid].append(sim.get_absolute_position().copy())

        bd = env.task._reward_breakdown
        totals['prog'] += bd.get('ProgressReward', {}).get('p0', 0)
        totals['ata'] += bd.get('ATAAlignmentReward', {}).get('p0', 0)
        totals['alt'] += bd.get('AltitudeDeviationPenalty', {}).get('p0', 0)
        totals['event'] += bd.get('EventReward', {}).get('p0', 0)

        if terms.get('__all__') or truncs.get('__all__'):
            break

    p0_arr = np.array(p0_traj); t0_arr = np.array(t0_traj)
    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    total_r = sum(totals.values())
    print(f"[{tag}] steps={len(p0_arr)} rew={total_r:+.0f} fires={fires_attempted} "
          f"launches={launches} {reason}")
    print(f"       prog={totals['prog']:+.0f} ata={totals['ata']:+.0f} "
          f"event={totals['event']:+.0f}")

    # ── Top-down plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    alpha_curve = np.linspace(0.25, 1.0, len(p0_arr))
    for i in range(len(p0_arr) - 1):
        ax.plot(p0_arr[i:i+2, 1], p0_arr[i:i+2, 0], color=COLORS['p0'],
                linewidth=2.0, alpha=alpha_curve[i])
        ax.plot(t0_arr[i:i+2, 1], t0_arr[i:i+2, 0], color=COLORS['t0'],
                linewidth=1.5, alpha=alpha_curve[i], linestyle='--')
    for uid, traj in missile_trajs.items():
        m_arr = np.array(traj)
        for i in range(len(m_arr) - 1):
            ax.plot(m_arr[i:i+2, 1], m_arr[i:i+2, 0], color=COLORS['missile'],
                    linewidth=1.0, alpha=0.6)

    ax.scatter(p0_arr[0,1], p0_arr[0,0], c=COLORS['p0'], marker='o', s=120,
               edgecolors='white', linewidth=0.8, zorder=5, label='P0 start')
    ax.scatter(p0_arr[-1,1], p0_arr[-1,0], c=COLORS['p0'], marker='s', s=120,
               edgecolors='white', linewidth=0.8, zorder=5, label='P0 end')
    ax.scatter(t0_arr[0,1], t0_arr[0,0], c=COLORS['t0'], marker='o', s=120,
               edgecolors='white', linewidth=0.8, zorder=5, label='T0 start')
    ax.scatter(t0_arr[-1,1], t0_arr[-1,0], c=COLORS['t0'], marker='X', s=140,
               edgecolors='white', linewidth=0.8, zorder=5, label='T0 end')
    for uid, traj in missile_trajs.items():
        m_arr = np.array(traj)
        ax.scatter(m_arr[0,1], m_arr[0,0], c=COLORS['missile'], marker='d', s=60,
                   edgecolors='white', linewidth=0.5, zorder=4)
        ax.scatter(m_arr[-1,1], m_arr[-1,0], c=COLORS['missile'], marker='x', s=60,
                   linewidth=1.0, zorder=4)

    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=8)

    stage_label = {'stage1': 'Stage 1 (d=0.0, straight target)',
                   'stage2': 'Stage 2 (d=0.3, evasive target)'}.get(tag, tag)
    ax.set_title(f'1v1 Shoot — WEZ-Masked MAPPO — {stage_label}\n'
                 f'seed={seed}  steps={len(p0_arr)}  '
                 f'fires={fires_attempted}  launches={launches}  {reason}  '
                 f'rew={total_r:+.0f}',
                 fontsize=10, pad=10)

    fig.tight_layout()
    plot_path = os.path.join(OUTDIR, f"topdown_wez_{tag}_s{seed}.png")
    fig.savefig(plot_path, dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[{tag}] ACMI: {acmi_path}")
    print(f"[{tag}] Plot: {plot_path}")
    env.close(); ray.shutdown()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for tag, ckpt, diff in [
        ("stage1", "marl_runs/shoot_v5_stage1_s42/checkpoints/best", 0.0),
        ("stage2", "marl_runs/shoot_v5_stage2_s42/checkpoints/best", 0.3),
    ]:
        render_and_plot(tag, ckpt, diff, 42)


if __name__ == "__main__":
    main()
