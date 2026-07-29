"""Render v8: ACMI + 3D + top-down for both stages."""
import os, sys, warnings, logging
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, ray, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask

ENV = "jsbsim_shoot_1v1_v1"
C = {'p0': '#377eb8', 't0': '#e41a1c', 'm': '#ff7f00'}
OUT = '/home/sean/jsbsim-marl-formation/results/shoot_training'


def render_all(tag, ckpt, diff, seed):
    register_env(ENV, lambda c: BaseEnv(task=SingleCombatShootTask(c)))
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level="ERROR")
    algo = PPO.from_checkpoint(os.path.abspath(ckpt))
    policy = algo.get_policy("default_policy")
    env = BaseEnv(task=SingleCombatShootTask({"difficulty_level": diff}))
    obs, _ = env.reset(seed=seed)
    p0 = env.pursuers[0]; t0 = env.targets[0]

    # ── ACMI ─────────────────────────────────────────────────────────────
    acmi = f'{OUT}/v8_{tag}_s{seed}.acmi'
    env.enable_acmi_logging(acmi); env.log_acmi_step()

    # ── Trajectory capture ──────────────────────────────────────────────
    p0_t = [p0.aircraft.position_ned.copy()]; t0_t = [t0.aircraft.position_ned.copy()]
    m_t = {}; fires = 0; launches = 0; total_r = 0; hits = 0
    alt_log = []

    for step in range(3000):
        a = policy.compute_single_action(obs['p0'], explore=False)[0]
        if isinstance(a, (int, np.integer)):
            a = np.array([a % 3, (a // 3) % 5, (a // 15) % 3, (a // 45) % 2], dtype=np.int64)
        else:
            a = np.asarray(a, dtype=np.int64).flatten()
        if a[3] == 1: fires += 1
        prev = len(env._tempsims)
        obs, rews, terms, truncs, info = env.step({'p0': a})
        env.log_acmi_step()
        if len(env._tempsims) > prev: launches += 1
        total_r += rews.get('p0', 0)
        p0_t.append(p0.aircraft.position_ned.copy())
        t0_t.append(t0.aircraft.position_ned.copy())
        alt_log.append((p0.aircraft.state['alt_m'], t0.aircraft.state['alt_m']))
        for uid, sim in env._tempsims.items():
            if sim.is_alive:
                if uid not in m_t: m_t[uid] = []
                m_t[uid].append(sim.get_absolute_position().copy())
        evt = env.task._reward_breakdown.get('EventReward', {}).get('p0', 0)
        if evt > 0: hits += 1
        if terms.get('__all__') or truncs.get('__all__'): break

    p0_a = np.array(p0_t); t0_a = np.array(t0_t)
    alts = np.array(alt_log)
    reason = info.get('p0', {}).get('termination_reason', '?')
    print(f'[{tag}] steps={len(p0_a)} rew={total_r:+.0f} fires={fires} launches={launches} hits={hits} {reason}')
    print(f'  P0 alt: {alts[:,0].min():.0f}~{alts[:,0].max():.0f}m  T0 alt: {alts[:,1].min():.0f}~{alts[:,1].max():.0f}m')

    # ── 3D Plot ────────────────────────────────────────────────────────
    origin = t0_a[0].copy()
    p0_r = p0_a - origin; t0_r = t0_a - origin
    z_min = min(p0_a[:, 2].min(), t0_a[:, 2].min()) - 200
    z_max = max(p0_a[:, 2].max(), t0_a[:, 2].max()) + 200

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    al = np.linspace(0.3, 1.0, len(p0_r))
    for i in range(len(p0_r) - 1):
        ax.plot(p0_r[i:i+2, 1], p0_r[i:i+2, 0], p0_a[i:i+2, 2], color=C['p0'], linewidth=1.8, alpha=al[i])
        ax.plot(t0_r[i:i+2, 1], t0_r[i:i+2, 0], t0_a[i:i+2, 2], color=C['t0'], linewidth=1.5, alpha=al[i], linestyle='--')
    for uid, traj in m_t.items():
        ma = np.array(traj) - origin
        ma_alt = np.array(traj)
        for i in range(len(ma) - 1):
            ax.plot(ma[i:i+2, 1], ma[i:i+2, 0], ma_alt[i:i+2, 2], color=C['m'], linewidth=1.0, alpha=0.6)
    ax.scatter(p0_r[0, 1], p0_r[0, 0], p0_a[0, 2], c=C['p0'], marker='o', s=60, edgecolors='white', linewidth=0.5, zorder=5)
    ax.scatter(p0_r[-1, 1], p0_r[-1, 0], p0_a[-1, 2], c=C['p0'], marker='s', s=60, edgecolors='white', linewidth=0.5, zorder=5)
    ax.scatter(t0_r[0, 1], t0_r[0, 0], t0_a[0, 2], c=C['t0'], marker='o', s=60, edgecolors='white', linewidth=0.5, zorder=5)
    ax.scatter(t0_r[-1, 1], t0_r[-1, 0], t0_a[-1, 2], c=C['t0'], marker='X', s=80, edgecolors='white', linewidth=0.5, zorder=5)
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_zlabel('Altitude (m)')
    ax.set_zlim(z_min, z_max)
    ax.grid(True, alpha=0.2); ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    leg = [Line2D([0], [0], color=C['p0'], linewidth=2, label='P0'),
           Line2D([0], [0], color=C['t0'], linewidth=1.5, linestyle='--', label='T0'),
           Line2D([0], [0], color=C['m'], linewidth=1, label='Missile')]
    ax.legend(handles=leg, loc='upper left', framealpha=0.9, edgecolor='none', fontsize=8)
    st = {'stage1': 'Stage 1 (d=0.0)', 'stage2': 'Stage 2 (d=0.3)'}[tag]
    ax.set_title(f'1v1 Shoot v8 — 3D — {st}\nseed={seed} steps={len(p0_a)} hits={hits}/{launches} rew={total_r:+.0f} {reason}\nP0 alt {alts[:,0].min():.0f}~{alts[:,0].max():.0f}m  T0 alt {alts[:,1].min():.0f}~{alts[:,1].max():.0f}m',
                 fontsize=9, pad=15)
    fig.tight_layout(); fig.savefig(f'{OUT}/v8_3d_{tag}_s{seed}.png', dpi=200, facecolor='white', edgecolor='none'); plt.close(fig)

    # ── Top-down Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    al = np.linspace(0.25, 1.0, len(p0_a))
    for i in range(len(p0_a) - 1):
        ax.plot(p0_a[i:i+2, 1], p0_a[i:i+2, 0], color=C['p0'], linewidth=2.0, alpha=al[i])
        ax.plot(t0_a[i:i+2, 1], t0_a[i:i+2, 0], color=C['t0'], linewidth=1.5, alpha=al[i], linestyle='--')
    for uid, traj in m_t.items():
        ma = np.array(traj)
        for i in range(len(ma) - 1): ax.plot(ma[i:i+2, 1], ma[i:i+2, 0], color=C['m'], linewidth=1.0, alpha=0.6)
        ax.scatter(ma[0, 1], ma[0, 0], c=C['m'], marker='d', s=50, edgecolors='white', linewidth=0.5, zorder=4)
        ax.scatter(ma[-1, 1], ma[-1, 0], c=C['m'], marker='x', s=50, linewidth=1.0, zorder=4)
    ax.scatter(p0_a[0, 1], p0_a[0, 0], c=C['p0'], marker='o', s=100, edgecolors='white', linewidth=0.8, zorder=5, label='P0 start')
    ax.scatter(p0_a[-1, 1], p0_a[-1, 0], c=C['p0'], marker='s', s=100, edgecolors='white', linewidth=0.8, zorder=5, label='P0 end')
    ax.scatter(t0_a[0, 1], t0_a[0, 0], c=C['t0'], marker='o', s=100, edgecolors='white', linewidth=0.8, zorder=5, label='T0 start')
    ax.scatter(t0_a[-1, 1], t0_a[-1, 0], c=C['t0'], marker='X', s=120, edgecolors='white', linewidth=0.8, zorder=5, label='T0 end')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=8)
    ax.set_title(f'1v1 Shoot v8 — Top-Down — {st}\nseed={seed} steps={len(p0_a)} hits={hits}/{launches} rew={total_r:+.0f} {reason}', fontsize=10, pad=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/v8_topdown_{tag}_s{seed}.png', dpi=200, facecolor='white', edgecolor='none'); plt.close(fig)

    print(f'[{tag}] ACMI={acmi} 3D+TopDown saved')
    env.close(); ray.shutdown()


def main():
    os.makedirs(OUT, exist_ok=True)
    for tag, ckpt, diff in [
        ("stage1", "marl_runs/shoot_v8_stage1_s42/checkpoints/best", 0.0),
        ("stage2", "marl_runs/shoot_v8_stage2_s42/checkpoints/best", 0.3),
    ]:
        render_all(tag, ckpt, diff, 42)


if __name__ == "__main__":
    main()
