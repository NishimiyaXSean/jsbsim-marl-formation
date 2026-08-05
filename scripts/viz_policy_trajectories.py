"""Render trained-policy episodes as 3D trajectories with launch/hit/WEZ markers.

Usage:
  python scripts/viz_policy_trajectories.py \
      --checkpoint marl_runs/shoot_v23_clogate_s42/checkpoints/best \
      --episodes 8 --difficulty 0.0 --seed 42 --outdir results/ctrl_viz/trajectories
"""
import os, sys, warnings, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.models.shoot_mask_model import ShootMaskModel

DIMS = [3, 5, 1, 2]


def decode(raw):
    if isinstance(raw, (int, np.integer)):
        flat = int(raw)
    else:
        arr = np.asarray(raw).reshape(-1)
        if arr.size == 4:
            return arr.astype(np.int64)
        flat = int(arr[0])
    out = np.zeros(4, dtype=np.int64)
    for i, d in enumerate(DIMS):
        out[i] = flat % d
        flat //= d
    return out


def run_episode(env, policy, include_closure):
    p0, t0 = env.pursuers[0], env.targets[0]
    obs, _ = env.reset(seed=42 + env._seed_offset)
    p_traj, t_traj = [], []
    m_trajs = {}
    launches, hits, wez_first = [], [], None
    fire_mask_open_steps = 0
    for step in range(1500):
        p_traj.append(p0.aircraft.position_ned.copy())
        t_traj.append(t0.aircraft.position_ned.copy())
        for uid, sim in env._tempsims.items():
            if sim.is_alive:
                m_trajs.setdefault(uid, []).append(sim.get_absolute_position().copy())
        # WEZ entry check (mask state this step)
        mask = env.task.get_action_mask(env, 'p0')
        if mask[10] == 1.0:
            fire_mask_open_steps += 1
            if wez_first is None:
                wez_first = step
        act = decode(policy.compute_single_action(obs['p0'], explore=False)[0])
        obs, rews, terms, truncs, info = env.step({'p0': act})
        if env.task._has_launched_this_step.get('p0', False):
            p_pos = p0.aircraft.position_ned.copy()
            launches.append((step, p_pos[0], p_pos[1], p_pos[2]))
        if env.task._hit_this_step.get('p0', 0) > 0:
            hits.append((step, p0.aircraft.position_ned[0], p0.aircraft.position_ned[1],
                         p0.aircraft.position_ned[2]))
        if terms.get('__all__') or truncs.get('__all__'):
            reason = info.get('p0', {}).get('termination_reason', 'unknown')
            break
    else:
        reason = 'timeout'
    p_traj.append(p0.aircraft.position_ned.copy())
    t_traj.append(t0.aircraft.position_ned.copy())
    return {
        'p': np.array(p_traj), 't': np.array(t_traj), 'm': m_trajs,
        'launches': launches, 'hits': hits, 'wez_first': wez_first,
        'fire_open': fire_mask_open_steps, 'reason': reason,
        'steps': len(p_traj) - 1,
    }


def plot_episode(ep, r, outdir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    origin = r['t'][0].copy()
    p = r['p'] - origin
    t = r['t'] - origin
    al = np.linspace(0.25, 1.0, len(p))
    for i in range(len(p) - 1):
        ax.plot([p[i, 1], p[i+1, 1]], [p[i, 0], p[i+1, 0]], [p[i, 2], p[i+1, 2]],
                color='#1f77b4', lw=1.6, alpha=al[i])
    for i in range(len(t) - 1):
        ax.plot([t[i, 1], t[i+1, 1]], [t[i, 0], t[i+1, 0]], [t[i, 2], t[i+1, 2]],
                color='#d62728', lw=1.2, alpha=al[i], linestyle='--')
    for uid, traj in r['m'].items():
        m = np.array(traj) - origin
        for i in range(len(m) - 1):
            ax.plot([m[i, 1], m[i+1, 1]], [m[i, 0], m[i+1, 0]], [m[i, 2], m[i+1, 2]],
                    color='#ff7f0e', lw=1.0, alpha=0.7)
        if len(m):
            ax.scatter([m[0, 1]], [m[0, 0]], [m[0, 2]], color='#ff7f0e', marker='D', s=50)
            ax.scatter([m[-1, 1]], [m[-1, 0]], [m[-1, 2]], color='#ff7f0e', marker='X', s=60)
    for (st, n, e, z) in r['launches']:
        ax.scatter([e], [n], [z], color='black', marker='^', s=90, zorder=6)
    for (st, n, e, z) in r['hits']:
        ax.scatter([e], [n], [z], color='gold', marker='*', s=180, edgecolors='black', zorder=7)
    if r['wez_first'] is not None:
        wf = min(r['wez_first'], len(p) - 1)
        ax.scatter([p[wf, 1]], [p[wf, 0]], [p[wf, 2]], color='green', marker='o', s=120,
                   edgecolors='white', zorder=6, label='first WEZ entry')
    ax.scatter([p[0, 1]], [p[0, 0]], [p[0, 2]], color='#1f77b4', marker='o', s=80,
               edgecolors='white', zorder=6, label='pursuer start')
    ax.scatter([t[0, 1]], [t[0, 0]], [t[0, 2]], color='#d62728', marker='o', s=80,
               edgecolors='white', zorder=6, label='target start')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_zlabel('Alt (m)')
    st = f'ep{ep}: {r["reason"]} | steps={r["steps"]} | WEZ_first={r["wez_first"]} | '
    st += f'launches={len(r["launches"])} hits={len(r["hits"])} | fire_open={r["fire_open"]}'
    ax.set_title(st, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    pth = os.path.join(outdir, f'episode_{ep:02d}.png')
    fig.savefig(pth, dpi=140, facecolor='white')
    plt.close(fig)
    print(f'  ep{ep}: {r["reason"]:<14s} steps={r["steps"]:>4d} WEZ_first={r["wez_first"]} '
          f'launches={len(r["launches"])} hits={len(r["hits"])}')
    return pth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--difficulty', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='results/ctrl_viz/trajectories')
    args = parser.parse_args()

    for name in ['jsbsim_shoot_1v1', 'jsbsim_shoot_v101', 'jsbsim_shoot_1v1_v1']:
        register_env(name, lambda c: BaseEnv(task=SingleCombatShootTask(c)))
    ModelCatalog.register_custom_model('shoot_mask_model', ShootMaskModel)
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')
    algo = PPO.from_checkpoint(os.path.abspath(args.checkpoint))
    policy = algo.get_policy('default_policy')
    include_closure = int(policy.observation_space.shape[0]) >= 39

    os.makedirs(args.outdir, exist_ok=True)
    print(f'checkpoint: {args.checkpoint}')
    print(f'obs dim: {int(policy.observation_space.shape[0])}, episodes: {args.episodes}, '
          f'difficulty: {args.difficulty}')

    env = BaseEnv(task=SingleCombatShootTask({
        'difficulty_level': args.difficulty,
        'obs_include_closure': include_closure,
    }))
    for ep in range(args.episodes):
        env._seed_offset = ep
        r = run_episode(env, policy, include_closure)
        plot_episode(ep, r, args.outdir)
    env.close()
    ray.shutdown()
    print(f'saved to {args.outdir}/')


if __name__ == '__main__':
    main()