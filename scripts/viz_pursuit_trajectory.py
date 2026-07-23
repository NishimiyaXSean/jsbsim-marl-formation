"""Plot 3D trajectory for SinglePursuitTask episode."""

import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for name in ['jsbsim', 'gymnasium']:
    logging.getLogger(name).setLevel(logging.CRITICAL)

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask

register_env('single_pursuit_v1', lambda c: BaseEnv(task=SinglePursuitTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv) > 1 else \
    'marl_runs/rllib_pursuit_0723_1042_s42/checkpoints/best'
algo = PPO.from_checkpoint(os.path.abspath(ckpt))

difficulty = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

env = BaseEnv(task=SinglePursuitTask({'difficulty_level': difficulty}), env_config={})
obs, _ = env.reset(seed=seed)

p_positions, t_positions = [], []
for st in range(500):
    acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
    obs, rews, terms, truncs, info = env.step(acts)
    p_positions.append(env.pursuers[0].aircraft.position_ned.copy())
    t_positions.append(env.targets[0].aircraft.position_ned.copy())
    if terms.get('__all__') or truncs.get('__all__'): break

p_arr = np.array(p_positions)
t_arr = np.array(t_positions)

# Save CSV for external plotting
out = f'results/pursuit_traj_s{seed}_d{difficulty}.csv'
with open(out, 'w') as f:
    f.write('step,px_N,px_E,pz,mx_N,mx_E,mz\n')
    for i in range(len(p_arr)):
        f.write(f'{i},{p_arr[i,0]:.1f},{p_arr[i,1]:.1f},{p_arr[i,2]:.1f},'
                f'{t_arr[i,0]:.1f},{t_arr[i,1]:.1f},{t_arr[i,2]:.1f}\n')

# Quick ASCII summary
distances = np.linalg.norm(p_arr - t_arr, axis=1)
min_idx = np.argmin(distances)
print(f'Steps: {len(p_arr)}  Min dist: {distances[min_idx]:.0f}m at step {min_idx}')
print(f'P0 start → end: ({p_arr[0,0]:.0f},{p_arr[0,1]:.0f}) → ({p_arr[-1,0]:.0f},{p_arr[-1,1]:.0f})')
print(f'T0 start → end: ({t_arr[0,0]:.0f},{t_arr[0,1]:.0f}) → ({t_arr[-1,0]:.0f},{t_arr[-1,1]:.0f})')
print(f'CSV saved: {out}')

env.close()
ray.shutdown()
