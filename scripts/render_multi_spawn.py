"""Generate Tacview for multiple spawn angles (check varied geometry)."""

import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'; warnings.filterwarnings('ignore')
for n in ['jsbsim','gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)
import numpy as np; import ray; from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask
register_env('single_pursuit_v1', lambda c: BaseEnv(task=SinglePursuitTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'marl_runs/rllib_pursuit_0723_1754_s42/checkpoints/best'
algo = PPO.from_checkpoint(os.path.abspath(ckpt))

os.makedirs('results/multi_spawn', exist_ok=True)

for seed in range(0, 500, 100):  # 0, 100, 200, 300, 400
    env = BaseEnv(task=SinglePursuitTask({'difficulty_level': 0}), env_config={})
    obs, _ = env.reset(seed=seed)
    # Log initial geometry
    t_pos = env.targets[0].aircraft.position_ned
    p_pos = env.pursuers[0].aircraft.position_ned
    rel = p_pos - t_pos
    angle = float(np.degrees(np.arctan2(rel[1], rel[0])))
    dist = float(np.linalg.norm(rel))

    out = f'results/multi_spawn/s{seed}_d{dist:.0f}m_a{angle:.0f}deg.acmi'
    env.enable_acmi_logging(out)
    env.log_acmi_step()

    for st in range(400):
        acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
        obs, rews, terms, truncs, info = env.step(acts)
        env.log_acmi_step()
        if terms.get('__all__') or truncs.get('__all__'): break

    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    t_end = env.targets[0].aircraft.position_ned
    p_end = env.pursuers[0].aircraft.position_ned
    end_dist = float(np.linalg.norm(p_end - t_end))
    print(f'seed={seed:3d}: spawn {dist:.0f}m @ {angle:.0f}° → end {end_dist:.0f}m ({reason})')
    env.close()

ray.shutdown()
