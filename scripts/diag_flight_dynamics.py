"""Diagnostic: record 5 key flight dynamics variables over 100 steps (20s)."""

import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for name in ['jsbsim','gymnasium']: logging.getLogger(name).setLevel(logging.CRITICAL)

import numpy as np, ray, json
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask

register_env('single_pursuit_v1', lambda c: BaseEnv(task=SinglePursuitTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv) > 1 else \
    'marl_runs/rllib_pursuit_0723_1534_s42/checkpoints/best'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
difficulty = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

algo = PPO.from_checkpoint(os.path.abspath(ckpt))
env = BaseEnv(task=SinglePursuitTask({'difficulty_level': difficulty}), env_config={})
obs, _ = env.reset(seed=seed)

log = []
for st in range(100):
    acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
    obs, rews, terms, truncs, info = env.step(acts)
    s = env.pursuers[0].aircraft.state
    ps = env.pursuers[0]
    t_alt = ps.ref_alt_m
    elev = getattr(ps, '_last_surfaces', None)
    elev_val = elev.elevator if elev else 0.0
    log.append({
        'step': st, 't': st * 0.2,
        'alt_m': float(s['alt_m']), 'target_alt_m': t_alt,
        'pitch_deg': float(s['pitch_deg']),
        'q_rps': float(s.get('q_rps', 0)),
        'airspeed_mps': float(s['airspeed_mps']),
        'elevator': float(elev_val),
    })
    if terms.get('__all__') or truncs.get('__all__'): break

out = f'results/flight_diag_s{seed}.json'
json.dump(log, open(out, 'w'))
print(f'Saved {len(log)} steps to {out}')
env.close()
ray.shutdown()
