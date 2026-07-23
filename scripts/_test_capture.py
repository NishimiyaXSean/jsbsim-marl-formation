"""Quick multi-seed capture test for SinglePursuitTask."""
import numpy as np, ray, sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for name in ['jsbsim', 'gymnasium']:
    logging.getLogger(name).setLevel(logging.CRITICAL)
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask
register_env('single_pursuit_v1', lambda c: BaseEnv(task=SinglePursuitTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')
ckpt = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'marl_runs/rllib_pursuit_0722_2220_s42/checkpoints/best')
algo = PPO.from_checkpoint(os.path.abspath(ckpt))
for seed in [100, 200, 300, 500, 999]:
    env = BaseEnv(task=SinglePursuitTask({'difficulty_level': 0}), env_config={})
    obs, _ = env.reset(seed=seed)
    for st in range(500):
        acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
        obs, rews, terms, truncs, info = env.step(acts)
        if terms.get('__all__') or truncs.get('__all__'): break
    reason = info.get('p0', {}).get('termination_reason', 'timeout')
    t_pos = env.targets[0].aircraft.position_ned
    d = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - t_pos))
    captured = 'YES' if 'capture' in str(reason) else 'no'
    sys.stdout.write(f'seed={seed}: steps={st+1} dist={d:.0f}m captured={captured} reason={reason}\n')
    sys.stdout.flush()
    env.close()
ray.shutdown()
