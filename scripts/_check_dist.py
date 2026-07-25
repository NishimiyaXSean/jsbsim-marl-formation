"""Quick min-distance check across seeds."""
import numpy as np, ray, sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG']='0'
warnings.filterwarnings('ignore')
for n in ['jsbsim','gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask
register_env('single_pursuit_v1',lambda c:BaseEnv(task=SinglePursuitTask(c),env_config=c))
ray.init(ignore_reinit_error=True,num_cpus=1,logging_level='ERROR')
ckpt='/home/sean/jsbsim-marl-formation/marl_runs/rllib_pursuit_0725_2019_s42/checkpoints/best'
algo=PPO.from_checkpoint(os.path.abspath(ckpt))
for seed in [42,100,200,300]:
    env=BaseEnv(task=SinglePursuitTask({'difficulty_level':0}),env_config={})
    obs,_=env.reset(seed=seed); min_d=99999
    for st in range(500):
        acts={aid:algo.compute_single_action(aobs,explore=False) for aid,aobs in obs.items()}
        obs,rews,terms,truncs,info=env.step(acts)
        tp=env.targets[0].aircraft.position_ned; pp=env.pursuers[0].aircraft.position_ned
        d=float(np.linalg.norm(tp-pp))
        if d<min_d: min_d=d
        if terms.get('__all__') or truncs.get('__all__'): break
    reason=info.get('p0',{}).get('termination_reason','timeout')
    print(f'seed={seed}: min={min_d:.0f}m steps={st+1} {reason}')
    env.close()
ray.shutdown()
