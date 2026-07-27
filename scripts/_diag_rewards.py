"""Diagnostic: per-module reward breakdown for FormationTask episode."""
import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG']='0'; warnings.filterwarnings('ignore')
for n in ['jsbsim','gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)
import numpy as np, ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask
from src.models.formation_rllib_model import RLlibAttentionActor
ENV='jsbsim_formation_base_v1'
ModelCatalog.register_custom_model('attention_formation',RLlibAttentionActor)
register_env(ENV,lambda c:BaseEnv(task=FormationTask(c),env_config=c))
ray.init(ignore_reinit_error=True,num_cpus=1,logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv)>1 else \
    '/home/sean/jsbsim-marl-formation/marl_runs/rllib_base_0727_0143_s42/checkpoints/checkpoint_final'
seed = int(sys.argv[2]) if len(sys.argv)>2 else 100
algo=PPO.from_checkpoint(os.path.abspath(ckpt))
env=BaseEnv(task=FormationTask({'curriculum_stage':1,'difficulty_level':0.0}),env_config={})
obs,_=env.reset(seed=seed)

# Accumulate per-module rewards
totals={f.__class__.__name__:{'p0':0,'p1':0} for f in env.task.reward_functions}
for st in range(500):
    acts={aid:algo.compute_single_action(obs[aid],policy_id='shared_policy',explore=False) for aid in env._agent_ids}
    obs,rews,terms,truncs,info=env.step(acts)
    for fn in env.task.reward_functions:
        sub=fn(env.task,env)
        totals[fn.__class__.__name__]['p0']+=sub.get('p0',0)
        totals[fn.__class__.__name__]['p1']+=sub.get('p1',0)
    if terms.get('__all__') or truncs.get('__all__'): break

print(f'Seed {seed}: {st+1} steps, reason={info.get("p0",{}).get("termination_reason","timeout")}')
print(f'{"Module":<30} {"P0":>12} {"P1":>12}')
print('-'*56)
for name,vals in totals.items():
    print(f'{name:<30} {vals["p0"]:12.1f} {vals["p1"]:12.1f}')
print('-'*56)
tp0=sum(v['p0'] for v in totals.values())
tp1=sum(v['p1'] for v in totals.values())
print(f'{"TOTAL":<30} {tp0:12.1f} {tp1:12.1f}')

# P1 runaway check
adp=totals.get('AltitudeDeviationPenalty',{'p1':0})
pincer=totals.get('PincerShapingReward',{'p1':0})
asym=totals.get('DistanceAsymmetryPenalty',{'p1':0})
prog=totals.get('ProgressReward',{'p1':0})
print(f'\nP1 diagnosis: Progress={prog["p1"]:.0f} AltPen={adp["p1"]:.0f} '
      f'Asym={asym["p1"]:.0f} Pincer={pincer["p1"]:.0f}')
env.close();ray.shutdown()
