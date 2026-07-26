"""Render FormationTask best episode to Tacview ACMI (3 aircraft)."""
import sys, os, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG']='0'; warnings.filterwarnings('ignore')
for n in ['jsbsim','gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)
import numpy as np, ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask
from src.models.formation_rllib_model import RLlibAttentionActor
from ray.rllib.models import ModelCatalog

ENV='jsbsim_formation_base_v1'
ModelCatalog.register_custom_model('attention_formation', RLlibAttentionActor)
register_env(ENV, lambda c: BaseEnv(task=FormationTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv)>1 else \
    'marl_runs/rllib_base_0726_0039_s42/checkpoints/checkpoint_final'
seed = int(sys.argv[2]) if len(sys.argv)>2 else 42
out = sys.argv[3] if len(sys.argv)>3 else 'results/formation_best.acmi'

algo = PPO.from_checkpoint(os.path.abspath(ckpt))
env = BaseEnv(task=FormationTask({'curriculum_stage':1,'difficulty_level':0.0}), env_config={})
obs,_ = env.reset(seed=seed)
env.enable_acmi_logging(out); env.log_acmi_step()

total_r = 0; min_d0 = 99999; min_d1 = 99999
for st in range(500):
    acts = {}
    for aid in env._agent_ids:
        acts[aid] = algo.compute_single_action(obs[aid], policy_id='shared_policy', explore=False)
    obs, rews, terms, truncs, info = env.step(acts)
    env.log_acmi_step()
    for r in rews.values(): total_r += r
    tp = env.targets[0].aircraft.position_ned
    d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - tp))
    d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - tp))
    if d0 < min_d0: min_d0 = d0
    if d1 < min_d1: min_d1 = d1
    if terms.get('__all__') or truncs.get('__all__'): break

reason = info.get('p0',{}).get('termination_reason','timeout')
print(f'Steps:{st+1} Rew:{total_r:+.0f} min_d0={min_d0:.0f}m min_d1={min_d1:.0f}m {reason}')
env.close(); ray.shutdown()
