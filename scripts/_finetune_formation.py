"""Conservative fine-tuning from Stage 4 checkpoint."""
import os, sys, warnings, logging, datetime, numpy as np, ray
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask
from src.models.formation_rllib_model import RLlibAttentionActor

ENV = 'jsbsim_formation_base_v1'
ModelCatalog.register_custom_model('attention_formation', RLlibAttentionActor)
register_env(ENV, lambda c: BaseEnv(task=FormationTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=4)

ts = datetime.datetime.now().strftime('%m%d_%H%M')
root = f'./marl_runs/rllib_base_{ts}_finetune_s42'
os.makedirs(f'{root}/checkpoints', exist_ok=True)

config = (PPOConfig()
    .environment(ENV, env_config={'curriculum_stage': 1, 'difficulty_level': 0.0})
    .framework('torch')
    .training(
        lr=5e-5, gamma=0.99, lambda_=0.95, clip_param=0.1,
        entropy_coeff=0.002, vf_clip_param=1000.0, grad_clip=0.5,
        train_batch_size=8192, minibatch_size=512, num_epochs=8,
        model={'custom_model': 'attention_formation', 'vf_share_layers': False},
    )
    .multi_agent(
        policies={'shared_policy'},
        policy_mapping_fn=lambda aid, *a, **kw: 'shared_policy',
    )
    .env_runners(num_env_runners=2, num_envs_per_env_runner=2)
    .resources(num_gpus=1)
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .debugging(log_level='WARN', seed=42)
)

algo = config.build()
algo.restore(os.path.abspath(
    'marl_runs/rllib_base_0727_1100_s42/checkpoints/checkpoint_000099'))
print(f'Fine-tuning: {root}  lr=5e-5 ent=0.002 batch=8192')

best = -float('inf')
for i in range(200):
    r = algo.train()
    rew = r.get('env_runners', {}).get('episode_reward_mean', float('nan'))
    length = r.get('env_runners', {}).get('episode_len_mean', 0)
    if i % 25 == 0:
        print(f'[iter {i:3d}] rew={rew:+.1f}  len={length:.0f}')
    if not np.isnan(rew) and rew > best:
        best = rew
        algo.save(f'{root}/checkpoints/best')

algo.save(f'{root}/checkpoints/checkpoint_final')
print(f'Best: {best:.1f} -> {root}/checkpoints/best')
ray.shutdown()
