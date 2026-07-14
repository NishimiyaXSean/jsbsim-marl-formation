"""Quick trajectory collection for AND-gate Stage 1 checkpoint."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault('JSBSIM_DEBUG', '0')

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from src.environment.formation_rllib_env import FormationRLlibEnv, COOP_PHASE_AND
from src.models.formation_rllib_model import RLlibAttentionActor

register_env('formation_2v1_rllib', lambda c: FormationRLlibEnv(c))
ModelCatalog.register_custom_model('attention_formation', RLlibAttentionActor)

ckpt = '/home/sean/jsbsim-marl-formation/marl_runs/rllib_formation_0713_1939_s42/checkpoints/best_iter_0125_rew_4283'
algo = PPO.from_checkpoint(os.path.abspath(ckpt))
S1 = {'and_dist': 2000.0, 'and_angle': 40.0, 'bearing_min': -20.0, 'bearing_max': 20.0}

eps = []
for ep in range(10):
    env = FormationRLlibEnv({'difficulty_level': 0.0, 'lock_altitude': True,
                             'record_tacview': False, 'cooperative_mode': True})
    env._difficulty = 0.0
    env.set_coop_phase(COOP_PHASE_AND)
    env.set_curriculum_stage(1, S1['and_dist'], S1['and_angle'],
                             S1['bearing_min'], S1['bearing_max'])
    obs_dict, _ = env.reset(seed=42 + ep)
    done, total_r, step = False, 0.0, 0
    pp0, pp1, pt, pc, d0s, d1s = [], [], [], [], [], []
    while not done:
        actions = {aid: algo.compute_single_action(obs_dict[aid], policy_id='shared_policy', explore=False)
                   for aid in env._agent_ids if aid in obs_dict}
        obs_dict, rewards, terms, truncs, _ = env.step(actions)
        total_r += sum(rewards.values())
        pp0.append(env.pursuers[0].aircraft.position_ned.copy())
        pp1.append(env.pursuers[1].aircraft.position_ned.copy())
        pt.append(env.targets[0].aircraft.position_ned.copy())
        d0s.append(np.linalg.norm(pp0[-1] - pt[-1]))
        d1s.append(np.linalg.norm(pp1[-1] - pt[-1]))
        p0, p1, t = pp0[-1], pp1[-1], pt[-1]
        n0, n1 = np.linalg.norm((t - p0)[:2]), np.linalg.norm((t - p1)[:2])
        if n0 > 1 and n1 > 1:
            cos_p = np.clip(np.dot((t-p0)[:2], (t-p1)[:2]) / (n0 * n1), -1, 1)
            pc.append(float(np.degrees(np.arccos(cos_p))))
        reason = getattr(env, '_last_termination_reason', 'timeout')
        done = terms.get('__all__', False) or truncs.get('__all__', False)
        step += 1
        if step >= 500: break
    eps.append({'rew': total_r, 'steps': step, 'reason': reason,
                'd0_min': min(d0s) if d0s else 0, 'd1_min': min(d1s) if d1s else 0,
                'pincer_max': max(pc) if pc else 0,
                'pp0': np.array(pp0), 'pp1': np.array(pp1), 'pt': np.array(pt),
                'd0s': np.array(d0s), 'd1s': np.array(d1s), 'pc': np.array(pc)})
    print(f'Ep {ep}: {step} steps, rew={total_r:.0f}, {reason}, '
          f'd0_min={min(d0s):.0f}m, pincer_max={max(pc) if pc else 0:.0f}deg')

save = {'n_episodes': len(eps)}
for i, e in enumerate(eps):
    p = f'ep{i}_'
    for k, v in [('n_steps', e['steps']), ('total_reward', e['rew']),
                 ('p0_positions', e['pp0']), ('p1_positions', e['pp1']),
                 ('target_positions', e['pt']), ('p0_distances', e['d0s']),
                 ('p1_distances', e['d1s']), ('pincer_angles', e['pc'])]:
        save[p + k] = v

np.savez_compressed('data/viz/and_stage1_best_4283_traj.npz', **save)
algo.stop()
sync = sum(1 for e in eps if e['reason'] == 'cooperative_success')
print(f'\nSync rate: {sync}/{len(eps)}. Rew range: [{min(e["rew"] for e in eps):.0f}, {max(e["rew"] for e in eps):.0f}]')
