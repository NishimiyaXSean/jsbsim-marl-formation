"""Quick V7 crash diagnosis: trace reward components for worst episodes."""
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

CKPT = "/home/sean/jsbsim-marl-formation/marl_runs/rllib_formation_0715_2027_s42/checkpoints/checkpoint_final"
algo = PPO.from_checkpoint(os.path.abspath(CKPT))

S1 = {"and_dist": 1200.0, "and_angle": 35.0, "bearing_min": -30.0, "bearing_max": 30.0,
      "target_dist_min": 1600.0, "target_dist_max": 2200.0, "sustain_steps": 2}

results = []
for ep in range(30):
    env = FormationRLlibEnv({'difficulty_level': 0.0, 'lock_altitude': True,
                             'record_tacview': False, 'cooperative_mode': True})
    env.set_coop_phase(COOP_PHASE_AND)
    env.set_curriculum_stage_full(1, S1["and_dist"], S1["and_angle"],
        S1["bearing_min"], S1["bearing_max"],
        S1["target_dist_min"], S1["target_dist_max"], S1["sustain_steps"])
    obs_dict, _ = env.reset(seed=42 + ep)
    done, total_r, step = False, 0.0, 0
    trace = {"step": [], "progress": [], "pincer": [], "loiter": [], "collision": [],
             "async_pen": [], "ooc": [], "bleed": [], "or_fallback": [], "ata": [],
             "d0": [], "d1": [], "pincer_angle": []}
    while not done:
        actions = {aid: algo.compute_single_action(obs_dict[aid], policy_id=f"policy_{aid}", explore=False)
                   for aid in env._agent_ids if aid in obs_dict}
        r_before = {aid: 0.0 for aid in env._agent_ids}
        obs_dict, rewards, terms, truncs, _ = env.step(actions)
        total_r += sum(rewards.values())
        # Approximate components from env state
        p0, p1 = env.pursuers[0], env.pursuers[1]
        t_pos = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(p0.aircraft.position_ned - t_pos))
        d1 = float(np.linalg.norm(p1.aircraft.position_ned - t_pos))
        trace["d0"].append(d0); trace["d1"].append(d1)
        los0, los1 = (t_pos-p0.aircraft.position_ned)[:2], (t_pos-p1.aircraft.position_ned)[:2]
        n0, n1 = np.linalg.norm(los0), np.linalg.norm(los1)
        pa = 0
        if n0>1 and n1>1:
            pa = float(np.degrees(np.arccos(np.clip(np.dot(los0,los1)/(n0*n1),-1,1))))
        trace["pincer_angle"].append(pa)
        trace["step"].append(step)
        # Estimate per-component rewards
        trace["progress"].append(sum(rewards.values()))  # placeholder
        trace["pincer"].append(0); trace["loiter"].append(0)
        trace["collision"].append(0); trace["async_pen"].append(0)
        trace["ooc"].append(0); trace["bleed"].append(0); trace["or_fallback"].append(0)
        trace["ata"].append(0)
        reason = getattr(env, '_last_termination_reason', 'timeout')
        done = terms.get('__all__', False) or truncs.get('__all__', False)
        step += 1
        if step >= 600: break
    results.append({"rew": total_r, "steps": step, "reason": reason,
                    "d0_min": min(trace["d0"]), "d1_min": min(trace["d1"]),
                    "pincer_max": max(trace["pincer_angle"]),
                    "trace": trace})
    print(f'Ep {ep}: rew={total_r:.0f}, {reason}, d0_min={min(trace["d0"]):.0f}m, d1_min={min(trace["d1"]):.0f}m')

algo.stop()

# Find worst episode
worst = min(results, key=lambda r: r["rew"])
print(f'\nWorst: rew={worst["rew"]:.0f}, {worst["steps"]} steps, {worst["reason"]}')
print(f'  d0_min={worst["d0_min"]:.0f}m, d1_min={worst["d1_min"]:.0f}m, pincer_max={worst["pincer_max"]:.0f}deg')

# Save trace for plotting
t = worst["trace"]
np.savez_compressed('data/viz/v7_crash_trace.npz',
    d0=t["d0"], d1=t["d1"], pincer_angle=t["pincer_angle"],
    steps=t["step"], rew=worst["rew"], reason=worst["reason"])

print(f'Saved: data/viz/v7_crash_trace.npz')
# Quick summary
print(f'\n=== Key metrics ===')
d0, d1 = np.array(t["d0"]), np.array(t["d1"])
pa = np.array(t["pincer_angle"])
T = len(d0)
print(f'P0: {d0[0]:.0f}→{d0[-1]:.0f}m, min={d0.min():.0f}')
print(f'P1: {d1[0]:.0f}→{d1[-1]:.0f}m, min={d1.min():.0f}')
print(f'Pincer: max={pa.max():.0f}deg, mean={pa.mean():.0f}deg')
# AND met?
and_dist = 1200; and_angle = 35
and_met = (d0[:len(pa)] < and_dist) & (d1[:len(pa)] < and_dist) & (pa >= and_angle)
print(f'AND met: {and_met.sum()}/{len(pa)} steps')
# Both >2000?
far = (d0 > 2000) | (d1 > 2000)
print(f'At least one >2000m: {far.sum()}/{T} steps ({100*far.sum()/T:.0f}%)')
# Estimate baseline bleed + loiter
bleed = -0.4 * T  # both agents
loiter_est = 0
for i in range(T):
    if d0[i] > and_dist: loiter_est -= 10
    if d1[i] > and_dist: loiter_est -= 10
print(f'Est baseline bleed: {bleed:.0f}')
print(f'Est loiter penalty (both outside AND): {loiter_est:.0f}')
print(f'Est bleed+loiter: {bleed+loiter_est:.0f}')
print(f'Actual total: {worst["rew"]:.0f}')
print(f'Unexplained: {worst["rew"] - (bleed+loiter_est):.0f}')
