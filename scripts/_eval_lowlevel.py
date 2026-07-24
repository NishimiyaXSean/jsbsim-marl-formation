"""Deterministic evaluation: trained low-level controller flight quality test."""
import numpy as np, ray, sys, os, warnings, logging, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']: logging.getLogger(n).setLevel(logging.CRITICAL)
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.heading_task import HeadingTrackingTask

ENV = 'lowlevel_control_v1'
register_env(ENV, lambda c: BaseEnv(task=HeadingTrackingTask(c), env_config=c))
ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')

ckpt = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/sean/jsbsim-marl-formation/marl_runs/rllib_lowlevel_0724_1535_s42/checkpoints/best'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
target_hdg = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0

algo = PPO.from_checkpoint(os.path.abspath(ckpt))
env = BaseEnv(task=HeadingTrackingTask({'target_heading': target_hdg}), env_config={})
obs, _ = env.reset(seed=seed)

print(f'Target: {target_hdg:.0f}°  Start hdg: {float(env.pursuers[0].aircraft.state["yaw_deg"]):.0f}°')
print('step  time   hdg    err   alt_m  roll   pitch  spd    rew')

log = []; total_r = 0
for st in range(500):
    acts = {aid: algo.compute_single_action(aobs, explore=False) for aid, aobs in obs.items()}
    obs, rews, terms, truncs, info = env.step(acts)
    s = env.pursuers[0].aircraft.state
    hdg = float(s['yaw_deg']); alt = float(s['alt_m']); spd = float(s['airspeed_mps'])
    roll = float(s['roll_deg']); pitch = float(s['pitch_deg'])
    err = abs((target_hdg - hdg + 180) % 360 - 180)
    r = rews.get('p0', 0); total_r += r
    log.append({'step':st, 't':st*0.2, 'hdg':hdg, 'err':err, 'alt':alt,
                'roll':roll, 'pitch':pitch, 'spd':spd, 'rew':r})
    if st % 50 == 0 or st < 10:
        print(f'{st:4d}  {st*0.2:5.1f}s  {hdg:5.1f}° {err:5.1f}° {alt:5.0f}m {roll:5.1f}° {pitch:5.1f}° {spd:5.0f}m/s {r:+.3f}')
    if terms.get('__all__') or truncs.get('__all__'): break

n = len(log)
errs = [d['err'] for d in log]
print(f'\n--- Summary (n={n}) ---')
print(f'Total reward: {total_r:+.1f}')
print(f'Heading MAE: {np.mean(errs):.1f}°  (first 50: {np.mean(errs[:50]):.1f}°  last 100: {np.mean(errs[-100:]):.1f}°)')
print(f'Final heading: {log[-1]["hdg"]:.1f}°  (target {target_hdg:.0f}°, error {log[-1]["err"]:.1f}°)')
print(f'Altitude: {np.mean([d["alt"] for d in log]):.0f}m  ±{np.std([d["alt"] for d in log]):.0f}m')
print(f'Reason: {info.get("p0",{}).get("termination_reason","timeout")}')

# Export ACMI
out = f'results/heading/lowlevel_eval.acmi'
env.enable_acmi_logging(out); env.log_acmi_step()
print(f'ACMI: {out}')

env.close(); ray.shutdown()
