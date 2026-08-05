"""Export Tacview ACMI logs for classic episodes.

Two modes:
  --checkpoint <path> : run the trained policy, classify episodes
                        (kill / fire / failure) and export ACMI for one of each.
  --openloop          : rule-based lead-pursuit chase through the exact
                        RL-used controller chain — validates the low-level
                        control with a clean trajectory.

Usage:
  python scripts/export_acmi_episodes.py --checkpoint marl_runs/.../best \
      --episodes 12 --outdir results/ctrl_viz/acmi
  python scripts/export_acmi_episodes.py --openloop --outdir results/ctrl_viz/acmi
"""
import os, sys, warnings, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')

import numpy as np

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.environment.formation_task import PHYSICS_DT
from src.dynamics.flight_controller import FlightControlTargets
from src.dynamics.controller_base import FlightTarget
from src.dynamics.pid_controller import PIDFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor

DIMS = [3, 5, 1, 2]


def decode(raw):
    if isinstance(raw, (int, np.integer)):
        flat = int(raw)
    else:
        arr = np.asarray(raw).reshape(-1)
        if arr.size == 4:
            return arr.astype(np.int64)
        flat = int(arr[0])
    out = np.zeros(4, dtype=np.int64)
    for i, d in enumerate(DIMS):
        out[i] = flat % d
        flat //= d
    return out


def bearing_deg(p, t):
    return float((math.degrees(math.atan2(t[1] - p[1], t[0] - p[0])) + 360.0) % 360.0)


def set_geometry(env, p0, t0, cmd_speed):
    t_hdg, t_alt = 0.0, 3000.0
    t0.aircraft.reset(lat_deg=30.02, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=t_hdg, speed_kts=int(200.0 / 0.5144), trim=False)
    t0.aircraft.position_ned = np.array([2224.0, 0.0, t_alt])
    t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt
    p_hdg = 30.0
    p_lat = 30.0 + (2224.0 - 4000.0) / 111320.0
    p0.aircraft.reset(lat_deg=p_lat, lon_deg=120.0, alt_ft=int(t_alt * 3.28084),
                      heading_deg=p_hdg, speed_kts=int(cmd_speed / 0.5144), trim=False)
    p0.aircraft.position_ned = np.array([2224.0 - 4000.0, 0.0, t_alt])
    p0.ref_hdg, p0.ref_alt_m = p_hdg, t_alt
    p0._cmd_speed = cmd_speed
    for _ in range(60):
        s = p0.aircraft.state
        target = FlightTarget(heading_deg=p_hdg, altitude_m=t_alt, speed_mps=cmd_speed)
        surf = p0.controller.predict(s, target, PHYSICS_DT)
        p0.aircraft.set_controls(float(np.clip(surf.throttle, 0, 1)),
                                 float(np.clip(surf.elevator, -1, 1)),
                                 float(np.clip(surf.aileron, -1, 1)),
                                 float(np.clip(surf.rudder, -1, 1)))
        ts = t0.aircraft.state
        tgt = FlightControlTargets(heading_deg=t_hdg, altitude_m=t_alt, speed_mps=200.0)
        thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
        t0.aircraft.set_controls(thr, elev, ail, rud)
        p0.aircraft.run()
        p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
        t0.aircraft.run()
        t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]


def step_aircraft(env, p0, t0, target):
    for _ in range(12):
        s = p0.aircraft.state
        surf = p0.controller.predict(s, target, PHYSICS_DT)
        p0.aircraft.set_controls(float(np.clip(surf.throttle, 0, 1)),
                                 float(np.clip(surf.elevator, -1, 1)),
                                 float(np.clip(surf.aileron, -1, 1)),
                                 float(np.clip(surf.rudder, -1, 1)))
        ts = t0.aircraft.state
        tgt = FlightControlTargets(heading_deg=0.0, altitude_m=3000.0, speed_mps=200.0)
        thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
        t0.aircraft.set_controls(thr, elev, ail, rud)
        p0.aircraft.run()
        p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
        t0.aircraft.run()
        t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]


def run_episode_acmi(env, p0, t0, policy, acmi_path, openloop=False, cmd_speed=280.0):
    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()
    reason = 'timeout'
    launches = 0
    wez_first = None
    min_dist = float('inf')
    obs, _ = env.reset()
    if openloop:
        set_geometry(env, p0, t0, cmd_speed)
        env.log_acmi_step()
    for step in range(1500):
        if openloop:
            p_pos = p0.aircraft.position_ned
            t_pos = t0.aircraft.position_ned
            dist = float(np.linalg.norm(p_pos - t_pos))
            min_dist = min(min_dist, dist)
            t_vel = t0.aircraft.velocity_ned
            lead = float(np.clip(dist / 1000.0 - 0.5, 0.0, 5.0))
            aim = t_pos + t_vel * lead
            hdg = bearing_deg(p_pos, aim)
            s = p0.aircraft.state
            hdg_err = abs((hdg - s['yaw_deg'] + 180) % 360 - 180)
            spd = cmd_speed if hdg_err < 15.0 else max(200.0, cmd_speed - 60.0)
            target = FlightTarget(heading_deg=hdg,
                                  altitude_m=float(t0.aircraft.state['alt_m']),
                                  speed_mps=spd)
            step_aircraft(env, p0, t0, target)
            env.log_acmi_step()
            if dist < 300.0:
                reason = 'caught'
                break
        else:
            act = decode(policy.compute_single_action(obs['p0'], explore=False)[0])
            mask = env.task.get_action_mask(env, 'p0')
            if mask[10] == 1.0 and wez_first is None:
                wez_first = step
            obs, rews, terms, truncs, info = env.step({'p0': act})
            env.log_acmi_step()
            if env.task._has_launched_this_step.get('p0', False):
                launches += 1
            if terms.get('__all__') or truncs.get('__all__'):
                reason = info.get('p0', {}).get('termination_reason', 'unknown')
                break
    env.close_acmi()
    return {'reason': reason, 'launches': launches, 'wez_first': wez_first,
            'min_dist': min_dist, 'steps': step + 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--openloop', action='store_true')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--difficulty', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='results/ctrl_viz/acmi')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    policy = None
    include_closure = True
    if not args.openloop:
        if not args.checkpoint:
            parser.error('--checkpoint required unless --openloop')
        import ray
        from ray.rllib.algorithms.ppo import PPO
        from ray.tune.registry import register_env
        from ray.rllib.models import ModelCatalog
        from src.models.shoot_mask_model import ShootMaskModel
        for name in ['jsbsim_shoot_1v1', 'jsbsim_shoot_v101', 'jsbsim_shoot_1v1_v1']:
            register_env(name, lambda c: BaseEnv(task=SingleCombatShootTask(c)))
        ModelCatalog.register_custom_model('shoot_mask_model', ShootMaskModel)
        ray.init(ignore_reinit_error=True, num_cpus=1, logging_level='ERROR')
        algo = PPO.from_checkpoint(os.path.abspath(args.checkpoint))
        policy = algo.get_policy('default_policy')
        include_closure = int(policy.observation_space.shape[0]) >= 40

    env = BaseEnv(task=SingleCombatShootTask({
        'difficulty_level': args.difficulty,
        'obs_include_closure': include_closure,
    }))
    p0, t0 = env.pursuers[0], env.targets[0]
    p0.controller = SafetyInterceptor(PIDFlightController())

    results = []
    tag = args.checkpoint.split("/")[-2] if args.checkpoint else "openloop"
    for ep in range(args.episodes):
        acmi = os.path.join(args.outdir, f'{tag}_ep{ep:02d}.acmi')
        r = run_episode_acmi(env, p0, t0, policy, acmi, openloop=args.openloop)
        results.append(r)
        print(f'  ep{ep}: {r["reason"]:<14s} launches={r["launches"]} '
              f'wez_first={r["wez_first"]} min_dist={r["min_dist"]:.0f}m')

    # pick classic representatives
    kills = [i for i, r in enumerate(results) if r['reason'] == 'target_killed']
    fires = [i for i, r in enumerate(results) if r['launches'] > 0]
    fails = [i for i, r in enumerate(results) if r['reason'] == 'lost_target']
    print(f'\nclassic episodes -> kill: {kills[:1]}  fire: {fires[:1]}  fail: {fails[:1]}')
    print(f'ACMI saved to {args.outdir}/')

    if not args.openloop:
        ray.shutdown()


if __name__ == '__main__':
    main()