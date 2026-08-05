"""Generate rule-expert data for the 1v1 shoot task (BC pretraining, v1).

The rule = open-loop lead-pursuit chaser (validated 30/30 catch) + two-layer fire:

    fire_allowed = env fire mask (ATA<15, DLZ, closure<0 i.e. closing, ammo)
    fire_desired = quality judgment (mask streak, stricter ATA, closure, DLZ depth,
                   premium zone, fallback after long wait)
    fire         = allowed AND desired (plus cooldown, matching task mechanics)

Discrete labels use the REFERENCE command (ref_hdg / _cmd_speed) with wrap-around,
deadband, and a one-step lookahead over candidate deltas (minimize next error +
command-change penalty).  Every transition also saves flags for downstream tuning:
fire_allowed, fire_desired, premium_window, launch_quality_score,
continuous_target_hdg, continuous_target_spd, phase, episode_id.

Usage:
  python scripts/generate_shoot_rule_expert.py --episodes 200 --out data/expert/shoot_rule_expert.npz
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore') if False else None
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.environment.formation_task import PHYSICS_DT
from src.dynamics.flight_controller import FlightControlTargets
from src.dynamics.controller_base import FlightTarget
from src.dynamics.pid_controller import PIDFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor

DELTA_HDG = [-10.0, -5.0, 0.0, 5.0, 10.0]
DELTA_SPD = [-20.0, 0.0, 20.0]
FIRE_IDX = 10          # flat fire index in the 11-dim mask
HDG_DEADBAND = 2.0     # deg — avoid -5/0/5 jitter
SPD_DEADBAND = 10.0    # m/s
COOLDOWN = 30          # decision steps (matches MIN_ATTACK_INTERVAL)
MASK_STREAK_REQ = 2    # consecutive open-mask frames before fire_desired
ATA_STRICT = 10.0      # stricter than the mask's 15 deg
CLOSURE_MIN = 5.0      # m/s
PREMIUM_LO = 2000.0
PREMIUM_HI = 4000.0
FALLBACK_WAIT = 75     # steps without a premium window -> use normal window


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def bearing_deg(p, t):
    return float((math.degrees(math.atan2(t[1] - p[1], t[0] - p[0])) + 360.0) % 360.0)


def chase_heading(p0, t0):
    p_pos = p0.aircraft.position_ned
    t_pos = t0.aircraft.position_ned
    t_vel = t0.aircraft.velocity_ned
    dist = float(np.linalg.norm(p_pos - t_pos))
    if dist > 6000.0:
        # far: lead pursuit to close in fast
        lead = float(np.clip(dist / 1000.0 - 0.5, 0.0, 5.0))
        aim = t_pos + t_vel * lead
    else:
        # inside the engagement band: PURE pursuit (aim at the target) so the
        # pursuer settles BEHIND the target with positive closure — the
        # lead-pursuit aim-ahead geometry put it abeam/ahead; with the corrected
        # closure sign (closing = negative) pure pursuit opens the fire mask.
        aim = t_pos
    return bearing_deg(p_pos, aim)


def chase_speed(p0, t0, cmd_speed):
    s = p0.aircraft.state
    hdg = chase_heading(p0, t0)
    err = abs(wrap180(hdg - s['yaw_deg']))
    return cmd_speed if err < 15.0 else max(200.0, cmd_speed - 60.0)


def hdg_label(err_deg, last_hdg_label):
    """Discrete heading delta from the AIRCRAFT's actual heading error.

    err_deg = wrapped(cmd_hdg - aircraft_yaw).  Thresholds + hysteresis so the
    reference is commanded to reduce the REAL misalignment (the aircraft lags
    the reference; a ref-based label left it 60+ deg behind).
    """
    if abs(err_deg) < 2.5:
        return 2  # 0 deg
    if err_deg > 7.5:
        want = 4  # +10
    elif err_deg > 2.5:
        want = 3  # +5
    elif err_deg < -7.5:
        want = 0  # -10
    else:
        want = 1  # -5
    # hysteresis: hold the current small-magnitude turn until the error grows
    if last_hdg_label in (1, 3) and want in (1, 3) and want != last_hdg_label:
        if abs(err_deg) < 5.0:
            return last_hdg_label
    return want


def spd_label(airspeed, target_speed):
    diff = target_speed - airspeed
    if abs(diff) < SPD_DEADBAND:
        return 1  # 0
    if diff > 0:
        return 2  # +20
    return 0      # -20


def quality_score(dist, ata_deg, closure, r_rps):
    dlz = np.clip(1.0 - abs(dist - 4750.0) / 3250.0, 0.0, 1.0)
    cl = np.clip(closure / 100.0, 0.0, 1.0)
    ata = np.clip(ata_deg / 15.0, 0.0, 1.0)
    hr = np.clip(abs(r_rps) / 0.2, 0.0, 1.0)
    return float(1.0 * dlz + 1.0 * cl - 0.6 * ata - 0.4 * hr)


def classify_phase(dist, ata_deg, closure, prev_closure, fire_allowed, fire_desired,
                   prev_phase, dist_history):
    if fire_desired:
        return 'launch_window'
    if dist <= 1500.0:
        return 'close_cross'
    if dist >= 8000.0:
        return 'far_approach'
    if prev_closure is not None and closure < 0 <= prev_closure:
        return 'closure_turn'
    if fire_allowed and ata_deg < 15.0:
        return 'wez_hold'
    if len(dist_history) >= 20 and dist > dist_history[-20] and dist > 6000.0:
        return 'pre_lost'
    if prev_phase == 'pre_lost' and dist < dist_history[-20] if len(dist_history) >= 20 else False:
        return 'recovery'
    return 'wez_approach' if ata_deg < 25.0 else 'far_approach'


def set_geometry(env, p0, t0, rng, cmd_speed, difficulty):
    """Randomized tail-chase geometry with wide coverage."""
    bias = rng.uniform(0.0, 90.0)
    if rng.random() < 0.2:
        bias = rng.uniform(90.0, 120.0)
    bias *= rng.choice([-1.0, 1.0])
    chase_dist = rng.uniform(2000.0, 5000.0)
    t_alt = 3000.0
    t_hdg = float(rng.uniform(0, 360))
    t_spd = float(rng.uniform(180.0, 240.0))
    t_hdg_rad = np.radians(t_hdg)
    t_north = 2224.0 + rng.uniform(-200, 200)
    t_east = rng.uniform(-200, 200)
    t_lat = 30.0 + t_north / 111132.0
    t_lon = 120.0 + t_east / 96420.0
    p_north = t_north - chase_dist * np.cos(t_hdg_rad)
    p_east = t_east - chase_dist * np.sin(t_hdg_rad)
    p_lat = 30.0 + p_north / 111132.0
    p_lon = 120.0 + p_east / 96420.0
    p_spd = float(rng.uniform(cmd_speed - 30.0, cmd_speed + 30.0))
    t0.aircraft.reset(lat_deg=t_lat, lon_deg=t_lon, alt_ft=int(t_alt * 3.28084),
                      heading_deg=t_hdg, speed_kts=int(t_spd / 0.5144), trim=False)
    t0.aircraft.position_ned = np.array([t_north, t_east, t_alt])
    t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt
    p_hdg = float((t_hdg + bias) % 360.0)
    p0.aircraft.reset(lat_deg=p_lat, lon_deg=p_lon, alt_ft=int(t_alt * 3.28084),
                      heading_deg=p_hdg, speed_kts=int(p_spd / 0.5144), trim=False)
    p0.aircraft.position_ned = np.array([p_north, p_east, t_alt])
    p0.ref_hdg, p0.ref_alt_m = p_hdg, t_alt
    p0._cmd_speed = p_spd
    for _ in range(60):
        s = p0.aircraft.state
        target = FlightTarget(heading_deg=p_hdg, altitude_m=t_alt, speed_mps=p_spd)
        surf = p0.controller.predict(s, target, PHYSICS_DT)
        p0.aircraft.set_controls(float(np.clip(surf.throttle, 0, 1)),
                                 float(np.clip(surf.elevator, -1, 1)),
                                 float(np.clip(surf.aileron, -1, 1)),
                                 float(np.clip(surf.rudder, -1, 1)))
        ts = t0.aircraft.state
        tgt = FlightControlTargets(heading_deg=t_hdg, altitude_m=t_alt, speed_mps=t_spd)
        thr, elev, ail, rud = t0.fc.compute(ts, tgt, PHYSICS_DT)
        t0.aircraft.set_controls(thr, elev, ail, rud)
        p0.aircraft.run()
        p0.aircraft.position_ned[0:2] += p0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        p0.aircraft.position_ned[2] = p0.aircraft.state["alt_m"]
        t0.aircraft.run()
        t0.aircraft.position_ned[0:2] += t0.aircraft.velocity_ned[0:2] * PHYSICS_DT
        t0.aircraft.position_ned[2] = t0.aircraft.state["alt_m"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--difficulty', type=float, default=0.0)
    parser.add_argument('--cmd-speed', type=float, default=280.0)
    parser.add_argument('--out', type=str, default='data/expert/shoot_rule_expert.npz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    env = BaseEnv(task=SingleCombatShootTask({'difficulty_level': args.difficulty,
                                               'obs_include_closure': True}))
    p0, t0 = env.pursuers[0], env.targets[0]
    p0.controller = SafetyInterceptor(PIDFlightController())
    rng = np.random.default_rng(args.seed)

    rec = {k: [] for k in ['obs', 'action', 'fire_allowed', 'fire_desired',
                           'premium_window', 'launch_quality', 'target_hdg',
                           'target_spd', 'phase', 'episode_id', 'mask']}
    episode = 0
    total_steps = 0
    for ep in range(args.episodes):
        episode += 1
        set_geometry(env, p0, t0, rng, args.cmd_speed, args.difficulty)
        obs, _ = env.reset()
        last_hdg_label = 2
        last_fire_step = -COOLDOWN
        mask_streak = 0
        prev_closure = None
        prev_phase = 'far_approach'
        dist_history = []
        fired = 0
        for step in range(1500):
            s = p0.aircraft.state
            p_pos = p0.aircraft.position_ned
            t_pos = t0.aircraft.position_ned
            dist = float(np.linalg.norm(p_pos - t_pos))
            los = t_pos - p_pos
            los_dir = los / max(dist, 1e-6)
            p_fwd = np.array([np.cos(np.radians(s['yaw_deg'])),
                              np.sin(np.radians(s['yaw_deg'])), 0.0])
            cos_ata = float(np.dot(p_fwd, los_dir))
            ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ata))))
            closure = float(np.dot(t0.aircraft.velocity_ned - p0.aircraft.velocity_ned, los_dir))
            r_rps = float(s.get('r_rps', 0.0))
            dist_history.append(dist)
            if len(dist_history) > 30:
                dist_history.pop(0)

            mask = env.task.get_action_mask(env, 'p0')
            fire_allowed = mask[FIRE_IDX] == 1.0
            cooldown_ok = (step - last_fire_step) >= COOLDOWN
            allowed = fire_allowed and cooldown_ok
            mask_streak = mask_streak + 1 if fire_allowed else 0

            # fire_desired: quality judgment (independent of the mask)
            in_premium = (closure < -CLOSURE_MIN and ata_deg < ATA_STRICT
                          and PREMIUM_LO < dist < PREMIUM_HI)
            in_normal = (closure < -CLOSURE_MIN and ata_deg < ATA_STRICT
                         and 2000.0 < dist < 5000.0)
            long_wait = (step - last_fire_step) > FALLBACK_WAIT
            fire_desired = (mask_streak >= MASK_STREAK_REQ and
                            (in_premium or (in_normal and long_wait)))
            fire = 1 if (allowed and fire_desired) else 0
            if fire:
                last_fire_step = step
                fired += 1

            cmd_hdg = chase_heading(p0, t0)
            target_spd = chase_speed(p0, t0, args.cmd_speed)
            yaw_err = wrap180(cmd_hdg - s['yaw_deg'])
            hdg_i = hdg_label(yaw_err, last_hdg_label)
            spd_i = spd_label(s['airspeed_mps'], target_spd)
            act = np.array([spd_i, hdg_i, 0, fire], dtype=np.int64)
            last_hdg_label = hdg_i

            phase = classify_phase(dist, ata_deg, closure, prev_closure,
                                   fire_allowed, fire_desired, prev_phase, dist_history)
            prev_closure = closure
            prev_phase = phase

            rec['obs'].append(obs['p0'].astype(np.float32))
            rec['action'].append(act)
            rec['fire_allowed'].append(float(allowed))
            rec['fire_desired'].append(float(fire_desired))
            rec['premium_window'].append(float(in_premium))
            rec['launch_quality'].append(quality_score(dist, ata_deg, closure, r_rps))
            rec['target_hdg'].append(cmd_hdg)
            rec['target_spd'].append(target_spd)
            rec['phase'].append(phase)
            rec['episode_id'].append(episode)
            rec['mask'].append(mask.astype(np.float32))

            obs, rews, terms, truncs, info = env.step({'p0': act})
            total_steps += 1
            if terms.get('__all__') or truncs.get('__all__'):
                break

        if (ep + 1) % 25 == 0:
            print(f'  ep {ep+1}: steps={step+1} fired={fired}')

    env.close()
    out = {k: np.array(v) for k, v in rec.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    n = len(out['obs'])
    print(f'\nsaved {args.out}: {n} transitions, {episode} episodes')
    print(f'  obs shape {out["obs"].shape}, action shape {out["action"].shape}')
    print(f'  fire rate: {out["action"][:,3].mean()*100:.2f}% '
          f'(allowed {out["fire_allowed"].mean()*100:.1f}%, '
          f'desired {out["fire_desired"].mean()*100:.1f}%)')
    ph, cnt = np.unique(out['phase'], return_counts=True)
    print('  phases:', dict(zip(ph.tolist(), cnt.tolist())))


if __name__ == '__main__':
    main()