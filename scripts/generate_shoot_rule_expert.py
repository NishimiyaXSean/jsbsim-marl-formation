"""Generate rule-expert data for the 1v1 shoot task (BC pretraining, v2 stateless).

Gate-2 compliance: every label is a PURE FUNCTION of the current 41-dim obs.
No history (mask streak, last-fire time, previous action) is used for labels —
history-dependent gates (launch cooldown) live in the env-side fire mask.

  fire_allowed = env fire mask (ATA<15, DLZ, closure<0 i.e. closing, cooldown, ammo)
  fire_desired = current-state quality judgment:
      ATA < 10 deg, closure < -5 m/s (closing), DLZ depth in [0.25, 0.75]
  fire         = allowed AND desired

Heading label = signed ATA thresholds (pure pursuit alignment; deadband).
Speed label   = delta toward cruise speed, or turn speed when |ATA| large.

--validate: run the discretized rule closed-loop and report Gate-1 metrics
(lost_target, WEZ reach, fire rate, launches, hits, ATA p90) WITHOUT saving data.

WARNING (2026-09-17): --validate builds ONE BaseEnv and reuses it for every
episode. The fresh-env eval paths (eval_bc_1v1.py, eval_paired_bc_vs_expert.py,
fire_oracle_audit.py) build a new env per episode, and the two disagree on
identical seeds (32.25% vs 36.75% kills at d=0). Treat --validate as a
rule-behaviour sanity check only; never pool its kill/CLR figures with the
fresh-env arms. Gate: tests/check_expert_path_consistency.sh.

Usage:
  python scripts/generate_shoot_rule_expert.py --episodes 200 --out data/expert/shoot_rule_expert.npz
  python scripts/generate_shoot_rule_expert.py --validate --episodes 100
"""
import os, sys, math, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
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
from scripts.eval_meta import build_run_meta

DELTA_HDG = [-10.0, -5.0, 0.0, 5.0, 10.0]
DELTA_SPD = [-20.0, 0.0, 20.0]
FIRE_IDX = 10
HDG_DEADBAND = 2.5
SPD_DEADBAND = 10.0
ATA_TURN = 15.0
ATA_STRICT = 10.0
CLOSURE_MIN = 5.0
DLZ_LO, DLZ_HI = 0.25, 0.75


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def signed_ata(obs):
    """Signed ATA (deg) from the obs: ATA * side.  Positive = target right."""
    ata = float(obs[17]) * 180.0
    side = float(obs[20])
    return ata * side


def dlz_depth(obs):
    """Normalized position inside the dynamic DLZ (0 at min, 1 at max)."""
    dist = float(obs[19]) * 15000.0
    aa = float(obs[18]) * 180.0
    rmax = 3000.0 + 5000.0 * (aa / 180.0)
    return (dist - 1500.0) / max(rmax - 1500.0, 1.0)


def hdg_label(obs):
    """Stateless heading delta — one-step lookahead on the REFERENCE command.

    cmd_hdg_err = wrapped(bearing - ref_hdg) is in the obs (index 21).
    Drive ref_hdg TO the bearing and HOLD (deadband), letting the aircraft
    catch up.  Using the aircraft-error (signed ATA) over-rotated the
    reference and oscillated (Gate-1 lost 97%).
    """
    err = wrap180(float(obs[21]) * 180.0)
    if abs(err) < HDG_DEADBAND:
        return 2  # 0 deg (hold)
    best_i, best_c = 2, float('inf')
    for i, d in enumerate(DELTA_HDG):
        next_err = abs(wrap180(float(obs[21]) * 180.0 - d))
        cost = next_err + 0.05 * abs(d)
        if cost < best_c:
            best_c, best_i = cost, i
    return best_i


def spd_label(obs, cmd_speed):
    ata = abs(float(obs[17]) * 180.0)
    airspeed = float(obs[8]) * 400.0  # airspeed/MAX_VEL
    target = cmd_speed if ata < ATA_TURN else max(200.0, cmd_speed - 60.0)
    diff = target - airspeed
    if abs(diff) < SPD_DEADBAND:
        return 1
    return 2 if diff > 0 else 0


def fire_desired(obs):
    ata = float(obs[17]) * 180.0
    closure = float(obs[22]) * 300.0  # closure/300; negative = closing
    d = dlz_depth(obs)
    return bool(ata < ATA_STRICT and closure < -CLOSURE_MIN and DLZ_LO <= d <= DLZ_HI)


def quality_score(obs):
    ata = float(obs[17]) * 180.0 / 15.0
    closure = float(obs[22]) * 300.0
    cl = float(np.clip(-closure / 100.0, 0.0, 1.0))
    d = float(np.clip(dlz_depth(obs), 0.0, 1.0))
    return float(1.0 * d + 1.0 * cl - 0.6 * np.clip(ata, 0.0, 1.0))


def classify_phase(obs, fire_allowed, fire_desired, prev_closure, prev_phase, dist_history):
    dist = float(obs[19]) * 15000.0
    ata = float(obs[17]) * 180.0
    closure = float(obs[22]) * 300.0
    if fire_desired:
        return 'launch_window'
    if dist <= 1500.0:
        return 'close_cross'
    if dist >= 8000.0:
        return 'far_approach'
    if prev_closure is not None and closure < 0 <= prev_closure:
        return 'closure_turn'
    if fire_allowed and ata < 15.0:
        return 'wez_hold'
    if len(dist_history) >= 20 and dist > dist_history[-20] and dist > 6000.0:
        return 'pre_lost'
    if prev_phase == 'pre_lost' and len(dist_history) >= 20 and dist < dist_history[-20]:
        return 'recovery'
    return 'wez_approach' if ata < 25.0 else 'far_approach'


def run_one(env, p0, t0, cmd_speed, record=None, episode_id=0, seed=None):
    """Run the discretized rule for one episode; optionally record data.

    ``seed`` (added 2026-09-16) makes the expert's episode reproduce a specific
    scenario. Without it the rule cannot be paired seed-by-seed against BC/SPC,
    which blocks any paired significance test involving the expert.
    """
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    bias_est = abs(float(obs['p0'][21]) * 180.0)
    prev_closure = None
    prev_phase = 'far_approach'
    dist_history = []
    fired = 0
    allowed = 0
    hits = 0
    first_fire = None
    wez_first = None
    ata_hist = []
    reason = 'timeout'
    for step in range(1500):
        mask = env.task.get_action_mask(env, 'p0')
        fire_allowed = mask[FIRE_IDX] == 1.0
        if fire_allowed:
            allowed += 1
        desired = fire_desired(obs['p0'])
        fire = 1 if (fire_allowed and desired) else 0
        if fire:
            fired += 1
            if first_fire is None:
                first_fire = step

        hdg_i = hdg_label(obs['p0'])
        spd_i = spd_label(obs['p0'], cmd_speed)
        act = np.array([spd_i, hdg_i, 0, fire], dtype=np.int64)

        dist = float(obs['p0'][19]) * 15000.0
        ata = float(obs['p0'][17]) * 180.0
        closure = float(obs['p0'][22]) * 300.0
        ata_hist.append(abs(ata))
        dist_history.append(dist)
        if len(dist_history) > 30:
            dist_history.pop(0)
        if fire_allowed and wez_first is None:
            wez_first = step

        phase = classify_phase(obs['p0'], fire_allowed, desired, prev_closure,
                               prev_phase, dist_history)
        prev_closure = closure
        prev_phase = phase

        if record is not None:
            record['obs'].append(obs['p0'].astype(np.float32))
            record['action'].append(act)
            record['fire_allowed'].append(float(fire_allowed))
            record['fire_desired'].append(float(desired))
            record['premium_window'].append(float(desired))
            record['launch_quality'].append(quality_score(obs['p0']))
            record['target_hdg'].append(
                float((p0.ref_hdg + wrap180(float(obs['p0'][21]) * 180.0)) % 360.0))
            record['target_spd'].append(float(cmd_speed if ata < ATA_TURN else max(200.0, cmd_speed - 60.0)))
            record['phase'].append(phase)
            record['episode_id'].append(episode_id)
            record['mask'].append(mask.astype(np.float32))

        obs, rews, terms, truncs, info = env.step({'p0': act})
        hits += env.task._hit_this_step.get('p0', 0)
        if terms.get('__all__') or truncs.get('__all__'):
            reason = info.get('p0', {}).get('termination_reason', 'unknown')
            break
    return {'reason': reason, 'fired': fired, 'allowed': allowed,
            'wez_first': wez_first,
            'hits': hits, 'first_fire': first_fire, 'bias_est': bias_est,
            'steps': step + 1,
            'ata_p90': float(np.percentile(ata_hist, 90)) if ata_hist else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--difficulty', type=float, default=0.0)
    parser.add_argument('--cmd-speed', type=float, default=280.0)
    parser.add_argument('--out', type=str, default='data/expert/shoot_rule_expert.npz')
    parser.add_argument('--validate', action='store_true',
                        help='run the discretized rule closed-loop and report '
                             'Gate-1 metrics. WARNING: this path reuses one '
                             'BaseEnv across episodes, so its numbers are NOT '
                             'comparable to the fresh-env eval paths '
                             '(eval_bc_1v1.py, eval_paired_bc_vs_expert.py, '
                             'fire_oracle_audit.py).')
    parser.add_argument('--seed', type=int, default=42,
                        help='base seed for --validate (episode e uses seed+e). '
                             'The data-generation path stays unseeded so that '
                             'shoot_rule_expert.npz remains reproducible as-is.')
    parser.add_argument('--out-json', type=str, default=None,
                        help='with --validate, also write a result JSON '
                             '(includes run_meta identity block + CLR)')
    parser.add_argument('--model-id', type=str, default='rule_expert')
    parser.add_argument('--min-heading-bias-deg', type=float, default=None,
                        help='Override the task minimum initial heading bias. '
                             'None keeps the default U(0,60); 30 reproduces '
                             'the pre-fb48155 geometry U(30,60).')
    args = parser.parse_args()

    task_cfg = {'difficulty_level': args.difficulty, 'obs_include_closure': True}
    if args.min_heading_bias_deg is not None:
        task_cfg['min_heading_bias_deg'] = float(args.min_heading_bias_deg)
    env = BaseEnv(task=SingleCombatShootTask(task_cfg))
    p0, t0 = env.pursuers[0], env.targets[0]
    p0.controller = SafetyInterceptor(PIDFlightController())

    if args.validate:
        reasons = {}
        wez_reach = 0
        fired_total = 0
        allowed_total = 0
        hits_total = 0
        kills = 0
        ata_p90s = []
        wez_firsts = []
        ep_detail = []
        n_ep = 0
        for ep in range(args.episodes):
            ep_seed = (args.seed + ep) if args.seed is not None else None
            r = run_one(env, p0, t0, args.cmd_speed, seed=ep_seed)
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
            if r['wez_first'] is not None:
                wez_reach += 1
                wez_firsts.append(r['wez_first'])
            fired_total += r['fired']
            allowed_total += r['allowed']
            hits_total += r['hits']
            if r['reason'] == 'target_killed':
                kills += 1
            ata_p90s.append(r['ata_p90'])
            ep_detail.append({
                'ep': ep,
                'seed': ep_seed,
                'kill': bool(r['reason'] == 'target_killed'),
                'reason': r['reason'],
                'launches': r['fired'],
                'hits': r['hits'],
                'steps': r['steps'],
                'wez_first_step': r['wez_first'],
                'fire_first_step': r['first_fire'],
                'fire_allowed_steps': r['allowed'],
                'fire_commanded_on_allowed': r['fired'],
            })
            n_ep += 1
        lost = reasons.get('lost_target', 0)
        clr = (fired_total / allowed_total) if allowed_total else float('nan')
        print(f'=== Gate-1 validate: discretized rule, {n_ep} episodes ===')
        print('  !! LIFECYCLE WARNING: this path reuses ONE BaseEnv for every '
              'episode,')
        print('     which is measurably different from the fresh-env evaluation '
              'paths')
        print('     (32.25% vs 36.75% kills at d=0 on identical seeds). These '
              'numbers')
        print('     must NOT be pooled with eval_bc_1v1.py / '
              'eval_paired_bc_vs_expert.py /')
        print('     fire_oracle_audit.py results. See '
              'tests/check_expert_path_consistency.sh.')
        print(f'  lost_target: {lost}/{n_ep} ({lost/n_ep*100:.1f}%)')
        print(f'  reasons: {reasons}')
        print(f'  kills: {kills}/{n_ep} ({kills/n_ep*100:.1f}%)')
        print(f'  WEZ reach: {wez_reach}/{n_ep} ({wez_reach/n_ep*100:.1f}%)')
        if wez_firsts:
            print(f'  first WEZ time: median={np.median(wez_firsts)*0.2:.1f}s')
        print(f'  launches: {fired_total} ({fired_total/n_ep:.2f}/ep)')
        print(f'  hits: {hits_total} ({hits_total/max(fired_total,1):.2f}/launch)')
        print(f'  CLR (fire | allowed): {clr*100:.2f}% '
              f'({fired_total}/{allowed_total} allowed steps)')
        print(f'  ATA p90: median across eps = {np.median(ata_p90s):.1f} deg')
        if args.out_json:
            meta = build_run_meta(
                model_id=args.model_id or 'rule_expert',
                checkpoint='',
                first_seed=(args.seed if args.seed is not None else -1),
                episodes=n_ep,
                difficulty=args.difficulty,
                min_heading_bias_deg=args.min_heading_bias_deg,
                action_mode='argmax',
                script='scripts/generate_shoot_rule_expert.py --validate',
                # HONESTY FLAG (2026-09-17). This path builds ONE BaseEnv
                # before the episode loop (see above) and reuses it for every
                # episode. That is measurably different from the fresh-env
                # paths: on identical seeds this script reports 32.25% kills at
                # d=0 where eval_paired_bc_vs_expert.py reports 36.75%.
                # Numbers from here must never be pooled with fresh-env arms.
                env_lifecycle=('reused across episodes (one BaseEnv for all '
                               'episodes; NOT comparable to fresh-env arms)'),
            )
            # This path only ever writes after the final episode, so the result
            # is complete by construction. Declare it explicitly so downstream
            # tooling (scripts/collect_matrix.py) does not have to guess.
            meta['complete'] = True
            meta['episodes_completed'] = n_ep
            out = {
                'run_meta': meta,
                'policy': 'discrete_rule_expert',
                'episodes': n_ep,
                'difficulty': args.difficulty,
                'termination_reasons': reasons,
                'lost_target_rate': lost / max(n_ep, 1),
                'kill_rate': kills / max(n_ep, 1),
                'wez_reach_rate': wez_reach / max(n_ep, 1),
                'launches_per_episode': fired_total / max(n_ep, 1),
                'hit_rate': hits_total / max(fired_total, 1),
                'hits': hits_total,
                'clr': clr,
                'clr_allowed_steps': allowed_total,
                'clr_fire_commands': fired_total,
                'episodes_detail': ep_detail,
            }
            os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
            with open(args.out_json, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2)
            print(f'[saved] {args.out_json}')
        env.close()
        return

    rec = {k: [] for k in ['obs', 'action', 'fire_allowed', 'fire_desired',
                           'premium_window', 'launch_quality', 'target_hdg',
                           'target_spd', 'phase', 'episode_id', 'mask']}
    episode = 0
    for ep in range(args.episodes):
        episode += 1
        run_one(env, p0, t0, args.cmd_speed, record=rec, episode_id=episode)
        if (ep + 1) % 25 == 0:
            print(f'  ep {ep+1} done')
    env.close()
    out = {k: np.array(v) for k, v in rec.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    n = len(out['obs'])
    print(f'\nsaved {args.out}: {n} transitions, {episode} episodes')
    print(f'  obs shape {out["obs"].shape}, action shape {out["action"].shape}')
    allowed = out['fire_allowed'].mean() * 100
    desired = out['fire_desired'].mean() * 100
    fired = (out['action'][:, 3] == 1).mean() * 100
    print(f'  fire: allowed {allowed:.2f}% desired {desired:.2f}% fired {fired:.2f}%')
    ph, cnt = np.unique(out['phase'], return_counts=True)
    print('  phases:', dict(zip(ph.tolist(), cnt.tolist())))


if __name__ == '__main__':
    main()