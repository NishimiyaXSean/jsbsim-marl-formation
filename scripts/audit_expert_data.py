"""Audit the rule-expert dataset against the BC hard gates.

Loads data/expert/shoot_rule_expert.npz, checks:
  1. data integrity (shape, NaN, episode boundaries, action-mask violations,
     stateless label determinism)
  2. dataset isolation (episode-level train/val/test, no crossover, strata)
  3. fire B/C distribution among fire_allowed samples
  4. key-phase coverage
  5. action label distribution (symmetry, 0-dominance, switch rate, buckets)
  E. launch efficiency via a fresh 100-episode validation (launches, hits,
     kills, first-fire, angle buckets)

Writes logs/expert_audit.txt and prints a PASS/FAIL summary.

Usage: python scripts/audit_expert_data.py [--data data/expert/shoot_rule_expert.npz] [--val-episodes 100]
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from scripts.generate_shoot_rule_expert import (
    run_one, set_geometry, BaseEnv, SingleCombatShootTask, PIDFlightController,
    SafetyInterceptor, hdg_label, spd_label, fire_desired, quality_score,
    dlz_depth, FIRE_IDX, ATA_TURN)

REPORT = []


def log(s=''):
    print(s)
    REPORT.append(s)


def recompute_labels(obs_rows, mask_rows, target_spd_rows):
    '''Recompute stateless expert labels from stored obs/mask/target_spd.'''
    hdgs, spds, fires, allw, desr, quals = [], [], [], [], [], []
    for o, m, ts in zip(obs_rows, mask_rows, target_spd_rows):
        al = m[FIRE_IDX] == 1.0
        de = fire_desired(o)
        hdgs.append(hdg_label(o))
        ata_deg = abs(float(o[17]) * 180.0)
        cmd = ts if ata_deg < ATA_TURN else ts + 60.0
        spds.append(spd_label(o, cmd))
        fires.append(1 if (al and de) else 0)
        allw.append(float(al))
        desr.append(float(de))
        quals.append(quality_score(o))
    return {'hdg': np.array(hdgs), 'spd': np.array(spds), 'fire': np.array(fires),
            'allowed': np.array(allw), 'desired': np.array(desr),
            'quality': np.array(quals)}


def gate(name, ok, detail):
    log(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/expert/shoot_rule_expert.npz')
    parser.add_argument('--val-episodes', type=int, default=100)
    parser.add_argument('--det-sample', type=int, default=30000,
                        help='rows to recompute for stateless-label determinism')
    parser.add_argument('--skip-validate', action='store_true',
                        help='skip the fresh launch-efficiency validation (E)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    d = np.load(args.data)
    obs, act, mask = d['obs'], d['action'], d['mask']
    eid, phase = d['episode_id'], d['phase']
    allowed, desired = d['fire_allowed'], d['fire_desired']
    quality = d['launch_quality']
    n = len(obs)
    log('=' * 66)
    log(f'EXPERT DATA AUDIT — {args.data}')
    log(f'transitions={n} episodes={eid.max()} obs={obs.shape}')
    log('=' * 66)

    # ---------- Gate 1: data integrity ----------
    log('\n--- Gate 1: data integrity ---')
    ok1 = obs.shape[1] == 41
    gate('obs dim 41', ok1, str(obs.shape))
    nan = not np.isfinite(obs).all() or not np.isfinite(act).all()
    gate('no NaN/Inf', not nan, '')
    # episode boundaries: episode_id monotonic, contiguous
    eps = np.unique(eid)
    contiguous = all((eid == e).sum() > 0 for e in eps) and \
        np.all(np.diff(np.concatenate([[0], np.flatnonzero(np.diff(eid) != 0) + 1, [n]])) > 0)
    gate('episode boundaries contiguous', contiguous, f'{len(eps)} episodes')
    # action-mask violations: fire=1 requires mask fire allowed
    viol = int(((act[:, 3] == 1) & (mask[:, 10] != 1.0)).sum())
    gate('action-mask violations = 0', viol == 0, f'{viol} violations')
    # obs tail must equal the saved action mask (RLlib concat convention)
    tail_ok = obs.shape[1] >= 11 and bool((obs[:, -11:] == mask).all())
    gate('obs tail == saved mask', tail_ok,
         f'max diff={np.abs(obs[:, -11:] - mask).max():.3g}')
    # stateless label determinism: recompute with the pure-function labeler
    # TWICE on a random sample; stored / run1 / run2 must all agree.
    rng_det = np.random.default_rng(20260806)
    n_det = min(args.det_sample, n)
    det_idx = rng_det.choice(n, n_det, replace=False)
    run1 = recompute_labels(obs[det_idx], mask[det_idx],
                            d['target_spd'][det_idx])
    run2 = recompute_labels(obs[det_idx], mask[det_idx],
                            d['target_spd'][det_idx])
    stored = {'hdg': act[det_idx, 1], 'spd': act[det_idx, 0],
              'fire': act[det_idx, 3], 'allowed': allowed[det_idx],
              'desired': desired[det_idx], 'quality': quality[det_idx]}
    det_ok = True
    det_msgs = []
    for k in stored:
        a, b, s = np.asarray(run1[k]), np.asarray(run2[k]), np.asarray(stored[k])
        tol = 1e-12 if k == 'quality' else 0
        for lbl, arr in (('run1', a), ('run2', b)):
            diff = float(np.abs(arr - s).max()) if len(s) else 0.0
            if diff > tol:
                det_ok = False
                det_msgs.append(f'{k} stored vs {lbl} diff {diff:.3g}')
        if not np.array_equal(a, b):
            det_ok = False
            det_msgs.append(f'{k} run1 vs run2 differ')
    gate('stateless label determinism (recompute x2)', det_ok,
         f'{n_det} rows checked'
         + ('' if det_ok else ' :: ' + '; '.join(det_msgs[:5])))
    g1 = ok1 and not nan and contiguous and viol == 0 and det_ok and tail_ok

    # ---------- Gate 2: dataset isolation ----------
    log('\n--- Gate 2: episode-level train/val/test isolation ---')
    rng = np.random.default_rng(0)
    eps_shuf = rng.permutation(eps)
    n_tr = int(0.8 * len(eps_shuf)); n_va = int(0.1 * len(eps_shuf))
    tr, va, te = set(eps_shuf[:n_tr]), set(eps_shuf[n_tr:n_tr+n_va]), set(eps_shuf[n_tr+n_va:])
    overlap = len(tr & va) + len(tr & te) + len(va & te)
    gate('no episode crossover', overlap == 0, f'train={len(tr)} val={len(va)} test={len(te)}')
    # strata coverage per split (dist/closure/ATA from obs)
    dist = obs[:, 19] * 15000.0
    closure = obs[:, 22] * 300.0
    ata = obs[:, 17] * 180.0
    strata = {'dist<1.5k': dist < 1500, '1.5-3k': (dist >= 1500) & (dist < 3000),
              '3-5k': (dist >= 3000) & (dist < 5000), '>8k': dist > 8000,
              'closure<0': closure < 0, 'closure>0': closure > 0,
              'ATA<10': ata < 10, 'ATA>45': ata > 45}
    log('  strata coverage per split (fraction of split transitions):')
    for name, m in strata.items():
        row = []
        for s in (tr, va, te):
            sel = np.isin(eid, list(s))
            row.append(f'{m[sel].mean()*100:.1f}%')
        log(f'    {name:<12s} train={row[0]} val={row[1]} test={row[2]}')

    # ---------- Gate 3: fire B/C among allowed ----------
    log('\n--- Gate 3: fire B/C distribution (fire_allowed only) ---')
    a = allowed == 1.0
    b = a & (desired == 0.0)
    c = a & (desired == 1.0)
    n_a, n_b, n_c = int(a.sum()), int(b.sum()), int(c.sum())
    log(f'  allowed samples: {n_a} ({n_a/n*100:.2f}%)')
    log(f'  B (allowed & not desired): {n_b} ({n_b/max(n_a,1)*100:.1f}% of allowed)')
    log(f'  C (allowed & desired):     {n_c} ({n_c/max(n_a,1)*100:.1f}% of allowed)')
    c_eps = set(eid[c].tolist())
    log(f'  C spread over {len(c_eps)}/{len(eps)} episodes')
    conc = 100.0
    if n_c:
        cid = eid[c]
        counts = np.bincount(cid, minlength=int(eid.max()) + 1)
        top = np.sort(counts)[::-1][:5]
        conc = top.sum() / n_c * 100.0
        log(f'  top-5 episodes C counts: {top.tolist()} (sum={top.sum()}, '
            f'concentration={conc:.0f}%)')

    # C counts per train/val/test split (same episode split as Gate 2)
    log('  split-level C counts (episode split rng=0, 80/10/10):')
    split_req = {'train': 600, 'val': 60, 'test': 60}
    split_c_ok = True
    for name, s in (('train', tr), ('val', va), ('test', te)):
        sel = np.isin(eid, list(s))
        cm = c & sel
        nc = int(cm.sum())
        na = int((a & sel).sum())
        ceps = len(set(eid[cm].tolist()))
        ok = nc >= split_req[name]
        split_c_ok &= ok
        log(f'    {name:<5s} C={nc:>5d} (need >= {split_req[name]:>3d}) '
            f'allowed={na:>6d} C-eps={ceps:>3d} {"PASS" if ok else "FAIL"}')

    # C geometry coverage per split: angle / DLZ depth / closure / ATA
    log('  C coverage per split by geometry bucket:')
    first_idx = np.concatenate([[0], np.flatnonzero(np.diff(eid) != 0) + 1])
    bias_all = np.abs(obs[first_idx, 21]) * 180.0
    bias_hist = np.histogram(bias_all, [0, 30, 60, 90, 120.001])[0]
    log(f'  NOTE: SingleCombatShootTask.reset overrides set_geometry — true '
        f'initial offset is [30,60] deg (max_heading_bias_deg=60). '
        f'obs[21]-based estimate after 1s warmup: {bias_hist.tolist()} '
        f'(min={bias_all.min():.0f} max={bias_all.max():.0f})')
    ep_bias = np.zeros(int(eid.max()) + 1)
    ep_bias[eid[first_idx]] = bias_all
    dist = obs[:, 19] * 15000.0
    aa = obs[:, 18] * 180.0
    rmax = 3000.0 + 5000.0 * (aa / 180.0)
    dlz = (dist - 1500.0) / np.maximum(rmax - 1500.0, 1.0)
    closure = obs[:, 22] * 300.0
    ata = obs[:, 17] * 180.0
    eb = ep_bias[eid]

    def _bucket_counts(cmask, val, bins):
        out = []
        for lo, hi in bins:
            m = val < hi if lo == -1e9 else (val >= lo) & (val < hi)
            out.append(int((cmask & m).sum()))
        return out

    angle_bins = [(0, 30), (30, 60), (60, 90), (90, 120)]
    nonempty_angle = [i for i, x in enumerate(bias_hist) if x > 0]
    dlz_bins = [(0.25, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.75)]
    closure_bins = [(-1e9, -30.0), (-30.0, -15.0), (-15.0, -5.0)]
    ata_bins = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    cov_ok = True
    for name, s in (('train', tr), ('val', va), ('test', te)):
        sel = np.isin(eid, list(s))
        cm = c & sel
        ang = _bucket_counts(cm, eb, angle_bins)
        dl = _bucket_counts(cm, dlz, dlz_bins)
        cl = _bucket_counts(cm, closure, closure_bins)
        at = _bucket_counts(cm, ata, ata_bins)
        ok = all(ang[i] > 0 for i in nonempty_angle) \
            and sum(x > 0 for x in dl) >= 2 \
            and sum(x > 0 for x in cl) >= 2 \
            and sum(x > 0 for x in at) >= 2
        cov_ok &= ok
        log(f'    {name:<5s} angle={ang} dlz={dl} closure={cl} ata={at} '
            f'{"PASS" if ok else "FAIL"}')

    g3_ok = n_c >= 600 and len(c_eps) >= 250 and conc < 5.0         and split_c_ok and cov_ok
    gate('B/C sufficient & spread (PASS-WITH-MONITORING)', g3_ok,
         f'C={n_c} over {len(c_eps)} eps, top5 conc={conc:.1f}%, '
         f'split C ok={split_c_ok}, coverage ok={cov_ok}')

    # ---------- Gate 4: phase coverage ----------
    log('\n--- Gate 4: key-phase coverage ---')
    focus = ['closure_turn', 'wez_approach', 'wez_hold', 'launch_window',
             'close_cross', 'pre_lost', 'recovery', 'far_approach']
    for ph in focus:
        m = phase == ph
        cnt = int(m.sum())
        pe = len(set(eid[m].tolist()))
        hdg = act[m, 1]
        fire_r = act[m, 3].mean() * 100 if cnt else 0
        log(f'  {ph:<14s} n={cnt:>7d} ({cnt/n*100:5.2f}%) eps={pe:>3d} '
            f'fire%={fire_r:5.2f} hdg0%={(hdg==2).mean()*100 if cnt else 0:5.1f}')
    g4_recovery = int((phase == 'recovery').sum())
    g4_turn = int((phase == 'closure_turn').sum())
    log(f'  NOTE: recovery={g4_recovery}, closure_turn={g4_turn} — a 0%-lost expert '
        f'produces few recovery states; DAgger with expert takeover is the source for these.')

    # ---------- Gate 5: action label distribution ----------
    log('\n--- Gate 5: action label distribution ---')
    hdg = act[:, 1]; spd = act[:, 0]; fire = act[:, 3]
    hc = np.bincount(hdg, minlength=5)
    sc = np.bincount(spd, minlength=3)
    fc = np.bincount(fire, minlength=2)
    log(f'  heading classes ([-10,-5,0,5,10]): {hc.tolist()} '
        f'({(hc/hc.sum()*100).round(1).tolist()}%)')
    log(f'  speed classes ([-20,0,20]): {sc.tolist()} '
        f'({(sc/sc.sum()*100).round(1).tolist()}%)')
    log(f'  fire (0/1): {fc.tolist()} ({fc[1]/n*100:.2f}% fired)')
    sym = abs((hdg <= 1).mean() - (hdg >= 3).mean())
    log(f'  left-right symmetry (left% - right%): {sym*100:.2f}pp')
    log(f'  0-dominance: heading-0 {hc[2]/n*100:.1f}%, speed-0 {sc[1]/n*100:.1f}%')
    sw = int((act[1:] != act[:-1]).any(axis=1).mean() * 100)
    log(f'  action switch rate: {sw}% of consecutive steps')
    # geometry buckets
    log('  action by dist bucket (heading-0%, speed-0%, fire%):')
    for lo, hi, name in [(0, 1500, '<1.5k'), (1500, 3000, '1.5-3k'),
                         (3000, 5000, '3-5k'), (5000, 8000, '5-8k'), (8000, 1e9, '>8k')]:
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        log(f'    {name:<6s} hdg0={(hdg[m]==2).mean()*100:5.1f}% '
            f'spd0={(spd[m]==1).mean()*100:5.1f}% fire={fire[m].mean()*100:.2f}%')

    if not args.skip_validate:
        # ---------- E: launch efficiency (fresh validation) ----------
        log('\n--- E: launch efficiency (100-ep validation, angle buckets) ---')
        from src.environment.base_env import BaseEnv as BE
        from src.environment.singlecombat_shoot_task import SingleCombatShootTask as SCT
        env = BE(task=SCT({'difficulty_level': 0.0, 'obs_include_closure': True}))
        p0, t0 = env.pursuers[0], env.targets[0]
        p0.controller = SafetyInterceptor(PIDFlightController())
        rng = np.random.default_rng(args.seed)
        buckets = {'0-30': [], '30-60': [], '60-90': [], '90-120': []}
        all_r = []
        for ep in range(args.val_episodes):
            bias, _ = set_geometry(env, p0, t0, rng, 280.0)
            r = run_one(env, p0, t0, 280.0)
            r['bias'] = abs(bias)
            all_r.append(r)
            key = '0-30' if bias < 30 else '30-60' if bias < 60 else '60-90' if bias < 90 else '90-120'
            buckets[key].append(r)
        env.close()
        n_ep = len(all_r)
        launches = sum(r['fired'] for r in all_r)
        hits = sum(r['hits'] for r in all_r)
        kills = sum(1 for r in all_r if r['reason'] == 'target_killed')
        log(f'  episodes={n_ep} launches={launches} ({launches/n_ep:.2f}/ep) '
            f'hits={hits} ({hits/max(launches,1):.2f}/launch) '
            f'kills={kills} ({kills/n_ep*100:.1f}% kill rate)')
        ff = [r['first_fire'] for r in all_r if r['first_fire'] is not None]
        if ff:
            log(f'  first-fire time: median={np.median(ff)*0.2:.1f}s mean={np.mean(ff)*0.2:.1f}s')
        ld = np.bincount([min(r['fired'], 4) for r in all_r], minlength=5)
        log(f'  launches-per-episode distribution (0-4+): {ld.tolist()}')
        log('  angle buckets (kills/episodes, launches/ep, hits/launch):')
        for k, rs in buckets.items():
            if not rs:
                continue
            kl = sum(1 for r in rs if r['reason'] == 'target_killed')
            la = sum(r['fired'] for r in rs)
            hi = sum(r['hits'] for r in rs)
            log(f'    {k:<7s} kills={kl}/{len(rs)} ({kl/len(rs)*100:.0f}%) '
                f'launches={la/len(rs):.2f}/ep hits/launch={hi/max(la,1):.2f}')


    # ---------- Overall ----------
    log('\n--- OVERALL ---')
    final = (g1 and overlap == 0 and viol == 0 and g3_ok)
    log(f'Gate1 integrity: {"PASS" if g1 else "FAIL"}')
    log(f'Gate2 isolation: {"PASS" if overlap == 0 else "FAIL"}')
    log(f'Gate3 fire B/C:  {"PASS-WITH-MONITORING" if g3_ok else "FAIL"} (C={n_c}, eps={len(c_eps)})')
    log(f'Gate4 phase coverage: {"PASS" if g4_recovery >= 20 and g4_turn >= 200 else "CHECK"} '
        f'(recovery={g4_recovery}, closure_turn={g4_turn})')
    log(f'Gate5 actions:   {"CHECK" if hc[2]/n > 0.85 or sc[1]/n > 0.85 else "PASS"} '
        f'(0-dominance hdg={hc[2]/n*100:.0f}% spd={sc[1]/n*100:.0f}%)')
    log(f'OVERALL: {"ALL GATES PASS (Gate3 = PASS-WITH-MONITORING) — proceed to BC" if final else "GATES NOT MET — review"}')

    os.makedirs('logs', exist_ok=True)
    with open('logs/expert_audit.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT))
    log(f'\n[report saved to logs/expert_audit.txt]')


if __name__ == '__main__':
    main()
