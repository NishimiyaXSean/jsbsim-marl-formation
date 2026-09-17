#!/usr/bin/env bash
# Why two "expert" evaluations disagree.
#
# E5's expert arm (generate_shoot_rule_expert.py --validate) reported 32.25%
# on seeds 20000-20399 while E3's expert arm (eval_paired_bc_vs_expert.py)
# reported 36.75% on the *same* seeds. Both import the same rule functions
# (hdg_label / spd_label / fire_desired) at the same CMD_SPEED, so the rule is
# not the cause. Two structural differences remain:
#
#   A) env lifecycle: generate_*.py --validate builds ONE BaseEnv and reuses it
#      for every episode; eval_paired_* builds a FRESH env per episode. Every
#      other eval path (eval_bc_1v1, fire_oracle_audit, eval_asap_baseline) also
#      builds a fresh env per episode.
#   B) controller: generate_*.py --validate re-assigns
#      p0.controller = SafetyInterceptor(PIDFlightController()). BaseEnv.__init__
#      already builds exactly that by default, so this should be a no-op -- but
#      it is an explicit second construction, so it is worth ruling out.
#
# This script isolates the effect on a small seed sample by running the expert
# through the fresh-env path with the *same* rule, and printing both numbers.
#
# Run:  bash tests/check_expert_path_consistency.sh
set -u

cd "$(dirname "$0")/.."
PY=/home/sean/miniconda3/envs/marl_env/bin/python
N=${N:-20}
SEED=${SEED:-20000}

echo "=== A) eval_paired_bc_vs_expert.py (FRESH env per episode) ==="
"$PY" -u scripts/eval_paired_bc_vs_expert.py --seeds "$N" --start-seed "$SEED" \
  --difficulty 0.0 --out /tmp/_path_fresh.json > /tmp/_path_fresh.log 2>&1
echo "  exit=$?"

echo "=== B) generate_shoot_rule_expert.py --validate (REUSED env) ==="
"$PY" -u scripts/generate_shoot_rule_expert.py --validate --episodes "$N" \
  --seed "$SEED" --difficulty 0.0 --out-json /tmp/_path_reuse.json \
  > /tmp/_path_reuse.log 2>&1
echo "  exit=$?"

echo
"$PY" -c "
import json
f = json.load(open('/tmp/_path_fresh.json'))   # fresh env, has per_seed
r = json.load(open('/tmp/_path_reuse.json'))   # reused env, has episodes_detail
ef = f['expert']
print('expert via FRESH env  : kill=%5.1f%%  launches=%.2f/ep  n=%d' % (
      100*ef['kill_rate'], ef['launches_per_episode'], ef['n']))
print('expert via REUSED env : kill=%5.1f%%  launches=%.2f/ep  n=%d' % (
      100*r['kill_rate'], r['launches_per_episode'], r['episodes']))
d = 100*(ef['kill_rate'] - r['kill_rate'])
print('difference            : %+.1f pp' % d)
print()
print('verdict:', 'PATHS DISAGREE - env lifecycle is a real confound'
      if abs(d) >= 1.0 else 'paths agree within noise on this sample')
print('note: n=%d is a screen, not a result; run n=400 to size the effect' % $N)
"
