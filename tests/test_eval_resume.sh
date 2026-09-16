#!/usr/bin/env bash
# Verifies that scripts/eval_bc_1v1.py survives an interruption and resumes.
#
# Why this test exists: long evaluations used to write their JSON only after the
# final episode, so a shutdown discarded the whole run. This checks the two
# guarantees that fix it: (1) an interrupted run leaves a valid partial file
# marked complete=false with accumulator state, and (2) --resume finishes it
# without recomputing the episodes already recorded.
#
# Run from anywhere:  bash tests/test_eval_resume.sh
set -u

cd "$(dirname "$0")/.."
PY=/home/sean/miniconda3/envs/marl_env/bin/python
W=data/expert/shoot_bc_round1_baseline.pth
OUT=/tmp/_resume_test.json
N=6

rm -f "$OUT" "$OUT.tmp" /tmp/_resume_1.log /tmp/_resume_2.log

echo "=== step 0: syntax ==="
"$PY" -m py_compile scripts/eval_bc_1v1.py && echo "  py_compile OK"

echo "=== step 1: start a ${N}-episode run and interrupt it mid-flight ==="
timeout 45 "$PY" -u scripts/eval_bc_1v1.py --weights "$W" --model-id resume_test \
  --episodes "$N" --seed 20000 --difficulty 0.0 --out "$OUT" > /tmp/_resume_1.log 2>&1
echo "  interrupted (timeout rc=$?)"

"$PY" -c "
import json
d = json.load(open('$OUT'))
m = d['run_meta']
print('  complete      =', m['complete'])
print('  eps_done      =', m['episodes_completed'])
print('  acc block     =', '_accumulators' in d)
print('  seeds         =', [e['seed'] for e in d['episodes_detail']])
assert m['complete'] is False, 'partial run must be marked incomplete'
assert '_accumulators' in d, 'partial run must carry accumulator state'
assert 0 < m['episodes_completed'] < $N, m['episodes_completed']
print('  [ok] partial file is valid and resumable')
"

echo "=== step 2: --resume must finish it without redoing recorded seeds ==="
"$PY" -u scripts/eval_bc_1v1.py --weights "$W" --model-id resume_test \
  --episodes "$N" --seed 20000 --difficulty 0.0 --out "$OUT" --resume \
  > /tmp/_resume_2.log 2>&1
grep -E '\[resume\]|kills:' /tmp/_resume_2.log | sed 's/^/  /'

"$PY" -c "
import json
d = json.load(open('$OUT'))
m = d['run_meta']
seeds = [e['seed'] for e in d['episodes_detail']]
print('  complete      =', m['complete'])
print('  episodes      =', d['episodes'])
print('  seeds         =', seeds)
print('  acc removed   =', '_accumulators' not in d)
assert m['complete'] is True, 'finished run must be marked complete'
assert d['episodes'] == $N, d['episodes']
assert seeds == list(range(20000, 20000 + $N)), seeds
assert '_accumulators' not in d, 'a complete run must not carry _accumulators'
print('  [ok] resumed run is complete and its seed set is exactly 20000..20005')
"
echo
echo "RESUME TEST PASSED"
