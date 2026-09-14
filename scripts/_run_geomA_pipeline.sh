#!/usr/bin/env bash
# geomA full BC+ASAP retrain - heading bias [0,60] instead of [30,60].
#
# Rationale: the reset geometry change alters the INITIAL-STATE DISTRIBUTION,
# so expert data, BC and ASAP distillation must all be regenerated under the
# new geometry for a clean comparison against the frozen 87%-kill baseline.
#
# FROZEN ARTEFACTS ARE NEVER OVERWRITTEN: every output uses a _geomA suffix.
#   data/expert/shoot_rule_expert_geomA.npz   (was: shoot_rule_expert.npz)
#   data/expert/shoot_bc_geomA.pth            (was: shoot_bc_round1_baseline.pth)
#   data/expert/shoot_bc_asap_geomA.pth       (was: shoot_bc_asap_distilled.pth)
#   results/shoot_eval/eval_geomA_d0_s42.json (was: eval_distilled_d0_s42.json)
#
# NOTE: distill_fire_asap.py hardcodes logs/p2a_distill.json; the pre-existing
# file is backed up first.
#
# Usage (WSL Ubuntu shell):
#   nohup bash scripts/_run_geomA_pipeline.sh > logs/geomA_nohup.log 2>&1 &

set -u
cd /home/sean/jsbsim-marl-formation || exit 1
PY=/home/sean/miniconda3/envs/marl_env/bin/python
D=logs/geomA_driver.log

mkdir -p logs results/shoot_eval data/expert

echo "=== geomA pipeline START $(date -Is) ===" > "$D"
[ -f logs/p2a_distill.json ] && cp -n logs/p2a_distill.json logs/p2a_distill_geomOLD_backup.json
echo "frozen baseline eval_distilled_d0_s42.json: $(ls -l results/shoot_eval/eval_distilled_d0_s42.json 2>/dev/null | awk '{print $5" bytes"}')" >> "$D"

# --- 1/4 rule-expert data regen (200 ep x 6.3 s = ~21 min) ------------------
echo "[1/4] expert regen START $(date -Is)" >> "$D"
$PY scripts/generate_shoot_rule_expert.py --episodes 200 \
    --out data/expert/shoot_rule_expert_geomA.npz > logs/geomA_1_expert.log 2>&1
echo "[1/4] rc=$? END $(date -Is)" >> "$D"

# --- 2/4 BC training (40 ep, ~3 min) ---------------------------------------
echo "[2/4] BC train START $(date -Is)" >> "$D"
$PY scripts/train_shoot_bc.py \
    --data data/expert/shoot_rule_expert_geomA.npz \
    --output data/expert/shoot_bc_geomA.pth \
    --metrics-out logs/geomA_bc_metrics.json > logs/geomA_2_bc.log 2>&1
echo "[2/4] rc=$? END $(date -Is)" >> "$D"

# --- 3/4 ASAP fire-head distillation (~2.8 h) ------------------------------
echo "[3/4] distill START $(date -Is)" >> "$D"
$PY scripts/distill_fire_asap.py \
    --weights data/expert/shoot_bc_geomA.pth \
    --out-weights data/expert/shoot_bc_asap_geomA.pth \
    --rollout-episodes 300 --eval-seeds 60 > logs/geomA_3_distill.log 2>&1
echo "[3/4] rc=$? END $(date -Is)" >> "$D"

# --- 4/4 100-ep holdout (the headline number, comparable to 87%) -----------
echo "[4/4] holdout eval START $(date -Is)" >> "$D"
$PY scripts/eval_bc_1v1.py \
    --weights data/expert/shoot_bc_asap_geomA.pth \
    --episodes 100 \
    --out results/shoot_eval/eval_geomA_d0_s42.json > logs/geomA_4_eval.log 2>&1
echo "[4/4] rc=$? END $(date -Is)" >> "$D"

echo "=== geomA pipeline DONE $(date -Is) ===" >> "$D"
