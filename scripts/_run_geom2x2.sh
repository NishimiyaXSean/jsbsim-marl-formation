#!/usr/bin/env bash
# 2x2 control for the heading-bias change (commit fb48155).
#
#   factor A (weights): shoot_bc_asap_distilled.pth  [base ]  vs  shoot_bc_asap_geomA.pth  [geomA]
#   factor B (geometry): min_heading_bias_deg = 30.0 [geoOld U(30,60)]  vs  0.0 [geoNew U(0,60)]
#
# Cells 1 and 2 are REPRODUCTION checks against the two frozen numbers
# (base@geoOld should equal 87/100, geomA@geoNew should equal 95/100).
# Cells 3 and 4 are the clean contrasts: same geometry, same seeds, so the
# two arms are genuinely paired and a McNemar test becomes possible.
#
# Resumable: a cell whose JSON already exists is skipped, so the script can be
# re-run after an interruption. All four cells are 100 episodes at seed 42,
# difficulty 0.0.
#
# WSL2 IDLE-SHUTDOWN: this must run under an ATTACHED client, otherwise the VM
# shuts down ~60 s after the launching wsl.exe detaches and kills the job.
#   Interactive WSL shell :  cd ~/jsbsim-marl-formation && bash scripts/_run_geom2x2.sh
#   From Windows terminal :  wsl -d Ubuntu-22.04 -- bash /home/sean/jsbsim-marl-formation/scripts/_run_geom2x2.sh
# Expected wall clock: ~1.0-1.2 h for all four cells on an RTX 3060 Laptop.

set -u
cd "$(dirname "$0")/.."

PY=/home/sean/miniconda3/envs/marl_env/bin/python
D=logs/geom2x2_driver.log
BASE_W=data/expert/shoot_bc_asap_distilled.pth
GEOA_W=data/expert/shoot_bc_asap_geomA.pth

mkdir -p logs results/shoot_eval

run_cell() {
    tag="$1"; weights="$2"; minbias="$3"; out="$4"; log="$5"
    if [ -s "$out" ]; then
        echo "[$tag] SKIP (output exists) $(date -Is)" >> "$D"
        return 0
    fi
    echo "[$tag] START $(date -Is) weights=$weights min_heading_bias_deg=$minbias" >> "$D"
    "$PY" scripts/eval_bc_1v1.py \
        --weights "$weights" \
        --episodes 100 \
        --seed 42 \
        --difficulty 0.0 \
        --min-heading-bias-deg "$minbias" \
        --out "$out" > "$log" 2>&1
    echo "[$tag] rc=$? END $(date -Is)" >> "$D"
}

echo "===== 2x2 RUN START $(date -Is) =====" >> "$D"

# Reproduction checks first, so a broken harness fails fast.
run_cell base_geoOld   "$BASE_W" 30.0 results/shoot_eval/eval_2x2_base_geoOld_s42.json   logs/geom2x2_base_geoOld.log
run_cell geomA_geoNew  "$GEOA_W"  0.0 results/shoot_eval/eval_2x2_geomA_geoNew_s42.json  logs/geom2x2_geomA_geoNew.log
# The two cells that isolate the policy effect.
run_cell base_geoNew   "$BASE_W"  0.0 results/shoot_eval/eval_2x2_base_geoNew_s42.json   logs/geom2x2_base_geoNew.log
run_cell geomA_geoOld  "$GEOA_W" 30.0 results/shoot_eval/eval_2x2_geomA_geoOld_s42.json  logs/geom2x2_geomA_geoOld.log

echo "===== 2x2 RUN DONE $(date -Is) =====" >> "$D"
