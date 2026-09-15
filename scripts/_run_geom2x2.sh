#!/usr/bin/env bash
# 2x2 control for the heading-bias change (commit fb48155).
#
#   factor A (weights): shoot_bc_asap_distilled.pth  [base ]  vs  shoot_bc_asap_geomA.pth  [geomA]
#   factor B (geometry): min_heading_bias_deg = 30.0 [geoOld U(30,60)]  vs  0.0 [geoNew U(0,60)]
#
# At the default EPISODES=100 the first two cells are REPRODUCTION checks
# against the two frozen numbers (base@geoOld = 87/100, geomA@geoNew = 95/100).
# At any other episode count they are simply the same two cells.
#
# Resumable: a cell whose JSON already exists is skipped, so re-running after
# an interruption costs nothing.
#
# Env overrides
#   EPISODES=100            episodes per cell. Any value other than 100 appends
#                           _n<EPISODES> to the output name, so the 100-ep
#                           artefacts are never clobbered.
#   SEED=42                 base seed; episode e is run with seed SEED+e
#   CELLS="base_geoNew geomA_geoNew"
#                           run only these cells (default: all four)
#
# WSL2: a detached `setsid nohup` launch has been observed to survive on this
# host, but verify liveness with `ps -eo pid,etime,cmd | grep eval_bc` rather
# than by log size (the eval prints are block-buffered).
# Wall clock on an RTX 3060 Laptop: roughly 13 min per cell per 100 episodes.

set -u
cd "$(dirname "$0")/.."

PY=/home/sean/miniconda3/envs/marl_env/bin/python
D=logs/geom2x2_driver.log
BASE_W=data/expert/shoot_bc_asap_distilled.pth
GEOA_W=data/expert/shoot_bc_asap_geomA.pth

EPISODES="${EPISODES:-100}"
SEED="${SEED:-42}"
CELLS="${CELLS:-base_geoOld geomA_geoNew base_geoNew geomA_geoOld}"

if [ "$EPISODES" = "100" ]; then
    SFX=""
else
    SFX="_n${EPISODES}"
fi

mkdir -p logs results/shoot_eval

run_cell() {
    tag="$1"; weights="$2"; minbias="$3"
    out="results/shoot_eval/eval_2x2_${tag}${SFX}_s${SEED}.json"
    log="logs/geom2x2_${tag}${SFX}.log"
    if [ -s "$out" ]; then
        echo "[$tag] SKIP (output exists) $(date -Is)" >> "$D"
        return 0
    fi
    echo "[$tag] START $(date -Is) w=$weights min_bias=$minbias ep=$EPISODES seed=$SEED" >> "$D"
    "$PY" scripts/eval_bc_1v1.py \
        --weights "$weights" \
        --episodes "$EPISODES" \
        --seed "$SEED" \
        --difficulty 0.0 \
        --min-heading-bias-deg "$minbias" \
        --out "$out" > "$log" 2>&1
    echo "[$tag] rc=$? END $(date -Is)" >> "$D"
}

echo "===== 2x2 RUN START $(date -Is) episodes=$EPISODES seed=$SEED cells='$CELLS' =====" >> "$D"

for cell in $CELLS; do
    case "$cell" in
        base_geoOld)  run_cell base_geoOld  "$BASE_W" 30.0 ;;
        geomA_geoOld) run_cell geomA_geoOld "$GEOA_W" 30.0 ;;
        base_geoNew)  run_cell base_geoNew  "$BASE_W"  0.0 ;;
        geomA_geoNew) run_cell geomA_geoNew "$GEOA_W"  0.0 ;;
        *) echo "[warn] unknown cell '$cell' ignored" >> "$D" ;;
    esac
done

echo "===== 2x2 RUN DONE $(date -Is) =====" >> "$D"
