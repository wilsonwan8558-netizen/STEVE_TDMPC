#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/stEVE_safety_head_1e1be81
out="$repo/RL_TDMPC/logs/generic_solver_phase_g1b_checkpoint_sweep"
manifest="$out/checkpoint_manifest.json"
cd "$repo"
source RL_TDMPC/setup_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$out/selection/mplconfig"
mkdir -p "$out/selection" "$MPLCONFIGDIR"

jq -e '.status == "PASS" and .success_mismatches == 0 and .length_mismatches == 0 and .max_reward_absolute_difference == 0 and .trajectory_hash_mismatches == 0' \
  "$out/reproducibility_sanity.json" >/dev/null
jq -e '.status == "ALL_SIX_CHECKPOINTS_VERIFIED" and (.checkpoint_records | length) == 6' "$manifest" >/dev/null

for step in 200000 220000 240000 260000 280000 300000; do
  checkpoint=$(jq -r --argjson step "$step" '.checkpoint_records[] | select(.step == $step) | .path' "$manifest")
  hash=$(jq -r --argjson step "$step" '.checkpoint_records[] | select(.step == $step) | .sha256' "$manifest")
  test -f "$checkpoint"
  test "${#hash}" -eq 64
  /home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
    RL_TDMPC/experiments/generic_solver_g1b/evaluate_navigation.py \
    --checkpoint "$checkpoint" --expected-sha256 "$hash" \
    --seed-start 91000 --episodes 100 \
    --output "$out/selection/step_${step}.csv" --resume \
    2>&1 | tee -a "$out/selection/step_${step}_console.log"
done
