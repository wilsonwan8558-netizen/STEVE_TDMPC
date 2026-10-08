#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/stEVE_safety_head_1e1be81
out="$repo/RL_TDMPC/logs/generic_solver_phase_g1b_checkpoint_sweep"
cd "$repo"
source RL_TDMPC/setup_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$out/confirmation/mplconfig"
mkdir -p "$out/confirmation" "$MPLCONFIGDIR"

jq -e '.status == "FROZEN_BEFORE_CONFIRMATION" and .selection_seeds == [91000,91099] and .confirmation_seeds_unseen_at_selection == true' \
  "$out/selected_checkpoint.json" >/dev/null
test -s "$out/selection_checkpoint_summary.csv"
test -s "$out/paired_checkpoint_comparison.csv"
checkpoint=$(jq -r '.path' "$out/selected_checkpoint.json")
hash=$(jq -r '.sha256' "$out/selected_checkpoint.json")
test -f "$checkpoint"
test "${#hash}" -eq 64

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1b/evaluate_navigation.py \
  --checkpoint "$checkpoint" --expected-sha256 "$hash" \
  --seed-start 90000 --episodes 200 \
  --output "$out/confirmation_200_episode_results.csv" --resume \
  2>&1 | tee -a "$out/confirmation/evaluation_console.log"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1b/analyze_confirmation.py --root "$out" \
  2>&1 | tee "$out/confirmation/analysis_console.log"
