#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/stEVE_safety_head_1e1be81
run="$repo/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006"
cd "$repo"
source RL_TDMPC/setup_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$run/mplconfig"
mkdir -p "$MPLCONFIGDIR"

if test -e "$run/logs/metrics.jsonl" || test -e "$run/checkpoints/step_10000.pt"; then
  echo "Refusing to overwrite or silently resume an existing G1 training run" >&2
  exit 2
fi
if ! jq -e '.status == "SMOKE_TRAINING_PASS" and .finite_losses and .checkpoint_round_trip' \
  "$run/smoke/smoke_summary.json" >/dev/null; then
  echo "G1 smoke training gate failed" >&2
  exit 3
fi
if ! test -s "$run/evaluation/episodes.csv" || ! test -s "$run/evaluation/summary.csv"; then
  echo "G1 smoke evaluation gate failed" >&2
  exit 4
fi

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1/audit_protocol.py \
  --config "$run/config.yaml" --output "$run/launch_audit.json" \
  2>&1 | tee "$run/launch_audit_console.log"

# The checkpoint-embedded config is the historical Original algorithm. This
# new config changes only the 300k budget and output directories. No --resume.
/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/train.py --config "$run/config.yaml" \
  2>&1 | tee "$run/train_console.log"
