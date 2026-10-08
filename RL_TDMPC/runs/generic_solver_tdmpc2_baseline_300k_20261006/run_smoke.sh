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
export MPLCONFIGDIR="$run/smoke/mplconfig"
mkdir -p "$run/smoke" "$MPLCONFIGDIR"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1/audit_protocol.py \
  --config "$run/config.yaml" --output "$run/protocol_audit.json" \
  2>&1 | tee "$run/smoke/protocol_audit_console.log"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1/smoke_test.py \
  --config "$run/config.yaml" --output-dir "$run/smoke" \
  2>&1 | tee "$run/smoke/train_smoke_console.log"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/evaluate.py \
  --checkpoint "$run/smoke/smoke_checkpoint.pt" \
  --episodes 1 --seed 1000 --device cuda:0 \
  2>&1 | tee "$run/smoke/evaluation_console.log"
