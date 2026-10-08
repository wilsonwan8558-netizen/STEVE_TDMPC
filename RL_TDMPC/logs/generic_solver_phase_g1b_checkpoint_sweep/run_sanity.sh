#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/stEVE_safety_head_1e1be81
out="$repo/RL_TDMPC/logs/generic_solver_phase_g1b_checkpoint_sweep"
checkpoint="$repo/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_260000.pt"
sha=3057c12acce930275ff1e6070d2591a6580b344e0afb18d4857ad23904e82cd0
cd "$repo"
source RL_TDMPC/setup_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$out/sanity/mplconfig"
mkdir -p "$out/sanity" "$MPLCONFIGDIR"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1b/evaluate_navigation.py \
  --checkpoint "$checkpoint" --expected-sha256 "$sha" \
  --seed-start 91000 --episodes 5 --output "$out/sanity/run_a.csv" \
  2>&1 | tee "$out/sanity/run_a_console.log"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1b/evaluate_navigation.py \
  --checkpoint "$checkpoint" --expected-sha256 "$sha" \
  --seed-start 91000 --episodes 5 --output "$out/sanity/run_b.csv" \
  2>&1 | tee "$out/sanity/run_b_console.log"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g1b/check_reproducibility.py \
  --first "$out/sanity/run_a.csv" --second "$out/sanity/run_b.csv" \
  --output "$out/reproducibility_sanity.json" \
  2>&1 | tee "$out/sanity/comparison_console.log"
