#!/usr/bin/env bash
set -euo pipefail

repo=/tmp/stEVE_safety_head_1e1be81
out="$repo/RL_TDMPC/logs/generic_solver_phase_g2_force_baseline"
cd "$repo"
source RL_TDMPC/setup_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$out/mplconfig"
mkdir -p "$MPLCONFIGDIR" "$out/plots"

jq -e '.status == "PASS" and .episodes == 10 and .steps_per_mode == 692' "$out/behavior_invariance.json" >/dev/null
jq -e '.status == "PASS" and .distal_region_mm == 3.0 and .live_pipeline_probe.observed_sofa_substeps == 2200' "$out/force_pipeline_audit.json" >/dev/null
jq -e '.status == "FROZEN_280K_VERIFIED" and .checkpoint_step == 280000' "$out/checkpoint_audit.json" >/dev/null
test "$(sha256sum RL_TDMPC/experiments/generic_solver_g2/collect.py | cut -d ' ' -f 1)" = "8b50f9c9d109d52514264affae2e75230209f263ab494d31eb351716fc9ce73c"
test "$(sha256sum RL_TDMPC/experiments/generic_solver_g2/force_reconstruction.py | cut -d ' ' -f 1)" = "76c920ce5122cb8da97ba100851985267e734b2877fc41ab6726217cdb13fa26"
test "$(sha256sum RL_TDMPC/experiments/generic_solver_g2/substep_monitor.py | cut -d ' ' -f 1)" = "744759896f36d284c9ba039a537a9f5334a5e6f5d99752f036062ae55ced3534"
test ! -e "$out/episode_level_force.csv"
test ! -e "$out/step_level_force.csv"

/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u \
  RL_TDMPC/experiments/generic_solver_g2/collect.py \
  --seed-start 90000 --episodes 200 --force on --output-dir "$out" \
  2>&1 | tee "$out/full_collection_console.log"
