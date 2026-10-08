# Phase G1B — Generic TD-MPC2 checkpoint maturity sweep

Status: **CONDITIONAL BASELINE READY**. The 280k checkpoint was frozen from the selection set before any Generic-solver confirmation episode was run. Its independent 200-seed confirmation is complete and satisfactory, with checkpoint-to-checkpoint nonmonotonicity retained as a maturity caveat.

## Protocol and integrity

All six required G1 format-1 TD-MPC2 checkpoints (200k, 220k, 240k, 260k, 280k, 300k) were present, and their file SHA256, model-state SHA256, internal step, embedded configuration, and training seed 1 were verified in [checkpoint_manifest.json](checkpoint_manifest.json). The frozen Generic solver adapter SHA256 is `f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67`; the evaluator SHA256 is `27ebf5d293206938a17a86e80966200f33dff4693e7688e0b1c7c1d303a3b54a`. Neither changed during the sweep. The scene check verified GenericConstraintSolver, ProjectedGaussSeidel, tolerance 1e-4, maxIterations 2000, computeConstraintForces=True, FrictionContactConstraint mu=0.001, and dt=0.006 s.

The evaluator followed the archived Original TD-MPC2 lifecycle: one SOFA scene per checkpoint run, reused across ordered episode seeds; `set_seed(seed)`, `env.reset(seed=seed)`, and `agent.act(..., eval_mode=True)` for navigation. G1B additionally called `env.action_space.seed(seed)` as required. It used the checkpoint's embedded config and the Original agent without safety MPC, Force Head, shield, CBF, force penalty, or force-based termination. It ran in `steve_tdmpc` via `conda run`, Python `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, torch `2.7.1+cu128`, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0`, physical/visible device 0 (NVIDIA GeForce RTX 5090), at git commit `59ea0ca569cdf2cc48095bd1dccc205fb6900ff7`. The exact commands and lifecycle are in [evaluation_protocol.json](evaluation_protocol.json) and [run_selection.sh](run_selection.sh). Saved-prefix resume replays all earlier seeds in the same newly created scene before appending, checking outcomes and trajectory hashes.

The exact launch entrypoints were `systemd-run --user --unit=steve-g1b-selection --working-directory=/tmp/stEVE_safety_head_1e1be81 /bin/bash /tmp/stEVE_safety_head_1e1be81/RL_TDMPC/logs/generic_solver_phase_g1b_checkpoint_sweep/run_selection.sh` and, only after [selected_checkpoint.json](selected_checkpoint.json) was written, `systemd-run --user --unit=steve-g1b-confirmation --working-directory=/tmp/stEVE_safety_head_1e1be81 /bin/bash /tmp/stEVE_safety_head_1e1be81/RL_TDMPC/logs/generic_solver_phase_g1b_checkpoint_sweep/run_confirmation.sh`. Those scripts contain the fully resolved `conda run --no-capture-output -n steve_tdmpc python -u .../evaluate_navigation.py` commands, checkpoint paths/hashes, seed ranges, and output files. Both units exited successfully.

The 260k checkpoint was evaluated twice on seeds 91000–91004. Success, length, reward, and full trajectory hashes matched exactly for all five seeds; see [reproducibility_sanity.json](reproducibility_sanity.json). The six completed selection runs each contain exactly seeds 91000–91099 in order, with zero simulation-error episodes. No force or safety metric was collected or used.

## Selection-set navigation results

All values below are from the fixed 100-seed selection set; success/failure conditional length is useful context, not a different selection objective. Per-episode rows are in [selection_episode_results.csv](selection_episode_results.csv), with the exact summary in [selection_checkpoint_summary.csv](selection_checkpoint_summary.csv).

| Checkpoint | Success | Mean length | Median length | Mean reward | Median reward | Mean success length | Mean failure length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200k | 82/100 | 85.10 | 59.0 | 3.373357 | 3.476273 | 59.88 | 200.0 |
| 220k | 86/100 | 74.99 | 57.5 | 3.430878 | 3.500512 | 54.64 | 200.0 |
| 240k | 85/100 | 78.80 | 59.0 | 3.426534 | 3.595836 | 57.41 | 200.0 |
| 260k | 75/100 | 91.86 | 59.5 | 3.174824 | 3.575699 | 55.81 | 200.0 |
| 280k | **88/100** | **72.53** | **56.0** | **3.460404** | 3.570087 | 55.15 | 200.0 |
| 300k | 82/100 | 85.64 | 59.5 | 3.397104 | 3.644155 | 60.54 | 200.0 |

Adjacent comparisons use paired seed outcomes and 1,000 fixed-seed episode-level bootstrap resamples (`20261007`). Differences below are later minus earlier; brackets are percentile 95% intervals. Complete paired success transitions, median differences, and bootstrap details are in [paired_checkpoint_comparison.csv](paired_checkpoint_comparison.csv) and [bootstrap_summary.json](bootstrap_summary.json).

| Pair | Success change, pp [95% CI] | Mean length change [95% CI] | Mean reward change [95% CI] |
|---|---:|---:|---:|
| 200k → 220k | +4 [-2, +12] | -10.11 [-20.71, -1.39] | +0.0575 [-0.0465, +0.1870] |
| 220k → 240k | -1 [-11, +8] | +3.81 [-10.09, +18.54] | -0.0043 [-0.1617, +0.1373] |
| 240k → 260k | -10 [-20, 0] | +13.06 [-0.99, +27.67] | -0.2517 [-0.4393, -0.0667] |
| 260k → 280k | +13 [+3, +23] | -19.33 [-33.44, -3.97] | +0.2856 [+0.0856, +0.4824] |
| 280k → 300k | -6 [-14, +3] | +13.11 [+0.66, +25.86] | -0.0633 [-0.1868, +0.0650] |

The 220k–240k high-performance pair is consistent with a local plateau; later performance is not monotonic. The 260k dip is substantial, 280k recovers, and 300k shows worse mean length with no success gain over 280k. This is a maturity region with checkpoint-to-checkpoint variability, not a smooth 200k–300k plateau.

## Training-curve support and frozen choice

The 180k–300k training log is summarized in [training_curve_summary.json](training_curve_summary.json). Across six consecutive 20k-step windows, training episode success ranged 90.2%–95.2%, mean episodic reward 3.445–3.529, and mean total loss 0.0666–0.0675. Loss remained finite and smooth and no simulator errors appeared. The 280k–300k window had lower mean success and reward than 260k–280k (90.3% versus 94.1%; 3.445 versus 3.520), a mild late-degradation signal. These noisy on-training-policy windows are supporting context only; they did not replace selection-set evaluation.

The selected **Generic TD-MPC2 Baseline** is **step 280000**, frozen in [selected_checkpoint.json](selected_checkpoint.json) before confirmation. Its path is `/tmp/stEVE_safety_head_1e1be81/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt`; file SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`; model-state SHA256 `d659a53bb5734f3354669bc5f8b8a3e2789a96a57d9096ac6ba4671b9dad934d`. Training seed is 1. The original checkpoint file was not renamed or modified.

The predeclared hierarchy was success rate, then mean episode length, then mean reward within the mature region. At 280k, all three metrics were best of the six on selection seeds; it was not chosen simply for later training step or for one isolated metric. Its 260k/300k neighbors expose nonmonotonicity, which remains a stability caveat and must be considered in final classification. No result from seeds 90000–90199 influenced selection.

## Final confirmation and historical LCP context

The one permitted 280k confirmation used a single reused SOFA scene for exactly 200 ordered seeds, 90000–90199. It completed with no simulation errors. The per-seed data are in [confirmation_200_episode_results.csv](confirmation_200_episode_results.csv), and aggregate values are in [confirmation_200_summary.json](confirmation_200_summary.json). An independent CSV read reproduced the summary. The selected checkpoint manifest SHA256 remained `de590aa75c6c13a1b3cfb3527449348f28e4fbf2b8560184030e8d686e441417` throughout confirmation; no other Generic checkpoint was evaluated on this set.

| Confirmation measure | 280k Generic baseline |
|---|---:|
| Success | **172/200 = 86.0%** |
| Mean episode length | **77.5** |
| Median episode length | **58** |
| Mean episode reward | **3.4300794215** |
| Median episode reward | 3.5279222817 |
| Mean successful-episode length | 57.56 |
| Mean failed-episode length | 200.0 |

Relative to its separate 100-seed selection set, confirmation success was 2.0 percentage points lower (86% versus 88%), mean length 4.97 steps longer (77.50 versus 72.53), and mean reward 0.0303 lower (3.4301 versus 3.4604). These are modest set-to-set differences, not a major generalization failure; the frozen checkpoint was not revisited.

The independently verified archived LCP Original reference in [historical_lcp_reference_verified.json](historical_lcp_reference_verified.json) is 167/200 successes (83.5%), mean length 82.705, median length 62, and mean reward 3.3946877875. On the same ordered confirmation seeds, Generic had +2.5 percentage points success, 5.205 fewer mean steps, and +0.035392 mean reward relative to the historical LCP configuration; see [historical_lcp_comparison.json](historical_lcp_comparison.json). Navigation performance under the new Generic solver remained broadly comparable and numerically slightly higher on these descriptive metrics. Because the solver changes the underlying contact dynamics, this is **not** a controlled solver-only or algorithm ablation and cannot establish that one method is inherently better.

## G1B classification

**B. CONDITIONAL BASELINE READY.** A usable, frozen 280k Generic TD-MPC2 navigation baseline is identified and independently confirmed at 86% success on 200 seeds, without a meaningful selection-to-confirmation collapse. The condition is the nonmonotonic checkpoint sweep: 260k dropped to 75% success, 280k recovered to 88%, and 300k fell to 82%, with a paired mean-length disadvantage for 300k. Thus a smooth, contiguous late-training plateau was not demonstrated, even though the selected checkpoint itself generalized adequately. Future work may use this exact frozen checkpoint while retaining that maturity caveat; no G1B force characterization, retraining, physics change, Force Head, shield, or CBF was performed.
