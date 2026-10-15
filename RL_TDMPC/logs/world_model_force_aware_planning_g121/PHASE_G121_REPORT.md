# Phase G12.1 — calibrated one-step force-aware world-model planning

**Final development classification: C — one-step force-aware planning is not supported under the frozen G12.1 protocol.** True Generic-SOFA high-force exposure fell, but every nonzero weight caused excessive navigation failure. No weight was selected, no pre-confirmation freeze was written, and no untouched confirmation cohort was opened. This is a development result, not a confirmation result or a clinical safety claim.

## Frozen model, calibration, and exact planning semantics

The TD-MPC2 checkpoint remains `step_280000.pt`, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. The inherited G12 force head remains SHA256 `b1d8f66b943efdab47c134f7126db0cb361270cabcc1e3e6899914219cd0d7e6`, architecture `258→64→64→1` with ReLU/ReLU/Softplus. Its train-set-only featurewise mean/std are stored as checkpoint buffers. It maps the frozen TD-MPC2 pre-transition latent `z_t[256]` and first normalized candidate action `a_t[2]` to the next RL interval's maximum distal contact force in nominal simulation N. The head, encoder, latent dynamics, reward predictor, value/Q heads, controller checkpoint, GenericConstraintSolver settings, and force reconstruction were not retrained or changed. See [`inherited_g12_audit.json`](inherited_g12_audit.json) and [`frozen_force_head.json`](frozen_force_head.json).

Calibration used **G12 validation episodes only**, choosing the higher-interpolation 95th percentile of `max(F_lim − F_hat_raw, 0)` on validation true-above-limit transitions. This froze `delta_F = 2.1008816303183657 N`, with `F_hat_cal = F_hat_raw + delta_F`. The threshold `F_lim = 3.685331040756903` nominal N remains the original simulation-derived statistical reference, not a clinical injury threshold. Held-out test labels were read only after freezing the offset. Detailed rows are in [`force_calibration.json`](force_calibration.json) and [`force_calibration_metrics.csv`](force_calibration_metrics.csv).

| One-step split | Raw false-safe / true high | Calibrated false-safe / true high | Raw false-warning / true safe | Calibrated false-warning / true safe | Raw → calibrated MAE (N) | Raw → calibrated RMSE (N) |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 49/130 | 6/130 | 34/2186 | 398/2186 | 0.560 → 2.304 | 1.018 → 2.440 |
| Held-out test | 45/105 | 4/105 | 22/2036 | 280/2036 | 0.455 → 2.260 | 0.908 → 2.369 |

Held-out above-limit coverage rose from `60/105 = 57.14%` to `101/105 = 96.19%`, but the held-out false-warning rate rose from `1.08%` to `13.75%`. The offset therefore improves conservative detection while substantially sacrificing absolute accuracy and specificity.

The original planner still scores horizon-3 trajectories with its frozen discounted predicted reward plus terminal Q; larger `J_task` is better. Original 512-candidate, 24-policy-prior, six-iteration, 64-elite sampling and optimized-mean action semantics are unchanged. **Only the first candidate action** receives a force score: `F_hat_cal_1 = head(z_t,a_t) + delta_F`, `c_force = max(F_hat_cal_1 − F_lim,0)^2`, `J_total = J_task − lambda_F·c_force`. The force cost horizon is exactly `H_force=1`; no second force or force sequence is predicted in G12.1. The implementation is [`g121_planner.py`](../../experiments/world_model_g121/g121_planner.py), a subclass that changes only `_estimate_value`; `lambda_F=0` returns the original value without invoking the head. This is an internal trajectory-scoring change, not post-hoc action repair or a CBF.

## Development objective-scale and planner gate

On 16 inherited G12 validation/development planner states (512 original final-iteration candidates each), `S_task = median_state[Q90(J_task)−Q10(J_task)] = 0.013568128645420091`; among the three force-active states, `S_force = median_state[Q90(c_force)−Q10(c_force)] = 0.03031464414257279`. This is well above the float32 rounding floor `2.78×10⁻⁷ N²`, so `lambda_base = S_task/S_force = 0.44757670852436304` is finite. The frozen test set is `{0, 0.04475767085243631, 0.13427301255730892, 0.44757670852436304}`. No other coefficient or force horizon was tested. See [`lambda_scale_definition.json`](lambda_scale_definition.json).

Thirteen of 16 original candidate sets were force-inactive (`c_force=0` for all candidates); three of 16 were force-active and force-discriminative. At state `90000_46`, even the smallest nonzero coefficient gave a weighted force-cost Q90−Q10 of `0.0511` against task-score spread `0.00699`; at the other two active states its weighted spread was `0.00136` and `0.000460`. The term was therefore heterogeneous but neither universally zero nor universally dominant. Offline candidate details are in [`objective_scale_reaudit.json`](objective_scale_reaudit.json) and [`planner_state_classification.csv`](planner_state_classification.csv).

Full CUDA MPPI was rerun from identical warm-start means and RNG states at the 16 development planner states. `lambda_F=0` reproduced all 16 archived baseline first actions exactly. Each nonzero coefficient changed the final optimized-mean first action in exactly the three force-discriminative states and **zero** force-inactive states. At all three changed states, the model predicted lower calibrated force for the executable first action. Both the top final-iteration candidate and the actual optimized-mean first action are logged separately; they are not interchangeable. See [`planner_mppi_reaudit.json`](planner_mppi_reaudit.json) and [`planner_selection_diagnostics.csv`](planner_selection_diagnostics.csv).

## Exact-parent true-SOFA one-step forks

One historical Generic scene was replayed in seed order from 90000 through decision 77 of 90199: 15,474 parent transitions. The three preselected discriminative states were forked from the **same live parent state** for baseline and every nonzero weight. Baseline forks, repeated baseline forks, and parent steps matched archived G2 signatures exactly, including true force at the selected states. See [`plan_fork_parent_replay_audit.json`](plan_fork_parent_replay_audit.json) and [`true_sofa_one_step_forks.csv`](true_sofa_one_step_forks.csv).

| State | Baseline true next-force (N) | Nonzero-weight true next-force range (N) | Immediate task observation |
|---|---:|---:|---|
| 90000, step 46 | 7.568 | 6.835–6.838 | Above-limit force fell ~0.73 N but stayed above limit; reward fell 0.02237→0.01186–0.01215 with strong translation/rotation reversal. |
| 90150, step 62 | 0.8528 | 0.8538–0.8642 | Small increases; all below limit. |
| 90199, step 77 | 1.2440 | 1.24405–1.24406 | Tiny increases; all below limit. |

The prespecified plan-fork gate passed **narrowly for development only** because one truly above-limit state had a useful local reduction without one-step termination or catastrophic immediate reward loss. The two below-limit ranking errors and absence of a threshold crossing were explicit limitations, not hidden successes. See [`plan_fork_gate_decision.json`](plan_fork_gate_decision.json).

## Fresh paired closed-loop development

The proposed 30-seed development range `112000–112029` was checked before any rollout against structured seed fields, CSV seed columns, and seed-named artifacts in the available branch and prior experiment trees; no episode-seed overlap was found in that scoped audit. The ordered seeds were then frozen in [`development_seed_manifest.json`](development_seed_manifest.json), with audit scope in [`development_seed_audit.json`](development_seed_audit.json). Each arm created one Generic-SOFA scene and reset it in the same 30-seed order. Episodes were saved incrementally; restart semantics replay preceding seeds before accepting new rows. Every RL step used true max-substep distal force. In B1, a same-state λ=0 counterfactual first action was calculated under the same warm start/RNG and discarded before the force-aware planner acted, solely to measure action changes.

| λ_F | Success | Mean reward | Mean steps | True-force Q95 / Q99 / max (N) | Above-limit steps / all steps | High-force events | Mean / Q90 event duration (steps) | Different first action at same state |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28/30 | 3.530 | 65.3 | 4.194 / 7.632 / 9.522 | 114/1960 (5.82%) | 57 | 2.00 / 4 | 0% |
| 0.044758 | 10/30 | 2.632 | 150.7 | 1.683 / 3.690 / 10.763 | 47/4522 (1.04%) | 37 | 1.27 / 2 | 67.34% |
| 0.134273 | 7/30 | 2.454 | 163.4 | 1.528 / 2.561 / 9.437 | 22/4902 (0.45%) | 18 | 1.22 / 2 | 67.28% |
| 0.447577 | 2/30 | 2.179 | 193.1 | 1.347 / 2.587 / 9.243 | 29/5793 (0.50%) | 29 | 1.00 / 1 | 57.79% |

The per-seed results are in [`development_episode_results.csv`](development_episode_results.csv), the true-force/action traces in [`development_step_results.csv`](development_step_results.csv), and the aggregate tables in [`development_force_metrics.csv`](development_force_metrics.csv) and [`development_task_metrics.csv`](development_task_metrics.csv). Event durations are contiguous above-limit RL steps; one RL step is `22×0.006 = 0.132 s`. Success-only mean time to target was `7.354 s` (baseline), then `6.890`, `5.695`, and `12.738 s`; the nonzero success-only values are heavily survivor-selected and must not be read as faster navigation overall. Relative to baseline, the nonzero arms lost 18, 21, and 26 paired successes and gained none. Their absolute above-limit step counts also fell, so the force reduction is not merely a larger denominator from longer episodes. Yet those longer episodes and widespread failure make the safety/task tradeoff unacceptable.

The B1 same-state action-change fractions were 58–67%. Mean absolute translation differences were `0.570`, `0.628`, `0.590`; mean absolute rotation differences were `0.541`, `0.523`, `0.522` in normalized action units. Mean signed translation differences were approximately `−0.569`, `−0.628`, `−0.587`: the planner frequently preferred withdrawal. This is consistent with the conservative calibration and force cost, but this phase does not isolate their separate causal contributions.

## Decision, remaining gates, and scope

[`lambda_selection.json`](lambda_selection.json) records **no selected nonzero λ**. The final G12.1 classification is **C: one-step world-model force-aware planning is not supported under this frozen configuration**. The model can distinguish some candidate actions, and conservative calibration reduces one-step false-safe predictions; at the audited states, force cost changes planner selection only when active. One exact-parent high-force fork improved true force. However the closed-loop B1 policies lose most successful navigation episodes. Thus `H_force=1` is not sufficient for a useful safety contribution *under this calibrated head and prescribed scale-matched weights*; this does not prove all one-step formulations impossible.

No full prior-seed-use audit for confirmation, `pre_confirmation_freeze.json`, fresh 200-seed confirmation cohort, or confirmation B0/B1 evaluation was performed because development did not support continuation. The inherited head/checkpoint/solver integrity checks passed; there is no selected λ to freeze. A formal CBF comparison is **not justified as the automatic next step** from this result. No reactive supervisor, post-hoc repair layer, TD-MPC2 retraining, alternate force horizon, new coefficient sweep, or CBF was added. Per the C outcome rule, the phase stops here; the dominant observed failure is the force/task tradeoff manifested as frequent withdrawal and navigation failure, with false warnings and heterogeneous objective scaling as plausible contributors that were not disentangled in this phase.
