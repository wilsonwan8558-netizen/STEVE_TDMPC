# Phase W1/G12 — world-model force-aware planning

**Status: STOP at the Step 12 objective-scale gate.** This is not a completed force-aware-planning evaluation. No force-aware MPPI, true-SOFA plan fork, closed-loop development, or confirmation was run. The original controller, solver, task, and force definition remain unchanged.

## Frozen planner and force-head alignment

The authoritative frozen model is `RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt` (SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`). The full pre-change audit is in [`tdmpc2_planner_audit.json`](tdmpc2_planner_audit.json). It was made before any G12 source modification. CUDA device 0 was a visible NVIDIA GeForce RTX 5090 under the `steve_tdmpc` environment, torch `2.7.1+cu128`.

`TDMPC2Agent.act` calls `_plan` in `RL_TDMPC/tdmpc2/agent.py`. The 14-dimensional observation is encoded to a 256-dimensional SimNorm latent `z_t`. The frozen world model scores a candidate action at each step using `reward(z_k,a_k)`, then computes `z_(k+1)=WorldModel.next(z_k,a_k)`. Termination is estimated from the next latent. At horizon 3, `pi(z_3)` supplies the terminal action for a two-of-five-critic averaged Q value. The same latent transition function also supports policy-prior candidate generation. There is no independent force dynamics model.

The original CUDA MPPI/CEM-like planner scores 512 length-3 action sequences per iteration: 24 policy-prior and 488 clipped Gaussian candidates. It runs six updates, keeps 64 elites, weights them with temperature 0.5, clips standard deviation to `[0.05,2]`, warm-starts the next control step from the shifted optimized mean, and returns the first action of the final optimized mean in deterministic evaluation. It maximizes discounted predicted reward plus terminal Q with discount 0.975, clips actions to `[-1,1]`, and uses the frozen checkpoint configuration.

The new *head only* maps `[z_k(256), a_k(2)]` through `258→64→64→1` with ReLU hidden layers and a Softplus output. It predicts the **maximum true distal contact force over the next RL interval**, in nominal simulation N: `F_hat_(t+k+1)=head(z_k,a_k)`. This is aligned with reward's pre-transition `(z,a)` and attaches to the exact TD-MPC2 latent rollout. The distal region is approximately 3 mm; the frozen reconstruction is `J_contact^T lambda / 0.006 s`, with the interval maximum over 22 SOFA substeps. `F_lim=3.685331040756903` nominal N is a simulation-derived statistical high-force reference, **not** a clinical injury threshold. See [`force_alignment_definition.json`](force_alignment_definition.json) and [`force_head_definition.json`](force_head_definition.json).

## Existing-data training and held-out diagnostics

The data join used 15,500 existing G2 true-force Generic-SOFA transitions from seeds 90000–90199 and G3 frozen-CUDA encoder latents. All source rows were checked for alignment; five latent rows were re-encoded exactly. The predeclared whole-episode G3 split was reused: 140 train, 30 validation, 30 held-out test seeds. There was no new SOFA data collection and no fabricated force label. The encoder, latent dynamics, original reward/value heads, and original planner were frozen. One head-only fit used train-only feature mean/std, raw nominal-N MSE, Adam at `1e-3`, batch 512, and validation early stopping; best epoch 6, validation MSE `1.0360 N²`. Head checkpoint SHA256 is `b1d8f66b943efdab47c134f7126db0cb361270cabcc1e3e6899914219cd0d7e6`. See [`force_dataset_manifest.json`](force_dataset_manifest.json), [`force_dataset_split.json`](force_dataset_split.json), and [`force_head_training_history.json`](force_head_training_history.json).

On 2,141 held-out one-step transitions, MAE was `0.4548 N` and RMSE `0.9082 N`. Of 105 true above-limit steps, 45 were false-safe (`42.86%`); of 2,036 true safe steps, 22 were false warnings (`1.08%`). The 105 above-limit steps had MAE `2.3051 N` and mean prediction error `-1.8665 N`. Separate no-contact, low-force, near-limit, and above-limit rows are in [`force_head_metrics.csv`](force_head_metrics.csv).

An actual frozen `WorldModel.next` rollout over held-out behavior action sequences, with no future observation re-encoding, showed rapid degradation: depth 1 MAE/RMSE `0.4451/0.9022 N`, depth 2 `3.1943/6.3637 N`, depth 3 `8.2189/16.6243 N` (2,081 complete windows). This is an offline behavior-action diagnostic, not a validation of candidate-plan forces in SOFA. See [`rollout_depth_metrics.csv`](rollout_depth_metrics.csv).

## Pre-rollout objective-scale gate

The intended cost is `J_force = Σ_(k=1..2) max(F_hat_k − F_lim, 0)²` in nominal `N²`, with `J_total = J_TDMPC2 − lambda_F × J_force` maximized. `H_force=2`; the task planner remains horizon 3. This formula was evaluated **offline on captured original-planner candidates only**. It was not integrated into MPPI. Frozen CUDA planner actions were checked against the archived G2 actions while replaying selected validation states, and the decomposed reward-plus-terminal-Q objective agreed with `_estimate_value` to `1e-6` tolerance.

The authoritative expanded audit sampled 16 states: four distinct validation seeds in each previous-force category, 512 candidates each, at both the first and final of six original planner iterations. The earlier two-seed samples in `objective_scale_audit.json` and `objective_scale_audit_initial.json` are retained but superseded. At the **final** iteration, median `J_TDMPC2` across states ranged `0.2718–1.8658`; its predicted-reward part ranged `−0.0497–0.8933`, terminal-Q part `0.2078–1.2483`. Within-state task-score `Q95−Q05` ranged `0.00704–0.09883`. These are the relevant scales for candidate discrimination, not just the absolute objective.

In 12 of 16 states **all 512 candidates had `J_force=0`**; 13 of 16 had median cost zero. Where the cost activated, `lambda_F=0.1` was often far larger than the task-score spread:

| Validation state | Median task objective | Task Q95−Q05 | Median `J_force` | Median penalty at `lambda_F=0.1` | Penalty / task spread |
|---|---:|---:|---:|---:|---:|
| 90075, step 89 | 0.6541 | 0.01172 | 0.5565 | 0.05565 | 4.75× |
| 90000, step 46 | 1.0554 | 0.01505 | 14.2672 | 1.42672 | 94.79× |
| 90000, step 56 | 1.8658 | 0.09883 | 22.2898 | 2.22898 | 22.55× |

`lambda_F=0.5` and `1.0` are proportionally more dominant in active states; no weight changes a zero-cost candidate set. The first-iteration audit shows the same heterogeneity. Same-candidate argmax comparisons in the audit are **not** new MPPI runs and are not planner-action comparisons. Exact candidate actions, reward/value terms, two predicted forces, penalties, and combined scores are saved in [`planner_candidate_diagnostics_expanded_final.csv`](planner_candidate_diagnostics_expanded_final.csv) and its `_first.csv` counterpart. The expanded summaries are [`objective_scale_audit_expanded_final.json`](objective_scale_audit_expanded_final.json) and [`objective_scale_audit_expanded_first.json`](objective_scale_audit_expanded_first.json).

The example set `{0, 0.1, 0.5, 1.0}` is therefore **numerically inappropriate for a rollout**: it is often dominant where active but blind in most audited states. The explicit gate decision is in [`objective_scale_decision.json`](objective_scale_decision.json). A provisional small scale-consistent set for a *future protocol decision* is `{0, 0.0005, 0.001, 0.002}`; **none was run or selected**. Smaller weights alone do not resolve zero-cost blindness or depth-2 model drift. The existing test is sufficient to stop, not to assert that a particular replacement weight works.

## Downstream status and interpretation

No planner integration or force-aware trajectory selection was performed. Thus there is no selected `lambda_F`, no true-SOFA two-action plan-fork comparison, no B0/B1 paired development safety/task metrics, and no evidence yet that predicted force changes the final MPPI first action or lowers true force. No new development cohort was run. A final seed-reuse audit, confirmation cohort, pre-confirmation freeze, and untouched confirmation remain contingent on successful development and were not reached. No A/B/C outcome classification can be assigned, and the study does **not** yet support or refute world-model force-aware planning. A later CBF formulation is not justified by these data. No reactive supervisor, post-hoc action repair, hard rejection, additional dynamics model, or CBF was added.

Any continuation needs an explicit decision on how to handle both objective scaling and two-step prediction behavior before re-entering the planner/true-SOFA gates; simply running the original example weights would violate the Step 12 STOP condition.
