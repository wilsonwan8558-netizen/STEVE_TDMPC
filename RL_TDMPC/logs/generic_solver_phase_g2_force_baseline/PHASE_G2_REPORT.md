# Phase G2 — Generic-solver distal contact-force baseline

Status: **A. FORCE BASELINE READY** for a separate one-step force-prediction study. This is a descriptive *simulation* baseline, not a calibrated injury-risk model or a clinical safety limit. All 200 ordered seeds completed; force monitoring was behaviorally passive and navigation reproduced the frozen G1B baseline exactly.

## Frozen provenance and protocol

G2 runs on `feature/generic-contact-force-safety` from the committed Generic baseline snapshot `69b51499881aa92bf43641393e6b61f82d9daef1`; the navigation-baseline branch was not edited. The only policy is the selected 280k TD-MPC2 checkpoint:

`/tmp/stEVE_safety_head_1e1be81/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt`

Checkpoint SHA256: `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. Model-state SHA256: `d659a53bb5734f3354669bc5f8b8a3e2789a96a57d9096ac6ba4671b9dad934d`; training seed 1. [checkpoint_audit.json](checkpoint_audit.json) pins the selected manifest and G1B task reference hashes. Evaluation uses the `steve_tdmpc` environment, PyTorch `2.7.1+cu128`, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0` / `cuda:0` on an NVIDIA GeForce RTX 5090. It preserves the G1B lifecycle: one SOFA scene reused for ordered seeds 90000–90199, per-episode `set_seed(seed)`, `env.action_space.seed(seed)`, and `env.reset(seed=seed)`, with `agent.act(..., eval_mode=True)`. No exploration or safety module is active.

The frozen scene is GenericConstraintSolver, ProjectedGaussSeidel, tolerance 1e-4, maxIterations 2000, computeConstraintForces=True, FrictionContactConstraint mu=0.001, SOFA dt 0.006 s. No solver, friction, device, vessel, observation, action, reward, target, termination, or control-rate change was made.

## Force reconstruction and release gates

At each completed SOFA substep, the passive monitor reads `Generic.constraintForces` and the current mapped guidewire collision Jacobian. It verifies `len(lambda) = currentNumConstraints = 3 × currentNumConstraintGroups`, mapping each active frictional group to a normal and two tangent rows. A group is active when its unilateral normal multiplier exceeds 1e-12; negative normals or tangential multipliers without an active normal are rejected. Point or barycentric line-contact location is measured along the current collision polyline from the distal tip. Only groups at arc length **≤ 3 mm** are included, with all three rows of each selected group. Their Cartesian `J_contact^T lambda` contributions are summed as vectors before taking the norm, then divided by the **0.006 s SOFA substep dt**. The force unit is nominal simulation N under the validated scene convention, not calibrated clinical force.

The primary step signal `F_t = distal_contact_force_max_N` is the maximum of **all 22** substep distal resultants in the subsequent RL control interval. Diagnostics retain substep mean, final, force-time integral `Σ_i F_i × 0.006 s`, and maximum active distal contact-group count. A contact step means at least one active distal group during any substep, even if vector cancellation theoretically makes the resultant zero. [force_pipeline_audit.json](force_pipeline_audit.json) and [dataset_indexing_definition.json](dataset_indexing_definition.json) give exact semantics. The 20 mm window in G0.5 was for broad diagnostics; it is not the G2 safety signal.

G0 scaling was rechecked with a diagnostic-only 0.1 nominal-N load: baseline-subtracted `J_contact^T lambda / dt` reconstructed 0.102650 N (2.65% relative error, direction cosine 0.99861), while unscaled `J_contact^T lambda` had 99.38% relative error. This load acted about 10.44 mm from the tip, so it validates the Generic force scale, not alone the 3 mm attribution. Eight pure mapping checks passed (including shaft exclusion, line-boundary weighting, tangent rows, and vector cancellation). A live 100-control-step probe observed all 2,200 substeps, including 47 distal-contact steps, with zero endpoint callback-vs-direct reconstruction difference and a maximum peak-minus-endpoint difference of 0.278944 nominal N. All scene settings matched G0/G0.5.

The first 10 seeds were run force OFF and ON on separate fresh runs, each reusing its own scene across the ordered 10 seeds. Both reproduced every frozen G1B trajectory hash. Across **692 paired RL steps**, maximum absolute differences in decision/next observations, selected/applied actions, rewards, tip positions, and insertion lengths were all **0**; discrete outcome and termination mismatches were also **0**. See [behavior_invariance.json](behavior_invariance.json). This passed before the 200-seed force-ON run began.

## Exact navigation and collection regression

The force-ON full run completed all seeds **90000–90199** in one reused scene. For each episode, the collector required the frozen G1B success, length, total reward (tolerance 1e-9), and exact action/observation/reward/termination trajectory SHA256 **before saving its rows**. An independent post-run analysis repeated those checks: 200/200 seeds matched, with **zero** success, length, or trajectory-hash mismatches and **zero** reward difference. There were no simulation errors. Results exactly reproduce **172/200 successes (86.0%)**, **15,500 RL steps**, mean length **77.5**, median length **58**, and mean reward **3.4300794215**. See [task_regression.json](task_regression.json), [episode_level_force.csv](episode_level_force.csv), and [step_level_force.csv](step_level_force.csv).

## Generic force and episode distributions

All thresholds and quantiles below are recomputed from **this Generic 200-seed dataset**, using NumPy linear quantiles. Values are nominal simulation N, except FTI in nominal N·s. A contact-only RL step has at least one active distal contact group in any of its 22 substeps. There were **9,402/15,500 contact steps = 60.66%**; positive-force steps were identical in count, with no observed active-contact/zero-resultant step. Complete values are in [force_distribution_summary.json](force_distribution_summary.json).

| Primary max-substep `F_t` | Count | Mean | Median | Q75 | Q90 | Q95 | Q97.5 | Q99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All RL steps | 15,500 | 0.841 | 0.334 | 1.086 | 2.127 | 3.685 | 5.926 | 7.467 | 19.038 |
| Contact-only | 9,402 | 1.387 | 0.883 | — | 3.134 | 5.477 | 6.960 | 7.979 | 19.038 |

The all-step **Q90=2.126516**, **Q95=3.685331**, **Q97.5=5.926250**, and **Q99=7.466569** nominal N are statistical high-force references, **not clinical safety thresholds**.

| Episode-level statistic (200 episodes) | Mean | Median | Q90 | Q95 | Q99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Peak `F_t` (nominal N) | 6.894 | 7.533 | 9.290 | 13.153 | 15.976 | 19.038 |
| Total FTI (nominal N·s) | 4.810 | 3.304 | 9.785 | 13.436 | 25.317 | 28.552 |
| Contact fraction | 0.509 | 0.454 | 0.845 | 0.890 | 0.935 | 0.945 |
| Contact steps | 47.01 | 25 | 169 | 178.1 | 187.0 | 189 |
| Longest contact run (steps) | 31.15 | 12 | 157 | 170 | 174 | 175 |

Pooled step contact fraction (60.66%) exceeds the mean *per-episode* fraction (50.87%) because episode lengths vary and long failure episodes receive more weight in pooled steps. Across all episodes there were **863** contiguous contact runs, median **5** steps, Q95 **24**, maximum **175**. Contact can be recurrent and sustained, not only isolated.

Episode-level uncertainty used **1,000 fixed-seed (`20261008`) whole-episode bootstrap replicates**; individual correlated RL steps were not resampled. The mean episode contact fraction was **0.509** (95% interval **0.482–0.536**), mean episode peak **6.894** nominal N (**6.468–7.357**), and mean episode FTI **4.810** nominal N·s (**4.163–5.558**). Median bootstrap intervals and method details are in [bootstrap_summary.json](bootstrap_summary.json).

Successful episodes (172) had median peak **7.654** nominal N, mean FTI **3.561** nominal N·s, and mean contact fraction **0.462**. Failed episodes (28, all 200-step timeouts) had median peak **2.600**, mean FTI **12.481**, and mean contact fraction **0.798**. The higher failure exposure is partly associated with their longer duration; these are descriptive groups and do **not** establish a causal effect of force on navigation outcome.

## Temporal structure, precursors, and actions

For all-step Generic quantiles, contiguous exceedance runs within episodes were:

| Reference | Exceeding steps | Episodes with any | Run count | Median run | Q95 run | Max run |
|---|---:|---:|---:|---:|---:|---:|
| Q90 (2.127 N) | 1,550 | 188 | 829 | 1 | 5 | 11 |
| Q95 (3.685 N) | 775 | 156 | 422 | 1 | 4 | 9 |
| Q97.5 (5.926 N) | 388 | 135 | 218 | 2 | 4 | 5 |

High-force runs are often short but can cluster for several control steps. By normalized episode-progress thirds, Q95-event fractions were **1.37% early**, **1.52% middle**, and **12.09% late**; contact fractions were **38.56%**, **62.16%**, and **81.26%**. The upper tail is concentrated late in these trajectories, without a causal interpretation.

Retrospective precursors for **775 Q95** and **155 Q99** event steps were:

| Event | Previous lag | Nonzero force | ≥ Generic Q75 | ≥ Generic Q90 |
|---|---:|---:|---:|---:|
| Q95 | 1 step | 85.81% | 78.97% | 60.13% |
| Q95 | 2 steps | 86.19% | 67.35% | 43.87% |
| Q95 | 3 steps | 84.13% | 62.84% | 34.32% |
| Q99 | 1 step | 89.03% | 87.74% | 76.77% |
| Q99 | 2 steps | 81.94% | 76.77% | 63.87% |
| Q99 | 3 steps | 87.74% | 68.39% | 40.65% |

All event steps had the corresponding 1–3 preceding rows available. Events can overlap within runs; these statistics are **not** evidence of prospective predictability. See [temporal_structure.json](temporal_structure.json).

Correlations between absolute normalized selected action and force were modest or rank-dependent. For `|translation|`, same-control-interval Pearson/Spearman were **−0.025/−0.324**, and following-control-interval **0.002/−0.292**. For `|rotation|`, they were **0.079/−0.101** and **0.082/−0.102**. For action norm, **0.039/−0.221** and **0.056/−0.208**. The same-interval pairing is `a_t` versus force produced by `env.step(a_t)`; the following-interval pairing uses the next control interval within the same episode. These are associations under one policy, not evidence of controllability or action-only predictive power. See [action_force_relationship.json](action_force_relationship.json).

## Max-substep versus endpoint sampling

The max-minus-final force difference across all 15,500 steps had mean **0.3668**, median **0.0010**, Q90 **1.0178**, Q95 **1.8011**, and maximum **17.6511** nominal N. Among the 8,078 steps with a final force above 1e-9 N, the median peak/final ratio was **1.187** and Q95 **5.995**. On **1,324 contact steps (14.08%)**, the step peak was positive but the endpoint force was near zero.

With event levels defined from max-substep `F_t`, endpoint-only sampling would miss **999/1,550 Q90 events (64.45%)**, **448/775 Q95 events (57.81%)**, and **131/155 Q99 events (84.52%)**. This strongly supports the all-substep maximum as the G2 primary signal; ratios with near-zero final denominators are excluded. See [substep_vs_endpoint.json](substep_vs_endpoint.json) and the [force distribution](plots/force_distribution.png), [endpoint comparison](plots/substep_vs_endpoint.png), and [episode-outcome](plots/episode_peak_by_outcome.png) plots.

## Historical LCP context and future indexing

The accepted archived LCP Phase 1 summary was independently provenance-checked and is kept separate in [historical_lcp_reference.json](historical_lcp_reference.json). Generic versus historical LCP had contact-step fractions **60.66% vs 61.60%**, all-step force medians **0.334 vs 0.465**, all-step Q95 **3.685 vs 3.309**, episode-peak medians **7.533 vs 7.058**, episode-peak Q95 **13.153 vs 9.087**, and episode-FTI medians **3.304 vs 4.838** nominal units. The Generic upper episode-peak tail is materially higher in this descriptive comparison, while contact incidence is similar and median cumulative FTI is lower. Because the solver and frozen policy checkpoint both differ, this is **not** a controlled solver-only or algorithm ablation and says nothing about clinical correctness; no LCP thresholds or rows were reused. Exact differences and ratios are in [historical_lcp_comparison.json](historical_lcp_comparison.json).

Each G2 step row stores the exact pre-action policy observation `s_t`, selected normalized action `a_t`, applied raw action, next observation `s_(t+1)`, and the force generated over that **immediately following** `env.step(a_t)` interval. The future one-step target is therefore unambiguously **`F_(t+1)`** from decision-state indexing (the row’s `distal_contact_force_max_N`). Episode identity and zero-based step timing are preserved. No latent hook was added because exact observation/action alignment suffices; no predictor was trained. See [dataset_indexing_definition.json](dataset_indexing_definition.json).

**Final classification: A. FORCE BASELINE READY** for the next, separate one-step predictor study. Force extraction, passive behavior, 200-seed task regression, substep accounting, and analysis integrity passed. This classification does *not* imply a validated clinical safety limit or authorize a Force Head, shield, CBF, physics change, or retraining within G2.
