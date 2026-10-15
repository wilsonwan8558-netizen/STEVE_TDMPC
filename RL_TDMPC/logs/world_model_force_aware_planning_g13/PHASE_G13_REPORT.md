# Phase G13 — geometry-enhanced force-aware world-model evaluation

**Final status: G13 force-representation gate FAIL; classification C.** Part A is complete. Part B/G14 was not started. This is a simulation force-representation result, not a claim about clinical injury or formal CBF guarantees.

## Frozen model and observation

The 300,000-step G13 CUDA training run completed before this evaluation. Its four predeclared late checkpoints (240k, 260k, 280k, 300k) passed a finite-tensor and format audit. No retraining or additional checkpoint candidates were used. On ordered development seeds 350000–350029, normal task-only MPPI selected **step 260000** by the frozen navigation-first rule: 30/30 successes, mean reward 3.709212, mean length 59.167, zero simulation errors. The selected checkpoint is `RL_TDMPC/runs/geometry_force_tdmpc2_g13_300k_20261014/checkpoints/step_260000.pt`, SHA256 `b2bba8be2e2722886c20952b15712deaa31ab612825999f1b3469a34fd6ef7be`.

| Checkpoint step | Successes / 30 | Mean reward | Mean steps | Simulation errors |
| ---: | ---: | ---: | ---: | ---: |
| 240000 | 27 | 3.570641 | 73.333 | 0 |
| **260000** | **30** | **3.709212** | **59.167** | **0** |
| 280000 | 29 | 3.663190 | 62.533 | 0 |
| 300000 | 29 | 3.637108 | 57.800 | 0 |

The exact 22-D float32 G13 observation order, dimensionality, normalization, and interpretation are frozen in `final_geometry_observation.json`. It contains the original 14 task features (five projected tracking points, projected target, device-rotation sine/cosine) followed by three normalized signed lumen-proxy clearances, projected wire–vessel tangent-angle sine/cosine, normalized distal bend, normalized local centerline radius, and a direction-validity bit. **The clearance is a signed centerline-radius lumen proxy, not exact collision-mesh wall distance.** It contains neither current force nor future information.

The environment and label remained GenericConstraintSolver/ProjectedGaussSeidel, tolerance `1e-4`, `maxIterations=2000`, `computeConstraintForces=True`, friction `mu=0.001`, SOFA `dt=0.006 s`. The target is next-RL-interval maximum distal (~first 3 mm) contact force reconstructed from `J_contact^T lambda / 0.006 s`. The frozen reference is `F_lim=3.685331040756903` nominal simulation N, a statistical high-force reference **not a clinical injury threshold**. All force predictions below use `H_force=1`.

## Navigation functionality

Paired task-only development evaluation used one reused scene per controller, the same ordered seeds 350000–350029, and no force-based action selection. G13 achieved 30/30 successes, mean reward 3.709212, mean length 59.167, zero truncations/errors, and mean success-only time to target 7.810 s. The frozen Generic controller achieved 27/30 successes, mean reward 3.536687, mean length 77.300, three truncations, zero simulation errors, and mean success-only time to target 8.404 s. **Navigation functionality passes.** These development results are not a force-safety claim.

## Matched G12 versus G13 held-out force heads

The two heads were tested against the same 2,141 archived G2 transitions across 30 held-out seeds, including 105 true above-`F_lim` steps. The G12 source labels are float64; the G13 geometry cache is an exact float32 cast of those labels (maximum rounding difference `4.7024e-7 N`). Both arms were scored against the same authoritative float64 truth. G12 uses the frozen post-hoc force head on the original task latent; G13 uses its jointly supervised geometry-enhanced latent and predicted post-action latent.

| Matched held-out metric | G12 | G13 |
| --- | ---: | ---: |
| MAE (N) | 0.454824 | 0.490549 |
| RMSE (N) | 0.908160 | 1.064555 |
| False-safe / 105 true high | 45 | 40 |
| Above-limit recall | 0.5714 | 0.6190 |
| False-warning / 2036 true safe | 22 | 35 |
| Near-boundary (`0.8–1.2 F_lim`) MAE (N) | 1.249880 | 1.297736 |
| Near-boundary false-safe / 20 true high | 13 | 15 |
| Near-boundary above-limit recall | 0.35 | 0.25 |

G13's modest overall false-safe improvement does **not** extend to the near-boundary subset, and overall MAE/RMSE worsen. Global error was not the primary gate.

## Primary normal-MPPI candidate test

At fixed decision indices 10, 25, 40, and 60 in the first ten G13 development episodes, 31 reachable parent states were available. Every final normal six-iteration MPPI first action (512 per parent) and the weighted-mean baseline action were executed from the **exact same live SOFA parent state**: 15,903 true one-step forks. Repeated baseline forks and parent execution were deterministic and matched; task-only episode outcomes reproduced the saved G13 navigation trajectories. Both force heads scored identical candidate actions. No force-aware planner was used.

The fork outcomes included 13,851 forces at or below `0.5 F_lim`, 1,018 between `0.5 F_lim` and `F_lim`, and 1,034 above `F_lim`. Three parent states had any above-limit candidate, but only **one** had both true safe and true above-limit candidates; no parent had an above-limit weighted-mean baseline with a sampled safe alternative. These are fixed-index development diagnostics, not population prevalence estimates.

| Candidate metric | G12 | G13 |
| --- | ---: | ---: |
| Aggregate candidate MAE (N) | 0.694706 | 0.197466 |
| Aggregate Pearson | 0.7084 | 0.9654 |
| True-high false-safe / 1034 | 513 | 466 |
| False-warning / 14869 true safe | 1531 | 0 |
| Mean within-state Spearman (19 defined states) | 0.2172 | 0.2808 |
| Median within-state Spearman | 0.2580 | 0.5718 |
| Overall pairwise force-order accuracy | 0.5948 | 0.6292 |
| Paired states where within-state Spearman is higher than the other head | 12 | 6 |
| High-vs-safe pairwise accuracy in the **only mixed state** | 1.0000 | 0.0307 |

Aggregate G13 predictions improve on this mostly low-force, on-policy candidate cohort. That between-state advantage does not establish reliable safety-relevant action ordering. In the only mixed-threshold parent (seed 350004, decision 40), the baseline true force was `3.640785 N`. G12's minimum-predicted sampled action truly produced `3.548442 N`; G13's minimum-predicted sampled action truly produced `3.693912 N` (above `F_lim` by only `0.008581 N`). The 4,040 high-vs-safe candidate pairs in this state are correlated comparisons within **one** parent, not 4,040 independent states. G13's within-state ranking was higher in only 6 of 19 comparable parents and lower in 12 (one tie).

## G12.3 true-high-parent failure-mode revisit

The separately frozen manifest selected 20 archived G2 true-high parent states and 20 matched low-force controls. Their historical reused-scene sequence was replayed through 15,239 parent steps. All archived behavior anchors reproduced exactly, including applied action, task observation/reward, distal force, tip position, and insertion length. At each parent, a normal G13 cold-start MPPI candidate distribution supplied 32 rank-spread first actions; the G13 weighted mean and archived G2 behavior anchor were also forked from the identical parent. This is a **true-force-enriched off-policy diagnostic**, never merged into on-policy prevalence.

At the 20 high parents, all 640 sampled candidate outcomes remained above `F_lim`; hence the test cannot show a one-step feasible action below the reference. Eighteen of the 20 parents nevertheless had a sampled action with **lower** true force than the archived behavior anchor. G12's minimum-predicted sampled candidate lowered that anchor force in 15/18 such parents; G13's did so in 14/18. Mean selected-sample-minus-anchor true force on those 18 parents was `-0.7603 N` for G12 and `-1.2216 N` for G13. However, this diagnostic paired-sample comparison favored G13 in only 6 parents, G12 in 9, with 3 ties; median G13-minus-G12 selected-sample true force was `+0.00047 N`. Thus a few large G13 gains drive its better mean, not a consistent directional advantage. This sampled-candidate argmin is **not** an implemented MPPI action rule. G12 versus G13 high-parent mean within-state Spearman was `0.5252` versus `0.1472`; median pairwise force-order accuracy was `0.8874` versus `0.5327`. False-safe counts among these high-parent forks (including anchors) were `136` versus `229`. The lack of a safe candidate is a candidate-availability limitation, while the worse G13 relative ranking and false-safe behavior are direct paired-head comparisons.

## Hard gate and interpretation

**FAIL — classification C: geometry-enhanced force-aware representation is not sufficient for this planner constraint.** G13 navigation works and some aggregate force metrics improve, but the predeclared gate requires *clear planner-relevant* force discrimination over G12. The held-out near-boundary false-safe count worsens, the only on-policy mixed-threshold state chooses a higher-force above-limit action under G13, paired within-state ordering does not consistently improve, and the enriched true-high-parent test favors G12 on ranking and false-safe behavior despite a better G13 mean chosen-force reduction. The informative mixed on-policy sample is small and barely crosses `F_lim`; this result is a failure to establish the required improvement, not proof that every G13 planner state is worse.

Accordingly **no G14 directory, CBF code, CBF fork, closed-loop force-constrained rollout, calibration sweep, or new training was created**. The frozen environment, force metric, `F_lim`, `H_force=1`, and normal MPPI semantics were not changed for this evaluation.

Machine-readable decision: `force_representation_gate_decision.json`. Detailed source artifacts: `checkpoint_integrity_audit.json`, `checkpoint_selection.json`, `navigation_baseline_summary.json`, `old_vs_g13_force_head.json`, `candidate_discrimination_summary.json`, `candidate_state_metrics.csv`, `high_force_candidate_results.csv`, `archived_high_candidate_summary.json`, and both fork-collection audits.
