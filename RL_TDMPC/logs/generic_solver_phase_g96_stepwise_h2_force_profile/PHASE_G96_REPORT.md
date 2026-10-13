# Phase G9.6 — stepwise H=2 force-profile comparison

**Final classification: C — stepwise decomposition does not solve the problem. G10 is not justified.** The valid untouched comparison is the **105000–105029** cohort in [`confirmation_v2/`](confirmation_v2/), frozen after the numerical protocol amendment and complete development reaudit. The earlier 104000–104029 cohort and its 36 saved forks are **diagnostic only** and are excluded from every final statistic.

## Completed development work

The exact G9.5 data were recovered without recollection. The audit matched all **7,193** complete planner rows to their original per-state JSON for the 298D input, original two-action sequence, `F1_true`, `F2_true`, and `R2_true=max(F1,F2)` at float32 equality; seven one-step terminal forks remain censored. It also matched all **14,300** G9 behavior training/validation/development rows' first two forces and actions to the archived G2 behavior trajectory. Hashes and state/episode-disjoint G9.5 splits are recorded in `g95_data_audit.json`. The TRAIN/VALIDATION/DEVELOPMENT-TEST planner splits remain 560/120/120 states, with 5,038/1,080/1,075 complete planner rows. No 104000-series label entered training or validation.

The frozen scalar baseline is the three G9.5 **M1 planner-conditioned** E1/H=2 replicas, not a retrained approximation. Each uses a 256D current TD-MPC2 latent, frozen 38D E1 estimate from three 14D observations and two prior 2D actions, and the original two candidate 2D actions (298D total); it predicts Q50/Q90/Q95/Q99 of `R2=max(F1,F2)`. The new profile model uses the **same inputs, train/validation rows, input normalization, 64–64 ReLU trunk, pinball objective family, Adam 0.001, batch 512, 60-epoch budget, patience 8, and seeds 260312/260313/260314**. Its only predictive target change is two independent noncrossing four-quantile heads for true `F1` and `F2`, with loss equal to the sum of their mean pinball losses. Checkpoint SHA256 values are:

| Stepwise training seed | SHA256 |
|---:|---|
| 260312 | `861223bd85d535d2f03faf7e26e04fcaec7e66e84431f27474f14ee593777030` |
| 260313 | `f092b2c9028ec1c1ebde404895bc6ee0f1ba8ad39b14c16df783d39bc2c8f5a6` |
| 260314 | `5c3ca6823b52d9f0419f02a2d9733c87a913814e48f987042dfc569b8929ebe7` |

The predeclared validation-only shared Q95 residual correction was **`delta1=0.2320449352 N`, `delta2=0.0515686274 N`**. Raw validation Q95 coverages for F1 were 88.3%, 90.4%, 91.8% and for F2 were 94.3%, 93.5%, 94.3%; both steps met the predeclared undercoverage criterion for correction. The stepwise safety decision requires **both** corrected Q95(F1) and corrected Q95(F2) to be at or below frozen `F_lim=3.685331040756903` nominal N. `max` of those two corrected scores is used only for ranking, not learned as another scalar output. `development_stepwise_coverage.csv` reports every F1/F2 Q50/Q90/Q95/Q99 coverage, calibration error, and pinball loss separately; after correction on held-out development, F1 Q95 coverage was 93.2%/95.0%/94.8% and F2 was 89.7%/87.9%/89.2%. The calibration was **not** retuned after seeing this undercoverage.

The 1,075 complete development-test planner plans comprised 916 SAFE-SAFE, 65 SAFE-HIGH, 33 HIGH-SAFE, and 61 HIGH-HIGH sequences. Results below compare paired fixed seeds on the exact same plans; true-safe is SAFE-SAFE, and unsafe is the union of the other three categories.

| Seed | Model | Safe plans rejected | Unsafe plans accepted | SAFE-HIGH missed | Mean within-state Spearman | Feasible states left usable | False predicted infeasible |
|---:|---|---:|---:|---:|---:|---:|---:|
| 260312 | Scalar | 88/916 | 20/159 | 11/65 | 0.309 | 96/107 | 11/107 |
| 260312 | Stepwise | 57/916 | 26/159 | 16/65 | 0.308 | 99/107 | 8/107 |
| 260313 | Scalar | 86/916 | 27/159 | 17/65 | 0.267 | 96/107 | 11/107 |
| 260313 | Stepwise | 82/916 | 26/159 | 17/65 | 0.312 | 96/107 | 11/107 |
| 260314 | Scalar | 80/916 | 26/159 | 17/65 | 0.293 | 96/107 | 11/107 |
| 260314 | Stepwise | 92/916 | 26/159 | 17/65 | 0.329 | 97/107 | 10/107 |

Thus development evidence was **mixed**: stepwise outputs improved mean ranking somewhat and one seed's safe-plan acceptance, but did not improve SAFE-HIGH detection (16/17/17 misses versus 11/17/17) and did not consistently improve safe rejection or unsafe acceptance. The full SAFE-SAFE, SAFE-HIGH, HIGH-SAFE, HIGH-HIGH rates, top-three containment, pairwise ranking, secondary temporal-severity ranking, plan-level scores, per-state plan availability, and three-seed mean/SD/worst values are saved in `temporal_event_metrics.csv`, `state_ranking_metrics.csv`, `development_plan_results.csv`, `development_plan_metrics.csv`, and `training_seed_robustness.json`. These development findings cannot substitute for untouched confirmation.

## Planner executable-action numerical semantics and prior stop

The original [`pre_confirmation_freeze.json`](pre_confirmation_freeze.json) (SHA256 `9f64a84557859a130a039c45be329a59b6a40f746d509384d8e77e681ed15793`) preceded the opened 104000–104029 cohort. Its second ordered scene pass stopped at **seed 104001, decision 8, state 4**, after only seed 104000's four states and 36 forks had been saved. [`confirmation_protocol_stop.json`](confirmation_protocol_stop.json) remains unchanged (SHA256 `93354dd1dca8440d43fe06b0d31a102c973fc01f2e8056477139c56eb848249c`). No model was scored on those partial forks.

The raw TD-MPC2 optimized mean's first action was `[1.000000238418579, 0.9999591708183289]`; the executed bounded action was `[1.0, 0.9999591708183289]`. The original exact-equality check rejected the raw first component's `2.384185791015625e-7` overshoot. The authorized [`protocol_amendment.json`](protocol_amendment.json) makes the ninth candidate's *first two 2D actions* componentwise `clip(raw_mean, -1, 1)` for both SOFA labeling and protocol identity, then matches its first action to the executed action using `np.allclose(rtol=0, atol=1e-6)`. This reflects the actual executable control, not a learned or outcome-selected substitution. The eight sampled candidates retain their frozen score/diversity selection, float32 values, order, and exact index identity. All 6,400 sampled development plans were already within action bounds and required no clipping. There is no approximate nearest-neighbor rematching of sampled candidates.

The full ordered, one-scene development reaudit in [`protocol_reaudit.json`](protocol_reaudit.json) replayed **15,500 parent steps, 800 selected states, and 7,200 candidate plans** against the archived G9.5 identities and G2 trajectories. Every parent continuous-field maximum difference was zero; no candidate identity changed. Previously valid states had **zero** matching changes due to clipping and **zero** due to `atol` alone. The observed 104001 boundary case is resolved by clipping. **Zero** newly ambiguous optimized-mean-versus-sampled full-plan matches occurred. The frozen sampled set already contained **42** sampled–sampled pairs numerically close under `1e-6`, but these retain distinct exact identities and are not subject to the new matching rule. The amendment did not introduce or rematch them.

## Untouched 105000–105029 confirmation protocol

[`pre_confirmation_freeze_v2.json`](pre_confirmation_freeze_v2.json) was written **before seed 105000 was opened** (SHA256 `1f1a79c93d7906f0d3dc6daeb79c37ba3abcb8cfbf70e5cc75790e78e4db354b`). It binds the clipping rule and tolerance, passing reaudit, controller/planner source, exact G9.5 dataset/candidate protocol, all three frozen scalar and stepwise checkpoint hashes, input normalization, shared F1/F2 validation corrections, F_lim, evaluation source, and development results. No model was retrained, recalibrated, or selected on confirmation.

The `steve_tdmpc` CUDA runtime used `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, Torch `2.7.1+cu128`, and visible CUDA device index **0**, `NVIDIA GeForce RTX 5090`. The 30 ordered behavior episodes used one reused Generic/SOFA scene. The frozen G3 progress/prior-force rule selected four states per episode *without candidate outcomes*. A second one-scene ordered pass replayed all **2,517 parent steps** with zero recorded continuous-field differences and forked the same nine planner plans at each of **120** selected states. There were **1,080** forks: **1,076** complete two-interval F1/F2/R2 profiles and **four** censored one-step terminal forks. The latter have no fabricated F2/R2 label and are excluded from scoring. No future behavior-action pair replaced a planner plan. Confirmation itself had zero optimized-mean matches requiring clipping, zero requiring tolerance alone, and zero mean–sampled ambiguities; the repaired rule nevertheless remained frozen throughout.

The frozen assembler wrote the complete verified NPZ, then failed only while serializing a derived JSON state count stored as `numpy.int64`. The JSON summary was mechanically regenerated from that unchanged NPZ and the frozen state selection, with explicit Python integer conversion; see [`assembly_serialization_repair.json`](confirmation_v2/assembly_serialization_repair.json). This did not change labels, model scores, evaluation source, or the v2 freeze. Both frozen models were then scored on the **same 1,076 rows**.

## Confirmation results: paired identical plans

The 1,076 complete plans comprised **946 SAFE-SAFE, 23 SAFE-HIGH, 57 HIGH-SAFE, and 50 HIGH-HIGH**. True-safe is SAFE-SAFE; the remaining 130 are unsafe. Values are counts unless indicated. `Usable` is the number of the **108** states with at least one true-safe plan where the model retains a predicted-safe true-safe plan. The full rates, AUROC/AUPRC, top-three containment, pairwise ranking, and per-plan predictions are in [`confirmation_model_metrics.csv`](confirmation_v2/confirmation_model_metrics.csv), [`confirmation_temporal_event_metrics.csv`](confirmation_v2/confirmation_temporal_event_metrics.csv), [`confirmation_state_metrics.csv`](confirmation_v2/confirmation_state_metrics.csv), and [`confirmation_plan_results.csv`](confirmation_v2/confirmation_plan_results.csv).

| Seed | Model | SAFE-SAFE rejected | SAFE-HIGH missed | HIGH-SAFE missed | HIGH-HIGH missed | Unsafe accepted | Mean within-state Spearman | False infeasible states | Usable feasible states |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 260312 | Scalar | 172/946 | 5/23 | 17/57 | 2/50 | 24/130 | 0.302 | 18/108 | 90/108 |
| 260312 | Stepwise | 131/946 | 4/23 | 17/57 | 1/50 | 22/130 | 0.318 | 16/108 | 92/108 |
| 260313 | Scalar | 130/946 | 0/23 | 17/57 | 1/50 | 18/130 | 0.271 | 13/108 | 95/108 |
| 260313 | Stepwise | 153/946 | 5/23 | 17/57 | 1/50 | 23/130 | 0.254 | 17/108 | 91/108 |
| 260314 | Scalar | 139/946 | 0/23 | 17/57 | 1/50 | 18/130 | 0.272 | 17/108 | 91/108 |
| 260314 | Stepwise | 121/946 | 5/23 | 17/57 | 2/50 | 24/130 | 0.263 | 13/108 | 95/108 |

Thus the stepwise model's three-replica mean safe-plan rejection was **14.27%** versus scalar **15.54%** (a modest 1.27-point reduction), but mean unsafe acceptance was **17.69%** versus **15.38%** (2.31 points worse). Mean SAFE-HIGH miss rate was **20.29%** versus **7.25%**, worsening from a total **5** to **14** misses across the three matched replicas. HIGH-SAFE misses remained 17/57 for every model/seed. Mean within-state Spearman was **0.278** versus **0.282**, not an improvement; pairwise accuracy was likewise mixed (stepwise 0.649/0.623/0.619 versus scalar 0.644/0.625/0.629). False predicted infeasibility averaged **14.20%** versus **14.81%**, only a 0.62-point reduction. Usable true-safe states were **92/91/95** stepwise versus **90/95/91** scalar, with no consistent seedwise retention gain. The seedwise SAFE-HIGH and usable-plan swings matter more than a best-seed comparison; no best seed was selected. [`stepwise_vs_scalar_comparison.json`](confirmation_v2/stepwise_vs_scalar_comparison.json) provides mean, population SD, and worst-replica values for the predeclared metrics.

### F1/F2 quantile calibration and temporal interpretation

The stepwise heads use the same E1 298D input and Q50/Q90/Q95/Q99 quantile levels as development. The validation-only shared corrections remained **+0.2320449352 N** for Q95(F1) and **+0.0515686274 N** for Q95(F2). On untouched confirmation, empirical Q95 coverages were:

| Seed | F1 raw | F1 corrected | F2 raw | F2 corrected |
|---:|---:|---:|---:|---:|
| 260312 | 90.43% | 93.49% | 88.29% | 90.99% |
| 260313 | 91.36% | 93.59% | 88.48% | 90.89% |
| 260314 | 93.22% | 94.24% | 90.89% | 92.38% |

All F2 corrected Q95 coverages remain below the nominal 95% level and below corresponding F1 coverages. [`confirmation_stepwise_coverage.csv`](confirmation_v2/confirmation_stepwise_coverage.csv) includes every step's Q50/Q90/Q95/Q99 empirical coverage, calibration error, and pinball loss; no post-confirmation correction was applied. The derived `max(corrected Q95(F1), corrected Q95(F2))` is a *decision/ranking diagnostic*, not a trained or calibrated scalar R2 quantile. Lower F2 coverage suggests second-step uncertainty/calibration remains important, but does not establish that F2 is universally harder under every loss: F1 and F2 force distributions differ. Nor is second-step uncertainty the sole scalar-model error: all models miss the same 17/57 HIGH-SAFE plans, and two scalar replicas miss zero of 23 SAFE-HIGH plans. The stepwise temporal signal did not translate into improved prospective safety decisions or ranking.

The representative examples in [`representative_planner_examples.json`](confirmation_v2/representative_planner_examples.json) are the **first complete plan in frozen state/candidate order** for each category, not cherry-picked for model outcome. For seed 260312, the first SAFE-SAFE plan at seed 105000/state 0 has F1=F2=0 and both models predict safe. The first SAFE-HIGH plan at seed 105000/state 3 has F1=0.541 N, F2=4.724 N and both predict unsafe. The first HIGH-SAFE plan at seed 105009/state 38 has F1=11.704 N, F2=1.935 N and both predict unsafe. The first HIGH-HIGH plan at seed 105005/state 23 has F1=5.639 N, F2=8.021 N and both predict unsafe. These examples illustrate temporal categories; they do not define the classification.

## Final G9.6 interpretation and stop

**C — STEPWISE DECOMPOSITION DOES NOT SOLVE THE PROBLEM.** On the predeclared untouched, identical-plan comparison, small and seed-dependent improvements in safe rejection and false infeasibility do **not** constitute a material prospective gain: unsafe acceptance and SAFE-HIGH misses worsen, mean ranking does not improve, F2 Q95 remains undercovered, and usable safe-plan retention is unstable across seeds. The dominant evidence-backed blockers are second-step calibration/uncertainty and weak candidate ranking with training-seed instability; the data do not isolate state ambiguity or candidate diversity as a cause. The development result was also mixed and, crucially, its SAFE-HIGH concern persisted on untouched confirmation. **G10 is not justified and was not launched.** No online safety filter, Safety Head, CBF, retraining, force-metric redesign, or later phase was performed.
