# Phase G10.2 — local translation–force response model

**Final classification: C — the current learned-force repair route is not sufficient. Do not start G10.3 online control.** The frozen local model selected **no nonzero translation correction** in either the 40-state development set or the 60-state untouched confirmation set. On the 40 nominally unsafe development states, 27 had a true-safe fixed-rotation translation on the finite SOFA grid, but the rule rescued **0/27**: it abstained on 38 states and passed through the unchanged, truly unsafe nominal action in the other two. The confirmation cohort contains **no recoverable nominally unsafe state**, so its 42 true-safe selections are nominal pass-throughs, **not** successful repairs. This is an offline model/decision-rule result, not evidence that translation-only physical regulation is impossible.

The force limit is `F_lim = 3.685331040756903` nominal N, a simulation-derived statistical reference **not a clinical injury threshold**. “No safe translation found” always refers to the finite tested grid and complete two-interval forks, not physical impossibility.

## Frozen protocol, alignment, and dataset

- Branch: `feature/generic-contact-force-safety`. Runtime: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, PyTorch `2.7.1+cu128`, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0` NVIDIA GeForce RTX 5090. No CPU fallback.
- TD-MPC2: frozen `RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt`, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. Original optimized-mean nominal action and fixed optimized-mean second action; no controller training or replanning.
- Physics and force: GenericConstraintSolver/projected Gauss–Seidel, tolerance `1e-4`, 2000 iterations, constraint forces enabled, friction `mu=0.001`, SOFA `dt=0.006 s`. Distal approximately 3 mm contact force uses `J_contact^T lambda / 0.006 s`; `F_t` is the maximum distal force over SOFA substeps. The true target is `R2(v)=max(F_(t+1),F_(t+2))` after candidate `[v,w_nom]` followed by the **unchanged full nominal second action**. A fork with fewer than two complete intervals is censored, never labeled safe.
- Dataset split is by **episode and parent state**, not perturbation row. Training: 560 G9.5 parent states from 140 episodes; validation: 120 states from 30 disjoint episodes. Selection uses episode progress and prior executed force, not future repair labels. The train/validation expansion replayed seeds 90000–90199 in one scene and matched all 15,500 archived parent steps exactly for observation, applied action, reward, tip, insertion, and force. It made 6,118 new local two-action forks; two were censored. Exact G9.5 nominal true-force forks were reused, not recollected. Total complete labeled rows: **6,796** (5,610 train; 1,186 validation), including 680 exact-nominal rows with zero loss weight.
- Complete-row realized `|delta_v|` coverage, with cumulative bands: train `<=0.1: 2,528`, `<=0.2: 3,317`, `<=0.5: 4,148`, `>0.5: 1,462`; validation `536, 700, 877, 309`, respectively. Larger stopping/withdrawal witnesses are retained with weight 0.25; local rows `|delta_v|<=0.5` have weight 1. Prior-force coverage is recorded in `local_dataset_summary.json`; train states include 229 zero-force, 247 positive-low, 37 near-boundary, and 47 high prior-force states, using the predeclared prior-force categories.
- G10.1's **40 already-inspected states** and their 4,241 aligned true-SOFA sweep forks were reused for **development testing only**, never model fitting or validation; this preserves the earlier held-out role but is not a new untouched test. They comprise 18 held-out unsafe-nominal states plus 11 each at difficult-event `t*−1` and `t*−2`. Replaying the 22 event features matched 1,848 parent steps and frozen G10 nominal predictor values exactly, with no new force forks. Source and file hashes are recorded in `local_dataset_manifest.json` and the two replay audits. The 107000–107199 online cohort remained unopened.

## Local model and frozen deployment rule

`x_t` is the frozen deployable 256D TD-MPC2 current latent plus 38D E1 current physical estimate. Input is `[x_t, v_nom, delta_v]` (296D), where `delta_v = clip(v_nom + requested_delta, -1, 1) - v_nom` in normalized action units. **Rotation stays exactly `w_nom`.** The target is `delta_R2 = R2_true(v_nom+delta_v)-R2_true(v_nom)`; the nominal point is overridden to exactly zero. There are no privileged future features. No `M_LOCAL_ABS` alternative was trained.

`M_DELTA` is a small 296→64→64→3 ReLU MLP with noncrossing Q50/Q90/Q95 outputs. Train-only input normalization, weighted quantile loss, three fixed training seeds, and validation-only Q95 residual corrections were frozen before development testing. Checkpoints:

| Training seed | Checkpoint SHA256 | Validation weighted pinball (N) | Validation Q95 residual correction (N) |
|---:|---|---:|---:|
| 261021 | `b782226943bd6f49b55cd9a6048cd0f8786fd0c660b237fed241831f828cd211` | 0.08269 | 0.11559 |
| 261022 | `1297b8ba8efd4da945b8767a401bbc5c4b2cf58984864897ac2b8f1337789900` | 0.08470 | 0.11004 |
| 261023 | `3d0d67e89539c0401204b69a944d363614e2d419145dbeda79766c72ff054b58` | 0.08072 | 0.15049 |

Normalization SHA256: `e1c1a2b49dabcf6454793955705376b4f6f7730436a722afc6a73169ee9a4d22`. No best seed is selected. The **deployable nominal baseline** is the maximum over the three frozen G10 single-action, validation-corrected Q95(R2) estimates at the exact nominal pair; it never uses future true `R2_nom`. A candidate's operational reconstructed force is that baseline plus the maximum of the three corrected local Q95(`delta_R2`) values. This sum of separately estimated quantiles is **not a mathematically guaranteed absolute Q95** or a clinical safety bound.

Search uses 101 uniform float32 translations in `[-1,1]` plus exact nominal and clipped nominal ±0.05/0.10/0.20, deduplicated by float32 value. Select minimum `|v-v_nom|` (lower `v` tie-break) only when reconstructed force `<=F_lim` **and** corrected-Q95 inter-replica range `<=0.4830828905 N`, the validation-only 95th-percentile spread. Otherwise **ABSTAIN**; there is no forced-withdrawal fallback. The full second action is fixed. The freeze file hashes the architecture, training range, normalization, all checkpoints, calibration/abstention rule, `F_lim`, H=2, grid, second-action semantics, development outputs, and evaluation code.

## Development: local response and actual repair

| Evaluation | States | True-safe fixed-rotation translation on finite grid | Predicted candidate | Nonzero translation correction | Truly safe selection | False-safe selection | Abstain | False-infeasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G10.1 frozen global predictor, same 40 development states | 40 | 27 | 5 | 3 | 2 | 3 | 35 | 22 |
| G10.2 frozen local rule, same 40 states | 40 | 27 | 2 | **0** | **0** | 2 | 38 | 25 |
| G10.2 difficult `t*−1` subset | 11 | 7 | 0 | 0 | 0 | 0 | 11 | 7 |
| G10.2 difficult `t*−2` subset | 11 | 9 | 0 | 0 | 0 | 0 | 11 | 9 |

All 40 development nominal actions were true-unsafe. G10.2's two candidate “repairs” were exactly unchanged nominal actions and both false-safe. Recoverable-state rescue is **0/27 = 0%**; the false-safe rate is **2/2 proposed = 100%**; false-infeasible is **25/27 = 92.6%** among recoverable states; abstention is **38/40 = 95%**. The 22 specifically requested G10.1 false-infeasible states were **all still abstained**, with zero true-safe repairs. G10.1 and G10.2 use different predicted search-grid densities, so this table is an operational comparison on the same states and force protocol, not a claim of pointwise identical candidate sets.

The only two states with both a true-safe boundary and a G10.2 predicted-safe point have absolute boundary errors **0.944** and **0.100** normalized translation units (mean 0.522); both predicted points are at `v_nom` while the true-safe point requires a decrease, so neither is on the correct correction side. In the other 25 recoverable states there is **no predicted-safe boundary** to compare; averaging boundary error only over the two proposed states would hide the dominant failure.

| Replica | Development Q50 `delta_R2` state-mean MAE / RMSE (N) | Local sign agreement | Decreased / increased translation sign | Local slope-sign agreement |
|---:|---:|---:|---:|---:|
| 261021 | 0.594 / 0.702 | 1067/1442 = 74.0% | 776/1071; 291/371 | 31/40 |
| 261022 | 0.612 / 0.728 | 1104/1442 = 76.6% | 830/1071; 274/371 | 31/40 |
| 261023 | 0.577 / 0.680 | 1063/1442 = 73.7% | 793/1071; 270/371 | 31/40 |

Median statewise local Spearman(`v,delta_R2`) is 0.980 true and 1.000 predicted for each replica on development. Broad directional agreement did **not** translate into safe-threshold crossing. The non-deployable diagnostic that substitutes **true future** nominal force for the baseline still finds only one proposal (true-safe), so nominal-baseline error alone does not explain the failure; local delta magnitude/boundary prediction remains inadequate.

At exact witness translations, G10.2 predicts only 2/40 `v=0`, 2/40 `v=-0.5`, and 1/40 `v=-1` as safe under its force and uncertainty rule, versus **16/40, 22/40, and 26/40** truly safe on the finite SOFA forks. Mean true versus mean replica-Q50 predicted `delta_R2` (N): `v=0` −1.914 vs −0.492; `v=-0.5` −2.677 vs −0.798; `v=-1` −3.267 vs −1.073. Mean true versus reconstructed conservative R2 (N): 4.238 vs 7.893; 3.475 vs 7.626; 2.885 vs 7.404. These are descriptive witness means, not policy actions; neither withdrawal nor stopping is universally safe.

## Untouched 108000–108029 confirmation and its limitation

`pre_confirmation_freeze.json` was written **before** any 108000-series scene was opened (SHA256 `b1db6ecf7bbf6639fb41bf48b17c1f1e9474abe022adf3e96b860c9502d94095`). A passive first pass selected exactly two parent states per episode by the predeclared quarter-progress and maximum-prior-executed-force slots, without candidate-force labels. A second ordered, single-reused-scene pass matched all **2,345** parent steps against the passive episodes with **zero** differences in observation, applied action, reward, tip, insertion, and force. It collected **6,354** candidate two-action forks; 29 one-interval forks were censored, all from one state. The exact freeze gate passed in the pre-frozen confirmation analyzer. No retraining, recalibration, threshold change, parent intervention, or online evaluation occurred.

The cohort composition is decisive:

| Confirmation category | States | True-safe translation found | Frozen rule selection |
|---|---:|---:|---|
| Nominal already true-safe | 54 | 54 (the nominal point itself) | 42 unchanged nominal pass-throughs; 12 abstentions |
| Nominal true-unsafe with complete two-interval nominal fork | 5 | **0 on finite grid** | 5 abstentions |
| Nominal two-interval fork censored | 1 | 0 found; 29 candidate forks censored | 1 abstention |

Thus the aggregate confirmation file's 54/60 “translation-recoverable,” 42/54 “rescue,” 42 true-safe selections, 0 false-safe selections, and zero boundary error reflect **nominally safe pass-through**. They are valid file-level counts but **not** evidence of repairing unsafe states. For the estimand “rescue among truly recoverable, nominally unsafe states,” the confirmation denominator is **zero** and the rate is **undefined**, not 77.8%. The 12 confirmation false-infeasible cases are rejections of already-safe nominal states. The five nominally unsafe complete states had no safe tested translation; the censored state cannot support a physical infeasibility claim. All 42 proposed actions had `|v_pred-v_nom|=0`; the 42 zero boundary errors are likewise pass-through artifacts.

`G101_vs_G102_comparison.json` is the hashed **pre-confirmation** comparison and correctly records that the 108000 series was unopened at freeze time. The separate `G101_vs_G102_final_comparison.json` records the post-confirmation cohort breakdown without altering that frozen artifact.

The independent confirmation response diagnostics still challenge local-response transfer:

| Replica | Q50 `delta_R2` state-mean MAE / RMSE (N) | Raw local sign agreement | Sign agreement excluding exact-zero true changes | Local slope-sign agreement |
|---:|---:|---:|---:|---:|
| 261021 | 0.119 / 0.157 | 722/2111 = 34.2% | 722/1235 = 58.5% | 19/59 |
| 261022 | 0.111 / 0.150 | 678/2111 = 32.1% | 678/1235 = 54.9% | 19/59 |
| 261023 | 0.123 / 0.163 | 694/2111 = 32.9% | 694/1235 = 56.2% | 19/59 |

There are 876 true local changes classified as zero under the predeclared `1e-6 N` sign tolerance among the 2,111 valid local nonnominal probes per replica; the model predicts a nonzero sign on all of them, so raw sign accuracy should not be interpreted without the nonzero-only column. Median statewise local Spearman is about **0.135 true versus 1.000 predicted**. On confirmation, exact `v=0/-0.5/-1` witnesses are true-safe in 53/60, 52/60, and 51/60 states, while the frozen rule labels only 36/60, 39/60, and 40/60 safe. These are mostly already-safe nominal states; confirmation witness rates do not establish a safe withdrawal policy.

## Decision

**C — local translation–force learning as currently formulated fails to produce usable repair.** The decisive development evidence is 0/27 recoverable unsafe states rescued, no nonzero correction at all, 2/2 false-safe nominal pass-throughs, and 25/27 false-infeasible recoverable states. The untouched confirmation is genuinely frozen and replay-valid, but its state selection yielded no recoverable unsafe case, so it cannot independently estimate unsafe-state rescue or overturn the development failure. The lower independent response-direction agreement and abstentions reinforce caution; the benign aggregate “success” must not be presented as a repair result. Dominant observed blockers are local delta magnitude/safe-boundary mismatch and conservative absolute reconstruction; state coverage/ambiguity remains plausible but is **not causally resolved** here.

No arbitrary numerical pass/fail threshold was introduced after seeing results. No G10.3 online test is justified from this evidence; no further force-predictor ablation is launched automatically. The finite SOFA sweeps show translation-only physical opportunities in the development set, but this learned predictor plus frozen abstention rule does not exploit them. The task stops at G10.2.
