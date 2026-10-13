# Phase G9.7 — privileged observable-state ceiling and H=2 risk identifiability

**Final classification: C — privileged state did not materially improve the tested prospective-risk ceiling.** In the untouched 106000–106029 confirmation cohort, M_PRIV reduced safe-plan rejection modestly, but did not improve unsafe-plan acceptance or SAFE-HIGH detection consistently and worsened within-state ranking and pinball loss for all three paired seeds. This is a result for the specified 3582D observable-state extraction, train-state PCA adapter, and matched 42,372-parameter model—not a proof about a complete, inaccessible SOFA Markov state or every possible privileged model.

## Frozen protocol and data integrity

- Branch: `feature/generic-contact-force-safety`; GenericConstraintSolver, projected Gauss–Seidel, tolerance `1e-4`, maximum 2000 iterations, constraint-force computation on, contact friction `0.001`, SOFA dt `0.006 s`.
- Frozen TD-MPC2 checkpoint: `step_280000.pt`, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. Frozen planner source SHA256 `b7ead501ad7607f2ce1b1a3a46c23d2f3e16827302a3d9f8c8862900e6b27ef5`.
- Exact scalar target: `R2_true = max(F_(t+1), F_(t+2))`, where each `F` is the maximum distal (~3 mm) `J_contact^T lambda / 0.006 s` nominal-N force over that RL interval. Both intervals require complete labels. `F_lim = 3.685331040756903` nominal N. The frozen nine G9.5/G9.6-v2 planner candidates are executed open-loop for their first two actions; no second-step replan, future behavior action substitution, changed horizon, or changed force definition.
- Original G9.5 planner dataset SHA256 `81fb1fbf4e9eab0aa1c17704bbfaae506c5f45bdfcd92f113a2e70bcffda1ee3`; 800 parent states, 7,193 complete H=2 candidate rows, seven terminal-censored forks. State-level train/validation/development-test split is 560/120/120 states and 5,038/1,080/1,075 complete planner rows. Existing G9.5 actions and F1/F2/R2 labels were reused, not reforked. An ordered 90000–90199 reused-scene replay collected only the previously missing *pre-action* simulator state; all 15,500 parent steps, G4 38D physical values, G9.5 latent/E1 values, and 7,193 row-to-state identities matched exactly. See `dataset_alignment_audit.json` and `development_privileged_replay_audit.json`.
- CUDA execution: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, torch `2.7.1+cu128`, CUDA `12.8`, `CUDA_VISIBLE_DEVICES=0`, visible device index `0`, NVIDIA GeForce RTX 5090. The code rejects CPU fallback.
- The G9.6-v2 executable planner source SHA256 `73cbbacd2b2800aa0c5a1f3f2eecb1b3172af8456fc316770f38853022d0623f` and amendment/reaudit hashes were checked against its prior freeze. `pre_confirmation_freeze.json` was written **before** opening 106000-series data; its SHA256 is `16e75d434a69044efd6ffd464cdf7f60532d8dc7508197f80d71af2900920679`. Post-confirmation verification of all frozen source, definition, model, calibration, and development-artifact hashes passed.

## Predictor states and simulator coverage

`X_DEPLOY` is exactly the frozen G9.5/G9.6 294D pre-action state: the TD-MPC2 encoder's 256D current-observation latent followed by the E1 mapper's 38D estimated current physical state, based on three 14D observations and two preceding 2D actions. Appending the same two 2D candidate actions gives the original 298D informative input. For matched model parameter count, M_DEPLOY inserts 294 fixed-zero channels *before* the unchanged four action channels, giving a 592D network input without adding deployable information.

`X_PRIV` includes that exact 294D deployable state, plus 38D directly measured current G4 physical values and 256 principal components fitted without labels on the 560 training parent states only. Those components compress the following **3582D raw pre-action privileged observable simulator state**; its full per-field source, index ranges, physical meanings, and pre-action flags are in `privileged_state_definition.json`.

| Raw simulator-only group | Dimensions | Pre-action source/meaning |
|---|---:|---|
| Exact G4 physical | 38 | Current tip/distal shape, velocity, distal contact force, insertion/rotation |
| All rigid guidewire positions / velocities | 441 / 378 | 63 parent generalized pose (xyz+quaternion) and linear/angular velocity nodes |
| Parent free positions / free velocities / forces | 441 / 378 / 378 | Exposed completed-substep unconstrained buffers and distributed generalized forces |
| Collision-node positions / velocities / free positions | 228 / 228 / 228 | All 76 mapped guidewire collision nodes |
| Contact-node normal / tangent forces / indicators | 228 / 228 / 76 | Current sparse contact Jacobian and `Generic.constraintForces`, reconstructed per node with `J^T lambda / dt`; current contact support |
| Nearest-vessel-vertex displacement / distance | 228 / 76 | Current collision nodes to fixed vessel mesh vertices; **vertex proxy, not exact signed wall distance** |
| Solver summary / device controller state | 4 / 4 | Exposed current constraint rows/groups/iterations/error and current insertion/rotation/first-node index |

These are all read at decision time *before* the candidate action. No future force, future contact, or post-candidate solver state enters the predictor. Exposed solver-internal values include current constraint multipliers/Jacobian, current iteration/error summaries, and free-state buffers. Warm-start history, contact-manifold/cache and collision-pipeline temporal state, opaque constraint-resolution/order details, integration scratch buffers, and possible unexposed material history are not known to be captured. Accordingly `X_PRIV` is a **privileged observable simulator state**, not demonstrated to be the full Markov state. The PCA explains 99.5220% of raw train-state variance, but low-variance predictive information could still be lost.

## Matched models, fitting, calibration, and hashes

Both families use the same 592D `Linear(592,64)-ReLU-Linear(64,64)-ReLU-Linear(64,4)` cumulative-softplus noncrossing quantile network (42,372 parameters), Q50/Q90/Q95/Q99 targets, equal-weight pinball loss, Adam `lr=0.001`, batch 512, maximum 60 epochs, validation-pinball checkpoint selection, patience 8, and the same three initialization/shuffle seeds. The sole experimental variable is additional pre-action state information. For each replica the nonnegative Q95 residual correction is the validation-only higher-order 95th percentile of `R2_true - raw_Q95`; the decision remains `corrected_Q95 <= F_lim`. No development-test or confirmation labels fit, selected, or recalibrated models.

| Family | Seed | Checkpoint SHA256 | Validation-only Q95 correction, N |
|---|---:|---|---:|
| M_DEPLOY | 260322 | `9f8b557643827c0bd4341e040444102c8cfcd5f925178057f6c0c366dfd04b59` | 0.692731 |
| M_DEPLOY | 260323 | `47618071683e1905e0bb969f75d76587f2ec6cc888a2714bd89bd8b8b4532b63` | 0.631872 |
| M_DEPLOY | 260324 | `8d71dfee7fd1d916e7ea5df0241ef9096110c4b81e8f40eafd83286f6ceeba88` | 0.583511 |
| M_PRIV | 260322 | `2ccccc546b9596b26bdeda6f2fdf9229780a14949defc4a2a9b75c787df94321` | 0.289888 |
| M_PRIV | 260323 | `2549cdfe12e5b9c91d065fcef7a8e3fc54195a74aff4e84f0fbd54804f1f655f` | 0.335331 |
| M_PRIV | 260324 | `af8d99aa3d1bd63ba9d4b1678b4944c58faa04bedc6cf26e3566c59249af2756` | 0.213685 |

## Development-test results (1,075 identical complete plans, 120 states)

Rates are fractions, not percentages. `False-safe` and `unsafe accept` are numerically the same event under this Q95 gate; both names are retained to match the requested audit. `Infeasible` is false predicted infeasibility among states containing a true-safe plan. `Usable` is the complementary fraction of those feasible states retaining at least one accepted true-safe plan. SAFE-HIGH is the count of truly unsafe plans whose first interval is safe and second is high but whose corrected Q95 is accepted.

| Family/seed | False-safe / unsafe accept | Safe reject | Raw / corrected Q95 coverage | Pinball N | Spearman | Pairwise rank | Infeasible | Usable | SAFE-HIGH misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D/260322 | .1258 | .1430 | .9014 / .9619 | .1671 | .1346 | .5642 | .1682 | .8318 | 11/65 |
| D/260323 | .1635 | .1648 | .9098 / .9563 | .1763 | .2156 | .5988 | .1589 | .8411 | 17/65 |
| D/260324 | .1635 | .1561 | .9265 / .9544 | .1835 | .0788 | .5397 | .1495 | .8505 | 17/65 |
| P/260322 | .2201 | .1692 | .9433 / .9563 | .1768 | .0269 | .5166 | .1869 | .8131 | 21/65 |
| P/260323 | .1635 | .1987 | .9237 / .9563 | .1788 | .0316 | .5194 | .2150 | .7850 | 17/65 |
| P/260324 | .2201 | .1496 | .9191 / .9405 | .1823 | .0115 | .5031 | .1589 | .8411 | 26/65 |

The privileged family did not achieve a clear development improvement. Its mean false-safe rate was .2013 versus .1509; mean safe rejection .1725 versus .1547; SAFE-HIGH misses 21.3 versus 15.0 of 65; mean Spearman .0234 versus .1430; and mean false infeasibility .1869 versus .1589. Per-replica data, population standard deviations and worst replicas are retained in `training_seed_robustness.json` and `development_deploy_vs_privileged_comparison.json`; no best seed was selected.

## Transition repeatability and identifiability diagnostics

Three predeclared development parent-state/action cases—low force, near `F_lim`, and high contact—were copied from the *same live pre-action process/scene snapshot* ten times each. All 30 two-action forks reproduced their archived G9.5 F1/F2/R2 values. Within each case, maximum absolute difference was **0** for F1, F2, R2, and every coordinate of both 3582D observable post-action state vectors. Mean absolute differences were also zero; reported force population standard deviations of order `1e-16` are only floating-point reduction artifacts on identical values. Repeated forks did not enter training. This supports exact repeatability for these three tested state/action pairs, **not** global determinism or complete Markov-state coverage; it gives no evidence for transition non-repeatability in the tested cases.

For 1,053 of 1,075 held-out plans, five other-state plans within 0.75 Euclidean action-pair distance were available. The diagnostic mean local true-R2 variance was **0.7497 N²** under deployable-state nearest neighbors and **0.7758 N²** under privileged-state neighbors; mean absolute query-to-neighbor R2 difference was **0.9750 N** versus **1.0618 N**. Thus these measured privileged neighborhoods did not reduce local conditional ambiguity. This is descriptive: neighborhood geometry, scaling, PCA compression, and finite samples prevent a causal conclusion about aliasing.

## Untouched confirmation (106000–106029)

The cohort was unopened until the freeze. One reused Generic scene collected behavior for 30 ordered seeds; a second reused scene replayed all **1,801** behavior parent steps exactly and ran the same nine frozen candidate plans at four outcome-blind selected states per episode. All **1,080** confirmation H=2 labels were complete; none censored. Maximum parent differences in observation, action, applied action, reward, tip position, insertion length, and next-interval force were zero. The same 1,080 candidate plans and 120 parent states were scored by all six frozen models, without retraining/recalibration. Temporal counts were SAFE-SAFE 881, SAFE-HIGH 53, HIGH-SAFE 63, HIGH-HIGH 83. Of the 120 states, 102 had at least one true-safe plan.

| Family/seed | False-safe / unsafe accept | Safe reject | Raw / corrected Q95 coverage | Pinball N | Spearman | Pairwise rank | Infeasible | Usable | SAFE-HIGH misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D/260322 | .0101 | .2350 | .9000 / .9565 | .2008 | .1155 | .5545 | .2451 | .7549 | 2/53 |
| D/260323 | .0101 | .2599 | .9130 / .9454 | .2064 | .2099 | .5887 | .2647 | .7353 | 2/53 |
| D/260324 | .0101 | .2611 | .9213 / .9537 | .2000 | .0958 | .5423 | .2647 | .7353 | 2/53 |
| P/260322 | .0101 | .2247 | .9491 / .9602 | .2577 | .0326 | .5229 | .2451 | .7549 | 2/53 |
| P/260323 | .0302 | .2509 | .9194 / .9426 | .2608 | -.0066 | .5082 | .2745 | .7255 | 6/53 |
| P/260324 | .0101 | .1975 | .9407 / .9602 | .2524 | .0443 | .5274 | .2157 | .7843 | 2/53 |

Paired privileged-minus-deployable safe-rejection changes were **−.0102, −.0091, −.0636**, but false-safe changes were **0, +.0201, 0**; SAFE-HIGH miss changes **0, +4, 0**; Spearman changes **−.0829, −.2164, −.0516**; pairwise-ranking changes **−.0317, −.0805, −.0148**; and false-infeasibility changes **0, +.0098, −.0490**. The confirmation means were safe rejection .2244 versus .2520, false-safe .0168 versus .0101, SAFE-HIGH misses 3.3 versus 2.0, Spearman .0234 versus .1404, pairwise ranking .5195 versus .5618, false infeasibility .2451 versus .2582, and pinball .2570 versus .2024 (privileged versus deployable). Detailed per-plan and per-state outcomes are in `confirmation_plan_results.csv`, `confirmation_state_ranking_metrics.csv`, `confirmation_temporal_category_metrics.csv`, `confirmation_metrics.csv`, and `deploy_vs_privileged_comparison.json`.

## Decision and next direction

**C.** The privileged family does not show the requested clear, consistent ceiling improvement across seeds and cohorts, especially for SAFE-HIGH detection, unsafe-plan acceptance, or within-state ranking. A small safe-rejection/usable-plan gain on confirmation trades against worse ranking, pinball loss, and one replica's additional unsafe acceptance. The tested richer observable state therefore does not justify attributing the main bottleneck simply to missing deployable features. The repeatability cases separate this conclusion from a claim of observed simulator stochasticity: those exact restored transitions were reproducible. Remaining possibilities include inaccessible simulator/contact history, sharp/non-smooth force boundaries, finite-data/model-adapter limitations, and the suitability of learned prospective-risk surrogates themselves.

**Stop after G9.7.** Reevaluate whether a learned H=2 prospective-risk surrogate is an appropriate safety interface before any G10 planner integration. No G10, online filter, controller change, or additional predictor ablation was launched.
