# Phase G7 — conditional next-force quantile prediction

**Classification: B — quantile modeling improves the tradeoff but remains insufficient.** All three G7 replicas passed the empirical ≤5% Q95 false-safe gate on untouched confirmation **without an additive residual** (`delta_cal = 0`), and accepted more actions than the three frozen G6B point predictors on the *same* 2,520 candidate rows. But G7 increased false-safe actions for every paired training seed, retained 12–14/120 zero-action states, and its empirical Q95/Q99 quantile coverage was below nominal on confirmation. It is not yet a calibrated safety boundary or a basis for an automatically launched shield.

## Exact frozen input, target, and training recovery

- Branch/worktree: `feature/generic-contact-force-safety`. The GenericConstraintSolver, projected Gauss–Seidel tolerance `1e-4`/2,000 iterations/constraint forces enabled, friction `0.001`, SOFA `dt=0.006 s`, frozen step-280000 TD-MPC2 navigation policy, task/reward/action/observation, and G3/G5 21-action counterfactual protocol were unchanged. The target remains maximum ~3-mm-distal guidewire–vessel contact force over all **22 SOFA substeps of the immediately following RL interval**, reconstructed as summed `J_contact^T lambda / 0.006 s` vector contributions before norm. `F_lim` is exact G2 Q95 `3.685331040756903` **nominal simulator N**; G2 Q99 is `7.466568793559142` nominal N. Neither is a clinical threshold.
- G7 calls the frozen G6B feature builder and reproduces its exact **308D** input: TD-MPC2 latent `Z[256]`, sorted G5 `K+D` physical indices `0–29,35–37` `[33]`, 17 current pre-action contact/wall quantities, then the proposed two-component action. The contact quantities include current reconstructed distal force vector/magnitude and contact indicator/count; geometry includes tip/five-node wall distance, outward normal, relative velocity/orientation, and force along the wall normal. They are all from the current decision state, after the preceding interval and before the candidate action; there is no future/post-action input. See `g6b_input_audit.json` and the unchanged G6B `full_contact_feature_definition.json` for exact ordering, frames, units, and timing. Exact solver-derived current contact force/count are **simulation-only**, not validated hardware measurements.
- The G6B 157-hash freeze passed before G7 work. The G6B matrix-building code and mesh/extractor hashes were checked; recomputed TRAIN-only mean/std arrays matched the G6B normalization **element-for-element**, SHA256 `f3600a518002269df1756d45f4ee908d56ece9a8227dd37a3a035adcc9756ce3`. Training used the same 10,903 behavior rows (`d>=1`, 140 episodes) and validation used the same 2,286 behavior rows (30 episodes), with zero counterfactual training rows. The target indexing, candidate-action coordinates, unnormalized target, Adam `lr=.001`, zero weight decay, batch 512, 10,903 samples/validation round, at most 60 rounds, minimum objective improvement `.0001`, and patience 8 match G6B. G7 changed only the scalar head/loss and the validation objective to mean four-quantile pinball. See `training_data_audit.json` for exact array/key hashes and `training_seed_manifest.json` for the protocol.
- CUDA runtime: `steve_tdmpc`, torch `2.7.1+cu128`, CUDA `12.8`, `CUDA_VISIBLE_DEVICES=0`, visible `cuda:0` NVIDIA GeForce RTX 5090; no CPU fallback.

## One non-crossing quantile model, three independent replicas

The unchanged hidden trunk is `Linear(308,64)-ReLU-Linear(64,64)-ReLU`. A four-output linear head emits raw values; `Q50=softplus(r0)`, `Q90=Q50+softplus(r1)`, `Q95=Q90+softplus(r2)`, and `Q99=Q95+softplus(r3)`. Thus Q50 is nonnegative and crossings are impossible by construction. The loss is the **equal-weight mean standard pinball loss** at `tau=[.50,.90,.95,.99]`, in nominal N. There is no classifier, ranking term, extra asymmetry, post-hoc sorting, or ensemble. Validation mean pinball alone selected one checkpoint per seed, at epochs `36/30/52` for seeds `260206/260207/260208`.

| Seed | Frozen G7 checkpoint SHA256 |
|---|---|
| 260206 | `f6200a931716fdf6b586510fb8bd7f0ffea2bdd2083c631697d118c23c89b3c3` |
| 260207 | `ce94ede9d2ef3a2075bb4516f691039d05b892b59d575ab8afe2a0f5ae2d846c` |
| 260208 | `fb0f44d4b2fb5c0a1b8d178005d8d09cef28e3d6ef2cc3763139a548ae60f5e0` |

The full architecture/loss definition is `quantile_model_definition.json`; hashes and validation selection are in `model_manifest.json` and each model directory. These are independent predictions, not a combined decision.

## Frozen development-test candidates

All replicas were evaluated on the same existing 120 states × 21 actions = 2,520 labels. Empirical coverage is `fraction(F_true <= Q_tau)`; the Q95 decision is predicted safe iff `Q95 <= F_lim`.

| Seed | Q50 coverage | Q90 coverage | Q95 coverage (error vs .95) | Q99 coverage | Q95 false-safe / 198 high-force | Acceptance / zero-action states | No-contact→high-force misses / 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 260206 | 65.52% | 87.50% | 90.79% (−4.21 pp) | 95.32% | 4/198 (2.02%) | 84.68% / 15 | 2/4 |
| 260207 | 70.00% | 91.27% | 93.33% (−1.67 pp) | 97.22% | 2/198 (1.01%) | 84.84% / 14 | 0/4 |
| 260208 | 71.11% | 89.84% | 92.82% (−2.18 pp) | 96.47% | 6/198 (3.03%) | 88.25% / 11 | 4/4 |

Q95 unsafe recall was `97.98%/98.99%/96.97%`; Q95 AUROC was `.990/.992/.995` and AUPRC `.928/.940/.965`. Development maximum missed excess above `F_lim` was `.577/.180/1.593` nominal N. The three 23-action Q99-extreme sets were all identified at the G2 Q99 reference, but 23 clustered positives are too few to establish tail reliability. Q50 overcoverage is influenced by the force distribution's zero-force mass; Q95/Q99 undercoverage is the more important upper-tail concern. Per-quantile mean pinball and mean predicted force, full confusion/availability/severity metrics, and crossing counts (all zero) are in `development_quantile_metrics.csv` and `development_boundary_metrics.csv`.

## New calibration and frozen residual

Seeds `99000–99029` yielded 30 ordered behavior episodes in one reused Generic scene: **29 successes, 1,799 parent steps**. A fresh reused scene replayed them in order and forked the exact four selected states/episode × 21 actions/state once: 120 states/2,520 labels. Observation, applied action, reward, tip, insertion, next-force, and pre-action previous-final-force replay/fork differences were **zero** (`calibration_replay_audit.json`). The three quantile replicas saw identical labels; 222 actions exceeded Q95 and 49 exceeded Q99.

Raw Q95 (`delta_cal=0`) passed the predeclared worst-replica empirical 5% gate: false-safe `3/222 = 1.35%`, `3/222 = 1.35%`, `8/222 = 3.60%`; candidate acceptance `84.80%/81.35%/83.65%`; zero-action states `14/15/13`. No-current-contact→Q95-high-force misses were `0/53`, `0/53`, `1/53`. The largest false-safe excess above Q95 was `1.173` nominal N for each replica. Since all three passed raw, the predeclared rule selected **one shared `delta_cal = 0` nominal N**—no needless residual margin. This is materially smaller than G6B's frozen **`1.579396897690497`-N** global point-prediction margin. The rule and raw/selected per-seed results are in `selected_residual_calibration.json` and `quantile_calibration_summary.json`.

Calibration Q95 empirical coverage was `95.16%/96.55%/93.37%`; Q99 coverage was `97.54%/97.82%/96.11%`. Q99 identified only **28/49** extreme-force actions at the G2 Q99 reference for each replica. Thus raw Q95 false-safe performance did not imply well-calibrated Q99 tails.

## Freeze before untouched confirmation

`pre_confirmation_freeze.json` passed with **166 SHA256 identities before any seed 100000–100029 was opened**. It binds the exact G6B input and normalization, all three G7 checkpoints/architecture/quantile levels, `delta_cal=0`, `F_lim`, calibration labels, state/action protocols, evaluation definitions, and the frozen G6B checkpoints/cutoff. No model, feature, or correction changed afterward.

The new 30 confirmation behavior episodes were evaluated in one ordered reused scene: **26 successes, 2,255 parent steps**. Four selected states/episode and 21 forks/state produced another 120 states/2,520 fresh labels, with 259 Q95-high-force and 70 Q99-extreme actions. Independent replay/fork validation again found **zero** differences in every audited field over all 2,255 parent steps and 2,520 forks. The same input rows and force labels were scored by each frozen G7 quantile replica and its same-seed frozen G6B point predictor.

## Paired untouched G7 versus G6B confirmation

G7 uses `Q95+0 <= F_lim`. G6B uses the already-frozen point rule `F_hat <= 2.105934143066406` nominal N. No training-seed selection, ensemble, retraining or recalibration occurred.

| Seed | G7 Q95 false-safe | G6B Q95 false-safe | G7 acceptance | G6B acceptance | G7 zero-action states | G6B zero-action states | No-contact→high-force misses (both) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 260206 | 10/259 (3.86%) | 5/259 (1.93%) | 88.33% | 84.88% | 13/120 | 16/120 | 0/42 |
| 260207 | 8/259 (3.09%) | 4/259 (1.54%) | 87.98% | 83.25% | 12/120 | 17/120 | 0/42 |
| 260208 | 7/259 (2.70%) | 0/259 (0%) | 88.17% | 80.32% | 14/120 | 18/120 | 0/42 |

G7 increased candidate acceptance by **3.45/4.72/7.86 percentage points** and reduced zero-action states by **3/5/4**. All three remained below the 5% empirical Q95 false-safe target and preserved zero observed no-contact→high-force misses. But G7 also added **5/4/7** Q95 false-safe candidates versus G6B. Therefore it did **not** increase acceptance without increasing false-safe error. G7's maximum false-safe excess was `.283/.283/.098` nominal N versus G6B's `.076/.038/none`; the largest G7 miss was not an extreme Q99 event. G7's remaining zero-action fraction was `10.0%–11.7%`, despite a median 21 accepted actions per state; that median masks the all-or-nothing state-level rejection pattern. Full severity, confusion, action-availability and no-contact counts are in `g7_confirmation_metrics.csv` and `g6b_same_data_baseline_metrics.csv`.

G7 had slightly less between-seed variation in Q95 false-safe *rates* on this set (`2.70%–3.86%`, sample SD about `.59` percentage points) than G6B (`0%–1.93%`, sample SD about `1.02` points), but at a higher mean miss rate. G7 acceptance was also steadier (`87.98%–88.33%`) than G6B (`80.32%–84.88%`). This does not erase the paired increase in misses.

| Seed | Untouched Q50 coverage | Q90 coverage | Q95 coverage (error vs .95) | Q99 coverage | Q99-extreme actions identified / 70 | Q95 AUROC / AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| 260206 | 63.29% | 88.97% | 92.54% (−2.46 pp) | 97.42% | 70/70 | .996 / .984 |
| 260207 | 67.18% | 91.31% | 94.09% (−.91 pp) | 96.87% | 70/70 | .997 / .987 |
| 260208 | 67.54% | 89.80% | 92.22% (−2.78 pp) | 95.40% | 67/70 | .996 / .983 |

Q99 extreme-action recall changed from **28/49 on calibration** to 70/70, 70/70, 67/70 on confirmation; nominal Q99 coverage was still missed by 1.6–3.6 percentage points. This variation and state clustering argue against interpreting the small extreme-action samples as a calibrated upper tail. Mean pinball by quantile and predicted quantile means are in `confirmation_quantile_metrics.csv`; crossing count was **zero** for all replicas and all rows.

## Decision and next methodological direction

1. Conditional quantiles replaced the `1.579`-N global margin with **zero** additive residual and improved paired action availability while keeping empirical Q95 false-safe rates below 5% for all replicas. They did **not** strictly dominate G6B: every G7 replica admitted more false-safe actions, and 12–14/120 states still had no accepted candidate.
2. The predicted conditional quantiles are not fully empirically calibrated in the tested distribution: Q95 and especially Q99 coverage are below nominal on untouched confirmation, and the calibration Q99 extreme recall was weak. An empirical candidate-level false-safe rate is not a formal safety guarantee; candidate outcomes share states/episodes.
3. **Classification B — quantile model improves the tradeoff but remains insufficient.** Do not start an offline or online shield, Safety Head, or CBF from this result. Do not start another feature ablation inside G7. A separately scoped next methodological direction should prioritize **temporal/state estimation of unobserved elastic/contact history**, followed by distributional calibration on independent data; the persistent tail miscoverage and zero-action states do not justify treating the present scalar next-force rule as ready for deployment.
4. The simulator was not assumed stochastic. The four quantiles represent uncertainty **conditional on the predictor's incomplete pre-action state**, not intrinsic randomness in deterministic SOFA physics. All force values are nominal simulator N, and current exact contact measurements remain simulator-derived.

**STOP after G7.** No TD-MPC2, Generic physics, force target, feature set, G6B checkpoint/cutoff, task, observation, reward, shield or CBF was changed.
