# Phase G6A — three-replica conservative force-boundary calibration

**Classification: B — CALIBRATION HELPS BUT TRAINING-SEED SENSITIVITY REMAINS.** One calibration-only shared cutoff lowered Q95 false-safe error for all three unchanged Z+K+D predictors on untouched confirmation. It did **not** preserve the predeclared `worst replica <=5%` criterion: the worst replica missed `18/307 = 5.86%` of Q95-positive actions, and missed `14/58` prospective no-contact→high-force actions. The cutoff also left 15–16 of 120 confirmation states with zero accepted candidate actions, depending on replica. This is a research result, not a validated safety boundary or a shield.

## Frozen system and predictor audit

- Branch: `feature/generic-contact-force-safety`. GenericConstraintSolver, projected Gauss–Seidel, tolerance `1e-4`, 2000 iterations, constraint forces enabled, friction `0.001`, SOFA `dt=0.006 s`. Frozen navigation checkpoint step 280000 SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`.
- Sole target: maximum distal (~3 mm) guidewire–vessel contact-force magnitude over all 22 SOFA substeps in the *immediately following* RL interval. Reconstructed with summed `J_contact^T lambda / 0.006 s` vector contributions before norm. All forces here are **nominal simulator N**.
- Exact G2 references: Q90 `2.1265156799878246`, Q95 `3.685331040756903`, Q97.5 `5.926250464744108`, Q99 `7.466568793559142` nominal N. Primary `F_lim` is the **exact** Q95 value, rather than its rounded 3.685 shorthand. These are statistical references, not clinical thresholds.
- Frozen G5 formulation is `Z+P_min = Z+K+D`: 256D TD-MPC2 latent, then P indices `0–29,35–37` in sorted G5 order (33D; tip xyz, four distal-node offsets, five distal-node velocities, insertion length, rotation sin/cos), then proposed 2D action. Input dimension `291`; train-behavior normalization SHA256 `1cd651d19d6d93c48a638c2824586a7a0313872b4b0edb5d931f1c16fb09f399`. Architecture `291-64-ReLU-64-ReLU-1`, G5 behavior-only training, no retraining in G6A. Most K+D 3D physical variables are only estimable in principle, not measured by the present controller.
- The three **independent**, non-ensembled checkpoints are training seed 260206 SHA256 `05efea349196c83e088fc0a8337539977c883ad89ef4899e07eb72cafac596a5`; 260207 `40f54351b98c19c2ee7a4662eb61156c0b8a2bb6c7b3b17967c3a41444ecb148`; 260208 `77f75420b6b31339fdb462b7d14fe5e8d306a567e5f4db6d657510ea0ee9f4aa`. No seed was selected using already-observed G5 confirmation results.
- The G5 187-hash freeze verified. A read-only audit reconstructed 63 existing G5 candidate inputs per replica and matched stored predictions and true labels **exactly** (maximum difference 0, Q95 truth mismatches 0). G5 confirmation was used only for this semantic audit, never for cutoff selection. See `g5_model_audit.json`.
- CUDA inference: `steve_tdmpc`, torch `2.7.1+cu128`, CUDA `12.8`, `CUDA_VISIBLE_DEVICES=0`, visible NVIDIA GeForce RTX 5090. No CPU predictor fallback.

## Data integrity and predeclared cutoff rule

The protocol was written before opening calibration seeds. Seeds 95000–95029 yielded 30 ordered behavior episodes in **one reused scene**, 26 successes, 2,240 parent steps, and exactly 4 selected states/episode × 21 G5 candidate actions = **120 states/2,520 one-time SOFA labels**. Seeds 96000–96029 were unopened until the `pre_confirmation_freeze.json` passed with **154 hashes**. They then yielded 30 ordered behavior episodes, 27 successes, 1,976 parent steps, and another 120 states/2,520 one-time labels. Both datasets had zero recorded replay/behavior-fork differences for observation, applied action, reward, tip, insertion, peak force, and pre-action previous-final force; see `calibration_replay_audit.json` and `confirmation_replay_audit.json`. The model did not control behavior.

True high force is `F_true > F_lim`; predicted safe is `F_hat <= c_pred`. The raw decision uses `c_pred=F_lim`. For each exact pooled three-replica prediction breakpoint at/below `F_lim` (6,796 sweep cutoffs including endpoints), the false-safe rate was computed **separately** for each replica. The predeclared rule selected the **largest shared cutoff** for which the *maximum* of three empirical Q95 false-safe rates was at most 5%; it did not use a mean rate, pooled predictions, RMSE, a favorable checkpoint, or confirmation data. False-safe 95% intervals below are two-sided exact Clopper–Pearson intervals; candidate outcomes are clustered within state, so those binomial intervals do not capture all dependence.

Selected `c_star = 2.976577758789062` nominal N, `Delta = F_lim-c_star = 0.7087532819678408` nominal N. The next prediction breakpoint, `2.9765777587890625`, would yield **13/251 = 5.18%** false-safe for seed 260206 and fail the worst-replica rule. The precise stored cutoff is the largest float64 predecessor of that breakpoint. **Do not round/cast it silently to float32:** float32 rounding reaches the failing prediction value and may change the decision. A later implementation would need a separately validated precision-safe representation; none is implemented here.

## Calibration: 251 Q95-positive candidates

| Replica | Raw FN/rate | Calibrated FN/rate [exact 95% CI] | False-unsafe raw→calibrated | Candidate acceptance raw→calibrated | States with ≥1 accepted / median accepted actions |
|---|---:|---:|---:|---:|---:|
| 260206 | 19 / 7.57% | 12 / 4.78% [2.49%, 8.20%] | 41→59 (1.81%→2.60% of true-safe) | 89.17%→88.17% | 107/120 / 21 |
| 260207 | 20 / 7.97% | 7 / 2.79% [1.13%, 5.66%] | 0→48 (0%→2.12%) | 90.83%→88.41% | 109/120 / 21 |
| 260208 | 20 / 7.97% | 7 / 2.79% [1.13%, 5.66%] | 30→52 (1.32%→2.29%) | 89.64%→88.25% | 108/120 / 21 |

The worst calibration false-safe rate was **4.78%**, worst acceptance **88.17%**, and worst state-level availability **107/120 = 89.17%**. Selected-cutoff rejection rates were **11.83%, 11.59%, 11.75%** by seed. Conversely, 11–13 calibration states had **zero** accepted actions. The mean ± sample SD across replicas at the selected cutoff is recorded for every decision metric in `calibration_tradeoff.json`; the full per-breakpoint/per-replica counts, confidence intervals, acceptance/rejection, precision, specificity, NPV, and severity appear in `cutoff_sweep.csv`. The tradeoff curve is `plots/calibration_acceptance_vs_false_safe.png`.

At the selected cutoff, false-safe severity (`F_true-F_lim`) was nonzero: seed 260206 count 12, mean `.251`, median `.146`, Q90 `.711`, max `.754` nominal N; seeds 260207/260208 each count 7, mean `.348`, median `.191`, Q90 `.750`, max `.754`. The largest missed true force was `4.439` nominal N. The no-current-contact→Q95-high-force stratum had **22 positive candidates** from 57 no-contact states; the selected cutoff missed **0/22 for each replica** (raw missed 0, 1, 0). These are candidate counts, not independent patient or vessel outcomes. Calibration Q99 descriptively had 67 positive candidates and 0 false-safe for each replica at the Q95-selected cutoff; Q99 was not separately tuned.

## Untouched confirmation: 307 Q95-positive candidates

The shared cutoff, all three checkpoints, G5 representation, metrics code, seed/state/action protocols, and calibration data were frozen **before** seed 96000 was opened. The comparison below uses that same cutoff without modification. Rejection rate is one minus acceptance; all precision, recall, specificity, NPV, true/false-positive counts and exact per-seed intervals are in `confirmation_metrics.csv` and `confirmation_summary.json`.

| Replica | Raw FN/rate [95% CI] | Calibrated FN/rate [95% CI] | False-unsafe raw→calibrated | Candidate acceptance raw→calibrated | States with ≥1 accepted / zero accepted / median accepted actions |
|---|---:|---:|---:|---:|---:|
| 260206 | 21 / 6.84% [4.28%, 10.27%] | 4 / 1.30% [0.36%, 3.30%] | 29→68 (1.31%→3.07% of true-safe) | 87.50%→85.28% | 104/120 / 16/120 / 21 |
| 260207 | 30 / 9.77% [6.69%, 13.66%] | 15 / 4.89% [2.76%, 7.93%] | 34→56 (1.54%→2.53%) | 87.66%→86.19% | 104/120 / 16/120 / 21 |
| 260208 | 46 / 14.98% [11.18%, 19.48%] | 18 / 5.86% [3.51%, 9.11%] | 18→57 (0.81%→2.58%) | 88.93%→86.27% | 105/120 / 15/120 / 21 |

Calibrated three-seed **mean ± sample SD** false-safe rate was `4.02% ± 2.40 percentage points`; worst replica `5.86%`. Mean acceptance was `85.91% ± 0.55 percentage points`; worst replica `85.28%`. Calibrated rejection rates were **14.72%, 13.81%, 13.73%** by seed. Calibration→confirmation calibrated false-safe rates were `4.78%→1.30%`, `2.79%→4.89%`, and `2.79%→5.86%`; acceptance was `88.17%→85.28%`, `88.41%→86.19%`, and `88.25%→86.27%`. Thus the shared cutoff improved every replica relative to raw, but **the empirical all-three <=5% criterion did not generalize**. Action availability also worsened: worst state-level fraction retaining any action fell from `89.17%` in calibration to `86.67%` in confirmation. Candidate acceptance masks a bimodal state-level outcome: median 21/21 actions accepted, yet 12.5–13.3% of states had none.

Confirmation calibrated false-safe severity was not confined to tiny threshold misses:

| Replica | Remaining FN | Mean / median / Q90 severity N | Maximum severity N | Maximum missed true force N |
|---|---:|---:|---:|---:|
| 260206 | 4 | .266 / .267 / .500 | .526 | 4.212 |
| 260207 | 15 | .521 / .274 / 1.542 | 2.317 | 6.002 |
| 260208 | 18 | .624 / .274 / 1.951 | 2.317 | 6.002 |

Raw maximum severity was `1.079`, `4.240`, `4.240` nominal N by seed. Full raw and calibrated count/mean/median/Q90/max/max-true-force records are in `calibration_false_safe_severity.json` and `confirmation_false_safe_severity.json`.

## Prospective contact emergence and upper tail

Pre-action current contact is defined by the frozen G5 P[34] count > 0; it is **not** a new predictor feature. Confirmation contained 62 no-current-contact states and **58** candidate actions that entered Q95-high force in the next interval. The per-replica misses were:

| Replica | Raw no-contact→high-force false-safe | Frozen-cutoff false-safe | Selected-cutoff false-safe rate |
|---|---:|---:|---:|
| 260206 | 0/58 | 0/58 | 0% |
| 260207 | 25/58 | 11/58 | 18.97% |
| 260208 | 25/58 | 14/58 | 24.14% |

This is the primary reason not to treat the low average Q95 rate as sufficient: two replicas still predict safety for newly emerging high-force actions. See `confirmation_contact_emergence.json` for accepted/rejected counts in the entire no-contact stratum. At the frozen Q95 cutoff, confirmation Q99 had **67** positive actions and **0** false-safe for each replica; the exact candidate-level 95% binomial upper bound for 0/67 is about **5.36%**, and clustered states add uncertainty. No Q99-specific cutoff was tuned. Q90 and Q97.5 descriptive results are in `calibration_reference_diagnostics.json` and `confirmation_reference_diagnostics.json`.

## Decision and limits

1. A single margin **substantially reduced** empirical Q95 false-safe rates on new data, at a modest aggregate acceptance cost of about 1.5–2.7 percentage points per replica.
2. It did **not** robustly meet the calibration target on untouched data (`5.86%` worst replica), and two replicas retained no-contact emergence misses and some high-severity misses. The exact binomial intervals are wider than the point estimates.
3. State-level action availability is a separate concern: 15–16/120 confirmation states had no predicted-safe candidate. No action repair or shield was tested.
4. Classification **B — calibration helps but training-seed sensitivity remains**, not A. It is not C because all replicas improved materially and retained >85% aggregate candidate acceptance; nevertheless, the stronger worst-replica criterion failed on confirmation.
5. The next separately predeclared research direction should test contact-aware state augmentation, particularly **tip-to-wall distance** and **current distal force/contact information**, and evaluate newly emerging contact and action-set coverage before any shield study. These features were **not** added in G6A. No checkpoint was selected or predictions ensembled.

**STOP after G6A.** No TD-MPC2, SOFA solver, task, reward, force target, predictor weights, input state, safety head, CBF or online shield was modified. The frozen cutoff was not retuned after confirmation.
