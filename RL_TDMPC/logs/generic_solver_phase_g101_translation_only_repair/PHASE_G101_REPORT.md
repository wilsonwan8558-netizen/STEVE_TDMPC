# Phase G10.1 — translation-only force-constrained repair

**Decision: B — translation-only physics is promising, but the current frozen predictor is not usable for repair. No online test is authorized.**

This is an offline diagnostic at the exact 40 G10 parent states: 18 held-out unsafe-nominal states and 11 G8.5 difficult events at both t*−2 and t*−1. The statistical reference is `F_lim = 3.685331040756903` nominal N, **not** a clinical injury threshold. “No safe translation” below means none on the finite tested grid, not mathematical or physical impossibility.

## Frozen protocol and audit

- Branch: `feature/generic-contact-force-safety`. Runtime: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, PyTorch `2.7.1+cu128`, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0`, logical device `cuda:0` (NVIDIA GeForce RTX 5090). No CPU fallback.
- Controller: `RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt`, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. No TD-MPC2 update.
- True physics: frozen GenericConstraintSolver / projected Gauss–Seidel, tolerance `1e-4`, 2000 iterations, constraint forces enabled, friction `mu=0.001`, SOFA `dt=0.006 s`; distal approximately 3 mm force from `J_contact^T lambda / 0.006 s`, RL-step maximum over SOFA substeps. True `R2=max(F1,F2)` requires two complete intervals.
- Predictor: exact G10 frozen G9.5 M1 planner-conditioned three-replica H=2 noncrossing quantile network, 298D input (256D TD-MPC2 latent, 38D E1 physical estimate, current candidate 2D action, fixed second 2D action). Replicas 260312/260313/260314 have checkpoint SHA256 values `7f2ce7d27a01a0772bfbcb103042d419c8718b73ce3b63354e7b8e1d9b6b9b66`, `decee08e203d53e828a95ccdf71f528492f11c07079a23779dc108bbc8ebaa20`, and `6932c520d8421cda7f9454248d4b79d86919e8bed67c47c5c5a98d67eb3330f0`. Frozen normalization SHA256: `697408d9f9acdbb63c4adfbaf0e3b39f4c3966d1212b86dcc566708969f94976`. Validation-only Q95 corrections and all-three-replica conservative rule are unchanged. See `frozen_predictor_definition.json` for paths and corrections.
- Repair formulation: choose tested `v∈[-1,1]` minimizing `|v-v_nom|²` subject to all three corrected Q95(R2) values `<=F_lim`. Current `w=w_nom` is the exact G9.6-v2 optimized-mean TD-MPC2 rotation. The **entire second action is the exact frozen nominal optimized-mean second action**, including its translation and rotation. Neither is replanned. Tie-break: lower `v`. No candidate is applied to the parent rollout.
- Predicted grid: 201 uniform float32 values, spacing `0.01`, plus exact nominal and clipped nominal ±0.05/0.10/0.20. True SOFA: 101 uniform float32 values, spacing `0.02`, the same exact probes, and the predicted closest-safe point if additional. Deduplication is by exact float32 value. Complete JSON curves include the extra points; NPZ arrays hold the uniform grids.
- Exact reused-scene parent replay passed for all 90000–90199 (15,500 parent steps) and 101000–101029 (1,848 parent steps). Parent observation, selected/applied action, reward, tip, insertion, and force differences were all **zero** against the frozen G10/G2 source. All 4,241 true-SOFA two-interval forks completed; none was censored. The 107000–107199 online cohort was not opened.
- A pre-result CUDA numerical amendment is documented in `numerical_protocol_amendment.json`: single-action nominal inference matches G10; batch-size-dependent float32 rounding in the sweep differed by at most `5.2453e-6 N`. The exact G10 single-action value replaces only the nominal curve point. No predicted point lay within `1e-4 N` of `F_lim`; model, calibration, threshold, and force labels did not change. The initial stopped attempt saved no state result.

## Recoverability and repair validation

| Distinct cohort | States | True-safe translation found | None found on finite grid | True-safe at `v=0` | True-safe at `v=-1` | Predictor-safe candidate | Candidate truly safe | False-safe | False-infeasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-out unsafe nominal | 18 | 11 | 7 | 8 | 11 | 3 | 1 | 2 | 8 |
| Difficult t*−2 | 11 | 9 | 2 | 7 | 9 | 1 | 1 | 0 | 8 |
| Difficult t*−1 | 11 | 7 | 4 | 1 | 6 | 1 | 0 | 1 | 6 |
| Descriptive total; cohorts not exchangeable | 40 | 27 | 13 | 16 | 26 | 5 | 2 | 3 | 22 |

At `v=-0.5`, 22/40 states were true-safe; some negative translation was true-safe in all 27 translation-recoverable states. Among the 34 states whose nominal translation was positive, stopping at `v=0` was safe in 16. Stronger withdrawal (`-1` versus `-0.5`) reduced true R2 in 34/40 states. This is a useful control tendency, not a guarantee: 13 states had **no** safe tested fixed-rotation translation, including 7/18 held-out states. A future safety method must recognize or abstain in such states, rather than assuming withdrawal always works.

G10 2D repair had one truly safe held-out repair, two unsafe nominal pass-throughs, and 15 states without a predictor-safe action. G10.1's narrower search still yields only **one** truly safe held-out selection, **two** false-safe selections, and **eight** false-infeasible held-out states. Reducing the optimization dimension therefore does **not** materially rescue the existing predictor-based repair, even though it exposes substantial true-SOFA translation controllability.

## Direction, slope, monotonicity, and boundary

- At exact nominal ±0.05/0.10/0.20 probes, predicted versus true force-change sign agreed in **189/240 (78.75%)**: increasing translation 88/120, decreasing translation 101/120; among decreasing probes ending at negative `v`, 15/18. Independently, reducing translation lowered true R2 in 101/120 probes. Cohort sign agreement was held-out 89/108, t*−2 55/66, t*−1 45/66.
- Local finite-difference slope signs agreed in **31/40 (77.5%)**; held-out 16/18, t*−2 8/11, t*−1 7/11. Signed slope Pearson correlation was `-0.087`; absolute-magnitude correlation was `-0.054` (descriptive only). A usually correct local sign is not sufficient to locate a safe threshold crossing.
- The predeclared “approximately nondecreasing” diagnostic allows at most 5% adjacent true-force decreases greater than `1e-6 N`; only **6/40** full curves and **6/40** local ±0.20 neighborhoods met it. Median global Spearman(v,R2) was `0.992`, local `0.903`, but median adjacent-decrease fractions were `0.16` and `0.211`, respectively. Thus there is a broad increasing trend with locally non-monotonic force response; a monotone-controller assumption is not supported.
- In the only **five** states with both a predicted-safe candidate and a true-safe tested translation, absolute closest-safe-translation error had mean `0.599`, median `0.620`, maximum `1.240` normalized action units. The predicted choice lay on the same side of `v_nom` in 3/5, yet only 2/5 choices were truly safe. In the 22 false-infeasible states there is no predicted safe boundary to compare at all. These finite-grid errors are more consequential than absolute force RMSE for this repair question.

## Stopping versus rotation, and difficult events

The G10 `[0,0]` action was true-safe in 15/40 states; **all 15** remained safe as `[0,w_nom]` with fixed nominal rotation. One additional state was safe at `[0,w_nom]` but not at `[0,0]`. Thus zeroing rotation was not required for those 15 known zero-action witnesses. This does **not** imply rotation cannot affect force; its effect was deliberately not swept here.

The following event table rounds normalized actions to two decimals; `—` means no safe point on the finite grid. The first pair in each cell is **closest true-safe v / predictor-selected safe v**. Exact `v_nom`, `w_nom`, full true and predicted curves, local direction results, and stopping/withdrawal witnesses for every event and lookback are in `difficult_event_results.csv` and `events_state_curves/`.

| Event state | t*−2 true / predicted v | t*−1 true / predicted v |
|---:|---:|---:|
| 3 | — / — | — / — |
| 27 | 0.00 / — | −0.42 / — |
| 35 | 0.76 / — | −0.42 / — |
| 39 | −0.22 / — | −0.98 / — |
| 47 | 0.12 / — | −0.94 / — |
| 55 | 0.82 / — | −0.90 / — |
| 67 | 0.90 / — | −0.44 / — |
| 75 | 0.62 / 0.00 (safe) | 0.20 / 0.29 (**false-safe**) |
| 83 | −0.54 / — | — / — |
| 91 | 0.20 / — | — / — |
| 99 | — / — | — / — |

Earlier correction matters under the frozen H=2 protocol: 9/11 event states were recoverable at t*−2, but only 7/11 at t*−1; `v=0` was safe in 7/11 earlier states and only 1/11 later states. Strong withdrawal at `v=-1` was safe in 9/11 earlier and 6/11 later states. These paired outcomes are not an online intervention study.

## Decision and next step

**B — translation-only physics is promising, but the current frozen force predictor is not a usable translation-only safety layer.** The majority of sampled states (27/40, including 9/11 t*−2 events) have a true-safe fixed-rotation translation, and reducing translation often lowers force. Yet the existing predictor declares 22 such states infeasible and produces three false-safe choices; the boundary errors are large when both boundaries exist. Class A is therefore unsupported. Class C would overstate the finite-sweep evidence because many states *are* recoverable, although the 13 nonrecoverable sampled states impose a serious limit on any translation-only controller.

**Recommended next phase, not performed here:** develop and validate an *offline local translation-force response model* for `delta_v -> delta_R2` or conservative threshold-crossing probability, with explicit detection/abstention for states where no fixed-rotation translation is safe; compare on held-out parent states and t*−1/t*−2 events before considering any online closed-loop test. Do not assume monotonicity or deploy a rule-based withdrawal fallback from these data alone. No training, recalibration, Safety Head, CBF, controller change, rotation optimization, or 107000-series evaluation occurred in G10.1.
