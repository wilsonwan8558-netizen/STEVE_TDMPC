# Phase G10 — H=2 force-constrained minimal current-action repair

**Offline gate: NO-GO. Final classification: C — the frozen learned force surrogate does not reliably support minimal-action repair under true Generic-SOFA evaluation.** No online closed-loop test or 107000–107199 confirmation seed was opened. This is a simulator result, not a clinical safety claim.

## Frozen force predictor and control problem

The predictor is the frozen G9.5 **M1 planner-conditioned scalar H=2** family, retained with **all three** fixed replicas (260312/260313/260314). This is the prior planner-conditioned scalar model; G9.6 stepwise decomposition and G9.7 privileged-state enlargement did not produce a justified replacement. It was not retrained, renormalized, or recalibrated in G10. The 298D input is the frozen TD-MPC2 current-observation 256D latent, frozen E1 pre-action estimated physical 38D state, and the proposed normalized two-action pair. Its noncrossing Q50/Q90/Q95/Q99 head predicts `R2 = max(F_(t+1), F_(t+2))` in nominal N. The two force intervals use maximum distal (~3 mm) `J_contact^T lambda / 0.006 s` over each interval's SOFA substeps. Each replica's Q95 receives its frozen nonnegative validation-only correction; **predicted safe means all three corrected Q95 values are at most `F_lim = 3.685331040756903` nominal N**. The limit is a Generic-simulation statistical high-force reference, *not* a clinical injury or vessel-damage threshold.

| Replica seed | Checkpoint SHA256 | Frozen Q95 correction, nominal N |
|---:|---|---:|
| 260312 | `7f2ce7d27a01a0772bfbcb103042d419c8718b73ce3b63354e7b8e1d9b6b9b66` | 0.298106 |
| 260313 | `decee08e203d53e828a95ccdf71f528492f11c07079a23779dc108bbc8ebaa20` | 0.139736 |
| 260314 | `6932c520d8421cda7f9454248d4b79d86919e8bed67c47c5c5a98d67eb3330f0` | 0.245044 |

The frozen input-normalization SHA256 is `697408d9f9acdbb63c4adfbaf0e3b39f4c3966d1212b86dcc566708969f94976`; E1 mapper and its normalization hashes are `1ab6ba33e15e79b0418084ad82830e8c46e9374ed3d279cdf482bf28fa22f665` and `b236fbf685a95f266ce1790e05724bc702f02dfbcfeb138b6b682115036db092`. The frozen TD-MPC2 `step_280000.pt` SHA256 is `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. The G9.6-v2 executable optimized-mean semantics were checked against its prior freeze. CUDA execution used `steve_tdmpc`, torch `2.7.1+cu128`, `CUDA_VISIBLE_DEVICES=0`, and visible GPU 0, NVIDIA GeForce RTX 5090; CPU fallback is rejected.

At each pre-action state, the exact frozen TD-MPC2 executable optimized mean supplies `a_nom(t)` and `a_nom(t+1)`. Only the first normalized 2D action is variable:

`min_a ||a − a_nom(t)||_2²`, subject to `a ∈ [−1,1]²` and every frozen replica's corrected `Q95[R2 | x_t, a, a_nom(t+1)] ≤ F_lim`.

The second action is fixed inside the H=2 prediction and true-SOFA fork; it is never optimized or replanned there. If later deployed, the controller would replan at the next *real* control step, but no such online operation was run. The objective remains predicted physical distal force relative to the statistical force reference, not relative plan ranking. A search finding no predicted-safe point explicitly returns `INFEASIBLE`, never an unlabeled risky action.

The predeclared search evaluates the nominal action, then a **41×41 float32 grid** over `[−1,1]²` (axis spacing 0.05). If a predicted-safe grid point exists, it selects minimum Euclidean deviation with translation/rotation lexicographic ties and refines around the current best with clipped 5×5 offsets at radii **0.025, 0.005, 0.001**. Duplicate executable float32 actions are skipped. Three repaired cases used 1,754 predictor action evaluations each (1 nominal + 1,681 grid + 72 distinct local); grid-infeasible cases used 1,682. A 41×41 no-safe finding is finite-search evidence, not mathematical infeasibility. The predeclared fallback candidate is normalized `[0,0]` (zero translation and rotation); its force was evaluated separately, not assumed safe or executed online.

## Offline cohorts and true-SOFA validity

The primary cohort comprised **all 18** G9.5 DEVELOPMENT-TEST states whose archived executable optimized-mean H=2 nominal plan was truly above `F_lim`. These were selected before G10 prediction/repair outcomes, preserving the 560/120/120 parent-state train/validation/development-test split. The second cohort comprised the 11 difficult G8.5 events at **t*−1 and t*−2** (22 earlier parent states). This is an event-enriched diagnostic, not exchangeable with the held-out development cohort. All 40 nominal H=2 plans were truly high force.

Every repaired or fallback action was executed only in a copy-on-write true Generic-SOFA fork; the original TD-MPC2 parent action always advanced the reused scene. The held-out replay matched **15,500** archived parent steps and all selected archived nominal F1/F2/R2 labels exactly. The difficult-event replay matched **1,848** archived parent steps exactly. For observation, applied action, reward, tip position, insertion length, and next-interval distal force, maximum parent difference was **0** in both passes. All two-step true-safe labels required two complete intervals. The simulator, friction, force reconstruction, checkpoint, task, and action bounds were unchanged. See the parent replay audits and per-state JSON artifacts.

## Primary held-out repair outcome

| Measure | 18 truly unsafe nominal states |
|---|---:|
| Predicted-safe repaired action found | **1** |
| Repaired action truly safe | **1** |
| False-safe *repaired* action | **0** |
| Unsafe nominal incorrectly predicted safe and passed through unchanged | **2** |
| No predicted-safe action on fixed 41×41 search | **15** |
| Zero-current-action true-safe witness, fixed nominal second action | **8** |
| Model-infeasible yet zero-action true-safe witness | **5** |

The only held-out repaired action was state 10 (seed 90007). Nominal action `[0.986902, −0.700905]` changed to `[0.343500, −0.062000]`, normalized Euclidean deviation **0.9067**. True R2 fell from **3.697** to **1.479** nominal N; corrected maximum replica Q95 for the repaired plan was **3.685287** nominal N. The held-out mean, median, Q90, and maximum action deviation are each 0.9067 because there is only one actual repaired action. The two unsafe nominal pass-throughs remained unsafe at true R2 **4.850** and **4.582** nominal N. The zero-action counterfactual was true-safe in 8/18 states but unsafe or censored in 10/18, so it is **not** a validated universal fallback.

The five held-out model-infeasible states with a true-safe zero-action witness establish a lower bound on fixed-second-action oracle-recoverable but model-missed cases. They cannot be dismissed as physically unrecoverable. Conversely, no safe result in this finite set cannot establish physical impossibility.

## Difficult G8.5 events and early intervention

| Position | Unsafe nominal states | Predicted-safe repairs | True-safe repairs | False-safe repairs | Search `INFEASIBLE` | Zero-action true-safe witnesses | Model-infeasible despite zero-safe witness |
|---|---:|---:|---:|---:|---:|---:|---:|
| t*−1 | 11 | 1 | **0** | **1** | 10 | 1 | 0 |
| t*−2 | 11 | 1 | **1** | 0 | 10 | 6 | **5** |

Both proposed early repairs occurred at event state 75. At t*−2 the nominal `[0.844549, −0.950305]` became `[0.166000, −0.635000]` (deviation **0.7482**); true R2 fell from **4.353** to **1.709** nominal N, so this was one successful early minimal repair under the tested search. At t*−1, nominal `[0.967231, −0.332479]` became `[0.355500, −0.144500]` (deviation **0.6400**) and was predicted safe at corrected maximum Q95 **3.685327** but had true R2 **4.414** nominal N: a false-safe repair. The same event's zero-action fork was true-safe at t*−1, underscoring a predictor/search miss. The detailed per-event classifications, nominal/repaired actions, predicted Q95, true F1/F2/R2, and force of the zero-action witness are in `difficult_event_repair_results.csv`.

The G8.5 historical statement that all 11 difficult events had a safe trajectory available from t*−2 allowed changing an action *trajectory*. G10 fixes the nominal second action and repairs only the current action. Its 1/11 successful t*−2 repairs should therefore not be conflated with G8.5's 11/11 existence result, but the 5/10 model-infeasible t*−2 cases with a zero-action fixed-second witness directly demonstrate failure even under the narrower G10 formulation.

## Oracle benchmark, intervention cost, and stop boundary

The predeclared potential closest-action oracle was a true-SOFA 21×21 grid plus three 5×5 local refinements around the closest true-safe grid action, with the **same fixed nominal second action**. The explicit offline NO-GO stop was reached from the already completed true-SOFA repair and zero-action forks, so that additional dense oracle was **not run**. `oracle_repair_results.csv` records the observed zero-action fixed-second witnesses and marks every closest-safe action and excess-deviation field as *not computed*. It reports a **lower bound**, not an exact oracle-recoverable count or globally closest action. Repair excess deviation versus a closest oracle is unavailable; inventing it would misstate the evidence. For the two true-safe repairs, zero action was also true-safe but farther from nominal (held-out: 1.2105 versus repair 0.9067; t*−2: 1.2714 versus repair 0.7482); those are merely observed witness comparisons, not proof of minimality.

Across the **descriptively combined, nonexchangeable** 40 offline states, there were three proposed repairs, two truly safe and one false-safe; two unsafe nominal pass-throughs; **35** predicted-infeasible searches, of which **at least 10** had true-safe zero-action witnesses. The three attempted repair deviations had mean **0.7650**, median **0.7482**, Q90 **0.8750**, and maximum **0.9067** in normalized action space. The zero-action counterfactual was true-safe in 15/40 and unsafe/censored in 25/40. The strong no-go evidence does not require an arbitrary numeric pass threshold: the proposed intervention is rarely found, misses demonstrably recoverable states, and includes a false-safe early repair. The fallback is not reliably force-safe either.

**Offline gate: NO-GO. Online closed-loop evaluation is not justified.** Consequently there is no B0/B1 task-performance or closed-loop force comparison, no intervention-frequency claim, no online fallback use, no pre-confirmation online freeze, and no untouched 107000–107199 result. Those metrics are **not measured**, not zero. Do not infer navigation-performance preservation from offline forks. Final classification **C**: the current frozen learned force surrogate is insufficient for force-constrained minimal current-action repair. The force-threshold-plus-minimum-correction method remains physically interpretable as a formulation, but its safety/control story is **not supported** as an effective deployed controller by this true-SOFA gate. Stop here; do not run online G10, redesign the predictor automatically, or make paper-level efficacy claims.
