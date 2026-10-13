# Phase G9 — direct short-horizon force-risk estimation

**Classification: B — temporal prediction helps, but early control remains uncertain.**
The historical Generic-solver simulation, frozen TD-MPC2 step-280000 checkpoint, and distal max-substep force definition were not changed. The 102000–102029 confirmation set was opened only after the corrected pre-confirmation freeze. No G10, shield, or controller change was run.

## Protocol and state estimation

- Runtime: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, PyTorch `2.7.1+cu128`, CUDA `12.8`, `cuda:0` on an NVIDIA GeForce RTX 5090. Checkpoint SHA256: `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`.
- Safety reference: `F_lim = 3.685331040756903` nominal N. Each `F` is the maximum reconstructed distal contact force over the 22 SOFA substeps of one 0.132 s control interval. The target is **direct** `R_H(d)=max(F_d,...,F_{d+H-1})` for `H=1,2,3,5`, using complete same-episode intervals. The five individual forces are retained for diagnostics.
- Development: 200 existing Generic behavior episodes, episode-disjoint TRAIN/VALIDATION/DEVELOPMENT-TEST with 10,203/2,136/1,961 common H5-complete rows. The original G2 next-interval force, G3 selected action, pre-action observation, and continuous same-episode indexing were audited exactly. Confirmation: 30 new ordered seeds, one reused scene, 2,196 steps, 27/30 navigation successes, mean episode reward 3.501, and 2,016 common H5-complete rows.
- Privileged supervision target: the frozen 38D pre-action contact-aware physical state, not an evaluation-time input. E0 maps the **current 14D observation** to that state with a 128–128 MLP. E1 maps three 14D observations and the two preceding 2D actions (46D) with the same-size MLP. E2 uses a 20D unscented Kalman filter on ten tracked 2D positions and ten velocities, followed by a 34D-to-38D MLP mapper. The UKF has a TRAIN-fitted nonlinear action-conditioned transition, sigma-point prediction, and direct correction from the **current** ten tracking measurements; it resets position, zero velocity, and covariance at each episode. This is appropriate to nonlinear guidewire motion without claiming that a future unobserved state is corrected. No estimator receives future observations/actions, true current force, or a force sensor.
- All risk models additionally use the frozen 256D TD-MPC2 current-observation latent and an **explicit** `H×2` action sequence. The direct 64–64 ReLU model emits noncrossing Q50/Q90/Q95/Q99 through cumulative softplus increments, fitted by equal-weight pinball loss; checkpoint and nonnegative Q95 additive correction were selected on VALIDATION. The primary decision score is validation-corrected Q95, above `F_lim` meaning predicted high risk. No recursive predicted-state rollout is used.
- The TD-MPC2 CEM planner is frozen at horizon **3**, 512 samples, 64 elites, six iterations; its executed evaluation action is the first optimized mean action. H5 realized-behavior sequences are retrospective diagnostics, **not** available five-step current plans. Training conditions on successive realized behavior actions and their matching SOFA forces; unexecuted plan sequences are assessed only with separately matched true-SOFA forks.

Validation state-estimation RMSE against privileged physical state:

| Estimator | Contact magnitude (N) | Distal velocity (mm/s) | Tip position (mm) |
|---|---:|---:|---:|
| E0 | 0.679 | 11.913 | 2.063 |
| E1 | 0.618 | 7.394 | 2.285 |
| E2 | 0.590 | 7.069 | 2.179 |

E2 modestly improves contact-magnitude and velocity estimation over E1, but **does not materially improve safety prediction**: on development-test H2, E1 and E2 had 11/140 and 12/140 false-safe high-risk windows; on confirmation, 10/228 and 12/228. For H3, confirmation E1 had 4/292 false-safe windows versus E2's 15/292. UKF diagnostics: 15,300 innovations, mean norm 0.0455, Q90 0.0978, no nonfinite values, minimum pre-projection covariance eigenvalue `6.95e-5`, maximum posterior diagonal variance 0.0177, and exact zero-velocity resets. Adding normalized tracking-measurement noise with sigma 0.005/0.010 raised validation contact-magnitude RMSE from 0.590 to 0.609/0.659 N and velocity RMSE from 7.069 to 7.393/8.274 mm/s. Thus filtering is stable, but noise-sensitive and not a demonstrated solution to future-force prediction.

## Direct prediction and untouched confirmation

The following uses the **same 2,016 confirmation behavior rows** at each horizon. False-safe is the count accepted by corrected Q95 among truly above-limit `R_H`; false-warning is the fraction of truly below-limit rows rejected. All coverage percentages are marginal over these behavior rows. Development-test pinball is reported in nominal N.

| Est. | H | Dev pinball | Confirm unsafe | False-safe | False-warning | Q95 raw/corrected coverage | Retrospective onset warned |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 1 | 0.092 | 152 | 14/152 | 6.7% | 93.0% / 95.9% | 0/71 |
| E0 | 2 | 0.110 | 228 | 16/228 | 9.7% | 92.2% / 94.9% | 65/71 |
| E0 | 3 | 0.119 | 292 | 7/292 | 10.8% | 92.2% / 93.8% | 70/71 |
| E0 | 5 | 0.148 | 407 | 5/407 | 16.2% | 90.0% / 94.1% | 71/71 |
| E1 | 1 | 0.091 | 152 | 14/152 | 6.4% | 87.7% / 94.8% | 0/71 |
| **E1** | **2** | **0.115** | **228** | **10/228** | **9.1%** | **89.2% / 95.3%** | **69/71** |
| E1 | 3 | 0.133 | 292 | 4/292 | 11.9% | 88.7% / 93.2% | 70/71 |
| E1 | 5 | 0.140 | 407 | 3/407 | 16.5% | 89.9% / 92.0% | 71/71 |
| E2 | 1 | 0.089 | 152 | 15/152 | 7.9% | 92.1% / 94.6% | 0/71 |
| E2 | 2 | 0.115 | 228 | 12/228 | 8.7% | 91.3% / 94.5% | 66/71 |
| E2 | 3 | 0.138 | 292 | 15/292 | 10.2% | 87.4% / 95.0% | 66/71 |
| E2 | 5 | 0.138 | 407 | 6/407 | 14.0% | 91.2% / 93.3% | 71/71 |

The Q50/Q90/Q95/Q99 confirmation empirical coverages for E1 were respectively H1 `0.600/0.836/0.877/0.960`, H2 `0.619/0.855/0.892/0.964`, H3 `0.576/0.830/0.887/0.955`, and H5 `0.539/0.845/0.899/0.960`. No quantile crossings occurred. Raw upper quantiles generally **undercover**; the validation-only additive correction brings E1/H2 marginal Q95 coverage to 95.3% on confirmation but does not fix H3/H5 or condition-specific calibration. The complete quantile-by-horizon tables are in `quantile_coverage_metrics.csv` and `confirmation_quantile_coverage.csv`.

On VALIDATION's 67 high-force onsets, the predeclared selection rule chose H2 for E0/E1/E2 as the shortest native-plan horizon with at least 10 percentage points more early-onset recall than H1 and no more than 20% false warnings. E1/H2 warned on 62/67 validation onsets and 69/71 confirmation onsets one step (about 0.132 s) before the onset; 55/71 confirmation warnings also occurred when the *preceding* interval's force was below the limit. H1 cannot include a later onset from an earlier decision by construction. Confirmation E1/H3 warned on 70/71 onsets, normally at two steps (0.264 s), with 64/71 warnings before a previously above-limit interval. H5 warned on 71/71 but its four-step (0.528 s) event-linked lead is **not a native five-action plan** and its false-warning/calibration tradeoff is worse. These onset numbers condition on the **future actions actually taken**; they are hindsight upper bounds, not deployable time-`d` warning recall.

## G8.5 difficult events: warning versus recoverability

The 11 G8.5 finite-search no-safe-one-step-found events were independently recovered. The 30 archived parent episodes were replayed in order in **one reused Generic scene**. All 1,848 parent steps matched archived observation, selected/applied action, reward, tip, insertion, and force with maximum recorded numerical difference **zero**. SOFA copy-on-write children evaluated counterfactual sequences without changing the parent. At each `t*−k`, the oracle included the force interval at `t*`, so its evaluation length was `k+1` intervals; in particular `t*−5` required six intervals. The finite family combined frozen G3 21 first-action candidates, a 5×5 coarse action grid, constant withdrawals, the current TD-MPC2 plan prefix, and open-loop continuations. A safe result required all checked intervals below `F_lim` and full horizon completion or successful terminal. Future archived-action continuations are **hindsight existence witnesses**, not executable time-`t` forecasts. No-safe-found is not a proof of infeasibility.

| Event state | Safe prior offsets found | Nearest found lead (steps) |
|---:|---|---:|
| 3 | 2 | 2 |
| 27 | 1, 2, 5 | 1 |
| 35 | 1, 2, 5 | 1 |
| 39 | 1, 2, 3 | 1 |
| 47 | 1, 2, 3, 5 | 1 |
| 55 | 1, 2, 3, 5 | 1 |
| 67 | 1, 2, 3, 5 | 1 |
| 75 | 1, 2, 3, 5 | 1 |
| 83 | 2, 3 | 2 |
| 91 | 1, 2, 3, 5 | 1 |
| 99 | 2, 3, 5 | 2 |

Recoverability counts are **8/11 at t*−1**, **11/11 at t*−2**, **8/11 at t*−3**, and **8/11 at t*−5**. Nearest found intervention lead: median **1 step** (0.132 s), Q90 **2 steps** (0.264 s), maximum **2 steps**. The non-monotone counts reflect different trajectory lengths and a finite action family, not a theorem that intervening earlier physically worsens safety. All 11 archived behavior intervals at `t*−1` already exceeded `F_lim`, underscoring that warnings near `t*` may be late relative to force escalation.

Using realized behavior actions in hindsight, event-linked H2 Q95 correctly warned at `t*−1` for all 11 events, H3 at `t*−2` for all 11, and H5 at `t*−4` for all 11; the latter cannot reach `t*` from `t*−5`. If a warning is also required to occur while the **previous** force interval remains below the limit, H2 covered **8/11**, H3 **10/11**, and H5 **10/11**; state 3 was already in an ongoing high-force episode. The primary event-linked table excludes warnings that concern an unrelated earlier high-force interval. These are retrospective leads, not on-policy anticipatory-control results.

To test actual current TD-MPC2 plan inputs, 44 prior states × H1/H2/H3 were forked under true SOFA, then scored against **their own** fork labels. This is an event-enriched diagnostic, not a population estimate. At H2, the 44 plans contained 32 true high-risk and 12 true safe sequences. E1/H2 produced **0/32 false-safe**, but accepted only **3/12 safe plans** (75% false warning); E0 and E2 were the same on these counts. At H3, E1 accepted **0/44** plans, despite six truly safe sequences, and raw/corrected Q95 coverage was only **77.3%/81.8%**. Planned-versus-subsequently-realized behavior action-sequence L2 distance averaged 0.372 at H2 and 0.745 at H3. Thus behavior-trajectory calibration does not establish useful candidate-plan availability.

## Decision and limits

1. Temporal information helps over instantaneous E0 for some outcomes: E1/H2 has better confirmation false-safe and onset recall, and E2 estimates certain hidden-state components more accurately. The UKF's state improvement does not consistently carry through to risk or candidate availability.
2. The shortest **validation-selected** horizon is H2, but it can supply at most one event-linked step of warning. Three difficult events required a found intervention two steps earlier, so H2 alone is insufficient for all 11. H3 is the shortest native TD-MPC2 horizon that retrospectively covers that two-step lead; its plan-fork calibration/availability remains inadequate. H5 adds nominal retrospective lead at a clear cost in drift/false warnings and lacks a native current plan.
3. Physical recoverability is **not** the dominant blocker in this finite study: all 11 had a safe witness by `t*−2`. The dominant unresolved blocker is **prospective action-sequence coverage and calibrated risk on actual TD-MPC2 candidate plans**; conditional upper-tail calibration and the limited filtering gain are secondary concerns.
4. Final classification is **B — temporal model helps, but early control remains uncertain**. G10 is **not justified for automatic launch** on this evidence. A separately scoped decision would need prospective, candidate-matched calibration/availability evidence and a control objective; none was implemented here.

The first pre-confirmation freeze attempt was rejected by its own guard because JSON changed an ordered seed tuple to a list. It is preserved as `pre_confirmation_freeze_rejected_typecheck.json`. The comparison was corrected and a new freeze passed integrity verification **before** any 102000–102029 seed was run. Confirmation episode output was flushed after each completed episode. All phase data and source scripts remain in this G9 output directory and `experiments/generic_solver_g9/`, respectively.
