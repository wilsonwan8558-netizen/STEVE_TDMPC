# Phase G12.3 — force-gated MPPI weighting

**Status: STOP at the true-SOFA plan-fork gate, before closed-loop development.**
The original weighted-mean MPPI action was reproduced exactly and the binary
mask was integrated without changing gate-inactive actions. The paired forks
did not demonstrate credible true-force reduction at above-limit states, and
severe masks collapsed effective sample size (ESS). This is a protocol stop,
not an A/B/C confirmation classification.

## Frozen inputs and original MPPI equations

The original TD-MPC2 checkpoint remains `step_280000.pt`, SHA256
`476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`.
The G12 force-head checkpoint remains SHA256
`b1d8f66b943efdab47c134f7126db0cb361270cabcc1e3e6899914219cd0d7e6`.
Its input is the same pre-transition planner latent `z_t[256]` and candidate
first action `a_t[2]`; frozen train-only mean/std buffers normalize the input.
The output predicts the next RL interval's maximum distal contact force in
nominal simulation N. G12.2-selected G1 calibration adds exactly
`2.1008816303183657` N. `H_force=1` and
`F_lim=3.685331040756903` nominal N. The Generic solver, contact-force
interface, task, TD-MPC2 checkpoint, and force head were unchanged.

The archived planner samples 512 horizon-3 trajectories per iteration (24
policy-prior and 488 Gaussian), scores each with the original frozen task
objective, and repeats six MPPI/CEM updates. For each iteration, `topk` keeps
64 elites. With task score `J_i`, temperature `T=0.5`, and
`J_max=max_elite J_i`, its code computes

```
u_i = exp(T * (J_i - J_max))
w_i = u_i / (sum_j u_j + 1e-9)
mean = (sum_i w_i * a_i) / (sum_i w_i + 1e-9)
```

Nonelites have zero original MPPI weight. The weighted mean updates the
trajectory distribution; deterministic evaluation executes the final mean's
first action, clamped to `[-1,1]`, not a sampled top candidate. The original
mean is also the planner's next-step warm start. Full source-path, variance,
and warm-start details are frozen in `mppi_weighting_audit.json`.

Before adding the mask, independent CUDA reconstruction from the archived
final candidate actions and task scores matched the archived applied action
**exactly in 16/16 states** (maximum absolute difference `0`, predeclared
`rtol=0`, `atol=1e-6`); see `baseline_action_reproduction.csv`.

## G12.3 mask and archived-state audit

For each final-iteration candidate, the frozen calibrated one-step force is
`F_hat_i = ForceHead(z_t,a_i) + 2.1008816303183657`. The binary mask is
`m_i=1[F_hat_i<=F_lim]`. After the original MPPI updates and task scores are
complete, only final elite weights are changed:

```
W_safe = sum_i w_i * m_i
w_i_safe = (w_i * m_i) / W_safe       if W_safe > 0
mean_safe = (sum_i w_i_safe * a_i) / (sum_i w_i_safe + 1e-9)
```

The masked mean's first action is executed and its sequence becomes the
normal planner warm start. If `W_safe=0`, the exact original MPPI action and
warm start are retained. This weight-empty fallback can occur even when
nonelite candidates pass the force test, because nonelites have zero original
weight; two such states occurred. When no elite weight is removed, the
implementation returns the original mean without epsilon-induced drift.
No force penalty, force-based ranking, sampled-candidate selection, or action
repair was used.

Exact CUDA replay of 46 archived validation planner states yielded:

| Retained-weight class | States | Action changed | Gate behavior |
| --- | ---: | ---: | --- |
| FORCE-INACTIVE (all 512 safe) | 20 | 0 | Exact baseline action in 20/20 |
| LIGHT-GATE (`W_safe>=0.8`) | 3 | 1 | Two masks removed only zero-weight nonelites |
| MODERATE-GATE (`0.2<=W_safe<0.8`) | 5 | 5 | Retained weight 0.422–0.734 |
| SEVERE-GATE (`0<W_safe<0.2`) | 4 | 4 | Retained weight 0.031–0.172 |
| GATE-EMPTY (`W_safe=0`) | 14 | 0 | Exact baseline fallback |

Across all 46 archived states, `W_safe` ranged 0–1 (median 0.797), baseline
ESS was approximately 64 (range 63.939–64), and gated ESS ranged 2–64
(median approximately 64). The four severe-gate ESS values were **2, 4, 7,
and 11**; this is substantial collapse relative to baseline ESS ≈64. The
largest action L2 difference was 0.605. The archived-state CSVs record safe
fraction, retained mass, baseline/gated ESS, translation/rotation difference,
baseline and gated predicted force, and exact inactive invariance. They also
show that removal strength did not translate monotonically into action size:
a severe state changed action by only 0.0068, while a moderate state changed
by 0.605.

## Paired true-SOFA forks

Before SOFA rollout, `fork_selection_manifest.json` froze all 12
positive-gate archived states, eight force-inactive controls, and two
weight-empty controls—22 states total. The latter controls were selected by
descending **archived baseline** true force, not by any gated outcome. One
reused Generic SOFA scene replayed 15,441 archived G2 transitions in order
through seed 90199/decision 44. All selected parent states, baseline forks,
repeated baseline forks, and actual parent transitions matched G2 exactly;
maximum observed discrepancy was zero. Each paired fork executed the baseline
and masked first action from the same parent for one RL interval. The full
row-level results, including true force, reward, tip displacement, insertion,
normalized projected tip-target distance, and next-observation deviation, are
in `true_sofa_masked_mppi_forks.csv`.

Among 12 positive-gate states, the action changed in 10. True force fell in
six, rose in four, and stayed identical in two unchanged-action states. For
the 10 changed-action states, mean gated-minus-baseline true force was
**+0.0051 N**, not a decrease. The paired immediate reward difference ranged
from -0.000137 to +0.000384; there was no obvious catastrophic one-step task
loss, though this cannot establish closed-loop navigation preservation.

The four positive-gate states whose baseline true force exceeded `F_lim` are
the critical safety subset:

| State | Gate | Retained weight | Gated ESS | Baseline force (N) | Gated force (N) | Difference (N) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90060/42 | Severe | 0.109 | 7 | 5.192958 | 5.190606 | -0.002352 |
| 90078/61 | Moderate | 0.594 | 38 | 4.678628 | 4.695414 | +0.016786 |
| 90126/154 | Severe | 0.031 | 2 | 4.134532 | 4.165250 | +0.030718 |
| 90165/43 | Light | approximately 1 | 64 | 3.951557 | 3.951557 | 0 |

Only one of these four forces fell, and only by 0.0024 N; two worsened and
none moved below `F_lim`. The eight other positive-gate states had truly
below-limit baseline force, consistent with G1's known validation
false-warning burden (398/2,186 safe validation transitions). This selected
fork sample does not estimate a population false-warning rate. The proposed
false-warning → low retained mass → large action change → navigation
degradation chain cannot be tested without closed-loop development; severe
mass/ESS loss is established, navigation degradation is not.

## Fork-gate decision and scope

The Step 14 gate **fails**. Although original MPPI semantics and inactive
invariance were recovered, the true-SOFA forks do not show credible force
reduction on the above-limit subset, and severe masks collapse ESS. No
development seed audit, B0/B1 closed-loop development, fallback tuning,
pre-confirmation freeze, or untouched confirmation was performed. Therefore
there are no development safety/task metrics or final A/B/C classification.
Preserving MPPI weighting alone has not rescued the one-step force-aware
planner on this gate. The result does not justify a formal CBF comparison as
the next automatic step. `fork_gate_decision.json` records the stop.
