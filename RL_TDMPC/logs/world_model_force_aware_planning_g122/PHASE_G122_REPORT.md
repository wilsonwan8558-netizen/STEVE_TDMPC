# Phase G12.2 — stopped at the true-SOFA plan-fork gate

**Status: STOP, before closed-loop development.** The frozen nine-state paired
fork audit did not establish credible true-force reduction when the gate was
active. This is a protocol stop, not an A/B/C confirmation classification.

## Frozen protocol and validation calibration

The TD-MPC2 checkpoint is `step_280000.pt`, SHA256
`476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`.
The G12 force-head checkpoint is `force_head.pt`, SHA256
`b1d8f66b943efdab47c134f7126db0cb361270cabcc1e3e6899914219cd0d7e6`.
Its train-only featurewise mean/std buffers normalize pre-transition planner
`z_t[256]` and first candidate action `a_t[2]`; its nonnegative output predicts
the next RL interval's maximum distal contact force in nominal simulation N.
`H_force=1`; no multi-step force rollout or continuous force penalty was used.
The Generic solver and `F_gate=F_lim=3.685331040756903` nominal N were unchanged.

The three allowed gate values were G0 raw (`delta=0`), G1 the frozen G12.1 full
additive calibration (`delta=2.1008816303183657` N), and G2 one predeclared
intermediate validation quantile (`delta=1.4631850065161804` N). The
intermediate was the higher-interpolated 80th percentile of validation
above-limit deficits, rounded one float upward. The source-frozen selection
rule required G2 to halve G0's false-safe **count** and halve G1's
false-warning count; otherwise it selected G1 if G1 halved G0's false-safe
count. Navigation outcomes were not used. G2 had 25 false-safe rows versus G0's
49, missing the integer half-count criterion by one row; G1 was therefore
selected. This stringent rule recreated the full calibration's substantial
false-warning burden, rather than achieving the intended compromise. It was
not changed after observing metrics or forks.

| Gate | Validation false-safe / 130 | False-warning / 2,186 | Safe acceptance | High-force recall | Safe specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 raw | 49 (37.7%) | 34 (1.6%) | 95.0% | 62.3% | 98.4% |
| G1 full, selected | 6 (4.6%) | 398 (18.2%) | 77.5% | 95.4% | 81.8% |
| G2 intermediate | 25 (19.2%) | 173 (7.9%) | 88.0% | 80.8% | 92.1% |

All validation metrics and the exact source-frozen rule are in
`gate_calibration_candidates.json`, `gate_calibration_metrics.csv`, and
`selected_gate_calibration.json`.

## Candidate availability

The archived 16 final-iteration validation candidate sets (512 candidates
each) classified under G1 as 13 inactive, 0 active, and 3 empty. The G1 safe
fraction was 0 or 1 in each of these states (median 1; mean 0.8125). To obtain
actual active examples **before any SOFA rollout**, an exact CUDA baseline
planner replay examined one validation state per validation seed nearest the
selected raw-force gate boundary. This targeted 30-state sample is deliberately
boundary-enriched, not a population estimate: 14 inactive, 7 active, 9 empty;
safe fraction min 0, median 0.2627, mean 0.4499, max 1. Its seven active
states had task-score loss from 0.000023 to 0.065487 (mean 0.012359), relative
to the highest-scoring unconstrained candidate. Candidate identity, rank, and
safe fractions are saved in `candidate_availability_audit.csv`,
`planner_state_gate_classes.csv`, and
`candidate_availability_boundary_validation.csv`.

Important planner-semantics caveat: archived TD-MPC2 applies the final
weighted MPPI **mean** action, not the highest-scoring sampled candidate. The
gate specification selects a sampled candidate. The top candidate's first
action differed from the archived applied action at all 16 original and all
30 boundary-audit states, including every gate-inactive state. Consequently,
"gate inactive" does not imply an unchanged applied action under this exact
candidate-selection formulation. This confounds any naive B0-vs-gated
attribution; the paired forks explicitly retain the archived applied-mean
baseline as a separate arm.

## Paired true-SOFA forks and decision

The first three seed-ordered states in each class were frozen before forking.
One reused Generic SOFA scene replayed 5,619 archived G2 parent transitions in
order through seed 90075, decision 118. The nine selected parent states,
baseline forks, repeated baseline forks, and actual parent transitions matched
G2 exactly; maximum observed difference was zero. At each parent we forked the
archived applied-mean action and the unconstrained best candidate, plus the
best safe candidate (inactive/active) or minimum-predicted-force candidate
(empty). Exact values are in `true_sofa_gate_forks.csv`.

| Gate-active state | Archived baseline true force | Gated true force | Gated minus baseline | Candidate task-score loss |
| --- | ---: | ---: | ---: | ---: |
| 90013/34 | 3.224165 | 3.233409 | +0.009245 | 0.000023 |
| 90044/44 | 0.983050 | 0.847009 | -0.136041 | 0.065487 |
| 90045/30 | 1.763570 | 1.854697 | +0.091126 | 0.013382 |

Only one of three gate-active selections lowered true force; two raised it.
All three archived baseline forces were already below `F_gate`. Gated-vs-top
candidate true force also improved in only one of the three states. This does
**not** satisfy the prespecified requirement for credible true-force reduction
in safety-active states. The plan-fork decision is **FAIL/STOP**.

In the three empty-gate states, F1 versus F0 true force was 5.554 versus
5.491 N, 2.181 versus 3.275 N, and 1.373 versus 1.619 N. Thus F1 was better
in two states but worse at the only above-limit baseline state; no fallback was
selected. This is descriptive fork evidence, not a development comparison.

No development seed audit or rollout was performed, so there are no B0/B1/B2
development success or safety metrics, selected fallback, pre-confirmation
freeze, or untouched confirmation results. G12.1 navigation preservation and a
safety-constrained planning contribution cannot be assessed from this stopped
phase. Formal CBF work is not justified by this G12.2 gate result alone.

The stop decision is recorded in `plan_fork_gate_decision.json`; the exact
parent replay audit is in `true_sofa_gate_parent_replay_audit.json`. No soft
penalty, action repair, reactive supervisor, controller retraining, or CBF was
implemented.
