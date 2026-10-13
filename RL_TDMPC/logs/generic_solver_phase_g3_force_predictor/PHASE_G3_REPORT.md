# Phase G3 — Generic one-step force predictor

**Final classification: C. PREDICTOR NOT READY.**

Offline predictor validation only. No shield, CBF, controller update, encoder fine-tuning, or clinical force threshold was used.

## Frozen data and protocol

- Navigation checkpoint: `/tmp/stEVE_safety_head_1e1be81/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt`; SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`; format v1; 280,000 environment steps; CUDA `cuda:0` on RTX 5090.
- Canonical row: pre-action `o_d` → frozen 256-D `z_d`; normalized 2-D `a_d`; execute one `env.step(a_d)`; `force_next_max_N` is the max 3 mm distal force over its 22 SOFA substeps. G2 source field: `distal_contact_force_max_N`. Force uses vector-sum-before-norm and `J_contact^T lambda / 0.006 s`.
- G2 development dataset: 200 ordered reused-scene episodes, 15,500 behavior rows. Episode split: 140 train / 30 validation / 30 development-test. Exact seeds: `development_split.json`; no step-level random split.
- Fork audit: `PASS_GENERIC_FORK_RESTORE_PARENT_AND_G2_FORCE` on 12 states across 4 seeds; maximum checked parent/fork/G2 differences `{'applied_action': 0.0, 'force_next_max_N': 0.0, 'insertion_length_mm': 0.0, 'observation': 0.0, 'reward': 0.0, 'tip_position_mm': 0.0}`. Development parent replay compared 15500 steps with G2.
- Candidate set: behavior action plus clipped ±0.2/±0.4 translation/rotation axis offsets, 0.2/0.4 signed diagonals, and deterministic seeded broad fills to 21 unique bounded float32 actions. Exact rule and RNG offsets: `candidate_action_protocol.json`.
- Behavior TRAIN rows: 11043; TRAIN counterfactual states/labels: 560/11760; VALIDATION states/labels: 120/2520; DEVELOPMENT-TEST states/labels: 120/2520; confirmation states/labels: 120/2520.
- SOFA one-step candidate evaluations: 16800 development + 2520 confirmation; parent trajectories were separately executed and audited. Confirmation parent replay: 2522 steps, maximum differences `{'applied_action': 0.0, 'force_next_max_N': 0.0, 'insertion_length_mm': 0.0, 'observation': 0.0, 'reward': 0.0, 'tip_position_mm': 0.0}`.
- Exact Generic G2 all-step statistical references (nominal N, not clinical): Q90 2.126515680, Q95 3.685331041, Q97.5 5.926250465, Q99 7.466568794.

## Matched M0/M1 training and freeze

Both models are newly initialized 258–64–64–1 ReLU MLPs with raw-force MSE, Adam 1e-3, batch 512, shared TRAIN-behavior-only normalization, and matched sample/optimizer-update budget per validation round. M0 uses TRAIN behavior; M1 additionally draws from TRAIN counterfactual labels. Validation behavior MSE selects each checkpoint; only validation candidate ranking selects the primary model. No LCP weights, dev-test labels, or confirmation labels enter fitting or selection.
Primary frozen predictor: **M1**. The comparison checkpoints and normalization files were frozen before confirmation; hashes are in `frozen_model_manifest.json`. Confirmation seed manifest matches the predeclared 92000–92029 range and records the freeze manifest hash.
Model checkpoint SHA256: M0 `545755b9996cc06831d0b3daf2d052ff9e423fd7388f27cd24faa9f9ea5b9d24`; M1 `466849cd83ca4f52ce747d94b034fc31ddcb4ef4c6b88f5de6a9843bb0f45224`. The shared normalization file SHA256 is `7da8007730456d773dd4a051d60266311583f2e33f80ce0d7957c3e99fd429a3`.
The recorded inference timings are single-call diagnostics, not a fair speed comparison: the first model call includes CUDA warm-up. Both heads have the same architecture.

| Model | Val candidate RMSE | Val median state Spearman | Val pairwise accuracy | Selected epoch | Training wall time (s) |
|---|---:|---:|---:|---:|---:|
| M0 | 1.131 | 0.565 | 0.636 | 31 | 0.5 |
| M1 | 1.250 | 0.649 | 0.670 | 14 | 0.2 |

## Development-test behavior prediction

| Model | RMSE N | MAE N | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| M0 | 0.874 | 0.435 | 0.648 | 0.815 | 0.752 |
| M1 | 0.947 | 0.477 | 0.587 | 0.772 | 0.719 |

Above-reference behavior MAE, mean/median underprediction (`true − predicted`), and false-safe counts at all four Generic references are in `development_behavior_tail_metrics.csv`.

## Development-test candidate-action prediction

| Model | RMSE N | Median state Spearman | Mean pairwise accuracy | Mean safe-choice regret N | Top-3 high/low containment |
|---|---:|---:|---:|---:|---:|
| M0 | 0.965 | 0.575 | 0.678 | 0.065 | 0.308 / 0.333 |
| M1 | 0.938 | 0.506 | 0.647 | 0.052 | 0.278 / 0.350 |

Per-state profile RMSE/MAE, predicted/true ranges, ranks of true force extrema, and individual regrets are in `development_candidate_metrics.csv`. Safe-choice regret is the true-force gap between the predicted-lowest candidate and the true-lowest candidate; no action was executed by a shield.

### Development-test boundary fidelity

Predicted exceedance means `F_hat > q`; true exceedance means `F_true > q`. False-safe rate is `P(predicted safe | true unsafe)`.

| Model | Reference | Positive count | False-safe count/rate | AUROC | AUPRC | Mean missed severity N |
|---|---|---:|---:|---:|---:|---:|
| M0 | Q90 | 326 | 94 / 0.288 | 0.947 | 0.824 | 0.722 |
| M0 | Q95 | 198 | 15 / 0.076 | 0.984 | 0.904 | 0.387 |
| M0 | Q97.5 | 99 | 1 / 0.010 | 0.983 | 0.643 | 0.514 |
| M0 | Q99 | 23 | 3 / 0.130 | 0.979 | 0.234 | 0.158 |
| M1 | Q90 | 326 | 77 / 0.236 | 0.933 | 0.797 | 0.824 |
| M1 | Q95 | 198 | 34 / 0.172 | 0.975 | 0.897 | 0.685 |
| M1 | Q97.5 | 99 | 14 / 0.141 | 0.985 | 0.594 | 0.771 |
| M1 | Q99 | 23 | 0 / 0.000 | 0.981 | 0.188 | undefined |

`development_boundary_metrics.csv` also gives recall, precision, specificity, NPV, false-unsafe count/rate, and full missed-exceedance severity. Q99 must be read with its positive count.

### Paired state-level bootstrap

1,000 fixed-seed replicates resample states, keeping each state's 21 actions together. Four states per episode mean these intervals do not fully account for episode clustering.

| M1 − M0 metric | Paired point difference | 95% percentile CI |
|---|---:|---:|
| within state Spearman | -0.009 | [-0.061, 0.001] |
| pairwise accuracy | -0.031 | [-0.080, 0.014] |
| safe choice regret N | -0.013 | [-0.039, 0.007] |
| Q95 false safe rate | 0.096 | [0.000, 0.300] |
| Q95 AUPRC | -0.007 | [-0.090, 0.054] |

## Untouched confirmation: seeds 92000–92029

The frozen model/protocol existed before these 30 episodes were opened: **True**. Exactly 120 states and 2520 fresh candidate labels were evaluated, with no retraining or recalibration.

| Model | Candidate RMSE N | Median state Spearman | Mean pairwise accuracy | Mean safe-choice regret N | Q95 false-safe rate | Q95 AUROC | Q95 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 1.202 | 0.575 | 0.660 | 0.109 | 0.232 (71/306) | 0.992 | 0.953 |
| M1 | 1.428 | 0.582 | 0.660 | 0.102 | 0.422 (129/306) | 0.976 | 0.886 |

| Model | Q97.5 positives | Q97.5 false-safe count/rate | Q97.5 AUROC | Q97.5 AUPRC | Top-3 high/low containment |
|---|---:|---:|---:|---:|---:|
| M0 | 169 | 47 / 0.278 | 0.986 | 0.766 | 0.278 / 0.344 |
| M1 | 169 | 69 / 0.408 | 0.980 | 0.820 | 0.247 / 0.394 |

Q99 has only 65 positive confirmation candidates; both models miss 38/65 at the raw Q99 reference. These extreme-tail estimates are descriptive only. Full Q90/Q95/Q97.5/Q99 diagnostics, candidate predictions, and state metrics are saved separately.

## Interpretation and limitations

The frozen Generic latent and action contain learnable one-step force signal: M0 achieves development behavior R2 0.648, and both models show nonrandom candidate ranking. Counterfactual supervision is not demonstrated to be necessary or reliably beneficial under this Generic protocol. Validation favored M1 ranking, but development-test paired intervals did not establish a ranking or regret gain, and M1 worsened Q95 false-safe behavior. On untouched confirmation, M1 and M0 have essentially identical within-state ranking and pairwise accuracy, while M1 has larger candidate RMSE and misses 129 of 306 true Q95 exceedances versus 71 for M0. Its mean missed-exceedance severity is 2.823 nominal N, with a maximum of 10.954 nominal N above Q95. Q97.5 has 169 confirmation positives and shows the same boundary concern; Q99 has 65 positives and remains descriptive. The validation-selected M1 is therefore not ready to define a safe action set or motivate shield integration from this evidence. The historical LCP counterfactual-ranking improvement did not reproduce robustly under Generic; simulator-specific nominal-N boundaries must not be compared numerically. No further model or hyperparameter changes are made after confirmation.

Historical LCP Phase 2D found weak behavior-only within-state candidate ranking and improved ranking with counterfactual supervision, but residual false-safe/tail and action-selection errors blocked a direct safety claim. The Generic qualitative pattern is compared here only as a methodological contrast; LCP and Generic nominal-N references are not numerically interchangeable.

This G3 study establishes offline one-step prediction only. It does not establish closed-loop safety, task success under intervention, calibrated physical force, or clinical safety boundaries. No shield/CBF or training of TD-MPC2 was performed.
