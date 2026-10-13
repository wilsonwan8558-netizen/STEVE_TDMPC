# Phase G5 — Privileged feature attribution and deployable-state study

**Result: B — DEPLOYABLE FEATURES HELP, BUT REMAIN INSUFFICIENT.** This is a simulator-only feature-attribution result, not authorization to use a force predictor as a safety controller. Frozen A/B features improve Q95 prediction, especially together with TD-MPC2 latent state, but Q95 and Q99 missed-force rates remain material. Most of the selected 33 physical coordinates are only *estimable in principle*, not measured by the present controller.

## Protocol and identity

- Branch: `feature/generic-contact-force-safety`; no controller, solver, reward, task, observation, termination, force-target or checkpoint change. GenericConstraintSolver, projected Gauss–Seidel, tolerance `1e-4`, 2000 iterations, `computeConstraintForces=True`, friction `0.001`, SOFA `dt=0.006 s`.
- Navigation checkpoint: step 280000, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`.
- Sole target: maximum reconstructed distal (~3 mm) guidewire–vessel contact-force magnitude over *all 22* SOFA substeps in the immediately following RL control interval, from summed `J_contact^T lambda / 0.006 s` vector contributions before norm. Units are **nominal simulator N**.
- G2 reference levels: Q90 `2.1265156799878246`, Q95 `3.685331040756903`, Q97.5 `5.926250464744108`, Q99 `7.466568793559142` nominal N. These are distribution descriptors, **not clinical safety thresholds**.
- G4 source/alignment and its 1,121-file freeze verified. The exact G5 Z/P train and validation behavior matrices and train candidate matrices matched G4 bitwise. Retrained Z, P, Z1 and P1 selected the same epoch/loss and had zero maximum weight difference from G4 for all three seeds. See `g4_reproduction_audit.json`, `reference_model_reproduction.json` and the preserved G4 artifacts.
- CUDA: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, torch `2.7.1+cu128`, CUDA `12.8`, `CUDA_VISIBLE_DEVICES=0`, visible `cuda:0 = NVIDIA GeForce RTX 5090`. The sandbox could not access the GPU, so approved CUDA runs used this explicit device; no CPU fallback.
- Exact G4 data reuse: 200 development behavior episodes, 15,500 parent steps, 15,300 `d>=1` physical-state rows; train/validation/development-test episodes `140/30/30`, behavior rows `10,903/2,286/2,111`; candidate states `560/120/120`, each with the same 21 G3 actions (`11,760/2,520/2,520` candidate labels). The 93000-series G4 confirmation was not the G5 confirmation.
- All models used the G4 `input-64-ReLU-64-ReLU-1` network, raw nominal-N MSE, Adam `lr=0.001`, batch 512, train-behavior mean/std with zero-std fallback 1, no target normalization, 60 maximum validation rounds, 8-round patience, lowest validation *behavior* MSE selection, and training seeds `260206, 260207, 260208`. Each validation round drew 10,903 behavior rows without replacement for a matched optimizer budget. Primary models used behavior-only supervision. Only Z1, P1 and P_min1 additionally used the exact 11,760 G3 train counterfactual labels after P_min selection.

## Exact G4 P inventory and predeclared semantic/deployability groups

All 38 P coordinates are float32 and extracted after the preceding control interval, **before** action `a_d` at `d>=1`. They are candidate-action independent. They contain no following-interval target or post-action contact. Model inputs append the proposed two-dimensional action *after* Z/P features. Each coordinate is normalized by its own train-behavior mean/std, recorded exactly in `privileged_feature_inventory.json`.

| P indices | Feature (dimension) | Source and frame | Group | Class |
|---|---|---|---|---|
| 0–2 | Distal tip xyz (3) | `CollisionDOFs.position[-1]`, global SOFA mm | K | B |
| 3–14 | Four distal-node offsets from tip (12) | `CollisionDOFs.position[-5:-1] - tip`, tip-relative SOFA mm | D | B |
| 15–29 | Five distal-node xyz velocities (15) | `CollisionDOFs.velocity[-5:]`, global scene mm/s | K | B |
| 30–32 | Current distal contact-force vector (3) | Generic multiplier/constraint reconstruction, global nominal N | C | C |
| 33 | Current distal contact-force magnitude (1) | Norm of current solver-reconstructed force, nominal N | C | C |
| 34 | Current distal contact-group count (1) | Generic current constraint manifold, count | C | C |
| 35 | Insertion length (1) | `inserted_lengths[0]`, SOFA mm | K | A |
| 36–37 | Rotation sin/cos (2) | Current device rotation / existing policy observation, dimensionless | K | A |

K = kinematics/pose/device state (21D); D = distal deformation/shape (12D); C = current solver-contact state (5D). Each scalar belongs to exactly one group. **No W/wall-relative feature exists in G4 P**, so wall geometry cannot be attributed in G5. A means already available from controller/tracking/command state; B means plausibly estimable or measurable in principle but *not presently measured in this exact form*; C requires Generic solver internals. Exact current force and contact count are C even if a future physical sensor might provide a different surrogate. `P_deployable = K+D` is 33D and excludes all C. The 2D tracking observation is not equivalent to the 3D tip/shape/velocity B coordinates. See frozen `semantic_feature_groups.json` and `deployability_classification.json` for per-feature rationale and exact indices.

## Validation-only P_min selection

All meaningful nonempty deployable group subsets (K, D, K+D) were trained under the same three-seed behavior-only protocol. The *predeclared* lexicographic rule used three-seed mean validation Q95 false-safe rate, candidate RMSE, negative pairwise accuracy, safe-choice regret, then group count/dimension, with **no development-test or confirmation labels**. For Z+subset, validation Q95 false-safe rates were K `0.2073`, D `0.3056`, K+D `0.1891`; corresponding candidate RMSEs were `1.0543`, `1.1572`, `0.9925` nominal N. Thus **P_min = K+D = P_deployable**, indices `0–29,35–37`, 33D. It is the minimal *tested group-level* winner, not a verified minimal scalar set. P_min alone is `PKD`; Z+P_min is `ZP_noC`. See `deployable_subset_metrics.csv` and `minimal_subset_selection.json`.

## Development feature attribution

Three-seed means below are on the identical 120 held-out development states / 2,520 candidate labels. Lower RMSE, false-safe rate and regret are favorable; higher pairwise accuracy is favorable. Full per-seed behavior, candidate, state, Q90/Q95/Q97.5/Q99 confusion, AUROC/AUPRC, top-3 high/low containment and contact-stratified metrics are in the corresponding CSVs.

| Model | RMSE N | Pairwise | Regret N | Q95 false-safe |
|---|---:|---:|---:|---:|
| Z | 1.101 | .632 | .061 | .200 |
| P | .634 | .624 | .125 | .096 |
| Z+P | .752 | .632 | .097 | .076 |
| Z+K | .956 | .634 | .088 | .116 |
| Z+D | 1.079 | .645 | .109 | .178 |
| Z+C | .814 | .661 | .075 | .143 |
| Z+P without K | .814 | .626 | .109 | .082 |
| Z+P without D | .722 | .621 | .109 | .077 |
| Z+P without C = Z+P_min | .971 | .641 | .094 | .108 |
| P_min alone | .782 | .611 | .138 | .091 |
| Z1 / P1 / P_min1 | .959 / .610 / .707 | .642 / .686 / .658 | .064 / .093 / .091 | .175 / .152 / .140 |

Paired effect signs are **right minus left**. Addition (`Z+group − Z`) Q95 false-safe changes were K `−.084`, D `−.022`, C `−.057`; RMSE changes K `−.145`, D `−.022`, C `−.287` N. Removal (`Z+P without group − Z+P`) Q95 changes K `+.007`, D `+.002`, C `+.032`; RMSE changes K `+.062`, D `−.030`, C `+.219` N. The 1,000 fixed-seed paired **state-level** bootstrap retained all 21 actions per sampled state and averaged matched effects over three independently trained seeds; see `feature_group_bootstrap.json` for every effect, per-seed SD, and percentile CI. For example, removal of C increased development RMSE `+.219 N`, 95% state-bootstrap CI `[+.106,+.328]`; its Q95 false-safe increase was `+.032`, CI `[0,+.111]`. K addition improved Q95 by `−.084`, CI `[−.238,+.008]`; uncertainty includes zero. This is not evidence of a clean single-group Q95 ranking.

## Frozen untouched confirmation

The 187-hash `pre_confirmation_freeze.json` was written and verified **before** opening seeds 94000–94029. A single reused Generic scene evaluated those 30 behavior episodes in order: 27 successes, 2,058 steps (mean 68.6), mean reward 3.5496. A new scene replayed the same ordered behavior for candidate forks. All 2,058 replayed parent steps and 2,520 candidate forks passed the G2/G4 parent–behavior-fork checks; maximum differences were **zero** for observation, applied action, reward, tip position, insertion length, next-interval peak force, and pre-action previous-final force. `confirmation_replay_audit.json` records this. Exactly 4 states/episode and 21 candidates/state yielded 120 states and 2,520 untouched labels. Frozen code/data/model hashes still verify after inference.

| Frozen model | Candidate RMSE N | MAE N | Median Spearman | Pairwise | Regret N | Q95 false-safe | Q95 AUROC / AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z | 1.105 | .628 | .492 | .621 | .165 | .293 | .975 / .894 |
| P | .742 | .409 | .467 | .645 | .112 | .212 | .989 / .954 |
| Z+P | .792 | .411 | .511 | .622 | .145 | .089 | .990 / .952 |
| Z+K | .935 | .507 | .469 | .618 | .158 | .140 | .988 / .944 |
| Z+D | 1.034 | .558 | .422 | .620 | .154 | .247 | .980 / .907 |
| Z+C | .794 | .426 | .517 | .633 | .146 | .181 | .984 / .928 |
| Z+P without K | .803 | .428 | .548 | .640 | .143 | .168 | .986 / .940 |
| Z+P without D | .815 | .389 | .523 | .634 | .132 | .120 | .987 / .934 |
| Z+P without C = Z+P_min | .874 | .445 | .537 | .636 | .155 | .108 | .991 / .954 |
| P_min alone | .935 | .537 | .371 | .600 | .157 | .182 | .990 / .957 |
| Z1 | 1.145 | .619 | .453 | .617 | .136 | .296 | — |
| P1 | .805 | .405 | .602 | .695 | .104 | .209 | — |
| P_min1 | .820 | .458 | .519 | .662 | .105 | .212 | — |

Three-seed means; **375** Q95-positive actions in the same 2,520-action confirmation set. For Z, P, Z+P, P_min, and Z+P_min, mean false-safe counts per seed were `110.0`, `79.3`, `33.3`, `68.3`, `40.7`. The table is not an ensemble: each of three independently trained models is evaluated separately and metrics averaged. Full per-seed counts, true/false positives, specificity, NPV, precision, four-reference AUROC/AUPRC and top-3 containment are in `confirmation_metrics.csv`, `confirmation_boundary_metrics.csv`, `confirmation_state_metrics.csv` and `confirmation_candidate_results.csv`.

At the higher Q99 reference (120 positive actions), mean false-safe rates were Z `.914`, P `.458`, Z+P `.475`, P_min `.833`, Z+P_min `.675`. Thus Q95 improvement **does not establish adequate tail safety**. Confirmation group effects also differ by metric: adding K to Z reduced Q95 false-safe rate by `.153` versus `.112` for C and `.046` for D; removing K from Z+P raised it by `.079`, versus `.019` for C and `.031` for D. Conversely, C produced the largest RMSE gain when added to Z (`−.311 N`) and the largest RMSE degradation when removed from Z+P (`+.082 N`). C is important simulator-internal information, but it does **not** uniquely dominate the Q95 boundary on these new seeds.

## Current-contact dependence and emerging high force

Current contact is defined *before* candidate execution by P[34] > 0. Development contained 59 no-contact and 61 contact states; the no-contact stratum had only **4** Q95-positive candidate actions, which cannot support a stable event-rate conclusion. Confirmation contained 58 no-contact and 62 contact states, with **22** and **353** Q95-positive actions respectively. The 22 are `no-current-contact → next-interval-Q95-high-force` events, not a new input feature.

| Model | Confirmation no-contact Q95 false-safe (22 events) | Current-contact Q95 false-safe (353 events) |
|---|---:|---:|
| Z | .455 | .283 |
| P | .939 | .166 |
| Z+P | .212 | .081 |
| Z+K | .182 | .138 |
| Z+C | .455 | .164 |
| P_min | .636 | .154 |
| Z+P_min | .182 | .104 |
| P1 / P_min1 | 1.000 / .803 | .160 / .175 |

Z+P_min did improve emerging-event recall relative to Z on confirmation, but still missed **4/22 events for each of its three seeds**. P without latent context was particularly poor on those events (mean `.939` false-safe), despite good overall RMSE. The near-absence of such events in development and their much larger confirmation count are distributional caveats; inspect `contact_conditioned_metrics.csv`, `no_contact_emergence_metrics.csv`, `confirmation_contact_conditioned_metrics.csv` and `confirmation_no_contact_emergence_metrics.csv` before relying on any rate.

## Interpretation and stopping decision

1. **Most useful addition to Z:** K for Q95 false-safe on both development and confirmation; C for candidate RMSE and development ranking. D alone had modest/unstable benefits. Neither a larger input nor one group is universally better across metrics.
2. **Largest removal cost:** C for RMSE; K for Q95 false-safe on untouched confirmation. Development Q95 removal of C was largest but uncertain by state bootstrap, so there is no consistent single-group Q95 dominance.
3. **Current solver-contact state:** materially useful for force magnitude regression, especially once contact exists; it is unavailable from the current physical observation stack. The best Q95 confirmation rate was Z+full P `.089`, but removing C retained `.108`, so C is not an absolute requirement for most of that Q95 improvement in this set.
4. **Kinematics and shape:** K materially helps; D alone is weak, but K+D wins the predeclared validation subset search. No wall-relative geometry was present in G4 P, so G5 cannot answer its value.
5. **Deployability:** P_min = K+D approaches Z+full P at Q95 only when combined with Z (`.108` vs `.089`), but has worse RMSE (`.874` vs `.792 N`), weak Q99 detection (`.675` false-safe), and contains 30 B-class coordinates not currently measured in exact 3D form. P_min alone is worse (`.182` Q95, `.833` Q99 false-safe). This supports classification **B**, not an identified ready-to-deploy minimal state.
6. **Counterfactual supervision:** P1 and P_min1 improved confirmation ranking and regret over P and P_min, but did not improve Q95 false-safe detection; P1 missed all 22 new-contact Q95 events in each seed. It does not remedy the boundary problem.
7. **Within-K exploratory sensitivity and granularity limit:** K combines tip position, distal velocities, insertion and rotation. After the freeze, a *development-only, no-retraining* mean-masking check removed one K family at a time from frozen P_min and Z+P_min inputs. For Z+P_min, masking velocities increased candidate RMSE by `.140 N` and Q95 false-safe rate by `.140` averaged over three seeds; masking tip, insertion or rotation did not increase Q95 false-safe rate. For P_min alone, all four masks raised RMSE, and insertion/tip masks especially raised Q95 false-safe rate (`+.492`/`+.286`). These are **not controlled retrained leave-family-out models**: mean masking breaks feature correlations and may create out-of-distribution states. The outputs are explicitly post-freeze exploratory diagnostics (`exploratory_postfreeze_K_family_mean_masking.csv/.json`) and were **not** used to alter P_min, models, or confirmation interpretation. The predeclared group-level search did not establish an indispensable K scalar family; a controlled family ablation would require a separately predeclared study with new untouched data. The 33D K+D winner is therefore *group-minimal among tested subsets*, not a compact scalar optimum.

**Stop after G5.** No Safety Head, CBF, shield, controller modification, retraining on confirmation, calibration, solver change, or new force metric was performed. The next evidence need is a separately designed physical-state sensing/estimation and tail-event study, not deployment of this predictor as a safety mechanism.
