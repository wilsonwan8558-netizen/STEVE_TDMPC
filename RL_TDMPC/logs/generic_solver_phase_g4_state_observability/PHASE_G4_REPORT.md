# Phase G4 — Generic force observability and state sufficiency

**Classification: B. PARTIAL OBSERVABILITY / MISSING PHYSICAL STATE, with important predictor limitations.** The diagnostic privileged physical state P consistently improves force-level regression and Q95 boundary fidelity on development-test and untouched confirmation data. Counterfactual-supervised P1 also has the best candidate ranking on both sets. Raw observation O is worse than the frozen navigation latent Z, so the evidence does **not** support a latent-information-loss bottleneck. Two-step history H helps mainly when paired with counterfactual supervision, but is not consistently competitive with Z or P. P is not deployable by assumption and none of these regressors is a validated safety filter.

## Frozen experiment and alignment

- Navigation: GenericConstraintSolver, frozen TD-MPC2 `step_280000.pt`, SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`. Runtime: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, PyTorch 2.7.1+cu128, CUDA 12.8, `CUDA_VISIBLE_DEVICES=0`, visible `cuda:0` = NVIDIA GeForce RTX 5090. See `training_runtime.json`.
- Target: **only** `force_next_max_N`, the maximum distal-guidewire (approximately 3 mm) Generic contact-force magnitude over the 22 SOFA substeps immediately after candidate action `a_d`. It uses the frozen G2 vector-sum-before-norm `J_contact^T λ / 0.006 s` reconstruction, in nominal N. Q90 = 2.1265156799878246, Q95 = 3.685331040756903, Q97.5 = 5.926250464744108, Q99 = 7.466568793559142 nominal N. These are statistical references, not clinical thresholds.
- G3 episode split and candidate labels were reused unchanged: 140 train / 30 validation / 30 development-test episodes. The common `d >= 1` rule excludes the first decision of each of 200 episodes for **all** representations. Matched behavior rows: train 10,903, validation 2,286, development-test 2,111; total 15,300. Candidate states/labels: train 560/11,760, validation 120/2,520, development-test 120/2,520. All cells see identical eligible states, actions and labels.
- P was extracted before `a_d` from one reused scene across ordered seeds 90000–90199. Alignment reproduced all 15,500 G2 parent steps, all 800 selected G3 states and **all 16,800 G3 candidate forks**. Maximum checked differences in observations, actions, reward, distal maximum force, tip position, insertion length and prior final force were all **0.0**. The 15,300 pre-action P rows are saved by episode. The initial 12-state physical-source probe also passed. See `physical_source_probe.json` and `physical_state_alignment_audit.json`.

## Representations and fitting

All models append the same normalized 2-D candidate action. `Z` is the exact frozen G3 256-D TD-MPC2 latent; `O` is the exact 14-D policy observation; `H` is exactly `[o_(d−1), o_d, a_(d−1)]` (30-D, same episode); `P` is the 38-D diagnostic pre-action SOFA vector. P contains distal collision-tip xyz position (3), four preceding distal-node positions relative to the tip (12), five distal-node velocities (15), the current distal contact-force vector from the *preceding interval’s final substep* (3), its magnitude (1), current distal contact-group count (1), insertion length (1), and sine/cosine of rotation (2). The xyz tip coordinate is in global SOFA mm; offsets are tip-relative mm, velocities are scene mm/s, and forces are nominal N. No candidate rollout, next state or next-step force enters P. Collision topology, finite values and previous-final-force alignment were audited. Vessel distance and variable-cardinality contact normals were omitted because no reliable compact fixed-dimensional extractor was established; this limits what a weak P result could imply. See `privileged_feature_definition.json`.

Suffix `0` uses train behavior labels only; suffix `1` adds the same 11,760 train candidate labels. Each of Z0/Z1/O0/O1/H0/H1/P0/P1 was trained from scratch with seeds **260206, 260207, 260208**. The architecture is input–64–ReLU–64–ReLU–1; raw nominal-N MSE, Adam 1e−3, zero weight decay, batch 512, at most 60 validation rounds, patience 8 and minimum validation-MSE improvement 1e−4. Each model receives 10,903 sampled examples and the same number of optimizer updates per round. Normalization is fitted on **train behavior only** and is identical across suffixes and seeds of a representation. Lowest validation behavior MSE selects each checkpoint; no development-test or new confirmation result selects a checkpoint. The Z features, target labels and archived candidate behavior were matched to G3 exactly; G4 Z fit values need not numerically equal G3 M0/M1 because G4 excludes all `d=0` rows and uses three training seeds.

Candidate tables below are **mean ± sample SD across three training seeds**; the behavior table shows means for compactness. Mean, sample SD and individual-seed values for every metric are in `development_three_seed_summary.json` and `confirmation_three_seed_summary.json`; per-state metrics and all four detailed boundary confusion/ranking measures are in the CSV artifacts.

## Development-test behavior and candidate results

Behavior regression (2,111 held-out transitions; behavior Q95/Q99 positives = 105/20):

| Cell | RMSE N | MAE N | R² | Pearson | Spearman | Q95 false-safe rate | Q99 false-safe rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 0.959 | 0.490 | 0.580 | 0.770 | 0.722 | 0.435 | 0.833 |
| Z1 | 0.909 | 0.464 | 0.623 | 0.796 | 0.726 | 0.457 | 0.767 |
| O0 | 1.039 | 0.525 | 0.507 | 0.712 | 0.675 | 0.638 | 1.000 |
| O1 | 1.050 | 0.499 | 0.497 | 0.706 | 0.717 | 0.594 | 1.000 |
| H0 | 1.007 | 0.522 | 0.537 | 0.736 | 0.682 | 0.521 | 1.000 |
| H1 | 1.010 | 0.506 | 0.534 | 0.734 | 0.700 | 0.543 | 0.867 |
| P0 | **0.649** | **0.290** | **0.808** | **0.899** | **0.843** | **0.238** | 0.533 |
| P1 | 0.700 | 0.307 | 0.776 | 0.884 | 0.832 | 0.270 | **0.450** |

Candidate actions (120 identical states × 21 actions = 2,520 labels; **primary evaluation**):

| Cell | RMSE N | MAE N | Median within-state Spearman | Mean pairwise accuracy | Mean regret N | Q95 false-safe rate |
|---|---:|---:|---:|---:|---:|---:|
| Z0 | 1.101±0.087 | 0.626±0.047 | 0.480±0.066 | 0.632±0.025 | **0.061±0.007** | 0.200±0.108 |
| Z1 | 0.959±0.055 | 0.523±0.020 | 0.470±0.068 | 0.642±0.040 | 0.064±0.013 | 0.175±0.006 |
| O0 | 1.239±0.002 | 0.754±0.015 | 0.306±0.173 | 0.604±0.039 | 0.145±0.038 | 0.643±0.028 |
| O1 | 1.141±0.014 | 0.662±0.020 | 0.435±0.051 | 0.630±0.010 | 0.126±0.019 | 0.544±0.025 |
| H0 | 1.140±0.033 | 0.677±0.016 | 0.318±0.044 | 0.589±0.012 | 0.148±0.010 | 0.429±0.013 |
| H1 | 1.009±0.033 | 0.580±0.025 | 0.532±0.045 | 0.658±0.023 | 0.104±0.014 | 0.369±0.028 |
| P0 | 0.634±0.029 | 0.367±0.023 | 0.404±0.095 | 0.624±0.027 | 0.125±0.024 | **0.096±0.027** |
| P1 | **0.610±0.044** | **0.346±0.027** | **0.614±0.085** | **0.686±0.024** | 0.093±0.023 | 0.152±0.063 |

The mean/median pairwise accuracies, mean/median regrets, top-three high/low containment and full per-state ranks are in `development_candidate_metrics.csv` and `development_state_metrics.csv`. The median regret is zero for most cells because many states contain nearly tied low-force candidates; the **mean** regret is more discriminating. P1 has the highest mean top-three high/low containment, 0.312/0.372; Z1 is 0.263/0.350. No proposed action was executed by a shield.

Candidate false-safe rates at all frozen nominal-N references (positive counts Q90/Q95/Q97.5/Q99 = 326/198/99/23):

| Cell | Q90 | Q95 | Q97.5 | Q99 |
|---|---:|---:|---:|---:|
| Z0 | 0.267 | 0.200 | 0.239 | 0.478 |
| Z1 | 0.263 | 0.175 | **0.141** | **0.101** |
| O0 | 0.377 | 0.643 | 0.976 | 1.000 |
| O1 | 0.300 | 0.544 | 0.707 | 0.971 |
| H0 | 0.274 | 0.429 | 0.815 | 1.000 |
| H1 | 0.349 | 0.369 | 0.502 | 0.551 |
| P0 | **0.194** | **0.096** | 0.155 | 0.304 |
| P1 | 0.243 | 0.152 | 0.158 | 0.377 |

`development_boundary_metrics.csv` provides each seed’s true-unsafe and false-safe counts, false-unsafe counts, recall, precision, specificity, NPV, AUROC and AUPRC at **all four** references. Q99 has only 23 development positives and is descriptive. Q90/Q97.5/Q99 are not selection thresholds.

### Paired uncertainty and supervision interaction

`bootstrap_summary.json` contains 1,000 fixed-seed paired **state-level** resamples of the 120 states, retaining all 21 candidate actions, with the three model-training seeds kept separate before averaging metric differences. Selected development differences (second minus first; 95% between-state percentile intervals):

| Comparison | Median state Spearman difference | Mean pairwise difference | Q95 false-safe-rate difference |
|---|---:|---:|---:|
| O0 − Z0 | −0.174 [−0.343, 0.008] | −0.028 [−0.076, 0.018] | +0.443 [0.225, 0.650] |
| H1 − O1 | +0.097 [−0.081, 0.221] | +0.027 [−0.010, 0.069] | −0.175 [−0.341, −0.019] |
| P1 − Z1 | +0.144 [0.021, 0.227] | +0.044 [−0.007, 0.087] | −0.024 [−0.140, 0.081] |
| H1 − H0 | +0.213 [0.036, 0.419] | +0.069 [0.028, 0.114] | −0.061 [−0.152, 0.034] |
| P1 − P0 | +0.210 [0.069, 0.314] | +0.061 [0.027, 0.096] | +0.056 [0.000, 0.132] |

The full file also includes Z↔P, O↔H, H↔P and all suffix comparisons, safe-choice regret intervals, and between-training-seed SDs. Four states come from each episode, so state-level intervals do not fully capture episode clustering. Representation and supervision interact: counterfactual labels materially help H/P ranking, but P1's better ranking comes with **more** Q95 misses than P0. More accurate mean-force regression alone does not guarantee better action ranking or boundary fidelity.

### Exact-action state aliasing

This diagnostic is independent of the neural models and uses train-behavior normalization. A query is compared only with a different state having **the exact same float32 2-D candidate action**. Among all 16,800 development candidate rows, just **368 (2.19%)** have such a peer; the only repeated actions are `(−1,−1)`, `(−1,+1)`, `(+1,−1)`, `(+1,+1)`. Therefore it probes saturated action corners, not general action-space observability.

| Representation | Matched development rows | Median / Q90 / Q95 nearest force disagreement N | Fraction >1 N | Q95 cross-boundary fraction |
|---|---:|---:|---:|---:|
| Z | 368 | 0.137 / 1.506 / 2.838 | 0.171 | 0.049 |
| O | 368 | 0.138 / 2.037 / 3.025 | 0.185 | 0.060 |
| H | 368 | 0.137 / 2.024 / 3.561 | 0.179 | 0.065 |
| P | 368 | 0.084 / 1.320 / 2.738 | 0.144 | 0.041 |

Small-distance bins and all query/neighbor distances are in `aliasing_summary.json` and `nearest_neighbor_aliasing.csv`. The closest development decile has median zero disagreement for every representation; the sparse corner-only coverage cannot establish a general mathematical non-observability result. Raw normalized distances also differ in dimension and should not be compared directly across representations.

## Untouched confirmation: seeds 93000–93029

`pre_confirmation_freeze.json` fixed **1,121** source/data/feature/normalization/checkpoint/metric identities before these seeds were opened. The confirmation seed manifest links to its SHA256. One reused-scene behavior pass was followed by an ordered deterministic replay from a fresh scene. All **2,276** replayed parent steps and behavior forks matched; 120 states and **2,520** new candidate labels were obtained. Pre-action P versus preceding final force difference was 0.0. The frozen navigation policy succeeded in 27/30 episodes (mean 75.87 steps and 3.524 reward), with no simulation error; these are contextual task results, not effects of a safety controller. No model, feature, hyperparameter, normalization or reference level was changed after freeze. Q95 has **328** positive confirmation candidates.

| Cell | Candidate RMSE N | MAE N | Median state Spearman | Mean pairwise | Mean regret N | Q95 false-safe rate | Q95 AUROC | Q95 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 1.224±0.148 | 0.688±0.091 | 0.580±0.117 | 0.642±0.022 | 0.114±0.008 | 0.249±0.014 | 0.959±0.014 | 0.851±0.012 |
| Z1 | 1.071±0.063 | 0.570±0.033 | 0.606±0.034 | 0.651±0.011 | 0.110±0.009 | 0.265±0.008 | 0.966±0.001 | 0.862±0.011 |
| O0 | 1.434±0.003 | 0.857±0.012 | 0.297±0.077 | 0.610±0.033 | 0.174±0.023 | 0.408±0.025 | 0.930±0.003 | 0.753±0.004 |
| O1 | 1.339±0.022 | 0.765±0.027 | 0.400±0.069 | 0.614±0.015 | 0.166±0.020 | 0.330±0.009 | 0.941±0.003 | 0.789±0.016 |
| H0 | 1.302±0.068 | 0.738±0.046 | 0.076±0.049 | 0.559±0.006 | 0.312±0.014 | 0.349±0.028 | 0.942±0.009 | 0.773±0.027 |
| H1 | 1.250±0.024 | 0.671±0.010 | 0.414±0.058 | 0.631±0.032 | 0.139±0.034 | 0.323±0.006 | 0.948±0.004 | 0.791±0.032 |
| P0 | **0.784±0.048** | 0.428±0.022 | 0.501±0.058 | 0.665±0.012 | 0.129±0.018 | **0.144±0.007** | **0.982±0.001** | **0.920±0.008** |
| P1 | 0.867±0.075 | **0.424±0.010** | **0.630±0.029** | **0.701±0.010** | **0.097±0.010** | 0.185±0.002 | 0.981±0.001 | 0.898±0.012 |

Confirmation Q95 false-safe counts (individual seeds 260206/260207/260208, out of 328 positives): Z0 87/80/78; Z1 89/84/88; O0 131/143/127; O1 109/105/111; H0 109/109/125; H1 104/106/108; P0 **46/50/46**; P1 61/60/61. `confirmation_metrics.csv`, `confirmation_candidate_results.csv` and `confirmation_state_metrics.csv` preserve all individual-seed results and candidate predictions. P1 again has the highest top-three high/low containment, 0.312/0.388. Median state-level regret is zero for all except O0 and H0, while mean regret reveals larger differences.

The broad development ordering reproduced: P0/P1 have the lowest RMSE and Q95 false-safe rates; P1 has the best ranking, pairwise accuracy and mean regret; O remains worse than Z. **H0 did not generalize in ranking** (median Spearman 0.318 development versus 0.076 confirmation), so history alone is not a stable remedy. H1 outperforms H0 but does not consistently surpass Z. P0 has better raw Q95 fidelity than P1; supervision improves P candidate ranking while trading off boundary misses.

Confirmation aliasing is even more coverage-limited: 46/2,520 (1.83%) new rows have a different-state exact-action peer within confirmation, and 47/2,520 (1.87%) have one among frozen development states. Within-confirmation median nearest force disagreement is Z 0.135, O 0.133, H 0.111, P 0.138 nominal N; nearest-to-development medians are Z/O/H 0.038 and P 0.033 N. Q95 crossing and small-distance-bin details are in `confirmation_aliasing_summary.json`. These sparse corner-action matches do not independently establish general state sufficiency or insufficiency.

## Interpretation and stop

**Z versus O:** O's weaker development and confirmation force RMSE, ranking, regret and Q95 fidelity argue *against* the idea that the navigation latent has discarded the force signal available in the 14-D observation. Z may encode a useful transformed representation of O.

**O versus H:** H1 gains some ranking and development Q95 fidelity over O1, and counterfactual supervision strongly helps H ranking. H0's poor confirmation ranking and H1's failure to surpass Z consistently make short history a partial, unstable solution—not proof that two observations suffice for contact dynamics.

**H/Z versus P:** Privileged distal geometry, velocity and previous-contact state materially improve force-level prediction and Q95 discrimination on both sets. State information is therefore a **major**, but not the sole demonstrated, predictor bottleneck. P1 can use candidate labels for better action ranking than P0, but P0 is the more faithful raw-Q95 classifier. Z1 lowers candidate RMSE relative to Z0 without a stable Q95 boundary gain, consistent with G3's warning that counterfactual supervision alone is insufficient. The persistent false-safe cases (P0 about 14%, P1 about 19% of 328 Q95-positive confirmation actions) and nonzero regret mean P is **not** a safe-set estimator. P omits vessel geometry and full contact topology, so the study does not isolate which physical variable is causal or prove scalar regression is intrinsically inadequate.

The most supported next **information class to investigate** is measurable or estimable contact-relevant physical state: distal velocity/loading history, contact-force surrogate and local guidewire–vessel geometry. This is a research direction, not an implemented controller change. P is privileged SOFA state, not assumed deployable in a real procedure. No shield, CBF, navigation training, encoder fine-tuning, solver change, reward change, force-metric redesign or clinical safety claim was made. **Stop after Phase G4.**
