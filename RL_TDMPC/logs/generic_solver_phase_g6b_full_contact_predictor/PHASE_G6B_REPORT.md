# Phase G6B — full pre-action contact-aware force predictor

**Classification: B — improved boundary behavior, still not reliable enough for an offline shield study.** The frozen shared cutoff met the *empirical* Q95 false-safe target on the new 98000–98029 confirmation set for all three independent replicas: `14/335 = 4.18%`, `7/335 = 2.09%`, and `0/335 = 0%`. It did so with a large `1.579396897690497` nominal-N margin, only `78.57%–80.20%` candidate acceptance, and **20/120 (16.67%) states with no accepted action for every replica**. Nonzero false-safe actions remain, including a maximum `1.534` nominal-N excess over the Q95 reference on confirmation and a `5.968` nominal-N excess on calibration. Do not treat this empirical calibration as a validated safety guarantee. No further feature ablations are justified under this phase; a different modeling formulation would be needed before reconsidering a shield.

## Frozen system and one full input

- Current branch/worktree: `feature/generic-contact-force-safety`; GenericConstraintSolver physics, TD-MPC2 navigation step-280000 checkpoint SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`, force reconstruction, ~3-mm distal region, 22-SOFA-substep next-interval maximum, task/reward/action/observation, and G3/G5 21-action protocol unchanged.
- Sole target `F_next(d)` is the maximum distal contact-force magnitude in the immediately following RL interval. The primary statistical reference is exact G2 Q95 `F_lim = 3.685331040756903` **nominal simulator N**, not a clinical threshold. G2 Q99 is `7.466568793559142` nominal N.
- The one prespecified 306-dimensional pre-action state is frozen TD-MPC2 latent `Z[256]`, sorted G5 `K+D` physical indices `0–29,35–37` `[33]`, and 17 current contact/wall features. The appended proposed action gives the 308-dimensional MLP input. The 17 features are current distal force vector/magnitude, current contact indicator/count, tip and five-node wall distances (absolute and signed), local outward wall normal, tip velocity toward that normal, tip-tangent/wall-normal orientation, and current force along that normal. See `full_contact_feature_definition.json` for exact order/units/source. No future or post-action quantity enters the input.
- The static closed AorticArch mesh is frozen at seed 30/scaling `[1,1,1,0.75]`; its parsed points/faces canonical SHA256 is `934640044a63635e2f17bdb2548e6f082b1b2d492ee8231b3533fd82e66ccb9f`. The same geometry extraction passed finite/defined checks for all **15,300** saved pre-action G4 P rows before fitting. The exact current force and contact count depend on Generic solver reconstruction: they are **simulation-only measurements**, not validated hardware force sensors. Shape, velocity and wall relation also need validated tracking/segmentation before physical deployment.
- Three independent behavior-only `308-64-ReLU-64-ReLU-1` scalar MLPs were trained from scratch with the exact G4/G5 raw-force MSE, optimizer, training/validation split, validation selection, normalization rule and seeds `260206/260207/260208`. Training used 10,903 behavior rows and validation 2,286; selected epochs were `27/26/33`. No candidate labels were used for training or checkpoint selection. Checkpoint hashes and normalization hash are in `model_manifest.json`. CUDA runtime was `steve_tdmpc`, Python `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, torch `2.7.1+cu128`, CUDA `12.8`, `CUDA_VISIBLE_DEVICES=0`, visible device `cuda:0` NVIDIA GeForce RTX 5090. There was no CPU fallback.

## Frozen development candidates

All three replicas were tested on the same 120 previously frozen development states/2,520 candidate labels. Q95 had 198 true-positive actions; Q99 had 23. Historical G5 `Z+K+D` is a read-only comparison, never retrained here.

| Training seed | Full-input RMSE N | MAE N | Median within-state Spearman | Pairwise accuracy | Mean safe-choice regret N | Q95 false-safe | Q95 AUROC / AUPRC | Q99 false-safe | No-current-contact → Q95-high-force false-safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 260206 | .672 | .392 | .587 | .689 | .056 | 15/198 (7.58%) | .990 / .942 | 9/23 | 4/4 |
| 260207 | .651 | .382 | .591 | .621 | .105 | 15/198 (7.58%) | .993 / .958 | 9/23 | 4/4 |
| 260208 | .620 | .373 | .556 | .662 | .131 | 15/198 (7.58%) | .995 / .962 | 9/23 | 4/4 |

Full-input mean ± sample SD RMSE was `.647 ± .026` N (worst `.672`), pairwise accuracy `.657 ± .034` (worst `.621`), Q95 false-safe `7.58% ± 0` (worst `7.58%`), and no-contact emergence false-safe `100% ± 0` (worst `100%`). Historical G5 `Z+K+D` on those same development labels had RMSE `1.089/.863/.961` N, Q95 false-safe `17.17%/7.58%/7.58%`, and the same `4/4` no-contact misses for each seed. Thus regression improved and one seed's Q95 rate improved, but the prospective-contact failure was not fixed on development. Full detail and historical rows are in `development_metrics.csv` and `training_seed_robustness.json`.

## Calibration — seeds 97000–97029 only

The 30 ordered behavior episodes used **one reused Generic scene** (27 successes, 2,021 parent steps). A fresh reused scene replayed every episode in order, selecting exactly four states/episode and forking the exact 21 candidate actions once per state: 120 states/2,520 labels. Observation, applied action, reward, tip, insertion, next-force, and pre-action previous-final-force replay/fork differences were all **zero**; see `calibration_replay_audit.json`. The three replicas scored identical labels.

Calibration had 348 true Q95-high-force candidate actions. Raw cutoff `F_hat <= F_lim` failed the all-three rule: Q95 false-safe counts/rates were `104/348 = 29.89%`, `78/348 = 22.41%`, `61/348 = 17.53%` for seeds 260206/7/8. Candidate acceptance was `89.60%/88.61%/87.46%`. No-current-contact→new-high-force raw misses were `2/37`, `1/37`, `19/37` respectively. The raw behavior is materially seed-sensitive.

The predeclared exact-event sweep selected one shared cutoff using only these labels: **`c_star = 2.105934143066406` nominal N**, **`Delta = 1.579396897690497` nominal N**. The next failing prediction breakpoint is `2.1059341430664062`; do not round the stored float64 cutoff to a coarser value in any future decision implementation. At the frozen cutoff:

| Training seed | Q95 false-safe | Candidate acceptance | States with ≥1 accepted / zero accepted | Median accepted actions | No-contact → high-force false-safe | Maximum missed excess above Q95 N |
|---|---:|---:|---:|---:|---:|---:|
| 260206 | 17/348 (4.89%) | 79.37% | 105/120 / 15/120 | 21 | 1/37 | 3.869 |
| 260207 | 17/348 (4.89%) | 79.33% | 100/120 / 20/120 | 21 | 1/37 | 5.968 |
| 260208 | 6/348 (1.72%) | 79.13% | 99/120 / 21/120 | 21 | 1/37 | 1.193 |

The largest missed calibration true force was `9.653` nominal N (seed 260207). All cutoffs/per-replica decision metrics are in `cutoff_sweep.csv`; severity, prospective-contact misses and state availability are in the adjacent calibration JSON files. The median 21/21 accepted actions is misleading by itself: rejection is concentrated in a significant minority of states.

## Pre-confirmation freeze and untouched confirmation

`pre_confirmation_freeze.json` passed with **157 file SHA256 identities** before any seed 98000–98029 was opened. It binds the complete feature definition/extractor, three model checkpoints, normalization, G2 reference, selected cutoff, calibration labels/metrics, state/action protocol and evaluation code. The cutoff and models were unchanged afterward.

The 30 confirmation behavior episodes used one ordered reused scene (27 successes, 2,389 parent steps). Their 120 selected states yielded 2,520 new one-time candidate labels, scored by all three frozen replicas. The independent confirmation replay/fork audit again found **zero** differences in all checked fields over 2,389 parent steps and 2,520 candidate forks. Confirmation had 335 Q95-positive and 73 Q99-positive actions.

| Training seed | Candidate RMSE / MAE N | Median Spearman / pairwise accuracy | Raw Q95 false-safe | Frozen-cutoff Q95 false-safe | Frozen-cutoff severity mean / max excess N | Q95 AUROC / AUPRC | Q99 false-safe | Acceptance / states with any action / zero-action states |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 260206 | .800 / .424 | .603 / .679 | 47/335 (14.03%) | 14/335 (4.18%) | .764 / 1.534 | .988 / .955 | 0/73 | 80.20% / 100/120 / 20/120 |
| 260207 | .801 / .407 | .523 / .648 | 31/335 (9.25%) | 7/335 (2.09%) | .395 / .697 | .994 / .966 | 0/73 | 78.61% / 100/120 / 20/120 |
| 260208 | .829 / .415 | .589 / .647 | 40/335 (11.94%) | 0/335 (0%) | none | .994 / .966 | 0/73 | 78.57% / 100/120 / 20/120 |

At the frozen cutoff, Q95 unsafe recall was `95.82%/97.91%/100%`, and median accepted actions per state was 21 for each replica despite **16.67%** of states having none. Q99 was descriptive, not a separately selected boundary. All 60 no-current-contact→next-Q95-high-force candidate actions were rejected by each replica at the frozen cutoff (`0/60` false-safe for all); raw misses were `16/60`, `0/60`, `19/60`. See `confirmation_metrics.csv` and `confirmation_contact_emergence.csv` for independent per-replica detail. The 0/60 result is encouraging but its candidate-level 95% exact upper bound is about 5.96%, and candidates within states are correlated.

## Comparison and decision

1. **Q95 false-safe error:** On matched development labels, full contact input improved one historical G5 seed and tied two; regression RMSE improved all. On different untouched confirmation seed sets, frozen-cutoff G6B's worst Q95 rate was `4.18%` versus historical G6A's `5.86%`. This is a descriptive, not paired, improvement. G6B raw confirmation rates were **not** consistently below historical G6A raw rates.
2. **Training-seed variability:** G6B confirmation frozen-cutoff rates ranged `0%–4.18%`, compared with G6A's `1.30%–5.86%`; the spread narrowed only modestly and the comparison is unpaired. Raw calibration varied widely (`17.53%–29.89%`), so the larger margin is doing much of the boundary work.
3. **Prospective contact:** Frozen-cutoff G6B missed `0/60` newly high-force actions for all replicas on confirmation, versus G6A's historical `0/58`, `11/58`, `14/58` at its cutoff. However, full G6B still missed `4/4` on development before calibration, and raw confirmation missed `16/60` and `19/60` for two seeds. The improvement is conditional on the much more conservative cutoff.
4. **Margin and availability:** The needed margin **increased** from historical G6A `.709` to G6B `1.579` nominal N. Candidate acceptance fell from historical G6A `85.28%–86.27%` to G6B `78.57%–80.20%`; zero-action states worsened from 15–16/120 to **20/120** for every replica. There is no fallback action protocol or online shield here.
5. **Decision B:** The empirical Q95 target and no-contact stratum pass on this one untouched set, but large calibration/confirmation margins, substantial zero-action state mass, nontrivial false-safe severity and simulation-only current-contact inputs do not support declaring the predictor ready for an offline action-shield study as a reliable boundary. The exact binomial intervals and state clustering add uncertainty. Stop further G6B feature engineering; reconsider only with a separately scoped change of modeling formulation and predeclared validation.

**STOP:** No Safety Head, CBF, online/offline shield, TD-MPC2 retraining, physics change, force-metric change, or additional feature ablation was performed.
