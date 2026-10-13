# Phase G8.5 — one-step continuous-action oracle feasibility

**Result: Case B, with a finite-search caveat.** The dense true-SOFA search found a safe one-step action in **5/16** G8 nominal-unsafe states. Four were already repairable in G8's 21-action set. Of the **12** old-set-infeasible states, **one (state 110) was a candidate-coverage false infeasible**; **11 still had no safe action found** after the 41×41 grid and deterministic local refinement. Thus sparse-candidate coverage explains **one**, not most, of G8's apparent infeasibility. The supported next methodological direction is **anticipatory multi-step safety planning**, not immediate online one-step shielding. A finite grid cannot prove that no safe point exists between samples, especially at near-boundary state 75.

## Frozen system, state recovery, and integrity

The study used the frozen GenericConstraintSolver, TD-MPC2 step-280000 checkpoint (SHA256 `476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99`), max-substep distal force reconstruction/approximately 3-mm distal region, and exact G2 reference **`F_lim = 3.685331040756903` nominal N**. No policy, physics, observation, reward, force target, or threshold changed. No learned force predictor was evaluated. All feasibility labels are true one-step Generic-SOFA forks. Execution used `conda run -n steve_tdmpc`, CUDA device index 0 (`CUDA_VISIBLE_DEVICES=0`, RTX 5090).

The exact G8 nominal-unsafe state IDs are **3, 14, 15, 27, 35, 39, 47, 55, 67, 75, 81, 83, 91, 99, 107, 110**. G8 did not store standalone serializable SOFA snapshots: its archived state files store decision observations, physical features, behavior actions, and one-step fork outcomes. We therefore restored each live simulator state by recreating one Generic scene and replaying **all 30 G8 seeds 101000–101029 in order**, resetting the same scene between seeds and advancing each parent with the archived nominal action. The [state manifest](recovered_g8_state_manifest.json) records every seed/decision index, archived state-file SHA256, nominal action/force, and old-set safe count. These are the same G8 states, not newly collected ones.

The mandatory pre-search [nominal replay audit](replay_audit.json) passed: **1,848 parent steps** and **16 unsafe-state nominal forks**, with **0.0** maximum difference for observation, applied action, reward, tip position, insertion length, physical state, pre-action force, and next-interval maximum force; discrete outcome/substep checks also passed. The [search-pass parent audit](search_parent_audit.json) independently passed with the same 1,848 steps and 16 forks, again zero differences. If either audit had failed, no feasibility result would be accepted. Oracle actions were never applied to the parent trajectory.

## Search protocol and evaluation count

For each of the 16 states, the global grid was the Cartesian product of `linspace(-1,1,41)` for normalized translation and rotation, **1,681 actions/state** and **26,896 new SOFA forks**. Coordinates were passed to the environment as realized float32 2-D actions. Every row of [global_grid_results.csv](global_grid_results.csv) records the action, true maximum-substep distal force, and `F_true <= F_lim` status.

For **every** state, local refinement used 5×5 offsets `[-0.02,-0.01,0,0.01,0.02]` around the three lowest-force global points and the three safe global points nearest the nominal action, if any. Actions were clipped to `[-1,1]^2`, deduplicated after float32 realization, and grid duplicates removed. This added **970 new SOFA forks**, for **27,866 new search evaluations** total. The 16 nominal integrity forks in each scene pass are reported separately; **336 archived G8 old-21 labels** were reused as already validated oracle samples, not counted as new evaluations. Nearest-safe ties use Euclidean normalized-action distance, then translation and rotation order. The returned “continuous oracle” is nearest **among these evaluated actions**, not a proven exact optimizer over the mathematical continuum. Search protocol and source hash were recorded before replay in [metadata.json](metadata.json).

## Per-state true-force result

`W*` is the minimum-force sampled action with **translation < 0**; `F(-1,0)` explicitly tests strong straight withdrawal. `Old` refers to G8's 21-action discrete oracle. `New` refers to the global grid, local refinement, **and** archived old-21 samples. Values below are nominal N, rounded for readability; exact values are in the CSVs.

| G8 state (seed:step) | Nominal F | Minimum sampled F | Best withdrawal W* `(t,r):F` | F(-1,0) | Old safe? | New safe? | Safe grid / 1681 |
|---|---:|---:|---|---:|---|---|---:|
| 3 (101000:51) | 7.879 | 6.004 | `(-1,-1):6.004` | 6.093 | No | No found | 0 |
| 14 (101003:78) | 5.118 | 1.751 | `(-1,0.26):1.751` | 1.753 | Yes | Yes | 996 |
| 15 (101003:51) | 4.091 | 3.476 | `(-1,-1):3.476` | 3.603 | Yes | Yes | 437 |
| 27 (101006:33) | 5.953 | 4.225 | `(-1,-1):4.225` | 4.344 | No | No found | 0 |
| 35 (101008:59) | 7.011 | 4.981 | `(-1,1):4.981` | 5.013 | No | No found | 0 |
| 39 (101009:46) | 6.587 | 6.388 | `(-1,-1):6.388` | 6.419 | No | No found | 0 |
| 47 (101011:70) | 6.545 | 6.430 | `(-1,-0.05):6.430` | 6.434 | No | No found | 0 |
| 55 (101013:45) | 7.028 | 4.841 | `(-1,-1):4.841` | 4.889 | No | No found | 0 |
| 67 (101016:45) | 6.853 | 4.209 | `(-1,-1):4.209` | 4.398 | No | No found | 0 |
| 75 (101018:44) | 7.138 | 3.807 | `(-1,1):3.807` | 3.812 | No | No found | 0 |
| 81 (101020:30) | 3.797 | 0.931 | `(-0.95,-0.40):0.931` | 1.042 | Yes | Yes | 1512 |
| 83 (101020:48) | 5.367 | 5.086 | `(-1,-1):5.086` | 5.170 | No | No found | 0 |
| 91 (101022:60) | 6.721 | 5.674 | `(-1,-1):5.674` | 5.726 | No | No found | 0 |
| 99 (101024:49) | 6.178 | 5.723 | `(-1,-1):5.723` | 5.775 | No | No found | 0 |
| 107 (101026:31) | 4.876 | 1.084 | `(-1,-1):1.084` | 1.105 | Yes | Yes | 1190 |
| 110 (101027:32) | 4.293 | 2.968 | `(-0.91,1):2.968` | 3.641 | **No** | **Yes** | **50** |

For every state, the best sampled action reduced force relative to the nominal action (reduction range approximately **0.115–3.792 N**), but in 11 states even the minimum sampled force remained above `F_lim`. The smallest residual among those 11 is state 75: **3.806739 N**, only **0.121408 N** above `F_lim`; 36 local refinement forks there did not lower the grid minimum. That near-boundary case deserves the strongest uncertainty warning.

## Withdrawal and nearest-safe oracle repair

Any negative-translation action was safe in **5/16 (31.25%)** states; `[-1,0]` was safe in the **same five**. Among the old 12 infeasible states, withdrawal rescued **only state 110 (1/12 = 8.33%)**. Best actions with translation `<= 0` are also recorded in [withdrawal_analysis.csv](withdrawal_analysis.csv); all state minima happened on the withdrawal side. Withdrawal is therefore sufficient in the five found-feasible states but is not a universal one-step recovery action in the other 11. State 47's nominal translation was already strongly negative (about −0.973), yet its minimum sampled force remained 6.430 N.

For the five feasible states, nearest **sampled** true-safe repairs (not minimum-force actions) are:

| State | Nearest safe `(translation, rotation)` | Safe F | L2 distance | Δtranslation | Δrotation | ΔF from nominal |
|---|---|---:|---:|---:|---:|---:|
| 14 | `(0.22,-0.03)` | 3.684895 | 0.777136 | −0.776185 | −0.038429 | −1.433131 |
| 15 | `(−0.23,0.19)` | 3.684372 | 0.789133 | −0.730788 | −0.297791 | −0.406996 |
| 81 | `(0.95,−0.97)` | 3.678892 | 0.046728 | −0.046534 | −0.004258 | −0.118295 |
| 107 | `(0.92,0.83)` | 3.619502 | 0.052080 | −0.049915 | +0.014861 | −1.256486 |
| 110 | `(−1,−0.02)` | 3.668877 | 0.521786 | −0.483541 | +0.196083 | −0.624341 |

Median nearest-safe action distance is **0.521786**, maximum **0.789133**; mean/median force change across the five nearest-safe repairs is approximately **−0.768/−0.624 N**. Their forces often lie very near the threshold because this oracle minimizes action deviation, not force. These are exact one-step force labels at the archived states, not robust margins or deployable actions. Full-precision component changes are in [continuous_oracle_repair.csv](continuous_oracle_repair.csv).

## Force-surface geometry and G8 correction

The 41×41 grid's safe fractions for states **14, 15, 81, 107, 110** are **59.25%, 26.00%, 89.95%, 70.79%, 2.97%**, respectively; all other states have 0 sampled safe grid points. State 110's 50 safe grid points form one 4-connected region confined to approximately **translation [−1,−0.85], rotation [0,1]**. Its local refinement extends the nearest sampled safe point to rotation −0.02. This is a real G8 joint-action coverage miss: the old 21 actions included `(-0.916,0.184)` but it produced **4.073 N**, while previously omitted `(-1,0)` produced a safe **3.641 N**. Rotation matters at state 110 (median true-force range across rotation at fixed translation **1.959 N**, maximum **5.407 N**); by contrast state 39's maximum such range is only **0.063 N**, and every sampled action remains unsafe. State 75 has substantial rotation-dependent variation yet remains 0.121 N above the limit at its best sampled action. All 16 heatmaps are saved under [force_landscapes/](force_landscapes/); numerical geometry is in [force_landscape_geometry.csv](force_landscape_geometry.csv).

The corrected G8 interpretation is therefore **mixed but strongly weighted toward one-step non-recoverability under this search**: candidate discretization caused one false-infeasible label (state 110), while 11/12 old-infeasible states still lack a true-safe sampled one-step action. This is **Case B**, not an assertion of mathematical impossibility across uncountably many actions. The recommendation is one separately scoped methodological direction—**anticipatory multi-step safety planning** to avoid entering states with poor one-step recovery. Do not start it under G8.5. No predictor evaluation, retraining, calibration, online shield control, or TD-MPC2 modification was performed.
