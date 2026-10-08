# Phase G0 — GenericConstraintSolver environment validation

**Decision: CONDITIONAL GO.** The installed Generic solver runs the tested open-loop scenes without numerical pathology, exposes a traceable contact multiplier vector, and passes a quasi-static known-load balance after `J_contact^T lambda / dt` reconstruction. The tracked production edit is limited to the SOFA solver/contact setup. The conditions are that signed guidewire–vessel penetration and broad-seed/long-horizon behavior remain unquantified, and the calibrated values are nominal forces in this simulation convention, not clinical force validation. From-scratch TD-MPC2 baseline retraining under this locked configuration is a justified *next* stage; no training was performed in G0. Do not designate it a fully validated safety benchmark on this screen alone.

## Provenance and scope

- Worktree: `/tmp/stEVE_safety_head_1e1be81`; branch `experiment/generic-solver-baseline`, at the exact `TDMPC2` base commit `59ea0ca569cdf2cc48095bd1dccc205fb6900ff7`. Other feature branches were untouched.
- Runtime: `steve_tdmpc` conda environment, Python `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, SOFA `/home/bizon/SOFA_TDMPC/install`, source commit `728003f7a7c16ceada5663f4787904742c726bab`. Diagnostic commands used `conda run -n steve_tdmpc` and `RL_TDMPC/setup_env.sh`.
- Sole tracked production file edited: `eve/intervention/simulation/sofabeamadapter.py`. Separate G0 scripts and output files are under `RL_TDMPC/experiments/generic_solver_g0/` and this directory. No device, material, geometry, task, observation, reward, controller, or training code was changed.

## 1. Original LCP scene, recorded before editing

The original `SofaBeamAdapter._basic_setup` in `eve/intervention/simulation/sofabeamadapter.py:189-226` used `FreeMotionAnimationLoop`, `DefaultPipeline(draw=0, depth=6, verbose=1)`, `BruteForceBroadPhase`, `BVHNarrowPhase`, `LocalMinDistance(contactDistance=0.3, alarmDistance=0.5, angleCone=0.02)`, and `DefaultContactManager(response="FrictionContactConstraint")` without `responseParams`. It constructed `LCPConstraintSolver(mu=0.001, tolerance=1e-4, maxIt=2000, name="LCP", build_lcp=False)`; `computeConstraintForces` was not enabled in the production baseline. The environment passes `friction=0.001`, simulation `dt=0.006 s`, and control frequency `7.5 Hz`.

The device path is `WireBeamInterpolation`/`AdaptiveBeamForceFieldAndMass`/`InterventionalRadiologyController`, with `LinearSolverConstraintCorrection(wire_optimization=true)` on `InstrumentCombined`, a `FixedConstraint`, then `MultiAdaptiveBeamMapping` into `CollisionDOFs` with line/point collision models. The vessel is the existing static triangle/line collision surface. These components remain unchanged. See `original_solver_audit.json` for the original file hashes and locations.

Important friction distinction: the original `mu=0.001` was passed to the **LCP solver**, whereas the original `FrictionContactConstraint` response had no explicit `mu` parameter. The installed SOFA `FrictionContact` component's default `mu` is `0.8`. Thus one should not infer that the old contact-response coefficient was explicitly `0.001`, nor regard LCP-vs-Generic trajectories as a controlled single-parameter solver comparison.

## 2. Generic scene and friction

The production adapter now instantiates `GenericConstraintSolver(name="Generic", resolutionMethod="ProjectedGaussSeidel", tolerance=1e-4, maxIterations=2000, computeConstraintForces=True)`. It retains the same animation/collision pipeline and correction. `DefaultContactManager(response="FrictionContactConstraint", responseParams="mu=0.001")` passes the environment's friction value to the contact response, where the installed SOFA source creates frictional contact constraints. The two required component plugins were made explicit. The runtime scene reports all requested values, with installed defaults `scaleTolerance=true`, `allVerified=false`, and `sor=1`. See `generic_solver_config.json`.

The field names and behavior were checked against installed SOFA source, notably `GenericConstraintSolver.cpp:72-99,499-507`, `DefaultContactManager.cpp:290-309`, `FrictionContact.inl:49,207-226`, and `UnilateralInteractionConstraint.inl`. The [SOFA Lagrange-constraint documentation](https://sofa-framework.github.io/doc/simulation-principles/constraint/lagrange-constraint/) provides the general constraint-pipeline context; the installed source is authoritative for this build.

## 3. Force access and units

With `computeConstraintForces=True`, `Generic.constraintForces` is a vector with one scalar Lagrange multiplier per active constraint row, refreshed by the solve after each SOFA `animate` substep. In a contact scene it had length 21, `currentNumConstraints=21`, and seven contact groups (normal plus two friction directions each). Across all 110 open-loop action-end samples, `len(constraintForces) == currentNumConstraints` with zero mismatches. An `env.step` sample sees only the **last** 0.006 s SOFA substep of that control interval, not its maximum.

The vector is **not** an already-aggregated distal contact force. The mapped guidewire collision `MechanicalObject.constraint` supplies sparse Jacobian rows with global row IDs. The diagnostic script matches those row IDs to solver multipliers, forms `J_contact^T lambda` on guidewire collision DOFs, and sums the guidewire reaction. Rows on the vessel or unrelated constraints must not be indiscriminately summed. In this scene, the calibrated `J_contact^T lambda` is impulse-like; division by SOFA `dt=0.006 s` gives the nominal contact-reaction force. This scaling is empirical G0 evidence, not imported from the earlier LCP audit. The exact reconstruction, scope, and limits are in `force_scaling_audit.json` and `calibrate_wire.py`.

## 4. Controlled guidewire load

The diagnostic-only `ConstantForceField` was attached to mapped guidewire collision DOF 53, about 10.44 mm proximal to the tip, against the unchanged rigid vessel surface. For each trial, seed 90008 was inserted for 14 actions, then a zero-action 20-step hold was run in its own scene. A separate no-load scene established the baseline; loaded forces opposed the initial contact normal. The final three control-end reactions were averaged and baseline-subtracted. Last-three loaded-node movement was at most `0.00082 mm` (versus `0.000062 mm` without load), supporting quasi-static interpretation. All loaded observations were finite and no simulation errors occurred.

| Applied nominal N | Reconstructed reaction N, `/dt` | Relative error | Opposing-direction cosine |
| ---: | ---: | ---: | ---: |
| 0.020 | 0.020328 | 1.64% | 0.99755 |
| 0.050 | 0.051074 | 2.15% | 0.99841 |
| 0.100 | 0.102650 | 2.65% | 0.99861 |
| 0.200 | 0.205565 | 2.78% | 0.99860 |

Without `/dt`, the reaction magnitudes were approximately 99.4% below the applied nominal values. Thus `J_contact^T lambda / dt` is required for a force readout in this scene. Only one load direction was tested; this is a total guidewire contact balance at a distal node, **not** tip-only force, cross-direction validation, or a physical/clinical calibration. Detailed per-step traces are in `controlled_load_equilibrium.json`; the requested concise table is `controlled_load_results.csv`.

## 5. Open-loop stability and old-LCP diagnostic

Six fixed scenarios covered zero action, moderate translation, rotation without insertion, simultaneous moderate translation/rotation, insertion followed by moderate rotation under contact, and aggressive insertion plus rotation. Seeds were 90000 or 90008. The no-insertion rotation case has no visible device motion, so the inserted-rotation case is the meaningful rotation test. All 110 Generic action samples had finite observations, reward, DOFs and multipliers, and no `simulation_error`, crash, gross coordinate/spacing anomaly, or tip jump above the diagnostic screen's thresholds. Sampled contact steps were respectively 0/12, 3/14, 0/14, 9/14, 20/28, and 20/28. The largest iteration count was 38 of the configured 2000; all reported errors were below the installed solver's `scaleTolerance=true` threshold when scaled by constraint count (largest ratio 0.998). Largest raw multiplier was 0.00801. No exploding multiplier trend was observed in these short traces.

The exact signed guidewire–vessel penetration depth was **not** measured: the available screen checked finite device geometry, adjacent DOF spacing, tip motion, and simulator errors. Consequently it cannot rule out local excessive penetration or longer-horizon oscillations. The SOFA adaptive-beam deprecation warnings and diagnostic `ConstantForceField` warnings are present, but none caused a test failure. `numerical_stability_results.json` contains the thresholds and per-scenario values.

The LCP and Generic traces used the same fixed actions and seeds from separate base/current worktrees; LCP multiplier exposure was enabled only in the diagnostic scene. Both were finite and had the same sampled contact counts. Final tip separation was `0.000086 mm` for simultaneous moderate motion, `0.000319 mm` for insert-then-rotate, and `0.000847 mm` for the contact-heavy sequence; the contact-heavy maximum intermediate separation reached `0.2213 mm`. These are descriptive observations, **not** an equivalence requirement or proof of identical friction/contact physics. See `lcp_vs_generic_diagnostic.csv` and the two complete rollout JSON files.

## 6. Environment invariants and decision

`environment_invariant_audit.json` records unchanged hashes for `RL_TDMPC/envs/steve_env.py`, vessel source, guidewire device source, and tracking source. Observation shape/order is still 14 (`position`, `target`, `rotation`); action shape/range is still 2/[-1,1]; reward, target sampling, episode termination/truncation, material parameters, tracking, `7.5 Hz` control, and `0.006 s` SOFA timestep are unchanged. `RL_TDMPC/smoke_test_env.py` passed seeded reset/step, finite observations, action range, a complete 200-step zero-action episode, and Gymnasium `check_env` under `steve_tdmpc`.

**CONDITIONAL GO** is justified for a new, from-scratch TD-MPC2 baseline experiment because the targeted Generic scene is stable and its contact reaction is internally interpretable and load-validated, without unintended environment/task changes. The conditions to carry forward are: quantify signed penetration and broader-seed/long-horizon stability before calling Generic a fixed benchmark; use `J_contact^T lambda / dt` only with an explicit guidewire/contact selection and SOFA-substep timing; do not make clinical-force or tip-only-safety claims from this G0 calibration. No TD-MPC2 training, Force Head, shield, CBF, or safety controller was run or added.

## Reproduction commands

Run from this worktree with `conda run -n steve_tdmpc bash -lc 'source RL_TDMPC/setup_env.sh >/dev/null && ...'` for every SOFA-dependent command. For example, the known-load equilibrium command inside that shell is:

```bash
python RL_TDMPC/experiments/generic_solver_g0/calibrate_wire.py \
  --hold-steps 20 \
  --magnitudes 0.02 0.05 0.1 0.2 \
  --output RL_TDMPC/logs/generic_solver_phase_g0_validation/controlled_load_equilibrium.json
```

`rollout_diagnostic.py` accepts `--source-root` (`/tmp/stEVE_safety_head_1e1be81` for Generic or `/tmp/stEVE_original_59ea0ca` for LCP), `--solver Generic|LCP`, and `--output` to recreate the two trace JSON files; `summarize_rollouts.py --output-dir RL_TDMPC/logs/generic_solver_phase_g0_validation` recreates the paired diagnostic and stability summaries. `python RL_TDMPC/smoke_test_env.py` repeats the environment contract check.
