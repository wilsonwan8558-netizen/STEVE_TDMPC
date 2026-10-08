# Phase G0.5 — long-horizon Generic solver and penetration validation

**Decision: CONDITIONAL GO for G1 retraining, not a clinical or finite-radius contact-safety validation.** The frozen G0 Generic configuration was reproduced. Across seven scripted programs (2,204 RL controls) and 50 independently seeded randomized programs (10,000 controls), there were no nonfinite states, simulator errors, sampled guidewire-centerline wall crossings, solver-cap hits, or observed geometry corruption. The remaining condition is geometric: the collision model treats the wire as a zero-radius line/point set with a 0.3 mm contact distance, while the mechanical beam radius is 0.35–0.445 mm. Thus the measured zero *centerline* crossing does not establish zero physical beam–vessel overlap. A matched-lifecycle 7,568-substep contact-heavy audit also showed a higher force peak between control endpoints than the endpoint trace captured. No training or solver tuning was performed.

## Provenance and frozen configuration

- Branch `experiment/generic-solver-baseline`, base commit `59ea0ca569cdf2cc48095bd1dccc205fb6900ff7`, `steve_tdmpc` conda environment, SOFA installation `/home/bizon/SOFA_TDMPC/install` (source commit `728003f7a7c16ceada5663f4787904742c726bab`).
- The production adapter `eve/intervention/simulation/sofabeamadapter.py` retained its G0 SHA256 `f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67` throughout G0.5. No physics, device, contact, material, task, action, observation, reward, termination, or controller setting was changed.
- `GenericConstraintSolver(name=Generic, resolutionMethod=ProjectedGaussSeidel, tolerance=1e-4, maxIterations=2000, computeConstraintForces=True)`; `DefaultContactManager(response=FrictionContactConstraint, responseParams=mu=0.001)`; SOFA `dt=0.006 s`; 7.5 Hz control; 22 SOFA substeps per control. The installed solver has `scaleTolerance=true`; reported scaled residual ratio is `currentError / (1e-4 × max(currentNumConstraints,1))`.
- The G0 controlled 0.1 nominal-N load was reproduced exactly: `J_contact^T lambda / 0.006 s = 0.10265011370558422` nominal N, matching the saved G0 value (2.65% above applied nominal load). See [frozen_solver_config.json](frozen_solver_config.json) and [force_sanity.json](force_sanity.json). These are simulator-convention nominal forces, not calibrated clinical Newtons.

## Penetration and contact measurement

The installed `LocalMinDistance.cpp` sets `DetectionOutput.value` to closest-point distance minus `contactDistance` and model proximities. A negative value is intrusion into the **contact envelope**, not signed physical wall crossing. The Python narrow-phase object did not expose a direct physical penetration depth. The diagnostic therefore uses the fixed closed vessel triangle mesh and mapped guidewire collision polyline:

`signed_exterior_mm = −vtkImplicitPolyDataDistance(vessel_surface, guidewire_centerline_sample)`.

For this mesh, the raw VTK sign is positive inside (checked at 12 vessel centerline points) and negative outside (checked at a distant point), so **positive signed exterior means the mapped wire centerline is outside the vessel lumen**; negative means inside clearance. `max(0, signed_exterior_mm)` is centerline wall-crossing depth in mm. The mesh is manifold with zero open edges (636 vertices, 1,268 triangles). Every endpoint samples all 76 collision vertices and midpoints of nondegenerate collision edges. The distal subset is within 20 mm polyline arc length of the tip. Binary crossing counts use only a `1e-6 mm` numerical-noise epsilon; raw signed values are retained. Active contact means `Generic.currentNumConstraintGroups > 0`, a separate contact-envelope indicator. See [penetration_metric_audit.json](penetration_metric_audit.json).

This **does not measure actual finite-radius beam/tissue intersection**. The mechanical tip radius is 0.35 mm and straight radius is 0.445 mm, but the SOFA collision model is line/point with zero proximity. A `+0.35 mm` spherical-radius offset of the nearest sampled centerline point is a *proxy*, not a lower bound on true cross-section overlap. A `+0.445 mm` offset is an upper bound on possible signed exterior excursion of any point within that radius of each sampled centerline point; it is neither measured overlap nor a guarantee for unsampled continuous segments. Cross-section orientation and continuous segment geometry were not resolved. Endpoints alone can miss transient extrema, so the contact-heavy case was independently sampled at every SOFA substep.

## Long-horizon scripted programs

All seven programs ran the full scripted duration even after the unchanged environment's nominal 200-step truncation; this is an open-loop simulator diagnostic, not a task-policy rollout. Each family used one newly constructed SOFA scene and its saved seed. The actions and exact seed assignments are in [collect_long_horizon.py](../../experiments/generic_solver_g05/collect_long_horizon.py).

| Program | Controls | Simulated s | Distal force median / Q95 / Q99 / max (nominal N) | Max solver iterations | Centerline crossing, distal/global |
|---|---:|---:|---:|---:|---:|
| Zero | 400 | 52.800 | 0 / 0 / 0 / 0 | 0 | 0 / 0 mm |
| Sustained moderate translation | 200 | 26.400 | 0.284 / 1.032 / 1.469 / 1.541 | 42 | 0 / 0 mm |
| Sustained moderate rotation | 320 | 42.240 | 0.435 / 0.795 / 0.861 / 1.359 | 38 | 0 / 0 mm |
| Translation + rotation | 300 | 39.600 | 0.232 / 1.154 / 1.285 / 1.323 | 54 | 0 / 0 mm |
| Alternating translation | 320 | 42.240 | 0.367 / 0.434 / 0.742 / 1.359 | 42 | 0 / 0 mm |
| Alternating rotation | 320 | 42.240 | 0.328 / 0.797 / 0.800 / 1.359 | 35 | 0 / 0 mm |
| Contact-heavy | 344 | 45.408 | 0.130 / 0.759 / 0.926 / 1.488 | 44 | 0 / 0 mm |

Across all 2,204 scripted endpoints, pooled distal-force median/Q95/Q99/max were `0.222/0.795/1.210/1.541` nominal N. The largest scaled residual ratio was `<1` and no solve reached the 2,000-iteration cap. Contact-heavy reached 132 mm insertion, had 341/344 contact-active endpoints, and still had no sampled centerline crossing. The largest sampled tip-radius offset proxy was `0.0579 mm`; the largest sampled max-radius exterior-extent upper bound was `0.1529 mm`. Neither number is an observed material penetration depth.

In the matched-lifecycle contact-heavy substep replay, the passive monitor matched the original endpoint trace **exactly** for distal/global force, contact groups, solver iterations, signed wall distance, insertion length, and tip coordinates. It sampled 7,568/7,568 SOFA substeps: all finite, no centerline crossing or force without contact, maximum 47 solver iterations, and scaled residual ratio `<1`. The maximum instantaneous distal force was `1.9403` nominal N at control step 4, substep 10, versus `1.4878` nominal N at control endpoints. See [substep_contact_heavy_summary.json](substep_contact_heavy_summary.json) and [substep_contact_heavy.csv](substep_contact_heavy.csv). Other trajectories were endpoint-sampled only.

## Fixed randomized stability test

Seeds `91000–91049` were fixed in [randomized_seed_plan.json](randomized_seed_plan.json). No TD-MPC2 policy was used. Every 20 controls, a seeded pseudo-random insertion target in 40–145 mm was chosen; translation moved toward it, clipped to normalized `[-0.3,0.3]`, and each rotation was sampled in `[-0.5,0.5]`. A **fresh SOFA scene per seed** produced 50 independent, complete 200-step runs (26.4 s each, 10,000 controls total). This lifecycle is a G0.5 environment-only stress test, not the historical Original TD-MPC2 evaluator. An interrupted three-seed reused-scene pilot remains in `shards/random_91000` as diagnostic data and was **excluded** from every merged statistic.

There were `0/50` numerical failures, simulation errors, solver-cap-hit seeds, residual-exceedance seeds, and distal/global centerline-crossing seeds. Pooled randomized endpoint force median/Q95/Q99/max were `0.303/0.866/1.677/2.393` nominal N. The maximum force occurred at seed 91040 step 8 with one/two contact groups in neighboring startup steps; the top 12 force samples across the full dataset were all randomized initial contact at steps 8–10. The maximum randomized force after step 20 was `1.065` nominal N. The largest solver iteration count was `207/2000` at seed 91040 step 61, with scaled residual ratio `0.9997` and six constraint rows; no cap or divergence followed. The randomized maximum sampled max-radius offset bound was `0.1501 mm`, again not measured physical overlap.

The optional old-LCP penetration subset was not run; no Generic-vs-LCP equivalence claim is made.

## Force consistency and geometry

No endpoint or audited contact-heavy substep had nonzero reconstructed distal force with zero active contact groups. Long nonzero-force runs coincided with long contact-active periods; for example, alternating rotation had a 314-control-step nonzero-force run and 314 contact-active endpoints. The longest run above each scripted trajectory's own Q95 force was 13 controls; that Q95 is **descriptive**, not a safety threshold. First- versus last-quarter force medians did not show a uniform secular rise: alternating translation `0.386→0.362`, alternating rotation `0.325→0.344`, and contact-heavy `0.184→0.077` nominal N. The largest randomized force spikes occurred at initial contact, with contact groups present and modest solver iteration counts. There was no observed persistent force after contact disappeared. The centerline crossing variable was identically zero, so a force-vs-penetration correlation is mathematically undefined; the [scatter plots](plots/force_vs_penetration/) show this explicitly beside the radius-offset diagnostic, without implying a linear force–depth law. A short-lived force artifact inside other, unsampled control intervals cannot be excluded.

Guidewire states stayed finite. The maximum parent quaternion-norm error was `2.38e-14`; maximum parent-tip/collision-tip positional mismatch was `7.98e-12 mm`. Collision polyline topology stayed 75 consecutive edges. Active collision length tracked insertion over the programs. Short active edges of 0.004–0.01 mm occurred near activation, while roughly 10–13 mm edges are consistent with the configured sparse straight-region collision sampling; neither by itself demonstrates accumulated collapse. Large local turn angles (up to `178.8°`) were already present in the first control samples and did not coincide with exploding coordinates or detached tip mapping. Self-intersection and exact parent-beam strain were **not** directly measured, so the geometry check is a numerical-integrity screen rather than a proof of all geometric validity. See [geometry_integrity_summary.json](geometry_integrity_summary.json).

## Required 16-point disposition

1. **Exact G0 configuration?** Yes; runtime settings and 0.1 nominal-N reconstruction reproduced exactly.
2. **Exact penetration metric?** Signed vessel-surface distance of mapped guidewire *centerline* vertices/edge midpoints, converted to positive outward crossing depth.
3. **Sign/unit?** Positive = outside vessel lumen; negative = inside clearance; millimeters.
4. **Durations?** Seven programs of 200–400 controls (`26.4–52.8 s`); 50 randomized programs of 200 controls (`26.4 s` each).
5. **NaN/inf/crashes?** None across 12,204 control endpoints and the separate 7,568-substep contact-heavy replay.
6. **Solver pathology?** No cap hits or scaled residual exceedances; max endpoint iterations 54 scripted, 207 randomized, versus 2,000 cap.
7. **Distal penetration?** Sampled centerline crossing max/median/Q95/Q99 = 0 mm; zero penetrating controls.
8. **Global penetration?** Same centerline result, 0 mm; sampled radius offsets do not establish physical zero overlap.
9. **Persistent penetration?** No sampled centerline crossing runs at endpoints or in the contact-heavy substep audit; other inter-endpoint excursions remain unmeasured.
10. **Force statistics?** Pooled scripted median/Q95/Q99/max `0.222/0.795/1.210/1.541`; pooled randomized `0.303/0.866/1.677/2.393` nominal N. Contact-heavy substep max `1.940` versus endpoint max `1.488`.
11. **Abnormal spikes/drift?** Startup-contact spikes were observed but contact-associated, finite, and not tied to solver failure; no uniform baseline drift or force without contact was observed.
12. **Guidewire valid?** No observed nonfinite, quaternion, topology, or tip-mapping corruption; self-intersection/physical strain remain unproven.
13. **Randomized seeds?** All 50 fixed independent seeds completed 200 controls, with 0/50 screened numerical/solver/centerline-crossing failures.
14. **Force–penetration coupling?** No pathological centerline crossing to correlate with force; force–physical-radius overlap coupling remains unresolved by this metric.
15. **Decision?** **CONDITIONAL GO**, because the frozen solver is numerically stable in this screen but physical finite-radius overlap is not directly measured and endpoint force can miss transient peaks.
16. **Proceed to G1 retraining?** Justified **only if** the project explicitly accepts the line/point collision abstraction and these measurement limits for simulator retraining. This is not evidence of clinical contact safety; no retraining was started in G0.5.

Machine-readable detail: [long_horizon_results.csv](long_horizon_results.csv), [randomized_seed_results.csv](randomized_seed_results.csv), [penetration_summary.json](penetration_summary.json), [force_stability_summary.json](force_stability_summary.json), [solver_convergence_summary.json](solver_convergence_summary.json), [long_horizon_steps.csv](long_horizon_steps.csv), and [randomized_steps.csv](randomized_steps.csv). Force, penetration, and solver-stress plots are under [plots](plots/).
