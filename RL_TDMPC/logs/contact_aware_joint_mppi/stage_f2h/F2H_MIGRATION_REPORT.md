# F2H — Clean G13 contact-reward research foundation

**Outcome:** the isolated `feature/g13-contact-reward` worktree at `/home/bizon/Desktop/stEVE_g13_contact_reward` reproduces the unchanged G13 baseline. **Pilot readiness remains conditional:** B0 checkpoint/replay loading is audited, but B1 training is intentionally disabled and its historical-replay FTI labels do not exist. No model training or full benchmark was run.

## Source and checkpoint

The worktree was created from `d2638d14ce57a288c3bd13e1c67a22d14eb86da0`, the first checked commit whose G13 config, model, environment adapter, and geometry-feature bytes match the pre-training freeze. The selected historical `step_260000.pt` remains external and unchanged, SHA256 `b2bba8be2e2722886c20952b15712deaa31ab612825999f1b3469a34fd6ef7be`. It is format-v2 G13, 22D observation, 2D insertion/rotation action, 256D latent, H=3, and contains Adam/replay/RNG state. Its complete-episode replay has 259,987 transitions, 3,800 episodes, and 252,387 eligible H=3 sequences.

The frozen training task was raw TargetReached plus `0.01 × PathLengthDelta`; MPPI scores task reward and value, not contact. The original `lambda_force_train=0.5` next-latent force-head Huber objective remains present in **both** proposed arms; its gradients reach shared encoder and dynamics. Generic SOFA remains ProjectedGaussSeidel, tolerance `1e-4`, 2,000 iterations, friction `0.001`, `dt=0.006 s`, and 22 substeps/control. The passive Generic G2 distal 3 mm force reconstruction remains unchanged. [Source provenance](source_provenance.json) lists hashes and the full observation layout.

Provenance limits matter: `d2638d14` is a content-matched core reference, **not** proof of the exact Git HEAD or training-runner bytes on 2026-10-14. Historical SOFA/BeamAdapter binary identity is not recorded. The 260k checkpoint was saved 13 steps into an unfinished episode; its complete-episode replay is available, but those 13 partial transitions and the live SOFA scene are not. The new runner therefore starts an independent clean scene and does not claim bitwise historical mid-episode training continuation.

## Zero-update baseline gate

The new worktree passed the predeclared CUDA gate in `steve_tdmpc` (`torch 2.7.1+cu128`, CUDA runtime 12.8, RTX 5090, device index 0; vendored `eve` imported from this worktree):

- Historical reused-scene seeds `350000–350009`: all 10 per-seed success, length, and task rewards matched exactly, 591 archived/reproduced total steps and maximum reward difference `0`.
- F2G fixed-observation model/MPPI probe: latent, 512 H=3 candidate-score, reward-logit and Q hashes, plus selected joint action matched.
- F2G first prospective case `1400000`, **regression only**: initial observation hash, complete action sequence, success, length and reward matched exactly. This reused diagnostic case is not independent confirmation.
- Every gate action returned 22 valid Generic SOFA substep samples; interval peak and FTI matched the passive substep force reconstruction. No simulation error occurred.

The archive lacks historical per-step candidate/action/SOFA-initial-state traces, so the 10-seed archived comparison is at episode level; the single F2G action-trace check supplies a separate, narrower step-level regression. Full results are in [baseline_equivalence.json](baseline_equivalence.json).

## Continuation infrastructure and arm separation

The new opt-in `continue_g13.py` imports the unchanged G13 agent/replay. Its default path is a **no-training audit**; execution requires an explicit `--execute` and a separately frozen/enabled B0 config. The B1 template is non-executable. The audit strictly restored model and both Adam optimizers, verified optimizer device placement, six model learning-rate groups, force-labeled replay, and CUDA availability. Five no-training tests passed for configuration guards, RNG round-trip, and observable scene-replay mismatch detection. No live training/resume cycle was performed in F2H.

For a future authorized B0 run, the runner stores the agent, optimizers, replay, Python/NumPy/PyTorch/CUDA RNG and accepted action history in an atomic checkpoint **only at completed episode boundaries**. On resume it replays preceding new-branch episodes in order, checks observations, targets, task rewards, peak force, FTI and termination, then restores the saved RNG state. Exact-step intermediate model snapshots are marked evaluation-only, not restartable. The replay check proves observable reconstruction, not inaccessible internal SOFA solver state; a live bounded resume smoke remains mandatory before relying on it. Training and evaluation are separate processes/protocols.

B0 is original task-only reward. B1 is a future single penalty `r_train = r_task − beta × (distal_force_time_integral_Ns / fixed_scale_Ns)`, with the **same** G13 auxiliary force-head loss and ordinary task-only MPPI. There is no new Contact Head, MPPI penalty, fuzzy/CBF filter, or second force penalty. Beta and scale are intentionally unset; the G1/F2C values are not assumed transferable.

The historical G13 replay stores **raw task rewards and interval peak-force labels**, not interval FTI. It cannot be silently mixed with B1 shaped rewards or exactly relabeled for an FTI penalty. B1 needs a separately audited, reward-consistent replay/data strategy before implementation or training. [Readiness](reward_experiment_readiness.json) and the [future matched protocol draft](future_pilot_protocol.json) record this blocker, require separate replay buffers and matched training/target seeds, and exclude F2G diagnostic cases from independent confirmation. The future primary checkpoint must be selected by a predeclared terminal budget, never by favorable intermediate navigation results.

## Decision and limits

**The clean G13 foundation is established; a bounded matched contact-reward pilot is not yet launch-ready.** The next stage must resolve B1 FTI replay semantics, freeze a training-only beta/scale, audit unused prospective seeds, and validate live episode-boundary resume. F2G's non-monotonic 19/24 intermediate B0 result prevents claiming stable long-term fine-tuning from this foundation alone.

Only the new isolated worktree received F2H files. The current research and legacy worktrees, historical checkpoints, SOFA physics, G13 model architecture, observation/action interfaces, reward, and MPPI implementation were not modified. Nothing was committed, pushed, merged, or trained.

## Verification commands

Run from the new worktree root with `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=0`, `SOFA_ROOT=/home/bizon/SOFA_TDMPC/install`, `SOFAPYTHON3_ROOT=/home/bizon/SOFA_TDMPC/install/plugins/SofaPython3`, `PYTHONPATH=/home/bizon/SOFA_TDMPC/install/plugins/SofaPython3/lib/python3/site-packages`, `LD_LIBRARY_PATH=/home/bizon/SOFA_TDMPC/install/lib:/home/bizon/SOFA_TDMPC/install/plugins/SofaPython3/lib:/home/bizon/SOFA_TDMPC/install/plugins/BeamAdapter/lib`, and `MPLCONFIGDIR=/tmp/f2h_mpl` for the SOFA gate:

```bash
conda run --no-capture-output -n steve_tdmpc python RL_TDMPC/experiments/g13_contact_reward/verify_baseline.py
```

The no-training runner audit and tests used `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0`:

```bash
conda run --no-capture-output -n steve_tdmpc python RL_TDMPC/experiments/g13_contact_reward/continue_g13.py --config RL_TDMPC/experiments/g13_contact_reward/b0_task_only.template.json --checkpoint /home/bizon/Desktop/stEVE_safety_head_1e1be81/RL_TDMPC/runs/geometry_force_tdmpc2_g13_300k_20261014/checkpoints/step_260000.pt --audit-output RL_TDMPC/logs/contact_aware_joint_mppi/stage_f2h/training_runner_audit.json
conda run --no-capture-output -n steve_tdmpc pytest -q RL_TDMPC/experiments/g13_contact_reward/test_continue_g13.py
```
