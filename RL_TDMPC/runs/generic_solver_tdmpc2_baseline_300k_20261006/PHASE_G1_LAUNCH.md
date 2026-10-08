# Phase G1 — Generic-solver Original TD-MPC2 baseline launch

**Status: RUNNING from step 0; no checkpoint selected.** The smoke test and separate one-episode evaluation passed before launch. The production run was started at **2026-10-06 22:25:52 EDT** as user-managed unit `steve-g1-generic-300k.service` on the current `experiment/generic-solver-baseline` worktree. Training logged through step 4,800 with `Device: cuda`, then reached step 5,000 and entered the historical 5,000-initial-update block; the service remained active at that check. This file documents the launch, not completion of 300k steps or those initial updates.

## Exact command and runtime

The exact durable launch command is:

```bash
systemd-run --user --unit=steve-g1-generic-300k --working-directory=/tmp/stEVE_safety_head_1e1be81 /bin/bash /tmp/stEVE_safety_head_1e1be81/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/launch_training.sh
```

The audited [launch_training.sh](launch_training.sh) sources `RL_TDMPC/setup_env.sh`, sets `CUDA_VISIBLE_DEVICES=0`, checks the smoke gate, reruns the historical-config/frozen-adapter/CUDA audit, and then executes:

```bash
/home/bizon/anaconda3/bin/conda run --no-capture-output -n steve_tdmpc python -u RL_TDMPC/train.py --config /tmp/stEVE_safety_head_1e1be81/RL_TDMPC/runs/generic_solver_tdmpc2_baseline_300k_20261006/config.yaml
```

The G1 config retains historical `training.device=auto`, while the preflight requires one visible CUDA device and confirms an **NVIDIA GeForce RTX 5090**, physical GPU index **0**, mapped to `cuda:0`. Runtime: `/home/bizon/anaconda3/envs/steve_tdmpc/bin/python`, PyTorch `2.7.1+cu128`, CUDA runtime `12.8`. The launch log printed `Device: cuda`; there is no silent CPU fallback. Branch `experiment/generic-solver-baseline`, base commit `59ea0ca569cdf2cc48095bd1dccc205fb6900ff7`. See [launch_audit.json](launch_audit.json) and [solver_environment.json](solver_environment.json).

## Protocol and smoke gate

The 260k Original checkpoint embedded the exact historical training configuration and seed 1; it equals the repository default. The literal historical shell invocation is not archived, so no missing CLI arguments were guessed. [PROTOCOL_RECOVERY.md](PROTOCOL_RECOVERY.md) records the recovered entrypoint, replay, batch, learning rates, planner, exploration, evaluation cadence, device semantics, and checkpoint cadence. [config.yaml](config.yaml) changes only the budget to **300,000 RL environment steps** and experiment-local output paths. No `--resume` is used. One environment step remains one TD-MPC2 action/`env.step` (22 SOFA substeps, 0.132 s simulated time).

The isolated CUDA smoke used the final model, batch, replay, planner, and environment settings. It took 200 real environment steps, filled 198 valid replay sequences, forced one diagnostic optimizer update to exercise that path before the *production* 5k random-action seed period, produced finite losses, planned a finite action, and round-tripped a 21 MB full-state checkpoint. A separate `evaluate.py` run of the smoke checkpoint completed 200 steps with finite reward `1.0750`, no simulation error, and valid CSV output. This untrained smoke checkpoint is isolated under [smoke](smoke/) and is **not** an input to the real run. The real run starts with update count zero and the historical 5k seed/5k initial-update schedule.

## Outputs and next gate

- [train_console.log](train_console.log): live console output; [logs](logs/) receives append-only `metrics.jsonl`, `episodes.csv`, and `updates.csv`.
- [checkpoints](checkpoints/) receives full-state checkpoints every 10k steps, including `step_200000.pt`, `step_220000.pt`, `step_240000.pt`, `step_260000.pt`, `step_280000.pt`, and `step_300000.pt` (and intervening 10k steps). No automatic extension beyond 300k.
- [evaluation](evaluation/) currently contains the smoke evaluation records; training itself has **no online evaluation cadence** in the historical `train.py`.
- Training ignores force in observation, reward, policy/planner objective, action, and termination. No Force Head, CBF, shield, or safety objective is active. The Generic solver and task remain frozen, carrying forward the documented G0.5 conditional finite-radius contact caveat.

After completion, a **separate** 200k–300k navigation checkpoint maturity sweep must choose the baseline by success, episode length, reward, and stability—not by force or safety performance. No checkpoint selection is performed here.
