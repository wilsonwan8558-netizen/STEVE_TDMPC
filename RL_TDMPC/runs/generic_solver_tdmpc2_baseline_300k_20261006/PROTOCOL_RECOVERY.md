# G1 Original TD-MPC2 protocol recovery

The trusted historical Original checkpoint `/home/bizon/Desktop/stEVE/RL_TDMPC/checkpoints/step_260000.pt` has SHA256 `4ee8567514d80cf545e53b0a93c9400aaf90fb0eaba086c760c5cc59e14c1778`, format version 1, algorithm `TD-MPC2`, and `total_env_steps=260000`. Its embedded configuration exactly equals `RL_TDMPC/configs/steve.yaml` in this worktree. `RL_TDMPC/train.py`, `RL_TDMPC/tdmpc2`, the task wrapper, and that YAML have no committed changes since the initial Original TD-MPC2 commit `2a26bd7`. The literal July shell invocation is not archived; the documented entrypoint is `python RL_TDMPC/train.py`. No unspecified historical CLI overrides are presumed.

| Item | Historical Original value recovered from checkpoint/code |
|---|---|
| Training seed | `1`; episode seed is `1 + episode_index` |
| Original configured budget | `1,000,000` RL environment steps; G1 deliberately stops at `300,000` |
| Step definition | One normalized TD-MPC2 action followed by one `env.step`, i.e. 22 SOFA substeps at 0.006 s each |
| Initial exploration | `5,000` random-action environment steps; thereafter stochastic latent MPPI, `eval_mode=False` |
| Initial updates | `5,000` updates when replay first becomes sampleable after seed period |
| Ongoing updates | `1` per environment step |
| Replay | Episode-aware, capacity `1,000,000` transitions, horizon `3`, batch `256`, seeded `1` |
| Optimizers | Adam model/policy, base LR `0.0003`, encoder LR multiplier `0.3`, policy Adam epsilon `1e-5` |
| Other training terms | reward/value coefficients `0.1/0.1`, termination `1`, consistency `20`, `rho=0.5`, grad clip `20`, target `tau=0.01`, entropy `0.0001` |
| Planner | MPC enabled, horizon `3`, `6` iterations, `512` samples, `64` elites, `24` policy trajectories, min/max std `0.05/2.0`, temperature `0.5` |
| Model | 256-dimensional latent/encoder/MLP, two encoder layers, five Q functions, 101 value bins, dropout `0.01` |
| Episode/task | 200-step maximum; exact historical navigation environment config and target/reward/termination wiring preserved |
| Logging | Episode and update JSONL/CSV, updates logged every 100 steps |
| Checkpoints | Every `10,000` environment steps, full agent/optimizers/replay/RNG, plus final checkpoint |
| Evaluation cadence | No evaluation call inside `train.py`; separate `evaluate.py` defaults to 10 episodes, seed `1000` |
| Device | Historical config `training.device=auto`; G1 preflight requires CUDA and exposes only physical GPU 0, so `auto` resolves to visible `cuda:0` on RTX 5090 |

The G1 [config.yaml](config.yaml) changes only `training.total_steps` from 1M to 300k and the logging/checkpoint/evaluation output directories. The 10k cadence naturally yields `step_200000.pt`, `step_210000.pt`, …, `step_300000.pt`, including every requested sweep checkpoint. No checkpoint will be selected by force metrics. The solver adapter is frozen at G0/G0.5 SHA256 `f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67`.

The Original training stack does not import force, a Force Head, shield, CBF, or safety objective. The physical Generic solver can compute contact multipliers, but they are not in the 14-value observation, navigation reward, planner objective, action transform, or episode termination. `audit_protocol.py` checks provenance/config/CUDA before smoke and before full launch. The G1 smoke forces one diagnostic optimizer update before the configured 5k random-action seed period solely to test the update path; its checkpoint is isolated and is never resumed into the real run.
