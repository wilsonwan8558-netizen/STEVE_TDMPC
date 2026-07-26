# TD-MPC2 for stEVE

This directory contains a minimal, single-task integration of the official
[TD-MPC2 implementation](https://github.com/nicklashansen/tdmpc2) with stEVE.
The algorithm code follows official commit
`e9f59321933cbc8e11a002b842adc7d4ffae8ff1`. It keeps the state encoder,
SimNorm latent dynamics, distributional reward and value models, Q ensemble,
Gaussian policy prior, TD learning losses, target critics, and latent MPPI
planner. Multi-task/offline training, pixels, external benchmark suites,
Hydra, TorchRL, W&B, and video code are omitted.

The adapted TD-MPC2 files retain the upstream MIT license in
`tdmpc2/LICENSE`.

## Environment

`envs/steve_env.py` follows `examples/function_check.py` and uses the same:

- `AorticArch(seed=30)` vessel tree and scaling;
- one `JShaped` guidewire;
- `SofaBeamAdapter(friction=0.001)`;
- `TrackingOnly` at 7.5 Hz and projection `[20, 5]`;
- random centerline target and branch set;
- 2-D tracking, target, and rotation observations;
- target plus path-length-delta reward;
- target-reached termination and 200-step truncation.

The one required task change is the start strategy. The reference script uses
`MaxDeviceLength(max_length=500)`, but the J-shaped wire is only 450 mm long,
so that condition never resets the physical device between episodes. This
adapter uses `InsertionPoint`, making RL episodes independent without changing
stEVE source code.

The observation dictionary is flattened in the stable order
`position -> target -> rotation` to a finite `float32` vector of shape `(14,)`.
The public action is a `float32` vector of shape `(2,)` in `[-1, 1]` and maps
to translation/rotation velocity limits of approximately
`[-50, 50] mm/s` and `[-3.14, 3.14] rad/s`.

stEVE's `MonoPlaneStatic.reset()` does not forward its seed to the SOFA
adapter. The wrapper seeds that adapter's existing random generator before the
device reset so initial rotations and evaluation episodes are reproducible.

## Quick start (`steve_tdmpc`)

Run all commands from the stEVE repository root. In every new terminal, first
activate Conda and source the supplied SOFA setup:

```bash
conda activate steve_tdmpc
source RL_TDMPC/setup_env.sh
```

The setup script deliberately removes inherited paths from the old `stenv`
SOFA installation. Mixing the new SOFA core with the old BeamAdapter causes
`undefined symbol` and `WireRestShape was not created` errors.

To use another SOFA installation, set it before sourcing the script:

```bash
export STEVE_SOFA_ROOT=/path/to/sofa/install
source RL_TDMPC/setup_env.sh
```

First-time Python package installation:

```bash
python -m pip install -e .
python -m pip install -r RL_TDMPC/requirements.txt
```

Verify that SofaPython3 and the matching BeamAdapter load correctly:

```bash
python -c "import Sofa, SofaRuntime; print(Sofa.__file__); SofaRuntime.importPlugin('BeamAdapter')"
```

Both paths printed by Python/SOFA must begin with
`/home/bizon/SOFA_TDMPC/install/` (or your `STEVE_SOFA_ROOT`).

### Daily commands

Train:

```bash
python RL_TDMPC/train.py
```

Resume the existing run to a final total of 1,000,000 environment steps:

```bash
python RL_TDMPC/train.py \
  --resume RL_TDMPC/checkpoints/step_270000.pt \
  --steps 1000000
```

Evaluate without a window:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_270000.pt
```

Evaluate with the SOFA/Pygame window:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_270000.pt \
  --render
```

Evaluate and save an MP4 video (this automatically opens the render window):

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_270000.pt \
  --episodes 1 \
  --video RL_TDMPC/logs/evaluation/step_270000.mp4
```

The default video rate is 7.5 FPS, matching the environment. Override it with
`--video-fps 15` if desired. Video recording requires a desktop/OpenGL display.

## Verify the wrapper first

Run the required pre-training acceptance test:

```bash
python RL_TDMPC/smoke_test_env.py
```

It checks `reset()` and `step()`, observation shape/dtype/finiteness, normalized
action conversion, physical reset behavior, the Gymnasium contract, and one
complete 200-step episode without a GUI.

After installing PyTorch, the algorithm-only smoke test verifies replay
sampling, one complete TD-MPC2 update, latent planning, and state restoration:

```bash
python RL_TDMPC/smoke_test_tdmpc2.py
```

The algorithm smoke test also runs the simulation-free Safety auxiliary replay
checks. They can be run separately while developing the data layer:

```bash
python RL_TDMPC/smoke_test_safety_aux.py
```

## Safety auxiliary dataset

`info["safety_metrics"]` exposes the intervention-time translation blockage
reason as one canonical ID/name pair:

```text
0 none
1 lower_insertion_boundary
2 device_length_limit
3 vessel_tree_end
4 other
```

This metadata is not part of the 14-D observation, reward, termination, or
two-channel Safety Head target. `SafetyAuxReplayBuffer` stores individual
`(observation_t, normalized_action_t, safety_cost_t)` transitions and supports
uniform, translation-balanced, curvature-balanced, and mixed sampling with an
RNG independent from the main temporal replay.

The auxiliary curvature strata are ordered as:

```text
0 low      curvature < 0.05 mm^-1
1 medium   0.05 <= curvature < 0.10 mm^-1
2 high     0.10 <= curvature < 0.25 mm^-1
3 extreme  curvature >= 0.25 mm^-1
```

Collect all supported scenarios and write the dataset/report under `/tmp`:

```bash
python RL_TDMPC/collect_safety_dataset.py
```

Collect selected modes with reproducible seeds:

```bash
python RL_TDMPC/collect_safety_dataset.py \
  --modes random lower-boundary vessel-tree-end curvature-coverage \
  --random-episodes 5 \
  --curvature-episodes 5 \
  --seed 7 \
  --output-dataset /tmp/steve_safety_aux_seed7.pt \
  --output-report /tmp/steve_safety_aux_seed7.json
```

The collector computes a conservative storage upper bound before creating the
SOFA environment and rejects an undersized `--capacity`; controlled records
therefore cannot be silently overwritten by the ring buffer. Dataset and
report paths must also be different. Tree-end and device-length attempts use
independent `--tree-end-seed` and `--device-length-seed` controls.

With the default controlled seed, the fixed vessel reaches a tree endpoint
before the 450 mm J-shaped guidewire limit, so `device_length_limit` is
reported as unsupported rather than fabricated. If an episode ends before any
blocker is observed, the report says `not_observed` instead. Curvature strata
are diagnostic sampling bins configured under `diagnostics`; they are not
clinical safety thresholds.

Normal training leaves `safety_aux.enabled: false`. If explicitly enabled, it
collects and checkpoints the auxiliary replay but never samples it in
`agent.update()`; main replay sampling, TD targets, losses, policy, and MPC
remain unchanged.

## Training

Start training with the main configuration:

```bash
python RL_TDMPC/train.py
```

Configuration is in `configs/steve.yaml`. Important settings include total and
random-seed steps, replay capacity, batch size, model dimensions, planner
samples, log interval, and checkpoint interval. Relative output paths are
always resolved inside `RL_TDMPC`, independent of the shell's working
directory.

By default checkpoints are written every 10,000 environment steps:

```text
RL_TDMPC/checkpoints/step_10000.pt
RL_TDMPC/checkpoints/step_20000.pt
```

Each checkpoint contains the model, target critics, both optimizers, value
scale, planner state, environment/update counters, RNG state, configuration,
and (by default) replay data. Checkpointing occurs at exact environment-step
boundaries. A resumed run starts a fresh physical episode because a live SOFA
scene cannot be serialized.

Full-state checkpoints use PyTorch serialization and should only be loaded
from trusted sources.

Useful command-line overrides:

```bash
python RL_TDMPC/train.py --steps 200000 --device cuda
python RL_TDMPC/train.py --checkpoint-interval 5000
```

For a short integration test rather than a meaningful training run:

```bash
python RL_TDMPC/train.py \
  --steps 205 --seed-steps 200 --initial-updates 1 \
  --batch-size 4 --num-samples 16 --iterations 1 \
  --checkpoint-interval 205 --device cpu
```

## Resume training

The configured `training.total_steps` is the final global step, not an
additional number of steps. It must be greater than the checkpoint step:

```bash
python RL_TDMPC/train.py \
  --resume RL_TDMPC/checkpoints/step_20000.pt
```

Unless `--config` is supplied, resumption reconstructs the model and training
settings from the configuration embedded in the checkpoint, then applies any
command-line overrides.

To extend beyond the value saved in the YAML/config checkpoint:

```bash
python RL_TDMPC/train.py \
  --resume RL_TDMPC/checkpoints/step_20000.pt \
  --steps 1000000
```

## Evaluation

Run deterministic/no-exploration evaluation with reproducible per-episode
seeds:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_20000.pt
```

Optional overrides:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_20000.pt \
  --episodes 20 --seed 123 --device cpu
```

To display the SOFA scene in a Pygame/OpenGL window during evaluation:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_20000.pt \
  --render
```

Rendering requires a desktop display and a working OpenGL context. Keep
`--render` disabled on headless machines unless display forwarding is set up.

To record the returned RGB frames as MP4, pass an output path. `--video`
automatically enables rendering, so `--render` is optional:

```bash
python RL_TDMPC/evaluate.py \
  --checkpoint RL_TDMPC/checkpoints/step_270000.pt \
  --episodes 1 \
  --video RL_TDMPC/logs/evaluation/step_270000.mp4
```

Evaluation uses the configuration embedded in the checkpoint unless
`--config` is supplied.

## Logs

Training writes append-only records under `RL_TDMPC/logs/`:

- `metrics.jsonl`: all episode and update records;
- `episodes.csv`: reward, length, success, cumulative success rate, and steps;
- `updates.csv`: total/model/policy losses and total environment steps.

Evaluation writes per-episode records and a summary under
`RL_TDMPC/logs/evaluation/`.
