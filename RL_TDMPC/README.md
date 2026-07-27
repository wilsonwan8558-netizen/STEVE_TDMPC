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
transitions with both requested and applied normalized actions and supports
uniform, translation-balanced, curvature-balanced, and mixed sampling with an
RNG independent from the main temporal replay.

The persisted `action` field is the normalized action requested by the caller.
The separate `applied_action` field is the intervention's command after its
translation constraint mask, normalized back to the public `[-1, 1]` action
range. It is not measured/realized guidewire motion: a blocked translation can
therefore have a nonzero requested component and a zero applied component,
while actual motion remains a property of the resulting simulation state.

The auxiliary curvature strata are ordered as:

```text
0 low      curvature < 0.05 mm^-1
1 medium   0.05 <= curvature < 0.10 mm^-1
2 high     0.10 <= curvature < 0.25 mm^-1
3 extreme  curvature >= 0.25 mm^-1
```

The collector can repeat controlled scenarios with deterministic, distinct
seeds and save every seed used in the report and dataset metadata. Its main
controls are:

- `--lower-boundary-repetitions`, `--tree-end-repetitions`, and
  `--device-length-repetitions`;
- `--random-episodes` and `--curvature-episodes`;
- `--seed-start` (with `--seed` retained as an alias),
  `--tree-end-seed`, and `--device-length-seed`;
- `--target-none`, `--target-lower-boundary`, and `--target-tree-end`;
- `--target-low`, `--target-medium`, `--target-high`, and
  `--target-extreme`;
- `--max-curvature-transitions`, `--capacity`, `--split-seed`, and
  `--validation-fraction`;
- `--observation-round-decimals` and the opt-in
  `--deduplicate-exact`.

Collect all supported scenarios with the small defaults and write the
dataset/report under `/tmp`:

```bash
python RL_TDMPC/collect_safety_dataset.py
```

A practical initial collection command is:

```bash
python RL_TDMPC/collect_safety_dataset.py \
  --modes all \
  --lower-boundary-repetitions 50 \
  --tree-end-repetitions 60 \
  --device-length-repetitions 1 \
  --random-episodes 3 \
  --curvature-episodes 6 \
  --max-curvature-transitions 1200 \
  --seed-start 7 \
  --tree-end-seed 301 \
  --device-length-seed 301 \
  --target-none 500 \
  --target-lower-boundary 100 \
  --target-tree-end 100 \
  --target-low 100 \
  --target-medium 500 \
  --target-high 300 \
  --target-extreme 100 \
  --capacity 3000 \
  --split-seed 4602 \
  --validation-fraction 0.20 \
  --observation-round-decimals 6 \
  --output-dataset /tmp/steve_commit46b_safety_aux.pt \
  --output-report /tmp/steve_commit46b_safety_aux_report.json
```

The collector computes a conservative storage upper bound before creating the
SOFA environment and rejects an undersized `--capacity`; controlled records
therefore cannot be silently overwritten by the ring buffer. Dataset and
report paths must also be different. Tree-end and device-length attempts use
independent `--tree-end-seed` and `--device-length-seed` controls.

Each lower-boundary repetition records moderate/maximum blocked retractions
and matched zero/forward controls. Each successful tree-end construction
records moderate/maximum blocked insertions and zero, rotation-only, and
retraction controls from the endpoint neighborhood. Labels always come from
the intervention; the collector never relabels a transition to satisfy a
target. Curvature collection stops when the requested strata have been
covered, the configured transition limit is reached, or the trajectory budget
is exhausted. The JSON report records requested, collected, and unmet counts
and the number of transitions examined.

With the default controlled seed, the fixed vessel reaches a tree endpoint
before the 450 mm J-shaped guidewire limit, so `device_length_limit` is
reported as unsupported rather than fabricated. If an episode ends before any
blocker is observed, the report says `not_observed` instead. Curvature strata
are diagnostic sampling bins configured under `diagnostics`; they are not
clinical safety thresholds.

### Dataset validation, split, and inspection

The saved file uses offline Safety dataset schema version 1 and embeds a
strict Safety auxiliary replay schema version 2. It contains:

- the ordered safety-cost, blockage-reason, and curvature-stratum names;
- curvature boundaries and observation/action dimensions;
- all dataset-generation seeds;
- one fixed joint `(blockage reason, curvature stratum)` split;
- explicit, disjoint train and validation indices;
- the split seed and requested validation fraction; and
- a SHA256 dataset fingerprint.

The default split is 80% training and 20% validation within each joint group.
A group with at least five transitions contributes at least one validation
transition while retaining training data; a smaller group remains train-only.
Membership is assigned deterministically from the split seed and saved once,
so validation data are never resampled by the training-buffer API.

Duplicate diagnostics report exact semantic duplicates based on observation,
requested action, safety cost, blockage reason, and curvature stratum;
repeated observation-action pairs; and unique observations after rounding to
`--observation-round-decimals`. Diagnostics do not remove samples.
`--deduplicate-exact` optionally retains the first exact representative and
reports removals, but is disabled by default because repeated controlled
commands can be legitimate data.

Inspect a saved dataset without constructing a stEVE/SOFA environment:

```bash
python RL_TDMPC/inspect_safety_dataset.py \
  --dataset /tmp/steve_commit46b_safety_aux.pt
```

Inspection first performs strict schema, ordered-name, curvature-boundary,
split-integrity, and fingerprint validation, then prints total/train/validation
sizes and their reason, curvature, joint, duplicate, and safety-cost
statistics as strict JSON.

### Balanced offline Safety supervision

Normal training leaves offline supervision disabled. Enable it by setting the
complete strict configuration:

```yaml
safety:
  loss_coef: 0.1

safety_aux:
  enabled: true
  dataset_path: /tmp/steve_commit46b_safety_aux.pt
  loss_coef: 1.0
  curvature_loss_coef: 1.0
  translation_loss_coef: 1.0
  translation_group_weights:
    none: 2.0
    lower_insertion_boundary: 1.0
    device_length_limit: 1.0
    vessel_tree_end: 1.0
    other: 1.0
  translation_zero_calibration_coef: 0.1
  validation_translation_threshold_candidates:
    [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.002, 0.005,
     0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  batch_size: 64
  update_interval: 1
  sampling_mode: mixed
  translation_fraction: 0.5
  curvature_fraction: 0.5
  sample_with_replacement: true
```

The main temporal replay and its RNG are unchanged. Each scheduled auxiliary
contribution samples only the dataset's stored training split. `mixed`
sampling allocates half of the batch across available translation-block
reasons and half across available curvature strata; absent groups are reported
and never fabricated. The complete stored validation split is evaluated
deterministically at `diagnostics.validation_interval`.

For an auxiliary transition, the encoder runs under `torch.no_grad()` and its
latent is detached before the existing Safety Head receives the requested
normalized action. The two targets use the same `log1p(cost / scale)`
transform and Smooth L1 loss as main Safety training, without temporal rho
weighting. Curvature loss is averaged within each available stored stratum and
then equally across available strata. Translation loss is averaged within each
available canonical blockage-reason group and combined as
`sum(weight[group] * loss[group]) / sum(available weights)`, so group size does
not silently become a loss weight. Missing groups are logged as unavailable
and are never fabricated.

`translation_zero_calibration_coef` adds an auxiliary-only squared penalty on
decoded translation predictions whose stored target is exactly zero. Positive
blockage targets are excluded. The channel coefficients, group weights, and
zero-calibration coefficient affect only offline auxiliary supervision; they
do not change the main-replay Safety loss. The default `none: 2.0` weight and
the threshold candidates are engineering diagnostics, not clinical limits.

Only the Safety trunk and its two output branches receive auxiliary gradients.
Encoder, dynamics, reward, termination, Q, and policy parameters receive only
their original main-update gradients. The existing model optimizer performs
one step; no second optimizer is introduced.

Enabled checkpoints store the resolved auxiliary configuration, dataset
fingerprint and fixed split identity, exact loss/calibration metadata,
sampler-only RNG state, and update counters. Dataset transitions are not
duplicated into checkpoints. Resume requires the same strict dataset,
coefficients, ordered group weights, zero calibration, and threshold candidate
list, and it reproduces the next auxiliary batch. Enabled checkpoints that
predate Commit 4.6D are intentionally rejected because they do not contain
this calibration contract. Existing format-v3 checkpoints created with
auxiliary supervision disabled remain compatible.

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

### Safety Head evaluation

Evaluate the complete fixed auxiliary validation split without constructing a
SOFA environment:

```bash
python RL_TDMPC/evaluate_safety_head.py \
  --checkpoint RL_TDMPC/checkpoints/step_20000.pt \
  --safety-dataset /tmp/steve_commit46b_safety_aux.pt \
  --output-json /tmp/steve_safety_offline.json
```

The report contains overall, per-translation-reason, and four-stratum
curvature metrics in original and transformed units. For translation blockage
it also reports the complete configured threshold table, confusion counts,
precision/recall/F1, false-positive/false-negative rates, specificity,
balanced accuracy, ROC AUC, PR AUC, prediction means, and the configured
percentiles for normal and positive samples. It selects three deterministic
diagnostic operating points: maximum F1, lowest-FPR among candidates with
recall at least 0.95, and highest-recall among candidates with false-positive
rate at most 0.05. Unavailable values are written as strict JSON `null`.

The legacy `1e-6` result remains in the report to show mathematically nonzero
predictions, but neither it nor a calibrated operating point is a clinical
safety limit. Requested actions are used as Safety Head inputs, and the fixed
validation split is never sampled or modified.

Add explicit policy episodes and targeted real blockage checks to run a
combined online/offline evaluation:

```bash
python RL_TDMPC/evaluate_safety_head.py \
  --checkpoint RL_TDMPC/checkpoints/step_20000.pt \
  --safety-dataset /tmp/steve_commit46b_safety_aux.pt \
  --episodes 3 \
  --targeted-blockage \
  --output-json /tmp/steve_safety_combined.json
```

Supplying `--safety-dataset` without `--episodes` or
`--targeted-blockage` is deliberately offline-only. Safety predictions remain
diagnostic and are never called by MPC or used to alter actions. When targeted
checks and a fixed dataset are requested together, the lower-boundary and
tree-end predictions are compared with every selected validation threshold;
the comparison is report-only.

## Logs

Training writes append-only records under `RL_TDMPC/logs/`:

- `metrics.jsonl`: all episode and update records;
- `episodes.csv`: reward, length, success, cumulative success rate, and steps;
- `updates.csv`: total/model/policy losses, per-group auxiliary losses,
  zero-calibration loss, prediction means, auxiliary-only gradient norms, and
  total environment steps;
- `safety_aux_sampling.csv`: sampled group counts, unavailable groups, unique
  exposure, and duplicate exposure for every scheduled auxiliary batch;
- `safety_aux_validation.csv`: complete-split validation summaries, selected
  calibration operating points, AUCs, and prediction distribution summaries.

Evaluation writes per-episode records and a summary under
`RL_TDMPC/logs/evaluation/`.
