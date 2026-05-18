# Getting Started

## Requirements

- Python `>=3.12,<3.14`
- `uv`
- a local checkout of `mjlab` at `../mjlab`

The local dependency is configured in the repository `pyproject.toml`.

## Install

```bash
uv sync
```

Optional zsh completion for project task IDs after `uv run train` / `uv run play`:

```bash
source scripts/zsh/mjlab-completion.zsh
```

This completion helper only customizes:

- task ID completion after `uv run train` / `uv run play`
- file completion after `--checkpoint-file`

All other `uv` arguments continue to use the normal `uv` shell completion.

## Train

```bash
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

Kinova training defaults to Weights & Biases when it is configured in your shell. If W&B is not set up, training falls back to TensorBoard automatically.

```bash
uv run train Mjlab-BallBalancing-Kinova
```

Canonical variants:

```bash
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-Cartesian --env.scene.num-envs 512
```

Cartesian batch runs:

```bash
uv run kinova-train-set config/training_sets/cartesian_ablation_simple_rewards.yaml \
  -- --env.scene.num-envs 4096 --video True --video-interval 1000 --video-length 500 --env.viewer.max-extra-envs 0
```

Batch runs for one variant are defined under `config/training_sets/` and launched with:

```bash
uv run kinova-train-set config/training_sets/joint_ablation.yaml -- --env.scene.num-envs 512
```

To launch only selected runs from a larger set:

```bash
uv run kinova-train-set config/training_sets/joint_randomization_ablation.yaml \
  --run ball_reset_only \
  -- --env.scene.num-envs 1 --agent.max_iterations 20
```

For extracted single-run configs under `config/training_sets/joint_randomization_ablation/`, use:

```bash
uv run kinova-train-run config/training_sets/joint_randomization_ablation/ball_reset_only.yaml \
  -- --env.scene.num-envs 1 --agent.max_iterations 20
```

To estimate per-run and whole-set min/max duration bounds without launching training:

```bash
uv run kinova-train-set-duration config/training_sets/joint_randomization_ablation.yaml
```

You can also provide an aggregate throughput estimate to convert environment steps into approximate wall-clock time:

```bash
uv run kinova-train-set-duration config/training_sets/joint_randomization_ablation.yaml \
  --env-steps-per-second 250000
```

That launcher merges:

- the base task parameters,
- zero or more preset YAMLs from `config/presets/`,
- per-run overrides from the training-set file.

For W&B-backed launches from `kinova-train-set`:

- the training-set file stem becomes the W&B project name with a `YYYYMMDD_HHMMSS` suffix
- the YAML `runs[].name` becomes the run display name
- a W&B-compatible run ID is derived from `runs[].name`

For `kinova-train-run`:

- `WANDB_PROJECT` is fixed to `kinova_ping_pong`
- the single-run name becomes the W&B display name
- the W&B-compatible run ID is still derived from the run name

Any arguments after `--` are forwarded to the underlying `train` command for every run in the set. This is the easiest way to apply a common CLI override such as:

```bash
uv run kinova-train-set config/training_sets/joint_randomization_ablation.yaml -- --env.scene.num-envs 512
```

Legacy aliases still exist for compatibility:

```bash
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRand --env.scene.num-envs 512
```

## Play a Trained Policy

```bash
uv run play Mjlab-BallBalancing-Kinova \
  --checkpoint-file logs/rsl_rl/kinova_ball_balancing_joint/.../model_*.pt
```

To resolve play parameters from a named run inside a training set:

```bash
uv run kinova-play-set config/training_sets/joint_randomization_ablation.yaml \
  --run ball_reset_only \
  -- --agent random --num-envs 1 --viewer native
```

Add `--ball-kick` before the `--` separator to replay with interval disturbance kicks enabled.

To resolve play parameters from one extracted single-run config:

```bash
uv run kinova-play-run config/training_sets/joint_randomization_ablation/ball_reset_only.yaml \
  -- --agent random --num-envs 1 --viewer native
```

`kinova-play-run` also supports `--ball-kick` before the `--` separator.

## GPU Fallback

If no CUDA device is visible to the training process, the Kinova package now falls back to CPU instead of crashing during GPU selection.

If you expected to use a GPU, verify that:

- `nvidia-smi` works in the same shell,
- `CUDA_VISIBLE_DEVICES` is not masking the device,
- your `uv run ...` process has access to the NVIDIA driver stack.

## Wiki Export

The project documentation is exported to GitHub Wiki-compatible Markdown files.

To regenerate the wiki pages from `docs/`:

```bash
python3 scripts/export_github_wiki.py --output-dir wiki
```

The generated files can be synced into the separate GitHub Wiki checkout of this repository.

## Configure Parameters

Task and training parameters are centralized in `src/mjlab_kinova/tasks/task_parameters.py`.

The default YAML file is:

- `config/task_parameters.yaml`

Preset YAMLs live in:

- `config/presets/`

Training-set batch definitions live in:

- `config/training_sets/`

Extracted one-run configs for direct `kinova-train-run` / `kinova-play-run` use live in:

- `config/training_sets/joint_randomization_ablation/`

The registered tasks now load that YAML automatically at import time. To point the task registry to a different file, set:

```bash
export MJLAB_KINOVA_TASK_PARAMS=/path/to/task_parameters.yaml
```

The repository also includes reusable presets under `config/presets/` and example batch experiment sets under `config/training_sets/`.

The migrated experiment presets such as `nominal`, `high_damping`, `high_disturbance`, `contact_focus`, and `stable_ppo` are set up as medium runs with `ppo.max_iterations: 3000`.

Those files are the easiest place to tune:

- ball properties,
- observation noise,
- action scales,
- reward weights and contact shaping,
- reset/randomization ranges,
- simulation timing,
- PPO hyperparameters.

For programmatic overrides, load the YAML and pass the resulting object into the builders:

```python
from mjlab_kinova.tasks.kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
    kinova_ppo_runner_cfg,
)
from mjlab_kinova.tasks.task_parameters import load_task_parameters

params = load_task_parameters("config/task_parameters.yaml")

env_cfg = kinova_ball_balancing_env_cfg(variant="joint", params=params)
rl_cfg = kinova_ppo_runner_cfg(variant="joint", params=params)
```

A training set file has the shape:

```yaml
variant: joint
global_overrides:
  ppo:
    max_iterations: 30000
runs:
  - name: default
    presets: []
    overrides: {}

  - name: no_model_rand
    presets: [no_model_rand]
    overrides: {}
```

`global_overrides` applies to every run in the set after the base config is loaded and before presets or per-run overrides are merged.

The `runs:` list can also contain string references to other training-set YAML files, as long as each referenced file contains exactly one run. `config/training_sets/joint_randomization_ablation.yaml` uses this pattern to point at the extracted one-run configs.

Each training set can declare a default stop policy, and each run can override it:

```yaml
variant: joint
default_stop_policy:
  min_iterations: 500
  max_iterations: 3000
  combine: all
  criteria:
    - metric: Train/mean_episode_length
      threshold: 900.0
      mode: max
      window: 5
      patience: 3
    - metric: Train/mean_reward
      threshold: 8.0
      mode: max
      window: 5
      patience: 3

runs:
  - name: all_randomization
    presets: [all_randomization]
    overrides: {}
    stop_policy:
      min_iterations: 1000
      max_iterations: 4000
      combine: any
      criteria:
        - metric: Train/mean_episode_length
          threshold: 950.0
          mode: max
          window: 5
          patience: 2
        - metric: Train/mean_reward
          threshold: 10.0
          mode: max
          window: 8
          patience: 2
```

Supported stop-policy fields are:

- `min_iterations`: earliest training iteration at which convergence stopping is allowed
- `max_iterations`: hard cap for that run, even if convergence criteria never trigger
- `combine`: `all` or `any` across the listed criteria
- `criteria`: list of metric thresholds to evaluate

Each criterion supports:

- `metric`: metric name produced by the runner, for example `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value_loss`, or episode extras such as `Episode_Reward/racquet_centering`
- `threshold`: numeric target value
- `mode`: `max` or `min`
- `window`: moving-average window in training iterations
- `patience`: number of consecutive windows that must satisfy the threshold

The older single-rule `stop_condition` form still works for compatibility, but `default_stop_policy` and `stop_policy` are now the preferred schema.

For the example above, runs are logged under the W&B project `joint_ablation`, and the `no_model_rand` entry uses `no_model_rand` as its run name and run ID.

## Reading Training Logs

The console output during training mixes PPO optimization metrics with episode statistics:

- `Mean reward`: average completed-episode return seen by the learner. Higher is better.
- `Mean episode length`: average episode duration in environment steps, not seconds.
- `Mean value loss`: critic fit error. Large early values are normal if reward scale is large.
- `Mean surrogate loss`: PPO policy objective. Near zero is common once updates are conservative.
- `Mean action noise std`: current policy exploration scale.

Reward breakdown lines such as `Episode_Reward/racquet_centering` are averaged episode-term contributions normalized by episode horizon. They are most useful for trend comparison across training, not as standalone “scores.”

Termination lines such as `Episode_Termination/ball_fell_off` count how often resets happened for each condition inside the current logging window. They are not probabilities.
