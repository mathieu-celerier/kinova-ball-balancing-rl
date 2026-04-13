# MjLab Kinova Ball Balancing

This repository provides reinforcement-learning environments for balancing a free ball on a plate mounted to a Kinova Gen3 end-effector in MuJoCo through MjLab.

## Documentation

The project now includes a proper docs-site structure under [`docs/`](./docs) with MkDocs configuration in [`mkdocs.yml`](./mkdocs.yml).

For task tuning, the main parameter entry points are [`config/task_parameters.yaml`](./config/task_parameters.yaml) and [`src/mjlab_kinova/tasks/task_parameters.py`](./src/mjlab_kinova/tasks/task_parameters.py).

The task registry loads `config/task_parameters.yaml` automatically by default. You can override the path with `MJLAB_KINOVA_TASK_PARAMS`.

The canonical control-space variants are:

- `Mjlab-BallBalancing-Kinova` for joint-space control
- `Mjlab-BallBalancing-Kinova-Cartesian` for Cartesian end-effector control

Joint-space ablations such as "no model randomization" and "no randomization" are now handled through preset YAMLs in [`config/presets/`](./config/presets) and batch training-set files in [`config/training_sets/`](./config/training_sets).

Single-run copies of the joint randomization ablations live in [`config/training_sets/joint_randomization_ablation/`](./config/training_sets/joint_randomization_ablation).

Main pages:

- [Home](./docs/index.md)
- [Getting Started](./docs/getting-started.md)
- [Task Overview](./docs/environment/overview.md)
- [MDP Design](./docs/environment/mdp.md)
- [Physics and Control](./docs/environment/physics.md)
- [Robot and MuJoCo Model](./docs/environment/robot-model.md)
- [Project Structure](./docs/code/structure.md)

## Install

```bash
uv sync
```

## Train

```bash
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

```bash
uv run train Mjlab-BallBalancing-Kinova-Cartesian --env.scene.num-envs 512
```

Kinova training now defaults to Weights & Biases when it is configured in your shell. If W&B is not set up, training falls back to TensorBoard automatically.

```bash
uv run train Mjlab-BallBalancing-Kinova
```

To launch a set of runs for one variant:

```bash
uv run kinova-train-set config/training_sets/joint_ablation.yaml -- --env.scene.num-envs 512
```

To launch one extracted run cleanly:

```bash
uv run kinova-train-run config/training_sets/joint_randomization_ablation/ball_reset_only.yaml \
  -- --env.scene.num-envs 512
```

To estimate per-run and whole-set min/max duration bounds:

```bash
uv run kinova-train-set-duration config/training_sets/joint_randomization_ablation.yaml
```

That workflow merges the base task parameters, any selected preset YAMLs, and the per-run overrides declared in the training-set file.

Arguments after `--` are forwarded to the underlying `train` command for every run in the set, so common CLI overrides such as `--env.scene.num-envs 512` can be applied once at launch time.

For W&B-backed launches from `kinova-train-set`:

- the training-set file stem becomes the W&B project name, with a `YYYYMMDD_HHMMSS` suffix added per invocation
- the YAML `runs[].name` becomes the run display name
- a W&B-compatible run ID is derived from `runs[].name` and exported to the training process

For `kinova-train-run`:

- `WANDB_PROJECT` is fixed to `kinova_ping_pong`
- the single run name becomes `WANDB_NAME`
- the W&B run ID is derived from that run name

Each training set can declare a metric-based `default_stop_policy`, and each run can override it with its own `stop_policy`, including multiple criteria plus explicit `min_iterations` and `max_iterations`.

## Play

```bash
uv run play Mjlab-BallBalancing-Kinova \
  --checkpoint-file logs/rsl_rl/kinova_ball_balancing_joint/.../model_*.pt
```

To play from a training-set entry:

```bash
uv run kinova-play-set config/training_sets/joint_randomization_ablation.yaml \
  --run ball_reset_only \
  -- --agent random --num-envs 1 --viewer native
```

To play from one extracted run config directly:

```bash
uv run kinova-play-run config/training_sets/joint_randomization_ablation/ball_reset_only.yaml \
  -- --agent random --num-envs 1 --viewer native
```

## Serve the Docs

If MkDocs is installed in your environment:

```bash
mkdocs serve
```

or install the docs tooling with:

```bash
uv sync --group docs
```
