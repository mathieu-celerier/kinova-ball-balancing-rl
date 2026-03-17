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

## Train

```bash
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

Other variants:

```bash
uv run train Mjlab-BallBalancing-Kinova-Baseline --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-Cartesian --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRand --env.scene.num-envs 512
```

## Play a Trained Policy

```bash
uv run play Mjlab-BallBalancing-Kinova --checkpoint-file logs/rsl_rl/kinova_ball_balancing/...
```

## Build the Documentation Site

The repository includes an MkDocs configuration.

To install the docs tooling through the project dependency group:

```bash
uv sync --group docs
```

If MkDocs is installed in your environment, run:

```bash
mkdocs serve
```

or:

```bash
mkdocs build
```

This serves the documentation as a normal project docs website instead of a single README page.

## Configure Parameters

Task and training parameters are centralized in `src/mjlab_kinova/tasks/task_parameters.py`.

The default YAML file is:

- `config/task_parameters.yaml`

The registered tasks now load that YAML automatically at import time. To point the task registry to a different file, set:

```bash
export MJLAB_KINOVA_TASK_PARAMS=/path/to/task_parameters.yaml
```

The repository also includes a small initial sweep under `config/sweeps/`.

Those sweep configs are set up as medium runs with `ppo.max_iterations: 3000`.

That file is the easiest place to tune:

- ball properties,
- observation noise,
- action scales,
- reward weights,
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

env_cfg = kinova_ball_balancing_env_cfg(variant="baseline", params=params)
rl_cfg = kinova_ppo_runner_cfg(variant="baseline", params=params)
```
