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

## Train

```bash
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

Other variants:

```bash
uv run train Mjlab-BallBalancing-Kinova-Baseline --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-Cartesian --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand --env.scene.num-envs 512
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

That file is the easiest place to tune:

- ball properties,
- observation noise,
- action scales,
- reward weights,
- reset/randomization ranges,
- simulation timing,
- PPO hyperparameters.

For programmatic overrides, create a modified parameter object and pass it into the builders:

```python
from dataclasses import replace

from mjlab_kinova.tasks.kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
    kinova_ppo_runner_cfg,
)
from mjlab_kinova.tasks.task_parameters import DEFAULT_TASK_PARAMETERS

params = replace(
    DEFAULT_TASK_PARAMETERS,
    rewards=replace(DEFAULT_TASK_PARAMETERS.rewards, ball_centering=60.0),
    ppo=replace(DEFAULT_TASK_PARAMETERS.ppo, learning_rate=1e-4),
)

env_cfg = kinova_ball_balancing_env_cfg(variant="baseline", params=params)
rl_cfg = kinova_ppo_runner_cfg(variant="baseline", params=params)
```
