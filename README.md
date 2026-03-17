# MjLab Kinova Ball Balancing

This repository provides reinforcement-learning environments for balancing a free ball on a plate mounted to a Kinova Gen3 end-effector in MuJoCo through MjLab.

## Documentation

The project now includes a proper docs-site structure under [`docs/`](./docs) with MkDocs configuration in [`mkdocs.yml`](./mkdocs.yml).

For task tuning, the main parameter entry points are [`config/task_parameters.yaml`](./config/task_parameters.yaml) and [`src/mjlab_kinova/tasks/task_parameters.py`](./src/mjlab_kinova/tasks/task_parameters.py).

The task registry loads `config/task_parameters.yaml` automatically by default. You can override the path with `MJLAB_KINOVA_TASK_PARAMS`.

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

Kinova training defaults to TensorBoard logging. To use Weights & Biases for a specific run:

```bash
uv run train Mjlab-BallBalancing-Kinova --agent.logger wandb
```

## Play

```bash
uv run play Mjlab-BallBalancing-Kinova \
  --checkpoint-file logs/rsl_rl/kinova_ball_balancing_baseline/.../model_*.pt
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
