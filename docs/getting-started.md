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
