# Mjlab Kinova Gen3 - Ball Balancing Task

## Install

```
uv sync
```

## Task

This package registers a Kinova ball balancing task in MjLab:
- `Mjlab-BallBalancing-Kinova` (train)
- `Mjlab-BallBalancing-Kinova-Play` (play alias)

Task objective:
- keep a free ball centered on the plate attached to the end-effector
- penalize ball speed and jerky arm actions
- terminate when the ball falls off the plate

## Training an agent

```
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

## Playing the environment

```
uv run play Mjlab-BallBalancing-Kinova --checkpoint-file logs/rsl_rl/kinova_ball_balancing/...
```
