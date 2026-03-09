# Mjlab Kinova Gen3 - Ball Balancing Task

## Install

```
uv sync
```

## Task

This package registers Kinova ball balancing tasks in MjLab:
- `Mjlab-BallBalancing-Kinova` and `Mjlab-BallBalancing-Kinova-Baseline`
- `Mjlab-BallBalancing-Kinova-Cartesian`
- `Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand`
- `Mjlab-BallBalancing-Kinova-Play` (baseline play alias)

Task objective:
- slow down and stabilize a free ball on the plate attached to the end-effector
- use force/torque sensing with either joint-space or cartesian control
- terminate when the ball falls off the plate

Policy variants:
- `Baseline`: joint-space policy with joint and end-effector observations, PD gain randomization, null-space reset randomization, and robot-model randomization.
- `Cartesian`: task-space end-effector policy anchored to the per-episode initial racket pose and realized through differential IK to the existing joint PD actuators, with PD gain and robot-model randomization.
- `BaselineNoRobotModelRand`: same as `Baseline` but disables robot-model randomization.

## Training an agent

```
uv run train Mjlab-BallBalancing-Kinova --env.scene.num-envs 512
```

Examples:

```
uv run train Mjlab-BallBalancing-Kinova-Baseline --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-Cartesian --env.scene.num-envs 512
uv run train Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand --env.scene.num-envs 512
```

## Playing the environment

```
uv run play Mjlab-BallBalancing-Kinova --checkpoint-file logs/rsl_rl/kinova_ball_balancing/...
```
