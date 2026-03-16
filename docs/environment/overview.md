# Task Overview

The task is to balance a free ball on a plate attached to the Kinova end-effector.

The robot does not actuate the ball directly. It acts on the ball only through:

- plate position,
- plate velocity,
- plate acceleration,
- contact and friction at the ball-plate interface.

## Objective

A successful controller should:

- keep the ball near the plate center,
- reduce the ball's linear and angular speed,
- maintain contact with the plate,
- avoid aggressive arm motion that solves the task in an unrealistic way.

## Why This Is Not a Simple Positioning Problem

A ball near the center is not necessarily stable. It may still have high speed and immediately roll away. That is why the task reward and the documentation focus on both:

- position error,
- kinetic state.

This is closer to dynamic stabilization than to static set-point regulation.

## Policy Variants

### Baseline

- action space: joint-space position commands
- actor observations: joint state, end-effector state, F/T wrench
- randomization: ball mass, PD gains, null-space reset, robot model

### Cartesian

- action space: end-effector position around the episode's initial frame
- actor observations: end-effector position, velocity, F/T wrench
- randomization: ball mass, PD gains, robot model

### BaselineNoRobotModelRand

- same observation/action interface as baseline
- disables robot inertial randomization for ablation
