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
- do so while the ball is actually supported by the plate,
- reduce the ball's linear and angular speed,
- maintain contact with the plate,
- avoid aggressive arm motion that solves the task in an unrealistic way.

## Why This Is Not a Simple Positioning Problem

A ball near the center is not necessarily stable. It may still have high speed and immediately roll away. That is why the task reward and the documentation focus on both:

- position error,
- contact support,
- kinetic state.

This is closer to dynamic stabilization than to static set-point regulation.

## Control-Space Variants

The repo now distinguishes between:

- control-space variants: `joint` and `cartesian`
- training presets: joint-space ablations such as `no_model_rand` and `no_rand`

### Joint

- action space: joint-space position commands
- actor observations: joint offsets `q - q_0`, joint velocity, end-effector state, racquet-weight-compensated F/T wrench, previous action
- ball centering reward: active only while ball-racquet contact is present
- randomization: ball mass and friction, PD gains, null-space reset, robot model

### Cartesian

- action space: 6D end-effector pose delta around the episode's initial racquet frame
- actor observations: end-effector pose relative to the nominal pose `(p_0, r_0)`, end-effector velocity, racquet-weight-compensated F/T wrench, previous action
- ball centering reward: active only while ball-racquet contact is present
- randomization: ball mass and friction, PD gains, robot model

### Joint Preset: NoRobotModelRand

- same observation/action interface as the joint variant
- disables PD-gain randomization, robot inertial randomization, and null-space reset randomization for ablation

### Joint Preset: NoRand

- same observation/action interface as the joint variant
- disables observation noise, stochastic resets, parameter randomization, and training kicks
