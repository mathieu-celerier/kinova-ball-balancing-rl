# Key Modules

## `src/mjlab_kinova/tasks/__init__.py`

Registers the canonical control-space variants through MjLab's task registry, plus a few compatibility aliases backed by presets.

## `src/mjlab_kinova/tasks/kinova_ball_balancing_env_cfg.py`

Defines:

- control-space variants
- observations
- actions
- events
- rewards
- terminations
- PPO runner configuration

This is the main file to read if you want to understand the environment at the configuration level.

## `src/mjlab_kinova/tasks/ball_balancing_mdp.py`

Defines task-local computational pieces such as:

- ball state in the plate frame
- wrench access
- contact-aware reward helpers
- termination helpers
- ball reset
- disturbance kicks
- inertial randomization helpers

This is where the code most directly meets the mechanics of the task.

## `src/mjlab_kinova/tasks/policy_actions.py`

Implements the custom Cartesian action term `InitialFramePositionAction`.

Its main role is to reinterpret policy outputs as end-effector position offsets around the initial episode frame and pass those references through differential IK.

## `src/mjlab_kinova/tasks/task_parameters.py`

Centralizes the tunable task, training-behavior, and PPO parameters.

This is the best place to start if you want to change:

- reward weights,
- observation noise ranges,
- action scales,
- reset and disturbance ranges,
- simulation timing,
- PPO hyperparameters.

The default values are also exposed as YAML in `config/task_parameters.yaml`.

Layered overrides from preset files and training-set launches are merged through the same module.

## `src/mjlab_kinova/train_set.py`

Launches multi-run experiment sets from `config/training_sets/`.

It is responsible for:

- loading the base config and global overrides
- resolving preset stacks and per-run overrides
- supporting `runs:` entries that point to extracted one-run YAML files
- generating timestamped W&B project names for set launches

## `src/mjlab_kinova/train_run.py`

Launches one extracted run config directly.

It reuses the same merge and stop-policy logic as `train_set.py`, but expects exactly one run and fixes the W&B project name to `kinova_ping_pong`.

## `src/mjlab_kinova/play_set.py`

Launches `play` by resolving parameters from a named run inside a training-set file.

## `src/mjlab_kinova/play_run.py`

Launches `play` directly from an extracted one-run config without needing `--run`.

## `src/mjlab_kinova/robot/kinova_constants.py`

Defines:

- the XML loader,
- the home pose,
- the actuator configuration,
- the reusable `KINOVA_CFG` entity configuration.

## `src/mjlab_kinova/robot/kinova.xml`

Contains the MuJoCo articulation tree, inertial values, sensors, and plate collision geometry.
