# Key Modules

## `src/mjlab_kinova/tasks/__init__.py`

Registers the environment variants through MjLab's task registry.

## `src/mjlab_kinova/tasks/kinova_ball_balancing_env_cfg.py`

Defines:

- policy variants
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
- reward helpers
- termination helpers
- ball reset
- disturbance kicks
- inertial randomization helpers

This is where the code most directly meets the mechanics of the task.

## `src/mjlab_kinova/tasks/policy_actions.py`

Implements the custom Cartesian action term `InitialFramePositionAction`.

Its main role is to reinterpret policy outputs as end-effector position offsets around the initial episode frame and pass those references through differential IK.

## `src/mjlab_kinova/robot/kinova_constants.py`

Defines:

- the XML loader,
- the home pose,
- the actuator configuration,
- the reusable `KINOVA_CFG` entity configuration.

## `src/mjlab_kinova/robot/kinova.xml`

Contains the MuJoCo articulation tree, inertial values, sensors, and plate collision geometry.
