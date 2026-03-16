# Project Structure

## Top-Level Layout

- `src/mjlab_kinova/tasks`: task registration, environment config, MDP terms, custom actions
- `src/mjlab_kinova/robot`: MuJoCo XML, meshes, actuator constants
- `typings/mujoco`: local typing support
- `logs`: training outputs

## Main Flow

The code path is:

1. tasks are registered in `src/mjlab_kinova/tasks/__init__.py`
2. environment configuration is created in `kinova_ball_balancing_env_cfg.py`
3. task-local reward/reset/termination helpers come from `ball_balancing_mdp.py`
4. the custom Cartesian action is implemented in `policy_actions.py`
5. the robot model and actuator configuration come from `robot/`

## Where To Edit What

If you want to change:

- task IDs: edit `src/mjlab_kinova/tasks/__init__.py`
- reward weights or observation sets: edit `kinova_ball_balancing_env_cfg.py`
- math/frame/reset logic: edit `ball_balancing_mdp.py`
- Cartesian action semantics: edit `policy_actions.py`
- geometry or inertial model: edit `robot/kinova.xml`
