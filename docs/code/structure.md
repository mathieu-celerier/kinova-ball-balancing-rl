# Project Structure

## Top-Level Layout

- `src/mjlab_kinova/tasks`: task registration, environment config, MDP terms, custom actions
- `src/mjlab_kinova/robot`: MuJoCo XML, meshes, actuator constants
- `config/presets`: reusable parameter presets for ablations and training conditions
- `config/training_sets`: batch launch definitions for one control-space variant
- `scripts/zsh`: project-local shell completion helpers
- `typings/mujoco`: local typing support
- `logs`: training outputs

## Main Flow

The code path is:

1. tasks are registered in `src/mjlab_kinova/tasks/__init__.py`
2. environment configuration is created in `kinova_ball_balancing_env_cfg.py`
3. layered task parameters, presets, and training-set overrides are merged through `task_parameters.py`
4. task-local reward/reset/termination helpers come from `ball_balancing_mdp.py`
5. the custom Cartesian action is implemented in `policy_actions.py`
6. the robot model and actuator configuration come from `robot/`

## Where To Edit What

If you want to change:

- task IDs: edit `src/mjlab_kinova/tasks/__init__.py`
- training presets and batch runs: edit `config/presets/` and `config/training_sets/`
- reward weights or observation sets: edit `kinova_ball_balancing_env_cfg.py`
- math/frame/reset logic: edit `ball_balancing_mdp.py`
- Cartesian action semantics: edit `policy_actions.py`
- geometry or inertial model: edit `robot/kinova.xml`
