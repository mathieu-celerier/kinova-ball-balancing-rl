# MDP Design

The environment is assembled in `kinova_ball_balancing_env_cfg.py` and task-local terms live in `ball_balancing_mdp.py`.

## Timing

- MuJoCo simulation step: `0.002 s`
- control decimation: `5`
- effective policy step: `0.01 s`
- episode length: `10 s`

## Observations

The task uses asymmetric actor-critic observations.

All observation terms are stacked with a 5-step history window at the policy rate, not the raw MuJoCo substep rate.

With `timestep = 0.002 s` and `decimation = 5`, each history entry is separated by the effective policy step of `0.01 s`, so a history length of `5` spans `0.05 s`.

### Actor Observations

Joint:

- relative joint positions
- relative joint velocities
- end-effector position in world frame
- end-effector orientation in world frame
- end-effector linear velocity in world frame
- end-effector angular velocity in world frame
- end-effector F/T wrench

Cartesian:

- end-effector position in world frame
- end-effector orientation in world frame
- end-effector linear velocity in world frame
- end-effector angular velocity in world frame
- end-effector F/T wrench

### Critic Observations

The critic receives privileged ball state:

- ball position in plate frame
- ball linear velocity in plate frame

This helps optimize the policy during simulation without forcing the deployed actor to depend on perfect ball-state access.

## Observation Noise

Training noise is injected into actor observations:

- joint position: `[-0.01, 0.01]`
- joint velocity: `[-0.1, 0.1]`
- end-effector position: `[-0.003, 0.003]`
- end-effector orientation quaternion: `[-0.01, 0.01]`
- end-effector velocity: `[-0.05, 0.05]`
- end-effector angular velocity: `[-0.1, 0.1]`
- F/T wrench: `[-0.1, 0.1]`

## Actions

### Joint-Space Variant

The joint-space variant uses `JointPositionActionCfg` over all joints with:

- per-joint scales derived from Kinova actuator limits using `0.25 * effort_limit / stiffness`
- default offsets enabled

The resulting action scales are:

- `joint_1` to `joint_4`: `0.59375`
- `joint_5` to `joint_7`: `0.75`

### Cartesian Variant

The Cartesian variant uses `InitialFramePositionAction`, which anchors commands to the initial end-effector frame pose of the episode.

The implemented command law is:

```text
x_ref = x_0 + a * delta_pos_scale
```

with:

- `x_0`: initial end-effector position
- `a`: policy action
- `delta_pos_scale = 0.04`

The code also stores the initial orientation reference, and the IK objective keeps that orientation active with `orientation_weight = 1.0`.

## Reward Terms

Positive terms:

- `is_alive`: `+0.2`
- `ball_centering`: `+200.0` in the default YAML
- `pre_contact_racquet_centering`: exponential reward toward the nominal racquet position
- `post_contact_racquet_centering`: weaker version of the same term after first contact
- `pre_contact_racquet_orientation_centering`: exponential reward toward the nominal racquet orientation
- `post_contact_racquet_orientation_centering`: weaker version after first contact
- `post_contact_racquet_lin_vel`: exponential reward for low racquet linear speed after contact
- `post_contact_racquet_ang_vel`: exponential reward for low racquet angular speed after contact

Negative terms:

- `ball_speed`: `-40.0`
- `ball_height_above_plate`: `-50.0`
- `ball_no_contact_penalty`: `-100.0`
- phase-gated action rate, action acceleration, and joint velocity penalties
- `joint_acc_l2`
- `joint_torque_l2`
- `joint_pos_limits`
- `plate_drop_under_ball`: `-20.0`
- `pre_contact_racquet_ang_vel_l2`: strong pre-contact angular-speed penalty
- `pre_contact_racquet_lin_vel_l2`: strong pre-contact linear-speed penalty

### Key Reward Intuition

`ball_centering` rewards small radial distance in the plate frame, but only while ball-racquet contact is active:

```text
r_center = contact(ball, plate) * exp(-(dx^2 + dy^2) / std^2)
```

`ball_speed` penalizes both translation and spin:

```text
penalty_speed = ||v_ball^plate||^2 + ||omega_ball^plate||^2
```

This makes the task care about damping, not only recentering.

`ball_height_above_plate` penalizes plate-frame height above a soft threshold before any termination is involved.

`plate_drop_under_ball` penalizes moving the plate down along its own normal while the ball is still close above it. This specifically discourages the local-minimum strategy of letting the racquet fall away from the ball while preserving short-term XY centering.

The racquet centering terms now use exponential shaping as well:

```text
r_pos = exp(-||p_racquet - p_nominal||^2 / std_pos^2)
r_ori = exp(-orientation_error / std_ori^2)
```

These are phase-gated so the policy is pulled toward a catch-ready pose before contact and still weakly regularized toward that pose after contact.

Post-contact racquet velocity also uses exponential shaping:

```text
r_v = exp(-||v_racquet||^2 / std_v^2)
r_w = exp(-||omega_racquet||^2 / std_w^2)
```

The pre-contact velocity terms stay as L2 penalties because that phase benefits from stronger pressure against aggressive diving motions.

This combination is deliberate:

- `ball_no_contact_penalty` makes unsupported flight expensive,
- contact-gated `ball_centering` prevents the agent from earning centering reward while the ball is no longer supported,
- `plate_drop_under_ball` discourages the racquet-drop local minimum.

## Terminations

Episodes terminate on:

- timeout
- ball falling off the plate support region

The loss condition includes:

- plate-frame radial distance above `0.16 m`
- plate-frame height below `-0.06 m`
- world height below `0.05 m`

There is no separate `ball_too_high` termination anymore. High plate-frame ball height is handled as a soft penalty rather than an immediate episode end.

## Training Diagnostics

The printed training summary mixes PPO optimization numbers with environment-side episode statistics.

Useful interpretation rules:

- `Mean reward` is average return over completed episodes collected recently.
- `Mean episode length` is measured in environment steps.
- `Episode_Reward/*` entries are averaged per-episode contributions normalized by the configured episode horizon.
- `Episode_Termination/*` entries are reset counts averaged over the current logging window, so they can be greater than `1.0`.

For this task, the most informative trends are usually:

- rising `Mean reward`,
- rising `Mean episode length`,
- less negative `Episode_Reward/ball_no_contact_penalty`,
- less negative `Episode_Reward/plate_drop_under_ball`,
- decreasing `Episode_Termination/ball_fell_off`.

## Reset and Randomization

The default training behavior for the joint-space variant includes:

- ball XY reset in `[-0.02, 0.02] m`
- ball Z offset `0.05 m`
- ball release delay in `[0.0, 0.25] s`
- randomized initial ball linear and angular velocity
- ball mass scaling in `[0.7, 1.3]`
- PD gain scaling in `[0.95, 1.05]`
- robot inertial randomization in `[0.9, 1.1]` for selected fields

The reset now behaves as a suspended drop rather than an immediate free-fall at reset:

- the ball is positioned above the racquet at reset
- a per-environment release delay is sampled
- the robot is free to move during that delay
- the ball is held relative to the racquet until release
- after release, gravity and the sampled reset velocity take over

During training only, the ball also receives interval velocity kicks every `0.4-1.0 s`.

Alternative joint-space ablations such as "no model randomization" and "no randomization" are now expressed as training presets rather than separate policy variants.
