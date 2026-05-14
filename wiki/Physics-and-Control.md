# Physics and Control

This environment is easiest to reason about as a ball-on-moving-plate system rather than as a generic robot RL benchmark.

## Core Mechanics

The robot controls the ball indirectly. The causal chain is:

1. the policy moves the robot,
2. the robot moves the plate,
3. the plate changes contact forces and effective acceleration,
4. the ball translates and rotates in response.

That means the system combines:

- rigid-body dynamics,
- unilateral contact,
- Coulomb-like friction effects,
- rolling/sliding transitions,
- robot actuation limits and kinematics.

## Plate-Frame Reasoning

Most task logic is expressed in the plate frame:

```text
p_ball^plate = R_world_to_plate * (p_ball^world - p_plate^world)
```

This is the right frame for the task because:

- "center of the plate" is defined there,
- "ball speed relative to support" is defined there,
- falling off the support region is naturally measured there.

Using world coordinates alone would mix the ball motion with the robot motion and make the signals much harder to interpret.

## Qualitative Dynamics

If the ball remains in contact with the plate, its in-plane motion can be understood qualitatively as:

```text
m * r_ddot ~= m * g_parallel + friction_terms + inertial_terms_from_plate_motion
```

where:

- `r`: ball position in the plate plane
- `g_parallel`: gravity projected onto the plate
- `friction_terms`: rolling/sliding contact effects
- `inertial_terms_from_plate_motion`: apparent forcing due to the plate moving and accelerating

This is not a closed-form model used directly by the code. MuJoCo integrates the full dynamics numerically. The equation is included to explain the structure of the control problem.

## Why Velocity Matters

A controller that only minimizes position error is incomplete. A ball can be exactly at the plate center while still carrying significant momentum.

That is why the environment penalizes:

- ball linear speed,
- ball angular speed,
- loss of contact.

In control terms, the policy must remove energy from the ball, not just drive its position toward zero.

## Why Contact Matters

The reward includes a penalty based on actual MuJoCo contact between:

- `ball/ball_geom`
- `robot/plate_collision`

This discourages a degenerate strategy where the robot impulsively throws the ball upward and only later tries to catch it.

From a physics perspective, sustained balancing means the plate is acting as support, not as a launcher.

The current reward shaping goes a step further:

- centering reward is only active while contact is present,
- no-contact states are penalized directly,
- dropping the plate down away from the ball is penalized separately.

That makes "supporting the ball near the center" the profitable behavior, not merely "keeping the plate-frame XY error small for a short time."

## Role of Force/Torque Sensing

The actor receives end-effector force/torque wrench measurements.

Those signals matter because they provide indirect information about:

- whether the ball is loading the plate,
- whether support contact is being maintained,
- whether the interaction is gentle or impulsive.

This is valuable both for robustness in simulation and for eventual hardware relevance, where direct ball-state measurements may be noisy or unavailable.
