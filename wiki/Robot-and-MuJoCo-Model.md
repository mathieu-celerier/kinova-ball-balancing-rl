# Robot and MuJoCo Model

The Kinova model is defined in `src/mjlab_kinova/robot/kinova.xml`.

## Robot

The articulated robot is a 7-DoF arm with MuJoCo position actuators configured in `kinova_constants.py`.

Actuator groups:

- joints `1-4`: stiffness `300`, damping `50`, effort limit `95`
- joints `5-7`: stiffness `200`, damping `40`, effort limit `45`

The home pose is also defined there and used as the nominal initial state and posture target.

## End-Effector Stack

The end-effector chain contains:

- adapter
- force/torque mounting
- F/T wrench body
- plate
- `racquet_frame`

The `racquet_frame` is the body used by the task as the balancing reference frame.

## Plate Geometry

The collision support surface is modeled by `plate_collision`, a MuJoCo box geom attached to the plate body.

The task uses that geom for:

- contact-aware reward terms,
- physical support of the ball,
- termination reasoning tied to the support region.

## Sensors

The MuJoCo model defines:

- force sensor `EEForceSensor_fsensor`
- torque sensor `EEForceSensor_tsensor`
- accelerometer
- gyroscope

The task currently uses the F/T pair as actor observations.

Before entering the policy observation, the raw F/T wrench is compensated by
subtracting the static weight effect of the bodies mounted after the sensor:
`FT_sensor_wrench`, `plate`, and `FT_sensor_imu`. This is the simulation-side
equivalent of using a gravity-compensated wrench measurement for the racquet
assembly, without introducing an additional learned or reset-time bias.

## Ball Model

The ball is created programmatically rather than embedded in the robot XML.

Key parameters:

- radius `0.0335 m`
- mass `0.0657 kg`
- free joint
- friction `(1.0, 0.2, 0.0005)`
- `condim = 6`

These parameters shape the rolling/sliding behavior and the sensitivity of the task to plate motion.
During training, the ball mass and friction triplet are randomized together to make the policy more robust to real-robot contact variation.
