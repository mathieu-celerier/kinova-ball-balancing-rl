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

The plate is a 3D-printed part weighing **226 g**, modeled from `meshes/plate.STL`
(authored in metres, in the `plate` body frame rotated -90 deg about Z). Its inertia is
computed from the mesh at uniform density, which is accurate here: 226 g over the
183.2 cm³ mesh volume is 1233 kg/m³, essentially solid PLA.

The blade is a 5 mm slab **offset to one side** of the wrist axis, spanning
`y = -0.035 … -0.030` in the plate frame. The striking surface is the plane
`y = -0.035`, and `racquet_frame` sits on the blade mid-plane at
`(0, -0.0325, 0.125)` with its `+Z` along that face normal.

The collision support surface is modeled by `plate_collision`, a MuJoCo mesh geom
attached to the plate body. It is the convex hull of the blade only (the blade proper
starts at exactly `z = 0.030` in the mesh frame), deliberately excluding the mounting
hub and the neck gusset below it: including
them would tilt the contact plane and make a resting ball creep across the racquet in
simulation when it does not on the real robot. The hull's striking face is exactly
planar to within float32 representation.

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

The actor F/T wrench is controlled by `ee_ft_wrench_mode`. `raw` uses the MuJoCo
sensor values directly. `weight_compensated` subtracts the static weight effect
of the bodies mounted after the sensor: `FT_sensor_wrench`, `plate`, and
`FT_sensor_imu`. The compensated mode mirrors a gravity-compensated wrench
measurement for the racquet assembly without introducing an additional learned
or reset-time bias.

> **Sim-to-real gap:** simulation compensates the modeled plate mass of 0.226 kg, but
> `config/calib/kinova_bota_plate/calib_data.EEForceSensor` still holds the *measured*
> payload of 0.137 kg from the previous plate revision. Until a fresh
> `ForceSensorCalibration` run is done on the real robot with the new plate, the
> `weight_compensated` wrench differs between sim and hardware by roughly 0.9 N.

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
