# Robot and MuJoCo Model

The Kinova model is defined in `src/mjlab_kinova/robot/kinova.xml`.

## Robot

The articulated robot is a 7-DoF arm with MuJoCo position actuators configured in `kinova_constants.py`.

Actuator groups:

- joints `1-4`: stiffness `40`, damping `15`, effort limit `95`, armature `0.01928`
- joints `5-7`: stiffness `15`, damping `8.5`, effort limit `45`, armature `0.015`

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
of the bodies mounted after the sensor: `FT_sensor_wrench` and `plate`. `mc_rtc_calibrated` — the default, and the one that transfers —
reproduces mc_rtc's `wrenchWithoutGravity` against the vendored calibration file,
so the policy trains on the signal the deployed controller produces.

### Modelling the real sensor

MuJoCo's force sensor is exact. The real Bota is not, and three of its properties
are reproduced deliberately (all measured in
`~/devel/ft-calib-investigation/README.md`, configured under `ft_sensor:` in
`config/task_parameters.yaml`):

- **A 3.67 % force gain.** The calibration file holds `k * true_mass`, because on
  hardware it subtracts a *measured* gravity wrench that is already scaled by `k`.
  Simulation therefore scales its own force by the same `k` before the
  compensation. Skip this and sim over-subtracts by 3.7 % while hardware comes out
  exact. The couple channel is left alone — no gain error was measurable there.
- **Two-mechanism noise.** A floor of `0.0785 N` / `0.00465 N·m` present with the
  arm parked, plus a broadband term growing at `0.3142 N` and `0.04713 N·m` per
  rad/s of joint speed. This replaced a single uniform `±0.1` applied to all six
  channels, which under-noised force while over-noising the couple by about 30x.
  The coefficients are quoted at the policy's own 100 Hz observation rate;
  decimating from the sensor's 1 kHz changes them by less than a part in a
  thousand, so there is no aliasing to model.
- **Anisotropy and torque cross-talk, randomized rather than fixed.** The measured
  principal gains span about 0.93–1.01, and roughly 1.6 N per N·m of torque leaks
  into the force channels. That leakage is far outside any plausible load-cell
  spec and probably comes from the *mount* deforming, so hard-coding it would train
  the policy against a fault we intend to fix. `randomize_ft_sensor_response`
  spans it per episode instead. It is on by default in training and off in play.

> **Residual gap:** back-solving the racquet calibration (divide by `k`, subtract
> the sensor's own 0.081117 kg) implies a plate of 0.2312 kg against the modelled
> 0.226 kg. That difference is left in place — it sits inside the fit's known bias
> from the sensor anisotropy — and leaves about 0.05 N of orientation-dependent
> residual in sim, below the sensor's own 0.067 N noise floor.

The tare needs no modelling: the sensor is re-zeroed at the Kinova home pose with
the racquet on at every startup, so there is no standing offset. Only drift since
the tare is randomized, and it is small.

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
