# F/T sensor calibration files

Vendored copies of the mc_rtc force-sensor calibrations, used by the
`mc_rtc_calibrated` F/T wrench observation mode so that policies train on the same
signal the mc_rtc controller produces at deploy time.

## `kinova_bota_plate/calib_data.EEForceSensor`

Copied 2026-07-15 from the mc_rtc workspace:

```
mc_rtc/src/catkin_data_ws/install/share/kortex_description/calib/kinova_bota_plate/calib_data.EEForceSensor
```

mc_rtc resolves this path itself as `<RobotModule::calib_dir>/calib_data.<sensor name>`
(`mc_rbdyn/Robot.cpp`), with `calib_dir` defaulting to `<kortex_description>/calib/<variant>`.
It is vendored rather than read in place so training runs stay reproducible and do not
depend on the mc_rtc workspace being installed. **If the sensor is recalibrated, re-copy
the file** — a stale copy silently trains against the wrong gravity model.

### Format

13 whitespace-separated floats, parsed by `ForceSensorCalibData::loadData`
(`mc_rtc/src/mc_rbdyn/ForceSensor.cpp`):

| Index | Meaning |
|---|---|
| 0 | `mass` — payload mass (kg) hanging off the sensor |
| 1-3 | rpy of `X_f_ds`, the model→actual sensor frame misalignment |
| 4-6 | `X_p_vb` translation — payload CoM in the parent frame |
| 7-9 | wrench offset **couple** (sensor bias) |
| 10-12 | wrench offset **force** (sensor bias) |

Note the offset is an `sva::ForceVecd`, which is **torque-first** — the opposite of this
repo's force-first wrench convention.

### What sim uses

`ee_ft_mc_rtc_calibrated_wrench` applies the misalignment and the mass/CoM gravity model,
but **not** the offset: on hardware the offset cancels a physical sensor bias, and the
MuJoCo sensor has no such bias to cancel. See `ball_balancing_mdp.py` for details.
