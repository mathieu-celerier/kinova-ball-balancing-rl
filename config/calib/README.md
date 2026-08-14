# F/T sensor calibration files

Vendored copies of the mc_rtc force-sensor calibrations, used by the
`mc_rtc_calibrated` F/T wrench observation mode so that policies train on the same
signal the mc_rtc controller produces at deploy time.

## `kinova_bota_plate/calib_data.EEForceSensor`

Copied 2026-08-13 from the mc_rtc workspace:

```
mc_rtc/src/catkin_data_ws/install/share/kortex_description/calib/kinova_bota_plate/calib_data.EEForceSensor
```

Fitted the same day with the current racquet mounted, tared at the Kinova home pose
`[0, 15, 180, -130, 0, 55, 90]` deg. The previous copy (2026-07-15) was for the *older*
plate revision — 0.137 kg with an 8.9°/14.7° claimed misalignment — and left roughly
0.9 N of constant sim-vs-real bias on this observation.

Two things about the numbers are worth knowing, both from
`~/devel/ft-calib-investigation/README.md`:

- **The mass is deliberately not the true mass.** The Bota's force channel reads 3.67 %
  low, and the calibration subtracts the *measured* gravity wrench, so the file carries
  `k * true_mass` with `k = 0.9633`. The simulator therefore scales its exact MuJoCo force
  by the same `k` before applying this file (`ee_ft_wrench`), or the compensation would be
  3.7 % over-subtracted in sim and exact on hardware.
- **The tare is what fixes the offset.** The sensor is zeroed at the home pose with the
  racquet on, so `f_measured` is zero there by construction and the offset is exactly minus
  the model wrench at that pose. As long as the real robot keeps taring at home with the
  racquet on, no extra bias term is needed to make sim and hardware agree.

Back-solving this file (divide by `k`, subtract `FT_sensor_wrench`'s 0.081117 kg) implies a
plate link of 0.2312 kg against the modelled 0.226 kg. That difference is left in place: it
is within the fit's known bias from the sensor's anisotropy, and it leaves ~0.05 N of
orientation-dependent residual in sim — below the sensor's own 0.067 N noise floor.

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
