"""The MuJoCo model and the calibration file have to agree on what the sensor carries.

`mc_rtc_calibrated` subtracts a gravity wrench built from the calibration file from a wrench
MuJoCo produces from the model. If the two disagree about the payload, the difference lands
straight on a policy observation as a pose-dependent bias, and nothing else in the suite
notices -- both sides stay internally consistent while being wrong together.

Both failure modes this guards against have already happened once:

  - a calibration file two plate revisions stale (0.137 kg against a 0.226 kg model, ~0.9 N);
  - `FT_sensor_imu` parented to `FT_sensor_wrench` in the MJCF, which put a body that sits
    below the load cell into the measured payload (8.4 g, ~0.08 N).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = REPO_ROOT / "src" / "mjlab_kinova" / "robot" / "kinova.xml"
CALIB_FILE = REPO_ROOT / "config" / "calib" / "kinova_bota_plate" / "calib_data.EEForceSensor"

# The Bota force channel reads 3.67 % low, and the calibration subtracts a *measured* gravity
# wrench, so the file holds k * true_mass. See config/calib/README.md.
FORCE_GAIN = 0.9633
# The sensor's own noise floor, from ~/devel/ft-calib-investigation/README.md section 0g. A
# disagreement smaller than this cannot be seen on hardware; a larger one is a modelling bug.
NOISE_FLOOR_N = 0.067
GRAVITY = 9.81


def _payload_above_load_cell():
    """Mass and CoM of every body the force sensor weighs, in the sensor frame."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sensor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FT_sensor_wrench")
    carried, stack = [sensor_body], [sensor_body]
    while stack:
        parent = stack.pop()
        for body in range(model.nbody):
            if model.body_parentid[body] == parent and body != parent:
                carried.append(body)
                stack.append(body)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in carried]
    mass = sum(model.body_mass[b] for b in carried)
    weighted = sum(model.body_mass[b] * data.xipos[b] for b in carried)
    rot = data.xmat[sensor_body].reshape(3, 3)
    com = rot.T @ (weighted / mass - data.xpos[sensor_body])
    return names, mass, com


def _calib_mass() -> float:
    return float(CALIB_FILE.read_text().split()[0])


class SensorPayloadMatchesCalibrationTests(unittest.TestCase):
    def test_imu_is_not_part_of_the_measured_payload(self) -> None:
        """FT_sensor_imu sits below the load cell, so the sensor must not weigh it.

        The bare-sensor run measured 0.077-0.078 kg above the cell against FT_sensor_wrench's
        0.081117 alone; including the imu would have predicted 0.0895 kg.
        """
        names, _, _ = _payload_above_load_cell()

        self.assertNotIn("FT_sensor_imu", names)
        self.assertIn("FT_sensor_wrench", names)
        self.assertIn("plate", names)

    def test_model_payload_agrees_with_the_calibration_file(self) -> None:
        """The residual the policy sees at deploy time must stay under the noise floor.

        This is a bound, not a discriminator. It catches a calibration that belongs to a
        different tool -- the 0.9 N stale-plate file, or a 52 g simulator-derived one -- but it
        cannot catch a small structural error: with the imu wrongly included the residual is
        0.031 N, *better* than the correct model's 0.049 N, because the plate's own mass is
        uncertain by more than the imu weighs. The structural check above is what guards that.
        """
        _, mass, _ = _payload_above_load_cell()

        # What the calibration subtracts, against what the model actually puts on the sensor.
        residual_n = abs(FORCE_GAIN * mass - _calib_mass()) * GRAVITY

        self.assertLess(
            residual_n,
            NOISE_FLOOR_N,
            f"model carries {mass:.6f} kg, so calibration should hold about "
            f"{FORCE_GAIN * mass:.6f} kg, but the file says {_calib_mass():.6f} -- "
            f"{residual_n:.4f} N of constant bias on a policy observation",
        )

    def test_payload_com_is_where_the_calibration_expects_it(self) -> None:
        """A CoM mismatch shows up in the couple channel even when the mass is right."""
        _, _, com = _payload_above_load_cell()
        calib_com = np.array([float(v) for v in CALIB_FILE.read_text().split()[4:7]])

        # The calibration was fitted on hardware, so it carries the real tool's CoM rather than
        # the model's; 5 mm is the accepted disagreement, well inside the couple noise.
        np.testing.assert_allclose(com, calib_com, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
