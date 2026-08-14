from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mjlab_kinova.tasks import ball_balancing_mdp as bb_mdp

CALIB_FILE = REPO_ROOT / "config" / "calib" / "kinova_bota_plate" / "calib_data.EEForceSensor"

# Ground truth generated from the real mc_rtc stack via the sva python bindings, by
# replicating ForceSensorCalibData::loadData + wfToSensor (ForceSensor.cpp:36-84) for the
# vendored calibration under gravity (0, 0, -9.81) and X_p_f = Identity, with the offset
# zeroed (sim has no sensor bias to cancel). Each entry is (site_quat_w wxyz, expected
# force xyz then torque xyz).
#
# Regenerate with ~/devel/ft-calib-investigation/analysis/gen_ft_reference.py whenever the
# vendored calibration changes -- it needs the mc_rtc bindings, which this venv does not
# have, and it self-checks its quaternion convention before emitting anything.
MC_RTC_REFERENCE = (
    (
        [-0.186516876395, -0.195973460027, 0.950047103190, 0.155616064417],
        [0.798082873136, 1.128069881203, -2.607835347977, -0.033897694766, 0.062174314618, 0.016520905780],
    ),
    (
        [0.514994212016, 0.382670106719, 0.688098228895, 0.338917952319],
        [-1.379050178564, 2.525573800404, -0.655729147692, -0.154299155690, -0.087084274597, -0.010906155638],
    ),
    (
        [0.790393982117, -0.464543402347, -0.345272690405, -0.200657792347],
        [2.206690816997, -1.733772021650, 0.913692150873, 0.098911619581, 0.139882612599, 0.026548545252],
    ),
    (
        [0.758033060637, 0.369010635945, -0.194996983103, -0.501191785768],
        [-0.216361283287, 2.204772154114, 1.949987352519, -0.172860051498, -0.021755840831, 0.005418727188],
    ),
)

# Kept in step with FtSensorParameters. The sensor reads 3.67 % low on force; see
# ~/devel/ft-calib-investigation/README.md section 0e.
FORCE_GAIN = 0.9633


class _FakeSensor:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


def _sensor_env(force: torch.Tensor, torque: torch.Tensor, joint_vel=None) -> SimpleNamespace:
    """Minimal env exposing what ee_ft_wrench reads for the raw readout and its noise."""
    num_envs = force.shape[0]
    if joint_vel is None:
        joint_vel = torch.zeros((num_envs, 7))
    return SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene={
            "robot/EEForceSensor_fsensor": _FakeSensor(force),
            "robot/EEForceSensor_tsensor": _FakeSensor(torque),
            "robot": SimpleNamespace(data=SimpleNamespace(joint_vel=joint_vel)),
        },
    )


def _calib_env(site_quat_w: torch.Tensor, gravity=(0.0, 0.0, -9.81)) -> SimpleNamespace:
    """Minimal env exposing what ee_ft_mc_rtc_calibrated_wrench reads."""
    num_envs = site_quat_w.shape[0]
    robot = SimpleNamespace(
        find_sites=lambda _name, preserve_order=True: ([0], ["ee_force"]),
        data=SimpleNamespace(site_quat_w=site_quat_w.unsqueeze(1)),
    )
    return SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene={"robot": robot},
        sim=SimpleNamespace(mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity))),
    )


class FtWrenchObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_sensor_type = bb_mdp.BuiltinSensor
        self._original_weight_effect = bb_mdp.ee_ft_racquet_weight_effect
        bb_mdp.BuiltinSensor = _FakeSensor

    def tearDown(self) -> None:
        bb_mdp.BuiltinSensor = self._original_sensor_type
        bb_mdp.ee_ft_racquet_weight_effect = self._original_weight_effect

    def test_raw_mode_returns_concatenated_sensor_values(self) -> None:
        force = torch.tensor([[1.0, 2.0, 3.0]])
        torque = torch.tensor([[4.0, 5.0, 6.0]])
        env = _sensor_env(force, torque)

        wrench = bb_mdp.ee_ft_wrench(env, mode="raw")

        torch.testing.assert_close(
            wrench,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

    def test_weight_compensated_mode_subtracts_racquet_weight_effect(self) -> None:
        force = torch.tensor([[1.0, 2.0, 3.0]])
        torque = torch.tensor([[4.0, 5.0, 6.0]])
        weight_effect = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        env = _sensor_env(force, torque)

        bb_mdp.ee_ft_racquet_weight_effect = lambda _: weight_effect
        wrench = bb_mdp.ee_ft_wrench(env, mode="weight_compensated")

        torch.testing.assert_close(
            wrench,
            torch.tensor([[0.9, 1.8, 2.7, 3.6, 4.5, 5.4]]),
        )

    def test_unknown_mode_raises_value_error(self) -> None:
        env = _sensor_env(torch.zeros((1, 3)), torch.zeros((1, 3)))

        with self.assertRaises(ValueError):
            bb_mdp.ee_ft_wrench(env, mode="unknown")

    def test_mc_rtc_calibrated_mode_requires_calib_file(self) -> None:
        env = _sensor_env(torch.zeros((1, 3)), torch.zeros((1, 3)))

        with self.assertRaises(ValueError):
            bb_mdp.ee_ft_wrench(env, mode="mc_rtc_calibrated", calib_file=None)

    def test_force_gain_scales_force_but_not_torque(self) -> None:
        """The Bota reads low on force only; the couple channel showed no gain error."""
        force = torch.tensor([[1.0, 2.0, 3.0]])
        torque = torch.tensor([[4.0, 5.0, 6.0]])
        env = _sensor_env(force, torque)

        wrench = bb_mdp.ee_ft_wrench(env, mode="raw", force_gain=FORCE_GAIN)

        torch.testing.assert_close(
            wrench,
            torch.tensor([[FORCE_GAIN, 2 * FORCE_GAIN, 3 * FORCE_GAIN, 4.0, 5.0, 6.0]]),
        )

    def test_noise_grows_with_joint_speed_and_separates_force_from_torque(self) -> None:
        """The two mechanisms of section 0g, checked as sample statistics."""
        floor_f, slope_f = 0.0785, 0.3142
        floor_t, slope_t = 0.00465, 0.04713
        speed = 1.5

        for expected_f, expected_t, joint_vel in (
            (floor_f, floor_t, torch.zeros((4096, 7))),
            # ||alpha|| = speed, spread over the 7 joints.
            (
                floor_f + slope_f * speed,
                floor_t + slope_t * speed,
                torch.full((4096, 7), speed / (7 ** 0.5)),
            ),
        ):
            env = _sensor_env(
                torch.zeros((4096, 3)), torch.zeros((4096, 3)), joint_vel=joint_vel
            )
            torch.manual_seed(0)
            wrench = bb_mdp.ee_ft_wrench(
                env,
                mode="raw",
                noise_force_floor=floor_f,
                noise_force_slope=slope_f,
                noise_torque_floor=floor_t,
                noise_torque_slope=slope_t,
            )

            self.assertAlmostEqual(float(wrench[:, :3].std()), expected_f, delta=0.05 * expected_f)
            self.assertAlmostEqual(float(wrench[:, 3:].std()), expected_t, delta=0.05 * expected_t)

    def test_calibration_and_gain_cancel_at_the_tare_pose(self) -> None:
        """The whole point of the exercise: sim must read ~zero where hardware does.

        On the real robot the sensor is tared at the Kinova home pose with the racquet on,
        so the calibrated wrench is zero there by construction. In sim the sensor is exact,
        which is why the raw force is scaled by k before the calibration -- the file
        deliberately holds k * true_mass. Feed in the gravity wrench of the *modelled*
        racquet and the two must very nearly cancel.
        """
        calib = bb_mdp.load_ft_calib(CALIB_FILE)
        # Gravity in the sensor frame at the home pose is exactly (0, 1, 0); README 0c.
        gravity_dir = torch.tensor([[0.0, 1.0, 0.0]])
        # What MuJoCo would report: the true modelled payload, unscaled.
        true_mass = 0.226 + 0.081117
        force = true_mass * 9.81 * gravity_dir
        com = torch.tensor([calib["com_p"]], dtype=torch.float32)
        torque = torch.cross(com, force, dim=-1)

        env = _sensor_env(force, torque)
        # Site aligned with world, and world gravity chosen so the calibrated payload's
        # weight lands along the same +y the sensor sees it on.
        calib_env = _calib_env(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), gravity=(0.0, -9.81, 0.0))
        env.scene["robot"] = calib_env.scene["robot"]
        env.sim = calib_env.sim

        wrench = bb_mdp.ee_ft_wrench(
            env,
            mode="mc_rtc_calibrated",
            calib_file=str(CALIB_FILE),
            force_gain=FORCE_GAIN,
        )

        # What is left is exactly the 5 g the model and the fit disagree by, scaled by k --
        # nothing else. Asserting the identity rather than just a bound is what makes this
        # test notice if the gain stops being applied: without it the residual is the same
        # size but the wrong number and the wrong sign.
        expected = (FORCE_GAIN * true_mass - calib["mass"]) * 9.81
        torch.testing.assert_close(
            wrench[0, :3],
            torch.tensor([0.0, expected, 0.0]),
            atol=2e-3,
            rtol=0,
        )
        # And it has to stay under the sensor's own 0.067 N noise floor to be ignorable.
        self.assertLess(abs(expected), 0.067)


class FtCalibParserTests(unittest.TestCase):
    def test_parses_vendored_calibration(self) -> None:
        calib = bb_mdp.load_ft_calib(CALIB_FILE)

        self.assertAlmostEqual(calib["mass"], 0.300848)
        self.assertEqual(calib["com_p"], (0.00244798, -0.014513, 0.0652359))
        # sva::ForceVecd is torque-first: params 7-9 are the couple, 10-12 the force.
        self.assertEqual(calib["offset_torque"], (-0.193675, -0.014485, -0.00055754))
        self.assertEqual(calib["offset_force"], (-0.112008, 3.01716, 0.0643274))

    def test_rotation_matches_mc_rtc_rpy_convention(self) -> None:
        calib = bb_mdp.load_ft_calib(CALIB_FILE)
        # RotX(r) * RotY(p) * RotZ(y) with SpaceVecAlg's spatial rotations, straight out of
        # mc_rbdyn.rpyToPT. The racquet fit claims well under 1.2 deg on every axis, so this
        # is near identity -- unlike the previous calibration, which claimed 9 and 15 deg
        # that section 4 of the investigation showed were the fit dumping offset error into
        # the mounting rotation.
        expected = torch.tensor(
            [
                [0.9996895727362972, -0.0155635247615828, 0.0194559723302416],
                [0.0157614878454220, 0.9998251372085113, -0.0100633248422295],
                [-0.0192959493992429, 0.0103668559832389, 0.9997600685333480],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(calib["rot_f_ds"], expected)

    def test_short_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".EEForceSensor", delete=False) as handle:
            handle.write("1.0 2.0 3.0\n")
        with self.assertRaises(ValueError):
            bb_mdp.load_ft_calib(handle.name)

    def test_malformed_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".EEForceSensor", delete=False) as handle:
            handle.write(" ".join(["nan_not_a_number"] * 13))
        with self.assertRaises(ValueError):
            bb_mdp.load_ft_calib(handle.name)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            bb_mdp.load_ft_calib("/nonexistent/calib_data.EEForceSensor")


class McRtcCalibratedWrenchTests(unittest.TestCase):
    def test_matches_mc_rtc_wf_to_sensor(self) -> None:
        """Zero raw wrench isolates the calibration model: output must be -wfToSensor."""
        quats = torch.tensor([q for q, _ in MC_RTC_REFERENCE], dtype=torch.float32)
        expected = -torch.tensor([w for _, w in MC_RTC_REFERENCE], dtype=torch.float32)
        env = _calib_env(quats)

        wrench = bb_mdp.ee_ft_mc_rtc_calibrated_wrench(
            env, str(CALIB_FILE), wrench=torch.zeros((len(MC_RTC_REFERENCE), 6))
        )

        torch.testing.assert_close(wrench, expected, atol=1e-5, rtol=1e-4)

    def test_zero_gravity_only_rotates_into_actual_sensor_frame(self) -> None:
        raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        env = _calib_env(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), gravity=(0.0, 0.0, 0.0))
        rot = bb_mdp.load_ft_calib(CALIB_FILE)["rot_f_ds"].to(torch.float32)

        wrench = bb_mdp.ee_ft_mc_rtc_calibrated_wrench(env, str(CALIB_FILE), wrench=raw)

        expected = torch.cat((raw[:, :3] @ rot.T, raw[:, 3:] @ rot.T), dim=-1)
        torch.testing.assert_close(wrench, expected)
        # The misalignment is ~9 deg / ~15 deg, so this must not be a no-op.
        self.assertFalse(torch.allclose(wrench, raw, atol=1e-3))

    def test_cache_is_rebuilt_when_calib_file_changes(self) -> None:
        env = _calib_env(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), gravity=(0.0, 0.0, 0.0))
        bb_mdp.ee_ft_mc_rtc_calibrated_wrench(env, str(CALIB_FILE), wrench=torch.zeros((1, 6)))

        with tempfile.NamedTemporaryFile("w", suffix=".EEForceSensor", delete=False) as handle:
            handle.write(" ".join(["0.0"] * 13))
        wrench = bb_mdp.ee_ft_mc_rtc_calibrated_wrench(env, handle.name, wrench=torch.zeros((1, 6)))

        # An all-zero calibration is identity rotation with no payload.
        torch.testing.assert_close(wrench, torch.zeros((1, 6)))


if __name__ == "__main__":
    unittest.main()
