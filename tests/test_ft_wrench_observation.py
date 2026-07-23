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
MC_RTC_REFERENCE = (
    (
        [-0.186516876395, -0.195973460027, 0.950047103190, 0.155616064417],
        [0.681206587512, 0.328210698919, -1.111811994873, 0.000241289318, 0.013711973764, 0.004195659326],
    ),
    (
        [0.514994212016, 0.382670106719, 0.688098228895, 0.338917952319],
        [-0.505471160081, 1.069182730628, -0.639716956592, -0.013510634382, -0.006024807351, 0.000605918064],
    ),
    (
        [0.790393982117, -0.464543402347, -0.345272690405, -0.200657792347],
        [0.841766681602, -0.683371855274, 0.795186991130, 0.006783359954, 0.010798538289, 0.002099393935],
    ),
    (
        [0.758033060637, 0.369010635945, -0.194996983103, -0.501191785768],
        [-0.321341762595, 1.130026811648, 0.653964422785, -0.020656055580, -0.006835158288, 0.001661036565],
    ),
)


class _FakeSensor:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


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
        env = SimpleNamespace(
            scene={
                "robot/EEForceSensor_fsensor": _FakeSensor(force),
                "robot/EEForceSensor_tsensor": _FakeSensor(torque),
            }
        )

        wrench = bb_mdp.ee_ft_wrench(env, mode="raw")

        torch.testing.assert_close(
            wrench,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

    def test_weight_compensated_mode_subtracts_racquet_weight_effect(self) -> None:
        force = torch.tensor([[1.0, 2.0, 3.0]])
        torque = torch.tensor([[4.0, 5.0, 6.0]])
        weight_effect = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        env = SimpleNamespace(
            scene={
                "robot/EEForceSensor_fsensor": _FakeSensor(force),
                "robot/EEForceSensor_tsensor": _FakeSensor(torque),
            }
        )

        bb_mdp.ee_ft_racquet_weight_effect = lambda _: weight_effect
        wrench = bb_mdp.ee_ft_wrench(env, mode="weight_compensated")

        torch.testing.assert_close(
            wrench,
            torch.tensor([[0.9, 1.8, 2.7, 3.6, 4.5, 5.4]]),
        )

    def test_unknown_mode_raises_value_error(self) -> None:
        env = SimpleNamespace(
            scene={
                "robot/EEForceSensor_fsensor": _FakeSensor(torch.zeros((1, 3))),
                "robot/EEForceSensor_tsensor": _FakeSensor(torch.zeros((1, 3))),
            }
        )

        with self.assertRaises(ValueError):
            bb_mdp.ee_ft_wrench(env, mode="unknown")

    def test_mc_rtc_calibrated_mode_requires_calib_file(self) -> None:
        env = SimpleNamespace(
            scene={
                "robot/EEForceSensor_fsensor": _FakeSensor(torch.zeros((1, 3))),
                "robot/EEForceSensor_tsensor": _FakeSensor(torch.zeros((1, 3))),
            }
        )

        with self.assertRaises(ValueError):
            bb_mdp.ee_ft_wrench(env, mode="mc_rtc_calibrated", calib_file=None)


class FtCalibParserTests(unittest.TestCase):
    def test_parses_vendored_calibration(self) -> None:
        calib = bb_mdp.load_ft_calib(CALIB_FILE)

        self.assertAlmostEqual(calib["mass"], 0.137062)
        self.assertEqual(calib["com_p"], (0.00641883, -0.00715995, 0.013377))
        # sva::ForceVecd is torque-first: params 7-9 are the couple, 10-12 the force.
        self.assertEqual(calib["offset_torque"], (-0.0426794, -0.0492642, 0.008956))
        self.assertEqual(calib["offset_force"], (-0.593463, 2.29036, -0.849233))

    def test_rotation_matches_mc_rtc_rpy_convention(self) -> None:
        calib = bb_mdp.load_ft_calib(CALIB_FILE)
        # RotX(r) * RotY(p) * RotZ(y) with SpaceVecAlg's spatial rotations, from sva.
        expected = torch.tensor(
            [
                [0.9672877928782722, -0.002468330347530886, -0.2536695352106286],
                [0.041910551156203055, 0.9877667493194842, 0.15020104740181636],
                [0.2501955863928543, -0.15591906966152297, 0.95555816791309],
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
