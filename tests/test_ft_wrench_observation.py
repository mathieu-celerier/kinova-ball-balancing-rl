from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mjlab_kinova.tasks import ball_balancing_mdp as bb_mdp


class _FakeSensor:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


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


if __name__ == "__main__":
    unittest.main()
