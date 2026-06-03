from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mjlab.utils.lab_api.math import apply_delta_pose, compute_pose_error, quat_from_angle_axis, quat_mul


def _axis_angle(axis: list[float], angle: float) -> torch.Tensor:
    axis_tensor = torch.tensor([axis], dtype=torch.float32)
    angle_tensor = torch.tensor([angle], dtype=torch.float32)
    return quat_from_angle_axis(angle_tensor, axis_tensor)


class CartesianControlLawConventionTests(unittest.TestCase):
    def test_pose_error_matches_target_minus_current_convention(self) -> None:
        current_pos = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
        target_pos = torch.tensor([[0.45, -0.18, 0.12]], dtype=torch.float32)
        current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        target_quat = _axis_angle([1.0, 0.0, 0.0], 0.1)

        pos_error, rot_error = compute_pose_error(
            current_pos,
            current_quat,
            target_pos,
            target_quat,
        )

        self.assertTrue(torch.allclose(pos_error, target_pos - current_pos, atol=1e-6))
        self.assertTrue(
            torch.allclose(rot_error, torch.tensor([[0.1, 0.0, 0.0]]), atol=1e-5)
        )

    def test_apply_delta_pose_uses_left_multiplication_as_documented(self) -> None:
        source_pos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        source_quat = _axis_angle([0.0, 0.0, 1.0], 0.4)
        delta_pose = torch.tensor([[0.02, -0.01, 0.03, 0.08, 0.0, 0.0]], dtype=torch.float32)

        target_pos, target_quat = apply_delta_pose(source_pos, source_quat, delta_pose)

        delta_quat = _axis_angle([1.0, 0.0, 0.0], 0.08)
        expected_quat = quat_mul(delta_quat, source_quat)

        self.assertTrue(
            torch.allclose(target_pos, source_pos + delta_pose[:, :3], atol=1e-6)
        )
        self.assertTrue(torch.allclose(target_quat, expected_quat, atol=1e-6))

    def test_orientation_error_recovers_applied_rotation_delta(self) -> None:
        source_pos = torch.zeros((1, 3), dtype=torch.float32)
        source_quat = _axis_angle([0.0, 1.0, 0.0], 0.35)
        delta_pose = torch.tensor([[0.0, 0.0, 0.0, 0.05, -0.02, 0.03]], dtype=torch.float32)

        _target_pos, target_quat = apply_delta_pose(source_pos, source_quat, delta_pose)
        _pos_error, rot_error = compute_pose_error(
            source_pos,
            source_quat,
            source_pos,
            target_quat,
        )

        self.assertTrue(torch.allclose(rot_error, delta_pose[:, 3:], atol=1e-5))

    def test_damping_sign_opposes_angular_velocity(self) -> None:
        orientation_weight = 3.0
        damping_ori = 8.0
        rot_error = torch.tensor([[0.0043, -0.0003, 0.0001]], dtype=torch.float32)
        frame_ang_vel = torch.tensor([[-6.2809, 0.0270, -0.0283]], dtype=torch.float32)

        rot_p = orientation_weight * rot_error
        rot_d = -damping_ori * frame_ang_vel
        rotational_task = rot_p + rot_d

        self.assertTrue(torch.allclose(rot_p, torch.tensor([[0.0129, -0.0009, 0.0003]]), atol=1e-6))
        self.assertTrue(
            torch.allclose(rot_d, torch.tensor([[50.2472, -0.2160, 0.2264]]), atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(
                rotational_task,
                torch.tensor([[50.2601, -0.2169, 0.2267]], dtype=torch.float32),
                atol=1e-4,
            )
        )
        self.assertGreater(rotational_task[0, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
