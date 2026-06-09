from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mjlab.utils.lab_api.math import (
    apply_delta_pose,
    compute_pose_error,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_mul,
)
from mjlab_kinova.tasks.policy_actions import (
    _rotation_matrix_error,
    _rotation_matrix_from_axis_angle,
)
from mjlab_kinova.tasks.ball_balancing_mdp import _axis_angle_from_rotation_matrix


def _axis_angle(axis: list[float], angle: float) -> torch.Tensor:
    axis_tensor = torch.tensor([axis], dtype=torch.float32)
    angle_tensor = torch.tensor([angle], dtype=torch.float32)
    return quat_from_angle_axis(angle_tensor, axis_tensor)


class CartesianControlLawConventionTests(unittest.TestCase):
    def test_rotation_matrix_error_matches_small_world_frame_delta(self) -> None:
        current_quat = _axis_angle([0.0, 1.0, 0.0], 0.35)
        delta = torch.tensor([[0.05, -0.02, 0.03]], dtype=torch.float32)
        _target_pos, desired_quat = apply_delta_pose(
            torch.zeros((1, 3), dtype=torch.float32),
            current_quat,
            torch.cat((torch.zeros((1, 3), dtype=torch.float32), delta), dim=-1),
        )

        rot_error = _rotation_matrix_error(
            matrix_from_quat(current_quat),
            matrix_from_quat(desired_quat),
        )

        expected = torch.sin(torch.linalg.norm(delta, dim=-1, keepdim=True)) * (
            delta / torch.linalg.norm(delta, dim=-1, keepdim=True)
        )
        self.assertTrue(torch.allclose(rot_error, expected, atol=1e-5))

    def test_rotation_matrix_error_body_frame_rotates_world_error(self) -> None:
        current_quat = _axis_angle([0.0, 0.0, 1.0], torch.pi / 2)
        delta = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
        _target_pos, desired_quat = apply_delta_pose(
            torch.zeros((1, 3), dtype=torch.float32),
            current_quat,
            torch.cat((torch.zeros((1, 3), dtype=torch.float32), delta), dim=-1),
        )

        body_error = _rotation_matrix_error(
            matrix_from_quat(current_quat),
            matrix_from_quat(desired_quat),
            body_frame=True,
        )

        self.assertTrue(
            torch.allclose(
                body_error,
                torch.tensor([[0.0, -torch.sin(torch.tensor(0.1)), 0.0]]),
                atol=1e-5,
            )
        )

    def test_axis_angle_matrix_matches_quaternion_rotation(self) -> None:
        axis_angle = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
        expected_quat = _axis_angle(
            (axis_angle / torch.linalg.norm(axis_angle, dim=-1, keepdim=True))[0].tolist(),
            float(torch.linalg.norm(axis_angle)),
        )

        rot = _rotation_matrix_from_axis_angle(axis_angle)

        self.assertTrue(
            torch.allclose(rot, matrix_from_quat(expected_quat), atol=1e-5)
        )

    def test_axis_angle_action_and_observation_representations_match(self) -> None:
        action_orientation = torch.tensor(
            [[0.2, -0.1, 0.3], [-0.4, 0.25, 0.1]],
            dtype=torch.float32,
        )

        observed_orientation = _axis_angle_from_rotation_matrix(
            _rotation_matrix_from_axis_angle(action_orientation)
        )

        self.assertTrue(
            torch.allclose(observed_orientation, action_orientation, atol=1e-5)
        )

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
