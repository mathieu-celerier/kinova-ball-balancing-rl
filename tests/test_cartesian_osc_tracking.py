from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mjlab.tasks  # noqa: F401
import mjlab_kinova  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab_kinova.tasks.kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
)
from mjlab_kinova.tasks.policy_actions import _rotation_matrix_error
from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _normalize_run,
)


def _pose_only_parameters():
    training_set_path = REPO_ROOT / "config/training_sets/cartesian_pose_only.yaml"
    training_set_cfg = _load_training_set(training_set_path)
    base_params = _base_parameters(
        training_set_cfg, training_set_path=training_set_path
    )
    name, presets, overrides = _normalize_run(training_set_cfg["runs"][0], index=0)
    return _build_run_parameters(
        run_name=name,
        preset_refs=presets,
        overrides=overrides,
        training_set_path=training_set_path,
        base_params=base_params,
    )


def _plot_output_path() -> Path | None:
    value = os.environ.get("MJLAB_KINOVA_OSC_TEST_PLOT", "").strip()
    if not value:
        return None
    if value.lower() in {"1", "true", "yes", "on"}:
        return REPO_ROOT / "artifacts/cartesian_osc_tracking.png"
    return Path(value).expanduser().resolve()


def _stats_output_path() -> Path | None:
    value = os.environ.get("MJLAB_KINOVA_OSC_TEST_STATS", "").strip()
    if not value:
        return None
    if value.lower() in {"1", "true", "yes", "on"}:
        return REPO_ROOT / "artifacts/cartesian_osc_tracking_stats.json"
    return Path(value).expanduser().resolve()


def _video_output_dir() -> Path | None:
    value = os.environ.get("MJLAB_KINOVA_OSC_TEST_VIDEO", "").strip()
    if not value:
        return None
    if value.lower() in {"1", "true", "yes", "on"}:
        return REPO_ROOT / "artifacts/cartesian_osc_tracking_videos"
    return Path(value).expanduser().resolve()


def _gravity_hold_test_enabled() -> bool:
    value = os.environ.get("MJLAB_KINOVA_OSC_TEST_GRAVITY_HOLD", "").strip()
    return value.lower() in {"1", "true", "yes", "on"}


def _reaching_gravity_enabled() -> bool:
    value = os.environ.get("MJLAB_KINOVA_OSC_TEST_REACHING_GRAVITY", "").strip()
    return value.lower() in {"1", "true", "yes", "on"}


def _save_tracking_videos(
    output_dir: Path,
    *,
    target_names: list[str],
    frames: list[list],
    fps: float,
) -> None:
    import mediapy

    output_dir.mkdir(parents=True, exist_ok=True)
    for target_name, target_frames in zip(target_names, frames, strict=True):
        filename = target_name.lower().replace(" / ", "_").replace(" ", "_")
        output_path = output_dir / f"{filename}.mp4"
        mediapy.write_video(output_path, target_frames, fps=fps)
        print(f"[OSC test] Tracking video written to {output_path}")


def _tracking_stats(
    *,
    target_names: list[str],
    positions: torch.Tensor,
    desired_positions: torch.Tensor,
    position_errors: torch.Tensor,
    orientation_errors: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    positions = positions.cpu()
    desired_positions = desired_positions.cpu()
    position_errors = position_errors.cpu()
    orientation_errors = orientation_errors.cpu()
    stats = []

    for target_idx in range(positions.shape[1]):
        trajectory = positions[:, target_idx]
        target_delta = desired_positions[target_idx] - trajectory[0]
        direct_distance = torch.linalg.vector_norm(target_delta)
        target_direction = target_delta / direct_distance
        relative_trajectory = trajectory - trajectory[0]
        axial_progress = relative_trajectory @ target_direction
        lateral_offsets = relative_trajectory - axial_progress[:, None] * target_direction
        path_length = torch.linalg.vector_norm(
            trajectory[1:] - trajectory[:-1], dim=-1
        ).sum()

        stats.append(
            {
                "target": target_idx + 1,
                "name": target_names[target_idx],
                "position_overshoot_percent": float(
                    100.0
                    * torch.clamp(axial_progress.max() - direct_distance, min=0.0)
                    / direct_distance
                ),
                "straightness_percent": float(100.0 * direct_distance / path_length),
                "max_lateral_deviation_m": float(
                    torch.linalg.vector_norm(lateral_offsets, dim=-1).max()
                ),
                "path_length_m": float(path_length),
                "direct_distance_m": float(direct_distance),
                "final_position_error_m": float(position_errors[-1, target_idx]),
                "final_orientation_error_rad": float(
                    orientation_errors[-1, target_idx]
                ),
            }
        )
    return stats


def _save_tracking_stats(
    output_path: Path, stats: list[dict[str, float | int | str]]
) -> None:
    header = (
        "target                overshoot [%]  straightness [%]  max lateral [m]  "
        "final pos [m]  final ori [rad]"
    )
    print(f"[OSC test] Tracking statistics\n{header}")
    for target in stats:
        print(
            f"{target['name']:<20}  "
            f"{target['position_overshoot_percent']:>13.3f}  "
            f"{target['straightness_percent']:>16.3f}  "
            f"{target['max_lateral_deviation_m']:>15.6f}  "
            f"{target['final_position_error_m']:>13.6f}  "
            f"{target['final_orientation_error_rad']:>15.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[OSC test] Tracking statistics written to {output_path}")


def _save_tracking_plot(
    output_path: Path,
    *,
    target_names: list[str],
    step_dt: float,
    positions: torch.Tensor,
    desired_positions: torch.Tensor,
    position_errors: torch.Tensor,
    orientation_errors: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = positions.cpu()
    desired_positions = desired_positions.cpu()
    position_errors = position_errors.cpu()
    orientation_errors = orientation_errors.cpu()
    time = torch.arange(positions.shape[0]) * step_dt

    figure = plt.figure(figsize=(12, 4 * positions.shape[1]), constrained_layout=True)
    for target_idx in range(positions.shape[1]):
        error_axis = figure.add_subplot(positions.shape[1], 2, 2 * target_idx + 1)
        error_axis.plot(time, position_errors[:, target_idx], label="position error [m]")
        error_axis.plot(
            time, orientation_errors[:, target_idx], label="orientation error [rad]"
        )
        error_axis.set_title(f"{target_names[target_idx]}: tracking errors")
        error_axis.set_xlabel("Time [s]")
        error_axis.set_yscale("log")
        error_axis.grid(True, which="both", alpha=0.3)
        error_axis.legend()

        trajectory_axis = figure.add_subplot(
            positions.shape[1], 2, 2 * target_idx + 2, projection="3d"
        )
        trajectory = positions[:, target_idx]
        target = desired_positions[target_idx]
        trajectory_axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            label="end-effector trajectory",
        )
        trajectory_axis.scatter(
            *trajectory[0], marker="o", s=40, label="start"
        )
        trajectory_axis.scatter(*target, marker="x", s=70, label="target")
        trajectory_axis.set_title(f"{target_names[target_idx]}: 3D trajectory")
        trajectory_axis.set_xlabel("X [m]")
        trajectory_axis.set_ylabel("Y [m]")
        trajectory_axis.set_zlabel("Z [m]")
        trajectory_axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"[OSC test] Tracking plot written to {output_path}")


class CartesianOscTrackingTests(unittest.TestCase):
    @unittest.skipUnless(
        _gravity_hold_test_enabled(),
        "Set MJLAB_KINOVA_OSC_TEST_GRAVITY_HOLD=1 to run the gravity-hold diagnostic.",
    )
    def test_holds_nominal_pose_under_gravity(self) -> None:
        params = _pose_only_parameters()
        cfg = kinova_ball_balancing_env_cfg(
            variant="cartesian",
            play=True,
            params=params,
        )
        cfg.scene.num_envs = 1

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
        try:
            env.reset()
            action = env.action_manager.get_term("ee_pos")
            robot = env.scene["robot"]
            zero_action = torch.zeros(1, 6, device=device)

            initial_frame_pos = action._get_frame_pose()[0].clone()
            initial_frame_rot = action._get_frame_rotation_matrix().clone()
            initial_joint_pos = robot.data.joint_pos[:, action._joint_ids].clone()

            max_position_error = torch.zeros(1, device=device)
            max_orientation_error = torch.zeros(1, device=device)
            max_joint_error = torch.zeros(1, device=device)

            for _ in range(500):
                env.step(zero_action)
                frame_pos = action._get_frame_pose()[0]
                frame_rot = action._get_frame_rotation_matrix()
                joint_pos = robot.data.joint_pos[:, action._joint_ids]
                max_position_error = torch.maximum(
                    max_position_error,
                    torch.linalg.vector_norm(frame_pos - initial_frame_pos, dim=-1),
                )
                max_orientation_error = torch.maximum(
                    max_orientation_error,
                    torch.linalg.vector_norm(
                        _rotation_matrix_error(initial_frame_rot, frame_rot), dim=-1
                    ),
                )
                max_joint_error = torch.maximum(
                    max_joint_error,
                    torch.linalg.vector_norm(joint_pos - initial_joint_pos, dim=-1),
                )

            final_frame_pos = action._get_frame_pose()[0]
            final_frame_rot = action._get_frame_rotation_matrix()
            final_joint_pos = robot.data.joint_pos[:, action._joint_ids]
            errors = {
                "nullspace_target_error_rad": float(
                    torch.linalg.vector_norm(
                        action._q_ns - initial_joint_pos, dim=-1
                    ).item()
                ),
                "max_position_error_m": float(max_position_error.item()),
                "final_position_error_m": float(
                    torch.linalg.vector_norm(
                        final_frame_pos - initial_frame_pos, dim=-1
                    ).item()
                ),
                "max_orientation_error_rad": float(max_orientation_error.item()),
                "final_orientation_error_rad": float(
                    torch.linalg.vector_norm(
                        _rotation_matrix_error(initial_frame_rot, final_frame_rot),
                        dim=-1,
                    ).item()
                ),
                "max_joint_error_rad": float(max_joint_error.item()),
                "final_joint_error_rad": float(
                    torch.linalg.vector_norm(
                        final_joint_pos - initial_joint_pos, dim=-1
                    ).item()
                ),
            }
            print(f"[OSC test] Nominal pose hold under gravity: {errors}")

            self.assertLess(errors["max_position_error_m"], 0.02, errors)
            self.assertLess(errors["max_orientation_error_rad"], 0.2, errors)
            self.assertLess(errors["max_joint_error_rad"], 0.2, errors)
        finally:
            env.close()

    def test_reaches_cartesian_position_and_orientation_targets(self) -> None:
        params = _pose_only_parameters()
        cfg = kinova_ball_balancing_env_cfg(
            variant="cartesian",
            play=True,
            params=params,
        )
        cfg.scene.num_envs = 9

        action_cfg = cfg.actions["ee_pos"]
        self.assertEqual(
            action_cfg.orientation_weight,
            params.cartesian_action.orientation_weight,
        )
        reaching_gravity = _reaching_gravity_enabled()
        if not reaching_gravity:
            # Isolate the OSC task-space tracking law from uncompensated gravity and
            # null-space posture torques. The opt-in gravity mode uses training gains.
            cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
            action_cfg.position_weight = 400.0
            action_cfg.orientation_weight = 400.0
            action_cfg.damping_pos = 40.0
            action_cfg.damping_ori = 40.0
            action_cfg.posture_weight = 0.0
            action_cfg.damping_null = 0.0

        video_dir = _video_output_dir()
        if video_dir is not None:
            cfg.viewer.max_extra_envs = 0

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env = ManagerBasedRlEnv(
            cfg=cfg,
            device=device,
            render_mode="rgb_array" if video_dir is not None else None,
        )
        try:
            env.reset()
            position_directions = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rotation_axes = torch.eye(3, device=device)
            distances_m = (0.05, 0.10, 0.20)
            angles_deg = (15, 30, 60)
            target_names = []
            action_rows = []
            for distance_m in distances_m:
                for angle_deg, position_direction, rotation_axis in zip(
                    angles_deg, position_directions, rotation_axes, strict=True
                ):
                    target_names.append(f"{distance_m * 100:.0f}cm / {angle_deg}deg")
                    action_rows.append(
                        torch.cat(
                            (
                                position_direction
                                * distance_m
                                / action_cfg.delta_pos_scale,
                                rotation_axis
                                * torch.deg2rad(torch.tensor(angle_deg, device=device))
                                / action_cfg.delta_ori_scale,
                            )
                        )
                    )
            actions = torch.stack(action_rows)
            action = env.action_manager.get_term("ee_pos")
            plot_path = _plot_output_path()
            stats_path = _stats_output_path()
            capture_trajectory = plot_path is not None or stats_path is not None
            video_fps = 25.0
            video_step_interval = max(1, round(1.0 / (video_fps * env.step_dt)))
            video_fps = 1.0 / (video_step_interval * env.step_dt)
            video_frames = [[] for _ in target_names]

            env.step(actions)
            positions = []
            position_errors = []
            orientation_errors = []

            def capture_state() -> tuple[torch.Tensor, torch.Tensor]:
                frame_pos = action._get_frame_pose()[0]
                pos_error = torch.linalg.vector_norm(
                    action._desired_pos - frame_pos, dim=-1
                )
                ori_error = torch.linalg.vector_norm(
                    _rotation_matrix_error(
                        action._get_frame_rotation_matrix(), action._desired_rot
                    ),
                    dim=-1,
                )
                if capture_trajectory:
                    positions.append(frame_pos.detach().cpu())
                    position_errors.append(pos_error.detach().cpu())
                    orientation_errors.append(ori_error.detach().cpu())
                return pos_error, ori_error

            def capture_video_frames() -> None:
                if video_dir is None:
                    return
                for target_idx, target_frames in enumerate(video_frames):
                    cfg.viewer.env_idx = target_idx
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("OSC test video renderer returned no frame.")
                    target_frames.append(frame.copy())

            initial_pos_error, initial_ori_error = capture_state()
            capture_video_frames()

            for step_idx in range(500):
                env.step(actions)
                if capture_trajectory:
                    capture_state()
                if (step_idx + 1) % video_step_interval == 0:
                    capture_video_frames()

            final_pos_error, final_ori_error = capture_state()

            if capture_trajectory:
                stacked_positions = torch.stack(positions)
                stacked_position_errors = torch.stack(position_errors)
                stacked_orientation_errors = torch.stack(orientation_errors)
                desired_positions = action._desired_pos.detach()

            if stats_path is not None:
                stats = _tracking_stats(
                    target_names=target_names,
                    positions=stacked_positions,
                    desired_positions=desired_positions,
                    position_errors=stacked_position_errors,
                    orientation_errors=stacked_orientation_errors,
                )
                _save_tracking_stats(stats_path, stats)

            if plot_path is not None:
                _save_tracking_plot(
                    plot_path,
                    target_names=target_names,
                    step_dt=env.step_dt,
                    positions=stacked_positions,
                    desired_positions=desired_positions,
                    position_errors=stacked_position_errors,
                    orientation_errors=stacked_orientation_errors,
                )

            if video_dir is not None:
                _save_tracking_videos(
                    video_dir,
                    target_names=target_names,
                    frames=video_frames,
                    fps=video_fps,
                )

            self.assertTrue(torch.all(initial_pos_error > 1.0e-2))
            self.assertTrue(torch.all(initial_ori_error > 5.0e-2))
            position_tolerance = 0.02 if reaching_gravity else 5.0e-4
            orientation_tolerance = 0.2 if reaching_gravity else 5.0e-4
            self.assertTrue(
                torch.all(final_pos_error < position_tolerance),
                dict(zip(target_names, final_pos_error.tolist(), strict=True)),
            )
            self.assertTrue(
                torch.all(final_ori_error < orientation_tolerance),
                dict(zip(target_names, final_ori_error.tolist(), strict=True)),
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
