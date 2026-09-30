"""Task-local MDP functions for Kinova ball balancing."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from math import cos, sin
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import axis_angle_from_quat, compute_pose_error
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.math import quat_conjugate, quat_from_matrix, quat_mul
from mjlab.utils.lab_api.math import sample_uniform
from mjlab_kinova.robot.kinova_constants import KINOVA_CFG

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _axis_angle_from_rotation_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to their principal axis-angle vectors."""
    return axis_angle_from_quat(quat_from_matrix(rot))


def _reset_debug_enabled() -> bool:
    return os.getenv("MJLAB_KINOVA_RESET_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _reset_debug_num_steps() -> int:
    raw = os.getenv("MJLAB_KINOVA_RESET_DEBUG_STEPS", "5").strip()
    try:
        return max(int(raw), 1)
    except ValueError:
        return 5


def _ensure_reset_debug_state(env: "ManagerBasedRlEnv") -> None:
    if hasattr(env, "_reset_debug_episode_id"):
        return
    env._reset_debug_episode_id = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )
    env._reset_debug_steps_remaining = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )


def _format_debug_vec(vec: torch.Tensor, precision: int = 4) -> str:
    values = vec.detach().reshape(-1).cpu().tolist()
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _format_debug_bool(name: str, value: bool) -> str:
    return f"{name}={'True' if value else 'False'}"


def _ensure_ball_release_state(env: "ManagerBasedRlEnv") -> None:
    if hasattr(env, "_ball_release_pending"):
        return
    env._ball_release_pending = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    env._ball_release_steps_remaining = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )
    env._ball_release_offset_plate = torch.zeros(
        env.num_envs, 3, device=env.device, dtype=torch.float32
    )
    env._ball_release_hold_pos_w = torch.zeros(
        env.num_envs, 3, device=env.device, dtype=torch.float32
    )
    env._ball_release_velocity_w = torch.zeros(
        env.num_envs, 6, device=env.device, dtype=torch.float32
    )


def mark_reset_debug_pending(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """Track envs whose first post-reset steps should be logged."""
    if not _reset_debug_enabled():
        return
    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    _ensure_reset_debug_state(env)
    env._reset_debug_episode_id[env_ids] += 1
    env._reset_debug_steps_remaining[env_ids] = _reset_debug_num_steps()


def log_first_step_after_reset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """Log the first few control steps after each reset for debugging."""
    del env_ids
    if not _reset_debug_enabled() or not hasattr(env, "_reset_debug_steps_remaining"):
        return

    pending = torch.nonzero(env._reset_debug_steps_remaining > 0, as_tuple=False).squeeze(-1)
    if pending.numel() == 0:
        return

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    force_sensor = env.scene["robot/EEForceSensor_fsensor"]
    torque_sensor = env.scene["robot/EEForceSensor_tsensor"]
    assert isinstance(force_sensor, BuiltinSensor)
    assert isinstance(torque_sensor, BuiltinSensor)
    plate_body_ids, _ = robot.find_bodies("racquet_frame", preserve_order=True)
    plate_body_id = plate_body_ids[0]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_body_id]
    plate_quat_w = robot.data.body_link_quat_w[:, plate_body_id]
    ball_pos_w = ball.data.root_link_pos_w
    ball_pos_plate = quat_apply_inverse(plate_quat_w, ball_pos_w - plate_pos_w)
    if hasattr(ball.data, "root_link_lin_vel_w"):
        ball_vel_w = ball.data.root_link_lin_vel_w
    else:
        ball_vel_w = ball.data.root_link_vel_w[:, :3]
    ball_vel_plate = quat_apply_inverse(plate_quat_w, ball_vel_w)
    raw_wrench = torch.cat((force_sensor.data, torque_sensor.data), dim=-1)
    racquet_weight_effect = ee_ft_racquet_weight_effect(env)
    corrected_wrench = raw_wrench - racquet_weight_effect
    no_contact = ball_no_contact_mujoco(env=env)
    try:
        joint_term = env.action_manager.get_term("joint_pos")
    except Exception:
        joint_term = None
    processed_actions = getattr(joint_term, "_processed_actions", None)
    radial_xy = torch.linalg.norm(ball_pos_plate[:, :2], dim=-1)
    below_racquet = ball_pos_plate[:, 2] < -0.06
    on_floor = ball_pos_w[:, 2] < 0.05
    fell_off = torch.logical_or(radial_xy > 0.16, torch.logical_or(below_racquet, on_floor))
    release_pending = getattr(env, "_ball_release_pending", None)
    release_steps_remaining = getattr(env, "_ball_release_steps_remaining", None)

    for env_id in pending.tolist():
        msg = (
            f"[RESET_DEBUG][STEP] env={env_id} "
            f"episode={int(env._reset_debug_episode_id[env_id].item())} "
            f"step={int(env.episode_length_buf[env_id].item())} "
            f"contact={bool((no_contact[env_id] == 0.0).item())} "
            f"raw_action={_format_debug_vec(env.action_manager.action[env_id])} "
            f"joint_target={_format_debug_vec(robot.data.joint_pos_target[env_id])} "
            f"joint_pos={_format_debug_vec(robot.data.joint_pos[env_id])} "
            f"plate_pos_w={_format_debug_vec(plate_pos_w[env_id])} "
            f"plate_quat_w={_format_debug_vec(plate_quat_w[env_id])} "
            f"ball_pos_w={_format_debug_vec(ball_pos_w[env_id])} "
            f"ball_plate={_format_debug_vec(ball_pos_plate[env_id])} "
            f"ball_vel_plate={_format_debug_vec(ball_vel_plate[env_id])} "
            f"ball_radial_xy={float(radial_xy[env_id].item()):.4f} "
            f"{_format_debug_bool('below_racquet', bool(below_racquet[env_id].item()))} "
            f"{_format_debug_bool('ball_on_floor', bool(on_floor[env_id].item()))} "
            f"{_format_debug_bool('ball_fell_off', bool(fell_off[env_id].item()))} "
            f"ee_ft_raw={_format_debug_vec(raw_wrench[env_id])} "
            f"ee_ft_racquet_weight={_format_debug_vec(racquet_weight_effect[env_id])} "
            f"ee_ft_corrected={_format_debug_vec(corrected_wrench[env_id])}"
        )
        if release_pending is not None and release_steps_remaining is not None:
            msg += (
                f" {_format_debug_bool('release_pending', bool(release_pending[env_id].item()))}"
                f" release_steps_remaining={int(release_steps_remaining[env_id].item())}"
            )
        if processed_actions is not None:
            msg += (
                f" processed_action={_format_debug_vec(processed_actions[env_id])}"
            )
        print(msg)

    env._reset_debug_steps_remaining[pending] -= 1


def ball_pos_in_plate_frame(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball position in the plate body frame."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)
    plate_quat_w = robot.data.body_link_quat_w[:, plate_asset_cfg.body_ids].squeeze(1)
    ball_pos_w = ball.data.root_link_pos_w

    return quat_apply_inverse(plate_quat_w, ball_pos_w - plate_pos_w)


def ball_lin_vel_in_plate_frame(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball linear velocity in the plate body frame."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    plate_quat_w = robot.data.body_link_quat_w[:, plate_asset_cfg.body_ids].squeeze(1)
    ball_vel_w = ball.data.root_link_lin_vel_w
    return quat_apply_inverse(plate_quat_w, ball_vel_w)


def ball_ang_vel_in_plate_frame(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball angular velocity in the plate body frame."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    plate_quat_w = robot.data.body_link_quat_w[:, plate_asset_cfg.body_ids].squeeze(1)
    if hasattr(ball.data, "root_link_ang_vel_w"):
        ball_ang_vel_w = ball.data.root_link_ang_vel_w
    else:
        ball_ang_vel_w = ball.data.root_link_vel_w[:, 3:]
    return quat_apply_inverse(plate_quat_w, ball_ang_vel_w)


def joint_torques(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Return actuator torques/forces for the robot."""
    robot: Entity = env.scene[robot_name]
    return robot.data.actuator_force


def _ensure_joint_torque_rate_state(env: "ManagerBasedRlEnv") -> None:
    if hasattr(env, "_joint_torque_rate_prev"):
        return
    robot: Entity = env.scene["robot"]
    env._joint_torque_rate_prev = torch.zeros_like(robot.data.actuator_force)
    env._joint_torque_rate_pending = torch.ones(
        env.num_envs, device=env.device, dtype=torch.bool
    )


def reset_joint_torque_rate_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice | None,
) -> None:
    """Reset the joint-torque history used by the torque-rate penalty."""
    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    _ensure_joint_torque_rate_state(env)
    robot: Entity = env.scene["robot"]
    env._joint_torque_rate_prev[env_ids] = robot.data.actuator_force[env_ids]
    env._joint_torque_rate_pending[env_ids] = True


def joint_accelerations(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return joint accelerations for the selected joints."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def joint_pos_rel(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return selected joint positions relative to the nominal configuration q0."""
    asset: Entity = env.scene[asset_cfg.name]
    return (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )


def body_position_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body position in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)


def body_orientation_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return body orientation as a quaternion constructed from MuJoCo's rotation matrix."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    body_rot_w = env.sim.data.xmat[:, body_ids].squeeze(1)
    return quat_from_matrix(body_rot_w)


def body_position_rel_nominal(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return selected body position relative to its nominal home pose."""
    if not hasattr(env, "_racquet_nominal_pose_w"):
        env._racquet_nominal_pose_w = _compute_nominal_racquet_pose_w(
            env=env,
            plate_asset_cfg=asset_cfg,
        )

    nominal_pos_w, _nominal_quat_w = env._racquet_nominal_pose_w
    return body_position_w(env, asset_cfg) - nominal_pos_w


def body_orientation_axis_angle_rel_nominal(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return nominal-local axis-angle orientation relative to the nominal home pose."""
    if not hasattr(env, "_racquet_nominal_pose_w"):
        env._racquet_nominal_pose_w = _compute_nominal_racquet_pose_w(
            env=env,
            plate_asset_cfg=asset_cfg,
        )

    _nominal_pos_w, nominal_quat_w = env._racquet_nominal_pose_w
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    current_rot_w = env.sim.data.xmat[:, body_ids].squeeze(1)
    nominal_rot_w = matrix_from_quat(nominal_quat_w)
    relative_rot_local = torch.matmul(nominal_rot_w.transpose(1, 2), current_rot_w)
    return _axis_angle_from_rotation_matrix(relative_rot_local)


def body_linear_velocity_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body linear velocity in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(asset.data, "body_link_lin_vel_w"):
        return asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids].squeeze(1)
    return asset.data.body_link_vel_w[:, asset_cfg.body_ids, :3].squeeze(1)


def body_angular_velocity_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body angular velocity in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(asset.data, "body_link_ang_vel_w"):
        return asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids].squeeze(1)
    return asset.data.body_link_vel_w[:, asset_cfg.body_ids, 3:].squeeze(1)


def body_linear_velocity_in_body_frame(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body linear velocity expressed in its own body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
    body_vel_w = body_linear_velocity_w(env, asset_cfg)
    return quat_apply_inverse(body_quat_w, body_vel_w)


def _batched_model_field(
    model_field: torch.Tensor,
    env: "ManagerBasedRlEnv",
    ids: torch.Tensor,
) -> torch.Tensor:
    """Return model field values for all environments and selected global ids."""
    if model_field.ndim == 1:
        return model_field[ids].unsqueeze(0).expand(env.num_envs, -1)
    if model_field.ndim == 2:
        return model_field[:, ids]
    if model_field.ndim == 3:
        return model_field[:, ids, :]
    raise ValueError(f"Unsupported model field rank: {model_field.ndim}")


def ee_ft_racquet_weight_effect(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
    sensor_site_name: str = "ee_force",
    # Everything above the load cell, and nothing else. FT_sensor_imu used to be listed here
    # because the MJCF wrongly parented it to FT_sensor_wrench; it sits on the fixed side, below
    # the cell, so the real sensor never weighs it. Keep this in step with the model.
    load_body_names: tuple[str, ...] = ("FT_sensor_wrench", "plate"),
) -> torch.Tensor:
    """Return the static wrench induced by the racquet-side weight at the F/T sensor."""
    robot: Entity = env.scene[robot_name]

    if not hasattr(env, "_ee_ft_weight_ids_cache"):
        site_ids, _ = robot.find_sites(sensor_site_name, preserve_order=True)
        if not site_ids:
            raise ValueError(f"Site '{sensor_site_name}' was not found in '{robot_name}'.")

        load_body_ids: list[int] = []
        for body_name in load_body_names:
            body_ids, _ = robot.find_bodies(body_name, preserve_order=True)
            if not body_ids:
                raise ValueError(f"Body '{body_name}' was not found in '{robot_name}'.")
            load_body_ids.extend(body_ids)

        load_local_ids = torch.tensor(
            tuple(dict.fromkeys(int(body_id) for body_id in load_body_ids)),
            device=env.device,
            dtype=torch.long,
        )
        env._ee_ft_weight_ids_cache = {
            "site_local_id": int(site_ids[0]),
            "load_local_ids": load_local_ids,
            "load_global_ids": robot.indexing.body_ids[load_local_ids].to(
                device=env.device, dtype=torch.long
            ),
        }

    cache = env._ee_ft_weight_ids_cache
    site_local_id = cache["site_local_id"]
    load_local_ids = cache["load_local_ids"]
    load_global_ids = cache["load_global_ids"]

    site_pos_w = robot.data.site_pos_w[:, site_local_id]
    site_quat_w = robot.data.site_quat_w[:, site_local_id]
    body_mass = _batched_model_field(env.sim.model.body_mass, env, load_global_ids)
    gravity_w = torch.as_tensor(
        env.sim.mj_model.opt.gravity,
        device=env.device,
        dtype=site_pos_w.dtype,
    ).unsqueeze(0).unsqueeze(0)
    gravity_force_w = body_mass.unsqueeze(-1) * gravity_w

    if hasattr(env.sim.data, "xipos"):
        body_com_w = env.sim.data.xipos[:, load_global_ids]
    else:
        body_pos_w = robot.data.body_link_pos_w[:, load_local_ids]
        body_quat_w = robot.data.body_link_quat_w[:, load_local_ids]
        body_ipos = _batched_model_field(env.sim.model.body_ipos, env, load_global_ids)
        body_com_w = body_pos_w + quat_apply(
            body_quat_w.reshape(-1, 4),
            body_ipos.reshape(-1, 3),
        ).reshape(env.num_envs, len(load_local_ids), 3)

    lever_arm_w = body_com_w - site_pos_w.unsqueeze(1)
    gravity_torque_w = torch.cross(lever_arm_w, gravity_force_w, dim=-1)
    force_w = -torch.sum(gravity_force_w, dim=1)
    torque_w = -torch.sum(gravity_torque_w, dim=1)
    return torch.cat(
        (
            quat_apply_inverse(site_quat_w, force_w),
            quat_apply_inverse(site_quat_w, torque_w),
        ),
        dim=-1,
    )


def _rpy_to_mat(rpy: tuple[float, float, float]) -> torch.Tensor:
    """Rotation matrix from roll-pitch-yaw, matching mc_rbdyn::rpyToMat.

    mc_rtc composes RotX(r) * RotY(p) * RotZ(y) using SpaceVecAlg's spatial rotations,
    which are the transpose of the usual vector-rotation matrices. The result maps
    vectors from the model sensor frame into the actual one.
    """
    r, p, y = rpy
    rot_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cos(r), sin(r)], [0.0, -sin(r), cos(r)]], dtype=torch.float64
    )
    rot_y = torch.tensor(
        [[cos(p), 0.0, -sin(p)], [0.0, 1.0, 0.0], [sin(p), 0.0, cos(p)]], dtype=torch.float64
    )
    rot_z = torch.tensor(
        [[cos(y), sin(y), 0.0], [-sin(y), cos(y), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    return rot_x @ rot_y @ rot_z


def load_ft_calib(path: str | os.PathLike[str]) -> dict[str, object]:
    """Parse an mc_rtc ``calib_data.<sensor>`` file.

    Mirrors ``ForceSensorCalibData::loadData``: 13 floats holding the payload mass, the
    rpy of the model-to-actual sensor misalignment, the payload CoM in the parent frame,
    and the sensor bias. Unlike mc_rtc, which only warns when the file is missing, a bad
    or absent calibration raises here -- training silently against an identity
    calibration would be far worse than failing.
    """
    calib_path = Path(path)
    try:
        tokens = calib_path.read_text().split()
    except OSError as exc:
        raise ValueError(f"Could not read F/T calibration file '{calib_path}': {exc}") from exc

    if len(tokens) != 13:
        raise ValueError(
            f"Invalid F/T calibration file '{calib_path}': expected 13 values, got {len(tokens)}."
        )
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid F/T calibration file '{calib_path}': {exc}") from exc

    # sva::ForceVecd is torque-first; this module's wrenches are force-first.
    return {
        "mass": values[0],
        "rot_f_ds": _rpy_to_mat((values[1], values[2], values[3])),
        "com_p": tuple(values[4:7]),
        "offset_torque": tuple(values[7:10]),
        "offset_force": tuple(values[10:13]),
    }


def _ft_calib_cache(env: "ManagerBasedRlEnv", calib_file: str, sensor_site_name: str) -> dict:
    cache = getattr(env, "_ee_ft_calib_cache", None)
    if cache is not None and cache["calib_file"] == calib_file:
        return cache

    robot: Entity = env.scene["robot"]
    site_ids, _ = robot.find_sites(sensor_site_name, preserve_order=True)
    if not site_ids:
        raise ValueError(f"Site '{sensor_site_name}' was not found in 'robot'.")

    calib = load_ft_calib(calib_file)
    rot_f_ds = torch.as_tensor(calib["rot_f_ds"], device=env.device, dtype=torch.float32)
    cache = {
        "calib_file": calib_file,
        "site_local_id": int(site_ids[0]),
        "rot_f_ds": rot_f_ds,
        # Gravity force of the calibrated payload, in world. mc_rtc bakes this in at load
        # time as ForceSensorCalibData::worldForce.
        "world_force": torch.as_tensor(
            [-calib["mass"] * g for g in env.sim.mj_model.opt.gravity],
            device=env.device,
            dtype=torch.float32,
        ),
        # Lever arm in the actual sensor frame. The parent body coincides with the sensor
        # site, so the CoM needs no world round-trip before being rotated into `ds`.
        "com_ds": rot_f_ds @ torch.as_tensor(
            calib["com_p"], device=env.device, dtype=torch.float32
        ),
    }
    env._ee_ft_calib_cache = cache
    return cache


def ee_ft_mc_rtc_calibrated_wrench(
    env: "ManagerBasedRlEnv",
    calib_file: str,
    *,
    wrench: torch.Tensor,
    sensor_site_name: str = "ee_force",
) -> torch.Tensor:
    """Return the wrench mc_rtc's ``wrenchWithoutGravity`` would report in sim.

    Reproduces ``ForceSensorCalibData::wfToSensor`` for the Kinova setup, where the sensor
    is declared with an identity parent transform on ``FT_sensor_wrench`` and the
    ``ee_force`` site sits at that body's origin. Model sensor frame, parent frame and site
    frame therefore coincide, and wfToSensor collapses to the reaction wrench of a point
    mass at the calibrated CoM, measured at the sensor origin and expressed in the actual
    sensor frame.

    The calibration offset is deliberately not subtracted: on hardware it cancels the
    physical sensor bias, and MuJoCo's sensor has no bias for it to cancel, so applying it
    would inject a constant error rather than remove one.
    """
    cache = _ft_calib_cache(env, calib_file, sensor_site_name)
    rot_f_ds = cache["rot_f_ds"]
    robot: Entity = env.scene["robot"]
    site_quat_w = robot.data.site_quat_w[:, cache["site_local_id"]]

    # Gravity of the calibrated payload, world -> site -> actual sensor frame.
    force_ds = quat_apply_inverse(site_quat_w, cache["world_force"].expand_as(wrench[:, :3]))
    force_ds = force_ds @ rot_f_ds.T
    torque_ds = torch.cross(cache["com_ds"].expand_as(force_ds), force_ds, dim=-1)

    # The raw MuJoCo wrench is reported in the model sensor frame; mc_rtc reports in the
    # actual one.
    raw_ds = torch.cat((wrench[:, :3] @ rot_f_ds.T, wrench[:, 3:] @ rot_f_ds.T), dim=-1)
    return raw_ds - torch.cat((force_ds, torque_ds), dim=-1)


FT_SENSOR_STATE_ATTR = "_ee_ft_sensor_state"


def _ft_sensor_state(env: "ManagerBasedRlEnv", force_gain: float) -> dict[str, torch.Tensor]:
    """Per-environment sensor response, defaulting to a clean isotropic gain.

    ``gain`` is the 3x3 mapping true force -> measured force and ``leakage`` the 3x3 mapping
    measured torque -> the force it bleeds into. Both are identity-like until
    :func:`randomize_ft_sensor_response` replaces them, so a run without that event still
    gets the deterministic ``force_gain`` and nothing else.
    """
    state = getattr(env, FT_SENSOR_STATE_ATTR, None)
    if state is not None and state["gain"].shape[0] == env.num_envs:
        return state

    eye = torch.eye(3, device=env.device, dtype=torch.float32)
    state = {
        "gain": (force_gain * eye).expand(env.num_envs, 3, 3).clone(),
        "leakage": torch.zeros(env.num_envs, 3, 3, device=env.device, dtype=torch.float32),
        "bias": torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32),
    }
    setattr(env, FT_SENSOR_STATE_ATTR, state)
    return state


def apply_ft_sensor_response(
    env: "ManagerBasedRlEnv",
    wrench: torch.Tensor,
    force_gain: float,
) -> torch.Tensor:
    """Turn MuJoCo's exact wrench into what the Bota would have measured.

    The real sensor reads about 3.7 % low on force, with a mild anisotropy, and leaks
    measured torque into the force channels at roughly 1.6 N per N.m -- see sections 0e
    and 0f of ~/devel/ft-calib-investigation/README.md. The isotropic part matters even
    without randomization: the calibration file deliberately holds ``k * true_mass``
    because it subtracts a *measured* gravity wrench, so leaving MuJoCo's force unscaled
    would over-subtract by 3.7 % in sim while hardware comes out exact.

    The torque channel keeps unit gain: the couple residuals against the rigid model
    (0.019-0.024 N.m) show no measurable error there.
    """
    state = _ft_sensor_state(env, force_gain)
    force, torque = wrench[:, :3], wrench[:, 3:]
    force = (
        torch.bmm(state["gain"], force.unsqueeze(-1)).squeeze(-1)
        + torch.bmm(state["leakage"], torque.unsqueeze(-1)).squeeze(-1)
    )
    return torch.cat((force, torque), dim=-1) + state["bias"]


def _ft_sensor_noise(
    env: "ManagerBasedRlEnv",
    noise_force_floor: float,
    noise_force_slope: float,
    noise_torque_floor: float,
    noise_torque_slope: float,
    robot_name: str = "robot",
) -> torch.Tensor:
    """Sample the measurement noise, which is two mechanisms rather than one.

    A floor that is there with the arm parked, and a broadband term from vibration and
    drivetrain noise that grows with how fast the arm is moving. Fitted on the racquet's
    own check-calibration run in section 0g of ~/devel/ft-calib-investigation/README.md,
    where the >5 Hz residual is linear in ||alpha|| over the whole speed range covered.

    Force and torque get their own floor and slope: they differ by more than a factor of
    ten and are not in the same units, which the single uniform range this replaced could
    not express -- it was over-noising the couple channels by about 30x.
    """
    robot: Entity = env.scene[robot_name]
    speed = torch.linalg.norm(robot.data.joint_vel, dim=-1, keepdim=True)
    sd = torch.cat(
        (
            (noise_force_floor + noise_force_slope * speed).expand(-1, 3),
            (noise_torque_floor + noise_torque_slope * speed).expand(-1, 3),
        ),
        dim=-1,
    )
    return torch.randn_like(sd) * sd


def ee_ft_wrench(
    env: "ManagerBasedRlEnv",
    force_sensor_name: str = "robot/EEForceSensor_fsensor",
    torque_sensor_name: str = "robot/EEForceSensor_tsensor",
    mode: str = "raw",
    calib_file: str | None = None,
    force_gain: float = 1.0,
    noise_force_floor: float = 0.0,
    noise_force_slope: float = 0.0,
    noise_torque_floor: float = 0.0,
    noise_torque_slope: float = 0.0,
) -> torch.Tensor:
    """Return the F/T wrench observation in the configured mode.

    The sensor response and its noise are applied to the raw wrench, before any gravity
    compensation, because that is where they happen on hardware. The compensation is
    linear, so for the noise this is bookkeeping, but for the gain it is not: applying it
    after the calibration would correct the wrong side of the subtraction.

    All of it defaults to off (unit gain, zero noise), so a direct call stays a pure
    readout; the environment config supplies the measured values.
    """
    force_sensor = env.scene[force_sensor_name]
    torque_sensor = env.scene[torque_sensor_name]
    assert isinstance(force_sensor, BuiltinSensor)
    assert isinstance(torque_sensor, BuiltinSensor)
    wrench = torch.cat((force_sensor.data, torque_sensor.data), dim=-1)
    wrench = apply_ft_sensor_response(env, wrench, force_gain)
    if noise_force_floor or noise_force_slope or noise_torque_floor or noise_torque_slope:
        wrench = wrench + _ft_sensor_noise(
            env,
            noise_force_floor,
            noise_force_slope,
            noise_torque_floor,
            noise_torque_slope,
        )
    if mode == "raw":
        return wrench
    if mode == "weight_compensated":
        return wrench - ee_ft_racquet_weight_effect(env)
    if mode == "mc_rtc_calibrated":
        if calib_file is None:
            raise ValueError("Mode 'mc_rtc_calibrated' requires a calib_file.")
        return ee_ft_mc_rtc_calibrated_wrench(env, calib_file, wrench=wrench)
    raise ValueError(
        f"Unsupported ee_ft_wrench mode {mode!r}. "
        "Expected 'raw', 'weight_compensated' or 'mc_rtc_calibrated'."
    )


def joint_torque_l2(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Penalty term on squared actuator torques/forces."""
    torques = joint_torques(env, robot_name=robot_name)
    return torch.sum(torch.square(torques), dim=-1)


def joint_torque_rate(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Return actuator torque rate normalized by the environment step."""
    _ensure_joint_torque_rate_state(env)
    torques = joint_torques(env, robot_name=robot_name)
    rate = (torques - env._joint_torque_rate_prev) / env.step_dt

    pending = env._joint_torque_rate_pending
    if pending.any():
        rate = rate.clone()
        pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        rate[pending_ids] = 0.0

    return rate


def joint_torque_rate_l2(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Penalty on squared torque rate, normalized by the environment step."""
    _ensure_joint_torque_rate_state(env)
    torques = joint_torques(env, robot_name=robot_name)
    pending = env._joint_torque_rate_pending
    prev = env._joint_torque_rate_prev
    delta = torques - prev
    rate = delta / env.step_dt

    if pending.any():
        pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        rate[pending_ids] = 0.0
        prev[pending_ids] = torques[pending_ids]
        env._joint_torque_rate_pending[pending_ids] = False

    prev[:] = torques
    return torch.sum(torch.square(rate), dim=-1)


def ball_fell_off(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    max_xy_radius: float,
    min_height: float,
    floor_height: float = 0.05,
) -> torch.Tensor:
    """Terminate when the ball leaves racquet support region or reaches floor."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    radial_xy = torch.linalg.norm(ball_pos_plate[:, :2], dim=-1)
    left_racquet_xy = radial_xy > max_xy_radius
    below_racquet = ball_pos_plate[:, 2] < min_height

    ball: Entity = env.scene[ball_name]
    on_floor = ball.data.root_link_pos_w[:, 2] < floor_height
    return torch.logical_or(torch.logical_or(left_racquet_xy, below_racquet), on_floor)


def ball_on_floor(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    floor_height: float = 0.05,
) -> torch.Tensor:
    """Penalty signal: 1.0 if ball center is below floor-height threshold."""
    ball: Entity = env.scene[ball_name]
    return (ball.data.root_link_pos_w[:, 2] < floor_height).float()


def ball_too_high(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    max_height: float,
    min_world_z_vel: float,
) -> torch.Tensor:
    """Terminate on upward ball escape rather than plate drop-away."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    ball: Entity = env.scene[ball_name]
    ball_lin_vel_w = (
        ball.data.root_link_lin_vel_w
        if hasattr(ball.data, "root_link_lin_vel_w")
        else ball.data.root_link_vel_w[:, :3]
    )
    return torch.logical_and(
        ball_pos_plate[:, 2] > max_height,
        ball_lin_vel_w[:, 2] > min_world_z_vel,
    )


def ball_lin_vel_l2(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Penalty on squared ball linear speed in world frame."""
    ball: Entity = env.scene[ball_name]
    if hasattr(ball.data, "root_link_lin_vel_w"):
        ball_lin_vel_w = ball.data.root_link_lin_vel_w
    else:
        ball_lin_vel_w = ball.data.root_link_vel_w[:, :3]
    return torch.sum(torch.square(ball_lin_vel_w), dim=-1)


def ball_lin_vel_in_plate_frame_l2(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on squared ball linear speed in the plate frame."""
    ball_lin_vel_plate = ball_lin_vel_in_plate_frame(
        env=env, ball_name=ball_name, plate_asset_cfg=plate_asset_cfg
    )
    return torch.sum(torch.square(ball_lin_vel_plate), dim=-1)


def plate_drop_under_ball_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    ball_height_threshold: float,
    xy_radius: float,
) -> torch.Tensor:
    """Penalty on moving the plate down along its normal while the ball is still overhead."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)
    ball_pos_w = ball.data.root_link_pos_w
    plate_vel_plate = body_linear_velocity_in_body_frame(env, plate_asset_cfg)

    radial_xy = torch.linalg.norm(ball_pos_plate[:, :2], dim=-1)
    ball_near_racquet = torch.logical_and(
        ball_pos_w[:, 2] - plate_pos_w[:, 2] > ball_height_threshold,
        radial_xy < xy_radius,
    )
    downward_plate_speed = torch.clamp(-plate_vel_plate[:, 2], min=0.0)
    return torch.where(
        ball_near_racquet,
        torch.square(downward_plate_speed),
        torch.zeros_like(downward_plate_speed),
    )


def plate_too_low(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
    min_plate_height: float,
) -> torch.Tensor:
    """Penalty signal: 1.0 when the plate height is below a minimum world-frame threshold."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)
    return (plate_pos_w[:, 2] < min_plate_height).float()


def _geom_id(env: "ManagerBasedRlEnv", geom_name: str) -> int:
    if not hasattr(env, "_geom_id_cache"):
        env._geom_id_cache = {}
    geom_id_cache = env._geom_id_cache

    geom_id = geom_id_cache.get(geom_name)
    if geom_id is None:
        geom_id = mujoco.mj_name2id(
            env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if geom_id < 0:
            raise ValueError(f"Geom '{geom_name}' was not found in the model.")
        geom_id_cache[geom_name] = int(geom_id)

    return int(geom_id)


def ball_no_contact_mujoco(
    env: "ManagerBasedRlEnv",
    ball_geom_name: str = "ball/ball_geom",
    racquet_geom_name: str = "robot/plate_collision",
    max_contact_dist: float = 0.0,
) -> torch.Tensor:
    """Penalty signal from MuJoCo contact list for a specific geom pair.

    Returns 1.0 for envs where ball-racquet contact is absent, otherwise 0.0.
    """
    ball_geom_id = _geom_id(env, ball_geom_name)
    racquet_geom_id = _geom_id(env, racquet_geom_name)

    ncon_raw = getattr(env.sim.data, "ncon", getattr(env.sim.data, "nacon", 0))
    ncon = int(torch.as_tensor(ncon_raw).max().item())
    no_contact = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    if ncon <= 0:
        return no_contact

    # MuJoCo only guarantees the first ncon slots in the contact buffer are valid.
    contact_geom = env.sim.data.contact.geom[:ncon]
    contact_worldid = env.sim.data.contact.worldid[:ncon]
    contact_dist = env.sim.data.contact.dist[:ncon]

    pair_match = torch.logical_or(
        torch.logical_and(
            contact_geom[:, 0] == ball_geom_id, contact_geom[:, 1] == racquet_geom_id
        ),
        torch.logical_and(
            contact_geom[:, 0] == racquet_geom_id, contact_geom[:, 1] == ball_geom_id
        ),
    )

    active_pair = torch.logical_and(
        pair_match,
        torch.logical_and(
            torch.logical_and(contact_worldid >= 0, contact_worldid < env.num_envs),
            contact_dist <= max_contact_dist,
        ),
    )

    if torch.any(active_pair):
        no_contact[contact_worldid[active_pair].long()] = 0.0
    return no_contact


def ball_contact_state_mujoco(
    env: "ManagerBasedRlEnv",
    ball_geom_name: str = "ball/ball_geom",
    racquet_geom_name: str = "robot/plate_collision",
    max_contact_dist: float = 0.0,
) -> torch.Tensor:
    """Binary contact state for the ball-racquet geom pair.

    Returns 1.0 when ball-racquet contact is present, otherwise 0.0.
    """
    no_contact = ball_no_contact_mujoco(
        env=env,
        ball_geom_name=ball_geom_name,
        racquet_geom_name=racquet_geom_name,
        max_contact_dist=max_contact_dist,
    )
    return (1.0 - no_contact).unsqueeze(-1)


def ball_centering_reward(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    std: float,
    ball_geom_name: str = "ball/ball_geom",
    racquet_geom_name: str = "robot/plate_collision",
    max_contact_dist: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> torch.Tensor:
    """Reward for keeping the ball centered on the racquet while in contact."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    dx = ball_pos_plate[:, 0] - center_x
    dy = ball_pos_plate[:, 1] - center_y
    radial_error_sq = torch.square(dx) + torch.square(dy)
    contact_state = ball_contact_state_mujoco(
        env=env,
        ball_geom_name=ball_geom_name,
        racquet_geom_name=racquet_geom_name,
        max_contact_dist=max_contact_dist,
    ).squeeze(-1)
    return contact_state * torch.exp(-radial_error_sq / (std**2))


def ball_no_contact_xy_proxy(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    max_xy_radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> torch.Tensor:
    """Penalty proxy: 1.0 when XY distance is outside contact radius."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    dx = ball_pos_plate[:, 0] - center_x
    dy = ball_pos_plate[:, 1] - center_y
    radial = torch.sqrt(torch.square(dx) + torch.square(dy))
    return (radial > max_xy_radius).float()


def ball_no_contact_z_proxy(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    contact_z: float,
    z_tolerance: float,
) -> torch.Tensor:
    """Penalty proxy: 1.0 when Z is outside contact-height tolerance."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    z_gap = torch.abs(ball_pos_plate[:, 2] - contact_z)
    return (z_gap > z_tolerance).float()


def racquet_lin_vel_l2(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on squared racquet (plate body) linear speed in world frame."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    if hasattr(robot.data, "body_link_lin_vel_w"):
        plate_vel_w = robot.data.body_link_lin_vel_w[
            :, plate_asset_cfg.body_ids
        ].squeeze(1)
    else:
        plate_vel_w = robot.data.body_link_vel_w[
            :, plate_asset_cfg.body_ids, :3
        ].squeeze(1)
    return torch.sum(torch.square(plate_vel_w), dim=-1)


def racquet_ang_vel_l2(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on squared racquet (plate body) angular speed in world frame."""
    plate_ang_vel_w = body_angular_velocity_w(env, plate_asset_cfg)
    return torch.sum(torch.square(plate_ang_vel_w), dim=-1)


def racquet_dist_from_initial_l2(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on racquet displacement from the nominal home-pose world position."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)

    if not hasattr(env, "_racquet_nominal_pos_w"):
        env._racquet_nominal_pos_w = _compute_nominal_racquet_pos_w(
            env=env,
            plate_asset_cfg=plate_asset_cfg,
        )

    nominal_pos_w = env._racquet_nominal_pos_w
    return torch.sum(torch.square(plate_pos_w - nominal_pos_w), dim=-1)


def racquet_centering_reward(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward for keeping the racquet position near its nominal home pose."""
    if not hasattr(env, "_racquet_nominal_pos_w"):
        env._racquet_nominal_pos_w = _compute_nominal_racquet_pos_w(
            env=env,
            plate_asset_cfg=plate_asset_cfg,
        )

    robot: Entity = env.scene[plate_asset_cfg.name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)
    nominal_pos_w = env._racquet_nominal_pos_w
    pos_error_sq = torch.sum(torch.square(plate_pos_w - nominal_pos_w), dim=-1)
    return torch.exp(-pos_error_sq / (std**2))


def racquet_ori_dist_from_initial_l2(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on racquet orientation error from the nominal home pose."""
    if not hasattr(env, "_racquet_nominal_pose_w"):
        env._racquet_nominal_pose_w = _compute_nominal_racquet_pose_w(
            env=env,
            plate_asset_cfg=plate_asset_cfg,
        )

    _nominal_pos_w, nominal_quat_w = env._racquet_nominal_pose_w
    plate_quat_w = body_orientation_w(env, plate_asset_cfg)
    plate_quat_w = plate_quat_w / torch.linalg.norm(
        plate_quat_w, dim=-1, keepdim=True
    ).clamp(min=1.0e-8)
    nominal_quat_w = nominal_quat_w / torch.linalg.norm(
        nominal_quat_w, dim=-1, keepdim=True
    ).clamp(min=1.0e-8)
    alignment = torch.sum(plate_quat_w * nominal_quat_w, dim=-1).abs().clamp(max=1.0)
    return 1.0 - torch.square(alignment)


def racquet_orientation_centering_reward(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward for keeping the racquet orientation near its nominal home pose."""
    ori_error = racquet_ori_dist_from_initial_l2(
        env=env,
        plate_asset_cfg=plate_asset_cfg,
    )
    return torch.exp(-ori_error / (std**2))


def _compute_nominal_racquet_pos_w(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute the racquet world position at the robot home joint configuration."""
    robot: Entity = env.scene[plate_asset_cfg.name]

    current_joint_pos = robot.data.joint_pos.clone()
    current_joint_vel = robot.data.joint_vel.clone()

    home_joint_map = KINOVA_CFG.init_state.joint_pos
    home_joint_pos = (
        torch.tensor(
            [home_joint_map[name] for name in robot.joint_names],
            device=env.device,
            dtype=current_joint_pos.dtype,
        )
        .unsqueeze(0)
        .expand(env.num_envs, -1)
    )
    zero_joint_vel = torch.zeros_like(current_joint_vel)

    robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
    env.sim.forward()
    nominal_pos_w = (
        robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1).clone()
    )

    robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel)
    env.sim.forward()

    return nominal_pos_w


def _compute_nominal_racquet_pose_w(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the racquet world pose at the robot home joint configuration."""
    robot: Entity = env.scene[plate_asset_cfg.name]

    current_joint_pos = robot.data.joint_pos.clone()
    current_joint_vel = robot.data.joint_vel.clone()

    home_joint_map = KINOVA_CFG.init_state.joint_pos
    home_joint_pos = (
        torch.tensor(
            [home_joint_map[name] for name in robot.joint_names],
            device=env.device,
            dtype=current_joint_pos.dtype,
        )
        .unsqueeze(0)
        .expand(env.num_envs, -1)
    )
    zero_joint_vel = torch.zeros_like(current_joint_vel)

    robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
    env.sim.forward()
    nominal_pos_w = (
        robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1).clone()
    )
    nominal_quat_w = body_orientation_w(env, plate_asset_cfg).clone()

    robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel)
    env.sim.forward()

    return nominal_pos_w, nominal_quat_w


def _as_env_ids(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _body_global_id(
    robot: Entity, body_ids: torch.Tensor | list[int] | tuple[int, ...] | slice
) -> int:
    if isinstance(body_ids, slice):
        local_body_id = 0
        return int(robot.indexing.body_ids[local_body_id].item())
    local_body_id = int(
        torch.as_tensor(body_ids, device=robot.data.body_link_pos_w.device)
        .flatten()[0]
        .item()
    )
    return int(robot.indexing.body_ids[local_body_id].item())


def _joint_ids_from_cfg(
    robot: Entity, asset_cfg: SceneEntityCfg, device: torch.device
) -> torch.Tensor:
    if isinstance(asset_cfg.joint_ids, slice):
        return torch.arange(robot.num_joints, device=device, dtype=torch.long)
    if asset_cfg.joint_ids is not None:
        return torch.as_tensor(asset_cfg.joint_ids, device=device, dtype=torch.long)
    if asset_cfg.joint_names is None:
        return torch.arange(robot.num_joints, device=device, dtype=torch.long)
    joint_ids, _ = robot.find_joints(asset_cfg.joint_names, preserve_order=True)
    return torch.tensor(joint_ids, device=device, dtype=torch.long)


def _accumulate_profile_stat(
    env: "ManagerBasedRlEnv",
    key: str,
    value: float,
) -> None:
    if not getattr(env, "_profile_timing", False):
        return
    if not hasattr(env, "_profile_stats"):
        return
    env._profile_stats[key] = env._profile_stats.get(key, 0.0) + value


def _solve_damped_system(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    base_damping: float,
    max_tries: int = 5,
) -> torch.Tensor:
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    damping = max(base_damping, 1e-6)
    for _ in range(max_tries):
        try:
            return torch.linalg.solve(matrix + damping * eye, rhs)
        except torch.linalg.LinAlgError:
            damping *= 10.0
    return torch.linalg.lstsq(matrix + damping * eye, rhs.unsqueeze(-1)).solution.squeeze(-1)


def _get_racquet_jacobian_buffers(
    env: "ManagerBasedRlEnv", body_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = getattr(env, "_racquet_jacobian_cache", None)
    cache_key = (body_id, env.num_envs, env.sim.mj_model.nv)
    if cache is None or cache.get("key") != cache_key:
        with wp.ScopedDevice(env.sim.wp_device):
            jacp_wp = wp.zeros((env.num_envs, 3, env.sim.mj_model.nv), dtype=float)
            jacr_wp = wp.zeros((env.num_envs, 3, env.sim.mj_model.nv), dtype=float)
            point_wp = wp.zeros(env.num_envs, dtype=wp.vec3)
            body_wp = wp.zeros(env.num_envs, dtype=wp.int32)
            body_wp.fill_(body_id)
        cache = {
            "key": cache_key,
            "jacp_wp": jacp_wp,
            "jacr_wp": jacr_wp,
            "point_wp": point_wp,
            "body_wp": body_wp,
            "jacp_torch": wp.to_torch(jacp_wp),
            "jacr_torch": wp.to_torch(jacr_wp),
            "point_torch": wp.to_torch(point_wp).view(env.num_envs, 3),
        }
        env._racquet_jacobian_cache = cache
    return cache["jacp_torch"], cache["jacr_torch"], cache["point_torch"]


def _compute_nullspace_direction(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    body_id: int,
    joint_dof_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    jacp_torch, jacr_torch, point_torch = _get_racquet_jacobian_buffers(env, body_id)
    point_torch[env_ids] = env.sim.data.xpos[env_ids, body_id]
    with wp.ScopedDevice(env.sim.wp_device):
        mjwarp.jac(
            env.sim.wp_model,
            env.sim.wp_data,
            env._racquet_jacobian_cache["jacp_wp"],
            env._racquet_jacobian_cache["jacr_wp"],
            env._racquet_jacobian_cache["point_wp"],
            env._racquet_jacobian_cache["body_wp"],
        )

    jacp = jacp_torch[env_ids][:, :, joint_dof_ids]
    jacr = jacr_torch[env_ids][:, :, joint_dof_ids]
    jac = torch.cat((jacp, jacr), dim=1)
    _u, _s, vh = torch.linalg.svd(jac, full_matrices=True)
    null_dir = vh[:, -1, :]
    null_dir = null_dir / torch.clamp(torch.linalg.norm(null_dir, dim=-1, keepdim=True), min=1e-6)
    return jacp, jacr, null_dir


def _max_alpha_along_direction(
    q: torch.Tensor,
    direction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_alpha_per_joint = torch.full_like(q, float("inf"))
    negative_alpha_per_joint = torch.full_like(q, float("inf"))
    positive_mask = direction > 1e-6
    negative_mask = direction < -1e-6
    positive_alpha_per_joint[positive_mask] = ((upper - q)[positive_mask] / direction[positive_mask])
    positive_alpha_per_joint[negative_mask] = ((lower - q)[negative_mask] / direction[negative_mask])
    negative_alpha_per_joint[positive_mask] = ((q - lower)[positive_mask] / direction[positive_mask])
    negative_alpha_per_joint[negative_mask] = ((upper - q)[negative_mask] / (-direction[negative_mask]))
    max_positive_alpha = torch.min(positive_alpha_per_joint, dim=-1).values
    max_negative_alpha = torch.min(negative_alpha_per_joint, dim=-1).values
    return max_positive_alpha, max_negative_alpha


def _correct_pose_to_target(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    robot: Entity,
    joint_ids: torch.Tensor,
    joint_dof_ids: torch.Tensor,
    body_id: int,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    q_pref: torch.Tensor,
    qd_full: torch.Tensor,
    finite_lower: torch.Tensor,
    finite_upper: torch.Tensor,
    max_iters: int,
    damping: float,
    max_dq: float,
    position_weight: float,
    orientation_weight: float,
    posture_weight: float,
    pose_tol: float,
    rot_tol: float,
) -> torch.Tensor:
    lam = max(damping, 1e-6)
    wp2 = position_weight * position_weight
    wo2 = orientation_weight * orientation_weight
    wpost2 = posture_weight * posture_weight

    for _ in range(max_iters):
        frame_pos_w = env.sim.data.xpos[env_ids, body_id]
        frame_quat_w = env.sim.data.xquat[env_ids, body_id]
        pos_error, rot_error = compute_pose_error(
            frame_pos_w,
            frame_quat_w,
            target_pos_w,
            target_quat_w,
        )
        if (
            torch.max(torch.linalg.norm(pos_error, dim=-1)).item() < pose_tol
            and torch.max(torch.linalg.norm(rot_error, dim=-1)).item() < rot_tol
        ):
            break

        jacp, jacr, _null_dir = _compute_nullspace_direction(env, env_ids, body_id, joint_dof_ids)
        q = robot.data.joint_pos[env_ids][:, joint_ids]
        JTJ = wp2 * torch.einsum("bti,btj->bij", jacp, jacp) + wo2 * torch.einsum(
            "bti,btj->bij", jacr, jacr
        )
        JTdx = wp2 * torch.einsum("bti,bt->bi", jacp, pos_error) + wo2 * torch.einsum(
            "bti,bt->bi", jacr, rot_error
        )
        if posture_weight > 0.0:
            JTJ.diagonal(dim1=-2, dim2=-1).add_(wpost2)
            JTdx.add_(wpost2 * (q_pref - q))
        JTJ.diagonal(dim1=-2, dim2=-1).add_(lam * lam)

        dq = _solve_damped_system(JTJ, JTdx, base_damping=lam * lam).clamp(-max_dq, max_dq)
        q_next = torch.minimum(torch.maximum(q + dq, finite_lower), finite_upper)
        q_full = robot.data.joint_pos[env_ids].clone()
        q_full[:, joint_ids] = q_next
        robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        env.sim.forward()

    return robot.data.joint_pos[env_ids][:, joint_ids].clone()


def _trace_nullspace_branch(
    env: "ManagerBasedRlEnv",
    robot: Entity,
    env_ids: torch.Tensor,
    joint_ids: torch.Tensor,
    joint_dof_ids: torch.Tensor,
    body_id: int,
    default_q_full: torch.Tensor,
    qd_full: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    finite_lower: torch.Tensor,
    finite_upper: torch.Tensor,
    direction_sign: float,
    max_iters: int,
    damping: float,
    max_dq: float,
    position_weight: float,
    orientation_weight: float,
    posture_weight: float,
    pose_tol: float,
    rot_tol: float,
    step_size: float = 0.15,
    max_steps: int = 64,
) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    q_current = default_q_full[:, joint_ids].clone()
    prev_dir: torch.Tensor | None = None

    for _ in range(max_steps):
        q_full = default_q_full.clone()
        q_full[:, joint_ids] = q_current
        robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        env.sim.forward()

        _jacp, _jacr, null_dir = _compute_nullspace_direction(env, env_ids, body_id, joint_dof_ids)
        if prev_dir is not None and torch.sum(null_dir * prev_dir, dim=-1).item() < 0.0:
            null_dir = -null_dir
        prev_dir = null_dir.clone()

        max_positive_alpha, max_negative_alpha = _max_alpha_along_direction(
            q_current, null_dir, finite_lower, finite_upper
        )
        travel = max_positive_alpha if direction_sign > 0.0 else max_negative_alpha
        step_alpha = torch.minimum(
            travel,
            torch.full_like(travel, step_size),
        )
        if step_alpha[0].item() < 1e-4:
            break

        q_pref = q_current + direction_sign * step_alpha.unsqueeze(-1) * null_dir
        q_pref = torch.minimum(torch.maximum(q_pref, finite_lower), finite_upper)

        q_full[:, joint_ids] = q_pref
        robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        env.sim.forward()
        q_corrected = _correct_pose_to_target(
            env=env,
            env_ids=env_ids,
            robot=robot,
            joint_ids=joint_ids,
            joint_dof_ids=joint_dof_ids,
            body_id=body_id,
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            q_pref=q_pref,
            qd_full=qd_full,
            finite_lower=finite_lower,
            finite_upper=finite_upper,
            max_iters=max_iters,
            damping=damping,
            max_dq=max_dq,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            posture_weight=posture_weight,
            pose_tol=pose_tol,
            rot_tol=rot_tol,
        )

        move_norm = torch.linalg.norm(q_corrected - q_current, dim=-1)
        if move_norm[0].item() < 1e-4:
            break

        q_current = q_corrected
        samples.append(q_current.squeeze(0).clone())

    return samples


def _get_racquet_nullspace_samples(
    env: "ManagerBasedRlEnv",
    robot: Entity,
    joint_ids: torch.Tensor,
    joint_dof_ids: torch.Tensor,
    body_id: int,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    finite_lower: torch.Tensor,
    finite_upper: torch.Tensor,
    max_iters: int,
    damping: float,
    max_dq: float,
    position_weight: float,
    orientation_weight: float,
    posture_weight: float,
    pose_tol: float,
    rot_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (
        tuple(joint_ids.tolist()),
        round(max_iters, 6),
        round(damping, 6),
        round(max_dq, 6),
        round(position_weight, 6),
        round(orientation_weight, 6),
        round(posture_weight, 6),
        round(pose_tol, 8),
        round(rot_tol, 8),
    )
    cache = getattr(env, "_racquet_nullspace_samples_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["default_q"], cache["negative"], cache["positive"]

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    default_q_full = robot.data.default_joint_pos[env_ids].clone()
    qd_full = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    default_q = default_q_full[:, joint_ids].clone()

    negative = _trace_nullspace_branch(
        env,
        robot,
        env_ids,
        joint_ids,
        joint_dof_ids,
        body_id,
        default_q_full,
        qd_full,
        target_pos_w[:1],
        target_quat_w[:1],
        finite_lower[:1],
        finite_upper[:1],
        direction_sign=-1.0,
        max_iters=max_iters,
        damping=damping,
        max_dq=max_dq,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        posture_weight=posture_weight,
        pose_tol=pose_tol,
        rot_tol=rot_tol,
    )
    positive = _trace_nullspace_branch(
        env,
        robot,
        env_ids,
        joint_ids,
        joint_dof_ids,
        body_id,
        default_q_full,
        qd_full,
        target_pos_w[:1],
        target_quat_w[:1],
        finite_lower[:1],
        finite_upper[:1],
        direction_sign=1.0,
        max_iters=max_iters,
        damping=damping,
        max_dq=max_dq,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        posture_weight=posture_weight,
        pose_tol=pose_tol,
        rot_tol=rot_tol,
    )

    cache = {
        "key": cache_key,
        "default_q": default_q.squeeze(0).clone(),
        "negative": torch.stack(negative) if negative else default_q.squeeze(0)[None].clone(),
        "positive": torch.stack(positive) if positive else default_q.squeeze(0)[None].clone(),
    }
    env._racquet_nullspace_samples_cache = cache
    return cache["default_q"], cache["negative"], cache["positive"]


def reset_joints_preserving_racquet_pose(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    plate_asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    max_iters: int = 16,
    damping: float = 0.05,
    max_dq: float = 0.2,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    posture_weight: float = 0.02,
    pose_tol: float = 1e-3,
    rot_tol: float = 1e-2,
) -> None:
    """Randomize joint posture while preserving the nominal racquet pose."""
    profile_start = time.perf_counter() if getattr(env, "_profile_timing", False) else 0.0
    del velocity_range  # Reset starts from rest regardless of sampled posture.

    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Entity = env.scene[asset_cfg.name]
    joint_ids = _joint_ids_from_cfg(robot, asset_cfg, env.device)
    joint_dof_ids = robot.indexing.joint_v_adr[joint_ids]

    if not hasattr(env, "_racquet_nominal_pose_w"):
        env._racquet_nominal_pose_w = _compute_nominal_racquet_pose_w(
            env=env,
            plate_asset_cfg=plate_asset_cfg,
        )
    nominal_pos_w_all, nominal_quat_w_all = env._racquet_nominal_pose_w
    target_pos_w = nominal_pos_w_all[env_ids]
    target_quat_w = nominal_quat_w_all[env_ids]

    # Use soft limits for posture sampling and fall back to a finite span for
    # joints that are unlimited in MuJoCo (stored as +/-inf).
    lower = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids, 0]
    upper = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids, 1]

    q_full = robot.data.default_joint_pos[env_ids].clone()
    qd_full = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    default_q = q_full[:, joint_ids]
    finite_lower = torch.where(torch.isfinite(lower), lower, default_q - torch.pi)
    finite_upper = torch.where(torch.isfinite(upper), upper, default_q + torch.pi)

    body_id = _body_global_id(robot, plate_asset_cfg.body_ids)
    default_sample_q, negative_samples, positive_samples = _get_racquet_nullspace_samples(
        env=env,
        robot=robot,
        joint_ids=joint_ids,
        joint_dof_ids=joint_dof_ids,
        body_id=body_id,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        finite_lower=finite_lower,
        finite_upper=finite_upper,
        max_iters=max_iters,
        damping=damping,
        max_dq=max_dq,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        posture_weight=posture_weight,
        pose_tol=pose_tol,
        rot_tol=rot_tol,
    )
    if abs(float(position_range[0])) < 1e-9 and abs(float(position_range[1])) < 1e-9:
        all_samples = default_sample_q.unsqueeze(0)
    else:
        all_samples = torch.cat(
            (
                negative_samples.flip(0),
                default_sample_q.unsqueeze(0),
                positive_samples,
            ),
            dim=0,
        )
    env._racquet_nullspace_sample_joint_ids = joint_ids.clone()
    env._racquet_nullspace_samples = all_samples.to(env.device)

    alpha_coeff = sample_uniform(
        position_range[0],
        position_range[1],
        (len(env_ids),),
        device=env.device,
    ).clamp(-1.0, 1.0)
    q_pref = default_sample_q.unsqueeze(0).expand(len(env_ids), -1).clone()
    negative_len = max(int(negative_samples.shape[0]), 1)
    positive_len = max(int(positive_samples.shape[0]), 1)
    negative_mask = alpha_coeff < 0.0
    positive_mask = alpha_coeff > 0.0
    if torch.any(negative_mask):
        neg_idx = torch.clamp(
            torch.round(torch.abs(alpha_coeff[negative_mask]) * (negative_len - 1)).long(),
            min=0,
            max=negative_len - 1,
        )
        q_pref[negative_mask] = negative_samples[neg_idx]
    if torch.any(positive_mask):
        pos_idx = torch.clamp(
            torch.round(torch.abs(alpha_coeff[positive_mask]) * (positive_len - 1)).long(),
            min=0,
            max=positive_len - 1,
        )
        q_pref[positive_mask] = positive_samples[pos_idx]
    q_full[:, joint_ids] = q_pref

    robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
    env.sim.forward()
    final_q_joint = _correct_pose_to_target(
        env=env,
        env_ids=env_ids,
        robot=robot,
        joint_ids=joint_ids,
        joint_dof_ids=joint_dof_ids,
        body_id=body_id,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        q_pref=q_pref,
        qd_full=qd_full,
        finite_lower=finite_lower,
        finite_upper=finite_upper,
        max_iters=max_iters,
        damping=damping,
        max_dq=max_dq,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        posture_weight=posture_weight,
        pose_tol=pose_tol,
        rot_tol=rot_tol,
    )

    final_q = robot.data.joint_pos[env_ids]
    final_qd = torch.zeros_like(robot.data.joint_vel[env_ids])
    robot.write_joint_state_to_sim(final_q, final_qd, env_ids=env_ids)
    robot.set_joint_position_target(final_q, env_ids=env_ids)
    robot.set_joint_velocity_target(final_qd, env_ids=env_ids)
    if not hasattr(env, "_racquet_nullspace_q_ns") or env._racquet_nullspace_q_ns.shape != robot.data.joint_pos.shape:
        env._racquet_nullspace_q_ns = torch.zeros_like(robot.data.joint_pos)
    # Reuse the sampled compatible posture as the null-space reference for the
    # cartesian controller.
    env._racquet_nullspace_q_ns[env_ids] = final_q

    if _reset_debug_enabled():
        _ensure_reset_debug_state(env)
        for idx, env_id in enumerate(env_ids.tolist()):
            print(
                "[RESET_DEBUG][JOINT_RESET] "
                f"env={env_id} "
                f"episode={int(env._reset_debug_episode_id[env_id].item()) + 1} "
                f"joint_pos={_format_debug_vec(final_q[idx])}"
            )

    if getattr(env, "_profile_timing", False):
        elapsed = time.perf_counter() - profile_start
        _accumulate_profile_stat(env, "pose_reset_ik_time", elapsed)
        _accumulate_profile_stat(env, "pose_reset_ik_count", 1.0)


def null_space_reset_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice,
    final_position_range: tuple[float, float],
    initial_scale: float,
    start_steps: int,
    duration_steps: int,
    event_name: str = "reset_robot_joints",
) -> dict[str, float]:
    """Expand the uniform null-space reset range linearly over training."""
    del env_ids
    start_steps = max(int(start_steps), 0)
    duration_steps = max(int(duration_steps), 1)
    initial_scale = min(max(float(initial_scale), 0.0), 1.0)
    progress = min(
        max(float(env.common_step_counter - start_steps) / duration_steps, 0.0),
        1.0,
    )
    scale = initial_scale + (1.0 - initial_scale) * progress
    position_range = (
        float(final_position_range[0]) * scale,
        float(final_position_range[1]) * scale,
    )
    env.event_manager.get_term_cfg(event_name).params["position_range"] = position_range
    return {
        "scale": scale,
        "position_min": position_range[0],
        "position_max": position_range[1],
    }


def body_external_force(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the external force vector applied to a selected body in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_external_force[:, asset_cfg.body_ids].squeeze(1)


def body_external_force_norm(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the norm of the external force applied to a selected body."""
    force = body_external_force(env, asset_cfg)
    return torch.linalg.norm(force, dim=-1)


def reset_ball_on_plate(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    xy_range: tuple[float, float],
    z_offset: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    racquet_x_radius: float | None = None,
    racquet_y_radius: float | None = None,
    release_delay_s: tuple[float, float] = (0.0, 0.0),
    lin_vel_x_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_z_range: tuple[float, float] = (0.0, 0.0),
    ang_vel_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset ball above racquet using uniform elliptical sampling in plate frame."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    # Ensure plate pose reflects any prior reset events (e.g. joint reset).
    env.sim.forward()

    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    _ensure_ball_release_state(env)

    plate_pos_w = robot.data.body_link_pos_w[env_ids][
        :, plate_asset_cfg.body_ids
    ].squeeze(1)
    plate_quat_w = robot.data.body_link_quat_w[env_ids][
        :, plate_asset_cfg.body_ids
    ].squeeze(1)
    env_id_list = env_ids.tolist()

    if _reset_debug_enabled():
        pre_ball_pos_w = ball.data.root_link_pos_w[env_ids].clone()
        pre_ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)[
            env_ids
        ].clone()
        pre_ball_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)[
            env_ids
        ].clone()

    if racquet_x_radius is not None and racquet_y_radius is not None:
        u = sample_uniform(0.0, 1.0, (len(env_ids),), device=env.device)
        theta = sample_uniform(0.0, 2.0 * torch.pi, (len(env_ids),), device=env.device)
        r = torch.sqrt(u)
        x = x_offset + racquet_x_radius * r * torch.cos(theta)
        y = y_offset + racquet_y_radius * r * torch.sin(theta)
    else:
        x = sample_uniform(xy_range[0], xy_range[1], (len(env_ids),), device=env.device)
        x = x + x_offset
        y = sample_uniform(xy_range[0], xy_range[1], (len(env_ids),), device=env.device)
        y = y + y_offset

    z = torch.full_like(x, z_offset)
    offset_plate = torch.stack((x, y, z), dim=-1)
    offset_w = quat_apply(plate_quat_w, offset_plate)
    ball_pos_w = plate_pos_w + offset_w

    quat_w = torch.zeros((len(env_ids), 4), device=env.device)
    quat_w[:, 0] = 1.0
    pose = torch.cat((ball_pos_w, quat_w), dim=-1)
    lin_vel_plate = torch.stack(
        (
            sample_uniform(
                lin_vel_x_range[0],
                lin_vel_x_range[1],
                (len(env_ids),),
                device=env.device,
            ),
            sample_uniform(
                lin_vel_y_range[0],
                lin_vel_y_range[1],
                (len(env_ids),),
                device=env.device,
            ),
            sample_uniform(
                lin_vel_z_range[0],
                lin_vel_z_range[1],
                (len(env_ids),),
                device=env.device,
            ),
        ),
        dim=-1,
    )
    ang_vel_plate = torch.stack(
        (
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
        ),
        dim=-1,
    )
    lin_vel_w = quat_apply(plate_quat_w, lin_vel_plate)
    ang_vel_w = quat_apply(plate_quat_w, ang_vel_plate)
    vel_w = torch.cat((lin_vel_w, ang_vel_w), dim=-1)

    release_delay = sample_uniform(
        release_delay_s[0],
        release_delay_s[1],
        (len(env_ids),),
        device=env.device,
    )
    release_delay_steps = torch.ceil(release_delay / env.step_dt).to(dtype=torch.long)
    held_vel = torch.zeros_like(vel_w)
    pose_vel = torch.where(
        (release_delay_steps > 0).unsqueeze(-1),
        held_vel,
        vel_w,
    )

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(pose_vel, env_ids=env_ids)
    env._ball_release_pending[env_ids] = release_delay_steps > 0
    env._ball_release_steps_remaining[env_ids] = release_delay_steps
    env._ball_release_offset_plate[env_ids] = offset_plate
    env._ball_release_hold_pos_w[env_ids] = ball_pos_w
    env._ball_release_velocity_w[env_ids] = vel_w
    mark_reset_debug_pending(env, env_ids)

    if _reset_debug_enabled():
        env.sim.forward()
        ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
        ball_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
        actual_ball_pos_w = ball.data.root_link_pos_w[env_ids]
        radial_xy = torch.linalg.norm(ball_pos_plate[env_ids, :2], dim=-1)
        below_racquet = ball_pos_plate[env_ids, 2] < -0.06
        on_floor = actual_ball_pos_w[:, 2] < 0.05
        for idx, env_id in enumerate(env_id_list):
            print(
                "[RESET_DEBUG][BALL_RESET] "
                f"env={env_id} "
                f"episode={int(env._reset_debug_episode_id[env_id].item())} "
                f"plate_pos_w={_format_debug_vec(plate_pos_w[idx])} "
                f"plate_quat_w={_format_debug_vec(plate_quat_w[idx])} "
                f"pre_ball_pos_w={_format_debug_vec(pre_ball_pos_w[idx])} "
                f"pre_ball_plate={_format_debug_vec(pre_ball_pos_plate[idx])} "
                f"pre_ball_vel_plate={_format_debug_vec(pre_ball_vel_plate[idx])} "
                f"target_offset_plate={_format_debug_vec(offset_plate[idx])} "
                f"target_ball_pos_w={_format_debug_vec(ball_pos_w[idx])} "
                f"written_vel_w={_format_debug_vec(pose_vel[idx])} "
                f"release_vel_plate={_format_debug_vec(torch.cat((lin_vel_plate[idx], ang_vel_plate[idx]), dim=-1))} "
                f"release_vel_w={_format_debug_vec(vel_w[idx])} "
                f"release_delay_s={float(release_delay[idx].item()):.4f} "
                f"release_delay_steps={int(release_delay_steps[idx].item())} "
                f"actual_ball_pos_w={_format_debug_vec(actual_ball_pos_w[idx])} "
                f"ball_plate={_format_debug_vec(ball_pos_plate[env_id])} "
                f"ball_vel_plate={_format_debug_vec(ball_vel_plate[env_id])} "
                f"ball_radial_xy={float(radial_xy[idx].item()):.4f} "
                f"{_format_debug_bool('below_racquet', bool(below_racquet[idx].item()))} "
                f"{_format_debug_bool('ball_on_floor', bool(on_floor[idx].item()))} "
                f"{_format_debug_bool('release_pending', bool(env._ball_release_pending[env_id].item()))}"
            )


def update_ball_release(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> None:
    """Keep the ball suspended above the plate until its randomized release step."""
    del env_ids, plate_asset_cfg
    if not hasattr(env, "_ball_release_pending"):
        return

    pending = torch.nonzero(env._ball_release_pending, as_tuple=False).squeeze(-1)
    if pending.numel() == 0:
        return

    ball: Entity = env.scene[ball_name]

    hold_ids = pending[env._ball_release_steps_remaining[pending] > 0]
    release_ids = pending[env._ball_release_steps_remaining[pending] == 0]

    if hold_ids.numel() > 0:
        hold_pos_w = env._ball_release_hold_pos_w[hold_ids]
        quat_w = torch.zeros((len(hold_ids), 4), device=env.device, dtype=hold_pos_w.dtype)
        quat_w[:, 0] = 1.0
        pose = torch.cat((hold_pos_w, quat_w), dim=-1)
        vel = torch.zeros((len(hold_ids), 6), device=env.device, dtype=hold_pos_w.dtype)
        ball.write_root_link_pose_to_sim(pose, env_ids=hold_ids)
        ball.write_root_link_velocity_to_sim(vel, env_ids=hold_ids)
        env._ball_release_steps_remaining[hold_ids] -= 1

    if release_ids.numel() > 0:
        release_vel_w = env._ball_release_velocity_w[release_ids]
        ball.write_root_link_velocity_to_sim(release_vel_w, env_ids=release_ids)
        env._ball_release_pending[release_ids] = False
        if _reset_debug_enabled():
            for env_id in release_ids.tolist():
                print(
                    "[RESET_DEBUG][BALL_RELEASE] "
                    f"env={env_id} "
                    f"episode={int(env._reset_debug_episode_id[env_id].item())} "
                    f"step={int(env.episode_length_buf[env_id].item())} "
                    f"release_vel_w={_format_debug_vec(env._ball_release_velocity_w[env_id])}"
                )

    if hold_ids.numel() > 0 or release_ids.numel() > 0:
        env.sim.forward()


def kick_ball_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    lin_vel_x_range: tuple[float, float] = (-0.2, 0.2),
    lin_vel_y_range: tuple[float, float] = (-0.2, 0.2),
    lin_vel_z_range: tuple[float, float] = (-0.05, 0.05),
    ang_vel_range: tuple[float, float] = (-1.0, 1.0),
    add_to_current: bool = True,
) -> None:
    """Apply random world-frame velocity kick to the free ball."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    ball: Entity = env.scene[ball_name]

    if hasattr(ball.data, "root_link_lin_vel_w"):
        lin_vel_w = ball.data.root_link_lin_vel_w[env_ids]
    else:
        lin_vel_w = ball.data.root_link_vel_w[env_ids, :3]

    if hasattr(ball.data, "root_link_ang_vel_w"):
        ang_vel_w = ball.data.root_link_ang_vel_w[env_ids]
    elif hasattr(ball.data, "root_link_vel_w"):
        ang_vel_w = ball.data.root_link_vel_w[env_ids, 3:]
    else:
        ang_vel_w = torch.zeros_like(lin_vel_w)

    kick_lin = torch.stack(
        (
            sample_uniform(
                lin_vel_x_range[0],
                lin_vel_x_range[1],
                (len(env_ids),),
                device=env.device,
            ),
            sample_uniform(
                lin_vel_y_range[0],
                lin_vel_y_range[1],
                (len(env_ids),),
                device=env.device,
            ),
            sample_uniform(
                lin_vel_z_range[0],
                lin_vel_z_range[1],
                (len(env_ids),),
                device=env.device,
            ),
        ),
        dim=-1,
    )

    kick_ang = torch.stack(
        (
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
            sample_uniform(
                ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device
            ),
        ),
        dim=-1,
    )

    out_lin = lin_vel_w + kick_lin if add_to_current else kick_lin
    out_ang = ang_vel_w + kick_ang if add_to_current else kick_ang
    out_vel = torch.cat((out_lin, out_ang), dim=-1)
    ball.write_root_link_velocity_to_sim(out_vel, env_ids=env_ids)


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def randomize_body_mass(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    operation: str = "scale",
) -> None:
    """Randomize body mass for the selected body set."""
    _randomize_body_mass_like_field(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        field_name="body_mass",
        value_range=mass_range,
        operation=operation,
    )


@requires_model_fields("geom_friction")
def randomize_ball_friction(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    geom_name: str,
    friction_scale: tuple[float, float],
    operation: str = "scale",
    racquet_asset_name: str | None = "robot",
    racquet_geom_name: str = "plate_collision",
    component_scales: Sequence[Sequence[float]] | None = None,
) -> None:
    """Randomize the friction triplet of the ball-racquet contact.

    Two things this has to get right, both of which it previously got wrong.

    Indexing: `Entity.find_geoms` returns positions within that entity's own `geom_names`, while
    `sim.model.geom_friction` is indexed by the model-global geom id. For the ball those differ
    (local 0, global 13), so the original `geom_friction[:, 0]` wrote the *terrain* and the ball
    was never touched. The mapping is `entity.indexing.geom_ids`.

    The pair: MuJoCo resolves a contact's friction as the element-wise maximum of the two geom
    frictions, and neither geom sets `priority` or `solmix`. Writing only one of them cannot pull
    the contact below the other's fixed 1.0/0.2/0.0005, so both are written, from the SAME
    per-environment draw - drawing them independently would bias every contact toward the higher
    of two samples.

    Pass `racquet_asset_name=None` to write the ball geom alone.

    `component_scales` gives (sliding, torsional, rolling) ranges separately, overriding
    `friction_scale`. The three are not physically interchangeable - sliding is dimensionless
    while torsional and rolling are lengths - so one shared range cannot be right for all of
    them, and scaling rolling by the factor that suits sliding lands far outside anything
    physical.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    def _global_geom_ids(asset_name: str, name: str) -> list[int]:
        entity: Entity = env.scene[asset_name]
        local_ids, _ = entity.find_geoms(name, preserve_order=True)
        table = entity.indexing.geom_ids
        return [int(table[i]) for i in local_ids]

    geom_ids = _global_geom_ids(ball_name, geom_name)
    if not geom_ids:
        return
    if racquet_asset_name is not None:
        geom_ids += _global_geom_ids(racquet_asset_name, racquet_geom_name)

    geom_ids = torch.as_tensor(geom_ids, device=env.device, dtype=torch.long)
    geom_friction = env.sim.model.geom_friction
    default_geom_friction = env.sim.get_default_field("geom_friction")

    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    base_values = default_geom_friction[geom_ids].unsqueeze(0).expand(len(env_ids), -1, -1)
    # One draw per (env, friction component), shared across both geoms of the pair.
    n_comp = base_values.shape[-1]
    if component_scales is None:
        samples = sample_uniform(
            friction_scale[0], friction_scale[1], (len(env_ids), 1, n_comp), device=env.device
        )
    else:
        if len(component_scales) != n_comp:
            raise ValueError(
                f"component_scales needs {n_comp} (lo, hi) pairs, got {len(component_scales)}"
            )
        samples = torch.cat(
            [
                sample_uniform(float(lo), float(hi), (len(env_ids), 1, 1), device=env.device)
                for lo, hi in component_scales
            ],
            dim=-1,
        )
    samples = samples.expand_as(base_values)

    if operation == "scale":
        geom_friction[env_grid, geom_grid] = base_values * samples
    elif operation == "abs":
        geom_friction[env_grid, geom_grid] = samples
    else:
        raise ValueError(f"Unsupported operation '{operation}'.")


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const_0)
def randomize_body_inertia(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_range: tuple[float, float],
    operation: str = "scale",
) -> None:
    """Randomize body inertia for the selected body set."""
    _randomize_body_mass_like_field(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        field_name="body_inertia",
        value_range=inertia_range,
        operation=operation,
    )


@requires_model_fields(
    "body_mass", "body_inertia", "dof_armature", recompute=RecomputeLevel.set_const
)
def randomize_robot_model(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    body_mass_range: tuple[float, float],
    body_inertia_range: tuple[float, float],
    dof_armature_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize robot inertial parameters and armature."""
    randomize_body_mass(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        mass_range=body_mass_range,
        operation="scale",
    )
    randomize_body_inertia(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        inertia_range=body_inertia_range,
        operation="scale",
    )
    _randomize_dof_armature(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        value_range=dof_armature_range,
        operation="scale",
    )


@requires_model_fields(
    "body_mass", "body_inertia", recompute=RecomputeLevel.set_const
)
def randomize_racquet_model(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    body_mass_range: tuple[float, float],
    body_inertia_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize the inertial properties of the racquet / plate body."""
    randomize_body_mass(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        mass_range=body_mass_range,
        operation="scale",
    )
    randomize_body_inertia(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        inertia_range=body_inertia_range,
        operation="scale",
    )


def randomize_ft_sensor_response(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    force_gain: float,
    anisotropy_range: tuple[float, float],
    crosstalk_norm_range: tuple[float, float],
    bias_force_range: tuple[float, float],
    bias_torque_range: tuple[float, float],
) -> None:
    """Randomize the part of the sensor's response that is real but not worth baking in.

    The F/T investigation measured both effects directly: principal gains of about
    0.997 / 0.992 / 0.931 around a mean of 0.9734, and a torque-to-force leakage of roughly
    1.6 N per N.m. The leakage is far outside any plausible load-cell cross-talk spec,
    which points at the *mount* deforming rather than the sensor element -- so it may
    vanish once the mechanical interface is inspected, and hard-coding it would train the
    policy against a fault we intend to fix. Randomizing over it instead makes the policy
    tolerant either way.

    The gain is built as ``force_gain * R diag(g) R^T`` with ``R`` a uniformly random
    rotation, so the anisotropy has no preferred direction -- the one measured weak axis
    was fitted on a single tool and should not be treated as a property of the hardware.

    The bias covers thermal drift accumulating after the tare. It is deliberately small:
    the sensor is re-zeroed at the home pose with the racquet on at every startup, so there
    is no standing offset for it to model, only drift since.
    """
    state = _ft_sensor_state(env, force_gain)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.numel())
    if n == 0:
        return

    # Uniformly random rotation, via a normalized Gaussian quaternion.
    quat = torch.randn(n, 4, device=env.device, dtype=torch.float32)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    rot = matrix_from_quat(quat)

    gains = sample_uniform(anisotropy_range[0], anisotropy_range[1], (n, 3), env.device)
    state["gain"][env_ids] = force_gain * (rot @ torch.diag_embed(gains) @ rot.transpose(-1, -2))

    # A random direction in 3x3 space, scaled to a drawn Frobenius norm.
    leak = torch.randn(n, 3, 3, device=env.device, dtype=torch.float32)
    leak = leak / torch.linalg.matrix_norm(leak, keepdim=True).clamp_min(1e-6)
    norms = sample_uniform(crosstalk_norm_range[0], crosstalk_norm_range[1], (n, 1, 1), env.device)
    state["leakage"][env_ids] = leak * norms

    state["bias"][env_ids] = torch.cat(
        (
            sample_uniform(bias_force_range[0], bias_force_range[1], (n, 3), env.device),
            sample_uniform(bias_torque_range[0], bias_torque_range[1], (n, 3), env.device),
        ),
        dim=-1,
    )


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_pd_gains(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    operation: str = "scale",
) -> None:
    """Randomize position-actuator PD gains for selected actuators."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    actuator_ids = asset.indexing.ctrl_ids

    gainprm = env.sim.model.actuator_gainprm
    biasprm = env.sim.model.actuator_biasprm
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")

    env_grid, act_grid = torch.meshgrid(env_ids, actuator_ids, indexing="ij")

    base_kp = default_gainprm[actuator_ids, 0].unsqueeze(0).expand(len(env_ids), -1)
    base_kd = (-default_biasprm[actuator_ids, 2]).unsqueeze(0).expand(len(env_ids), -1)

    kp_samples = sample_uniform(
        kp_range[0],
        kp_range[1],
        base_kp.shape,
        device=env.device,
    )
    kd_samples = sample_uniform(
        kd_range[0],
        kd_range[1],
        base_kd.shape,
        device=env.device,
    )

    if operation == "scale":
        out_kp = base_kp * kp_samples
        out_kd = base_kd * kd_samples
    elif operation == "abs":
        out_kp = kp_samples
        out_kd = kd_samples
    else:
        raise ValueError(f"Unsupported operation '{operation}'.")

    gainprm[env_grid, act_grid, 0] = out_kp
    biasprm[env_grid, act_grid, 1] = -out_kp
    biasprm[env_grid, act_grid, 2] = -out_kd


def randomize_osc_pd_gains(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    action_name: str,
) -> None:
    """Randomize Cartesian OSC task-space and null-space PD gains."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "set_pd_gain_scales"):
        raise TypeError(
            f"Action term `{action_name}` does not support OSC PD gain randomization"
        )

    kp_scales = sample_uniform(
        kp_range[0],
        kp_range[1],
        (env_ids.numel(), 2),
        device=env.device,
    )
    kd_scales = sample_uniform(
        kd_range[0],
        kd_range[1],
        (env_ids.numel(), 2),
        device=env.device,
    )
    action_term.set_pd_gain_scales(env_ids, kp_scales, kd_scales)


def _randomize_body_mass_like_field(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    field_name: str,
    value_range: tuple[float, float],
    operation: str,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    model_field = getattr(env.sim.model, field_name)
    default_field = env.sim.get_default_field(field_name)

    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    base_values = default_field[body_ids]
    if len(model_field.shape) == 3:
        base_values = base_values.unsqueeze(0).expand(len(env_ids), -1, -1)
    else:
        base_values = base_values.unsqueeze(0).expand(len(env_ids), -1)

    samples = sample_uniform(
        value_range[0],
        value_range[1],
        base_values.shape,
        device=env.device,
    )

    if operation == "scale":
        model_field[env_grid, body_grid] = base_values * samples
    elif operation == "abs":
        model_field[env_grid, body_grid] = samples
    else:
        raise ValueError(f"Unsupported operation '{operation}'.")


def _randomize_dof_armature(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    value_range: tuple[float, float],
    operation: str,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    dof_ids = asset.indexing.joint_v_adr
    model_field = env.sim.model.dof_armature
    default_field = env.sim.get_default_field("dof_armature")

    env_grid, dof_grid = torch.meshgrid(env_ids, dof_ids, indexing="ij")
    base_values = default_field[dof_ids].unsqueeze(0).expand(len(env_ids), -1)

    samples = sample_uniform(
        value_range[0],
        value_range[1],
        base_values.shape,
        device=env.device,
    )

    if operation == "scale":
        model_field[env_grid, dof_grid] = base_values * samples
    elif operation == "abs":
        model_field[env_grid, dof_grid] = samples
    else:
        raise ValueError(f"Unsupported operation '{operation}'.")


