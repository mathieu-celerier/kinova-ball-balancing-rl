"""Task-local MDP functions for Kinova ball balancing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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


def joint_torques(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Return actuator torques/forces for the robot."""
    robot: Entity = env.scene[robot_name]
    return robot.data.actuator_force


def ee_ft_wrench(
    env: "ManagerBasedRlEnv",
    force_sensor_name: str = "robot/EEForceSensor_fsensor",
    torque_sensor_name: str = "robot/EEForceSensor_tsensor",
) -> torch.Tensor:
    """Return end-effector F/T wrench [Fx, Fy, Fz, Tx, Ty, Tz]."""
    force_sensor = env.scene[force_sensor_name]
    torque_sensor = env.scene[torque_sensor_name]
    assert isinstance(force_sensor, BuiltinSensor)
    assert isinstance(torque_sensor, BuiltinSensor)
    return torch.cat((force_sensor.data, torque_sensor.data), dim=-1)


def joint_torque_l2(
    env: "ManagerBasedRlEnv",
    robot_name: str = "robot",
) -> torch.Tensor:
    """Penalty term on squared actuator torques/forces."""
    torques = joint_torques(env, robot_name=robot_name)
    return torch.sum(torch.square(torques), dim=-1)


def ball_centering_reward(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    std: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> torch.Tensor:
    """Reward for keeping the ball near the plate center in XY."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    dx = ball_pos_plate[:, 0] - center_x
    dy = ball_pos_plate[:, 1] - center_y
    radial_sq = torch.square(dx) + torch.square(dy)
    return torch.exp(-radial_sq / (std**2))


def ball_speed_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on ball speed in plate frame."""
    ball_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
    return torch.sum(torch.square(ball_vel_plate), dim=-1)


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


def plate_too_low(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
    min_plate_height: float,
) -> torch.Tensor:
    """Penalty signal: 1.0 when the plate height is below a minimum world-frame threshold."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)
    return (plate_pos_w[:, 2] < min_plate_height).float()


def ball_no_contact_proxy(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    contact_z: float,
    z_tolerance: float,
    max_xy_radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> torch.Tensor:
    """Penalty proxy for missing plate contact using a plate-frame contact band.

    Returns 1.0 when the ball center is outside the expected contact band.
    """
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    dx = ball_pos_plate[:, 0] - center_x
    dy = ball_pos_plate[:, 1] - center_y
    radial = torch.sqrt(torch.square(dx) + torch.square(dy))
    z_gap = torch.abs(ball_pos_plate[:, 2] - contact_z)
    in_contact_band = torch.logical_and(radial <= max_xy_radius, z_gap <= z_tolerance)
    return torch.logical_not(in_contact_band).float()


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
        plate_vel_w = robot.data.body_link_lin_vel_w[:, plate_asset_cfg.body_ids].squeeze(1)
    else:
        plate_vel_w = robot.data.body_link_vel_w[:, plate_asset_cfg.body_ids, :3].squeeze(1)
    return torch.sum(torch.square(plate_vel_w), dim=-1)


def racquet_dist_from_initial_l2(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on racquet displacement from per-episode initial world position."""
    robot: Entity = env.scene[plate_asset_cfg.name]
    plate_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1)

    if not hasattr(env, "_racquet_init_pos_w"):
        env._racquet_init_pos_w = plate_pos_w.clone()
        return torch.zeros(env.num_envs, device=env.device)

    init_pos_w = env._racquet_init_pos_w
    return torch.sum(torch.square(plate_pos_w - init_pos_w), dim=-1)


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
) -> None:
    """Reset ball above racquet using uniform elliptical sampling in plate frame."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    # Ensure plate pose reflects any prior reset events (e.g. joint reset).
    env.sim.forward()

    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    plate_pos_w = robot.data.body_link_pos_w[env_ids][:, plate_asset_cfg.body_ids].squeeze(1)
    plate_quat_w = robot.data.body_link_quat_w[env_ids][:, plate_asset_cfg.body_ids].squeeze(1)

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

    if not hasattr(env, "_racquet_init_pos_w"):
        env._racquet_init_pos_w = torch.zeros(
            (env.num_envs, 3), device=env.device, dtype=plate_pos_w.dtype
        )
    env._racquet_init_pos_w[env_ids] = plate_pos_w

    quat_w = torch.zeros((len(env_ids), 4), device=env.device)
    quat_w[:, 0] = 1.0
    pose = torch.cat((ball_pos_w, quat_w), dim=-1)
    vel = torch.zeros((len(env_ids), 6), device=env.device)

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
