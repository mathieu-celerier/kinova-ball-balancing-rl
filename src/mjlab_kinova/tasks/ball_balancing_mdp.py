"""Task-local MDP functions for Kinova ball balancing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.math import sample_uniform
from mjlab_kinova.robot.kinova_constants import KINOVA_CFG

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


def body_position_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body position in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)


def body_linear_velocity_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the selected body linear velocity in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(asset.data, "body_link_lin_vel_w"):
        return asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids].squeeze(1)
    return asset.data.body_link_vel_w[:, asset_cfg.body_ids, :3].squeeze(1)


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
    lin_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> torch.Tensor:
    """Penalty on ball linear and angular speed in plate frame."""
    ball_lin_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
    ball_ang_vel_plate = ball_ang_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
    lin_penalty = torch.sum(torch.square(ball_lin_vel_plate), dim=-1)
    ang_penalty = torch.sum(torch.square(ball_ang_vel_plate), dim=-1)
    return lin_weight * lin_penalty + ang_weight * ang_penalty


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
) -> torch.Tensor:
    """Terminate when the ball rises too high above the plate in plate frame."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    return ball_pos_plate[:, 2] > max_height


def ball_height_above_plate_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    soft_height: float,
) -> torch.Tensor:
    """Penalty on ball height above a soft plate-frame threshold."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    excess = torch.clamp(ball_pos_plate[:, 2] - soft_height, min=0.0)
    return torch.square(excess)


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
        geom_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
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
        torch.logical_and(contact_geom[:, 0] == ball_geom_id, contact_geom[:, 1] == racquet_geom_id),
        torch.logical_and(contact_geom[:, 0] == racquet_geom_id, contact_geom[:, 1] == ball_geom_id),
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


def _compute_nominal_racquet_pos_w(
    env: "ManagerBasedRlEnv",
    plate_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute the racquet world position at the robot home joint configuration."""
    robot: Entity = env.scene[plate_asset_cfg.name]

    current_joint_pos = robot.data.joint_pos.clone()
    current_joint_vel = robot.data.joint_vel.clone()

    home_joint_map = KINOVA_CFG.init_state.joint_pos
    home_joint_pos = torch.tensor(
        [home_joint_map[name] for name in robot.joint_names],
        device=env.device,
        dtype=current_joint_pos.dtype,
    ).unsqueeze(0).expand(env.num_envs, -1)
    zero_joint_vel = torch.zeros_like(current_joint_vel)

    robot.write_joint_state_to_sim(home_joint_pos, zero_joint_vel)
    env.sim.forward()
    nominal_pos_w = robot.data.body_link_pos_w[:, plate_asset_cfg.body_ids].squeeze(1).clone()

    robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel)
    env.sim.forward()

    return nominal_pos_w


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

    quat_w = torch.zeros((len(env_ids), 4), device=env.device)
    quat_w[:, 0] = 1.0
    pose = torch.cat((ball_pos_w, quat_w), dim=-1)
    vel = torch.stack(
        (
            sample_uniform(lin_vel_x_range[0], lin_vel_x_range[1], (len(env_ids),), device=env.device),
            sample_uniform(lin_vel_y_range[0], lin_vel_y_range[1], (len(env_ids),), device=env.device),
            sample_uniform(lin_vel_z_range[0], lin_vel_z_range[1], (len(env_ids),), device=env.device),
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
        ),
        dim=-1,
    )

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


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
            sample_uniform(lin_vel_x_range[0], lin_vel_x_range[1], (len(env_ids),), device=env.device),
            sample_uniform(lin_vel_y_range[0], lin_vel_y_range[1], (len(env_ids),), device=env.device),
            sample_uniform(lin_vel_z_range[0], lin_vel_z_range[1], (len(env_ids),), device=env.device),
        ),
        dim=-1,
    )

    kick_ang = torch.stack(
        (
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
            sample_uniform(ang_vel_range[0], ang_vel_range[1], (len(env_ids),), device=env.device),
        ),
        dim=-1,
    )

    out_lin = lin_vel_w + kick_lin if add_to_current else kick_lin
    out_ang = ang_vel_w + kick_ang if add_to_current else kick_ang
    out_vel = torch.cat((out_lin, out_ang), dim=-1)
    ball.write_root_link_velocity_to_sim(out_vel, env_ids=env_ids)


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


def randomize_robot_model(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    body_mass_range: tuple[float, float],
    body_inertia_range: tuple[float, float],
    dof_armature_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize robot inertial parameters and armature."""
    from mjlab.envs import mdp

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
    mdp.randomize_field(
        env=env,
        env_ids=env_ids,
        field="dof_armature",
        ranges=dof_armature_range,
        operation="scale",
        distribution="uniform",
        asset_cfg=asset_cfg,
    )


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


randomize_body_mass.model_fields = ("body_mass",)
randomize_body_inertia.model_fields = ("body_inertia",)
randomize_robot_model.model_fields = ("body_mass", "body_inertia", "dof_armature")
