"""Task-local MDP functions for Kinova ball balancing."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any, Callable

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import compute_pose_error, quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.math import sample_uniform
from mjlab_kinova.robot.kinova_constants import KINOVA_CFG

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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
    ft_bias = getattr(env, "_ee_ft_bias", None)
    if ft_bias is None:
        bias_wrench = raw_wrench
    else:
        bias_wrench = raw_wrench - ft_bias
    no_contact = ball_no_contact_mujoco(env=env)
    try:
        joint_term = env.action_manager.get_term("joint_pos")
    except Exception:
        joint_term = None
    processed_actions = getattr(joint_term, "_processed_actions", None)

    for env_id in pending.tolist():
        msg = (
            f"[RESET_DEBUG][STEP] env={env_id} "
            f"episode={int(env._reset_debug_episode_id[env_id].item())} "
            f"step={int(env.episode_length_buf[env_id].item())} "
            f"contact={bool((no_contact[env_id] == 0.0).item())} "
            f"raw_action={_format_debug_vec(env.action_manager.action[env_id])} "
            f"joint_target={_format_debug_vec(robot.data.joint_pos_target[env_id])} "
            f"joint_pos={_format_debug_vec(robot.data.joint_pos[env_id])} "
            f"ball_plate={_format_debug_vec(ball_pos_plate[env_id])} "
            f"ball_vel_plate={_format_debug_vec(ball_vel_plate[env_id])} "
            f"ee_ft_raw={_format_debug_vec(raw_wrench[env_id])} "
            f"ee_ft_bias={_format_debug_vec(bias_wrench[env_id])}"
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
    """Return the selected body orientation quaternion in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)


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
    wrench = torch.cat((force_sensor.data, torque_sensor.data), dim=-1)
    if not hasattr(env, "_ee_ft_bias"):
        env._ee_ft_bias = torch.zeros_like(wrench)

    pending = getattr(env, "_ee_ft_bias_pending", None)
    if pending is not None:
        pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        if pending_ids.numel() > 0:
            env._ee_ft_bias[pending_ids] = wrench[pending_ids]
            env._ee_ft_bias_pending[pending_ids] = False
            if _reset_debug_enabled():
                _ensure_reset_debug_state(env)
                for env_id in pending_ids.tolist():
                    print(
                        "[RESET_DEBUG][FT_BIAS_LATCH] "
                        f"env={env_id} "
                        f"episode={int(env._reset_debug_episode_id[env_id].item())} "
                        f"stored_bias={_format_debug_vec(env._ee_ft_bias[env_id])} "
                        f"raw_wrench={_format_debug_vec(wrench[env_id])}"
                    )

    return wrench - env._ee_ft_bias


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


def ball_centering_contact_reward(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    ball_geom_name: str,
    racquet_geom_name: str,
    max_contact_dist: float,
    std: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> torch.Tensor:
    """Reward centering only while ball-racquet contact is active."""
    centering = ball_centering_reward(
        env=env,
        ball_name=ball_name,
        plate_asset_cfg=plate_asset_cfg,
        std=std,
        center_x=center_x,
        center_y=center_y,
    )
    no_contact = ball_no_contact_mujoco(
        env=env,
        ball_geom_name=ball_geom_name,
        racquet_geom_name=racquet_geom_name,
        max_contact_dist=max_contact_dist,
    )
    return centering * (1.0 - no_contact)


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


def plate_drop_under_ball_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    ball_height_threshold: float,
    xy_radius: float,
) -> torch.Tensor:
    """Penalty on moving the plate down along its normal while the ball is still above it."""
    ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
    plate_vel_plate = body_linear_velocity_in_body_frame(env, plate_asset_cfg)

    radial_xy = torch.linalg.norm(ball_pos_plate[:, :2], dim=-1)
    ball_near_racquet = torch.logical_and(
        ball_pos_plate[:, 2] > ball_height_threshold,
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


class ball_no_contact_after_first_contact(ManagerTermBase):
    """Penalty is inactive until an env has established ball-racquet contact once."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self.params = cfg.params
        self._has_seen_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._has_seen_contact[env_ids] = False

    def __call__(self, env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
        no_contact = ball_no_contact_mujoco(env=env, **kwargs)
        has_contact = no_contact == 0.0
        self._has_seen_contact |= has_contact
        return no_contact * self._has_seen_contact.float()


class contact_phase_reward(ManagerTermBase):
    """Gate an arbitrary reward term by whether first contact has already happened."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self.params = cfg.params
        self._has_seen_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _resolve_nested_term_kwargs(self, value: Any) -> Any:
        if isinstance(value, SceneEntityCfg):
            value.resolve(self._env.scene)
            return value
        if isinstance(value, dict):
            return {
                key: self._resolve_nested_term_kwargs(sub_value)
                for key, sub_value in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_nested_term_kwargs(sub_value) for sub_value in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_nested_term_kwargs(sub_value) for sub_value in value)
        return value

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._has_seen_contact[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        term_func: Callable[..., torch.Tensor],
        term_kwargs: dict[str, Any] | None = None,
        ball_geom_name: str = "ball/ball_geom",
        racquet_geom_name: str = "robot/plate_collision",
        max_contact_dist: float = 0.0,
        activate_after_contact: bool = True,
    ) -> torch.Tensor:
        no_contact = ball_no_contact_mujoco(
            env=env,
            ball_geom_name=ball_geom_name,
            racquet_geom_name=racquet_geom_name,
            max_contact_dist=max_contact_dist,
        )
        has_contact = no_contact == 0.0
        self._has_seen_contact |= has_contact

        resolved_term_kwargs = self._resolve_nested_term_kwargs(term_kwargs or {})
        base_reward = term_func(env, **resolved_term_kwargs)
        if base_reward.ndim > 1:
            base_reward = base_reward.reshape(base_reward.shape[0], -1).sum(dim=-1)
        phase_mask = (
            self._has_seen_contact
            if activate_after_contact
            else ~self._has_seen_contact
        )
        return base_reward * phase_mask.float()


class observation_history(ManagerTermBase):
    """Stack a fixed recent timestep window for an arbitrary observation term."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self.params = cfg.params
        history_length = int(self.params.get("history_length", 1))
        self._history_length = max(history_length, 1)
        self._history: torch.Tensor | None = None

        term_func = self.params["term_func"]
        term_kwargs = self._resolve_nested_term_kwargs(
            self.params.get("term_kwargs") or {}
        )
        if isinstance(term_func, type) and issubclass(term_func, ManagerTermBase):
            wrapped_cfg = SimpleNamespace(params=term_kwargs)
            self._term = term_func(wrapped_cfg, env)
            self._term_is_manager = True
        else:
            self._term = term_func
            self._term_is_manager = False
        self._term_kwargs = term_kwargs

    def _resolve_nested_term_kwargs(self, value: Any) -> Any:
        if isinstance(value, SceneEntityCfg):
            value.resolve(self._env.scene)
            return value
        if isinstance(value, dict):
            return {
                key: self._resolve_nested_term_kwargs(sub_value)
                for key, sub_value in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_nested_term_kwargs(sub_value) for sub_value in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_nested_term_kwargs(sub_value) for sub_value in value)
        return value

    def _ensure_history(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(-1)
        else:
            obs = obs.reshape(obs.shape[0], -1)

        if self._history is None or self._history.shape[-1] != obs.shape[-1]:
            self._history = torch.zeros(
                self.num_envs,
                self._history_length,
                obs.shape[-1],
                device=self.device,
                dtype=obs.dtype,
            )
        return obs

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self._history is not None:
            self._history[env_ids] = 0.0
        if self._term_is_manager:
            self._term.reset(env_ids=env_ids)

    def __call__(self, env: "ManagerBasedRlEnv", **_kwargs) -> torch.Tensor:
        if self._term_is_manager:
            obs = self._term(env, **self._term_kwargs)
        else:
            obs = self._term(env, **self._term_kwargs)
        obs = self._ensure_history(obs)

        assert self._history is not None
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1, :] = obs
        return self._history.reshape(self.num_envs, -1)


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
    nominal_quat_w = (
        robot.data.body_link_quat_w[:, plate_asset_cfg.body_ids].squeeze(1).clone()
    )

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


def reset_ee_ft_bias(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    force_sensor_name: str = "robot/EEForceSensor_fsensor",
    torque_sensor_name: str = "robot/EEForceSensor_tsensor",
) -> None:
    """Arm a bias reset to be captured from the first post-reset observation."""
    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    env.sim.forward()
    force_sensor = env.scene[force_sensor_name]
    torque_sensor = env.scene[torque_sensor_name]
    assert isinstance(force_sensor, BuiltinSensor)
    assert isinstance(torque_sensor, BuiltinSensor)
    wrench = torch.cat((force_sensor.data, torque_sensor.data), dim=-1)
    if not hasattr(env, "_ee_ft_bias"):
        env._ee_ft_bias = torch.zeros_like(wrench)
    if not hasattr(env, "_ee_ft_bias_pending"):
        env._ee_ft_bias_pending = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    env._ee_ft_bias_pending[env_ids] = True

    if _reset_debug_enabled():
        for env_id in env_ids.tolist():
            print(
                "[RESET_DEBUG][FT_BIAS_ARM] "
                f"env={env_id} "
                f"episode={int(getattr(env, '_reset_debug_episode_id', torch.zeros(1, dtype=torch.long))[env_id].item()) + 1} "
                f"pending=True "
                f"raw_wrench={_format_debug_vec(wrench[env_id])}"
            )


def clear_ball_for_ee_ft_bias_reset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    plate_asset_cfg: SceneEntityCfg,
    clear_height: float = 0.30,
) -> None:
    """Move the ball away from the racquet before capturing the EE F/T zero."""
    env_ids = _as_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    env.sim.forward()

    robot: Entity = env.scene[plate_asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    plate_pos_w = robot.data.body_link_pos_w[env_ids][:, plate_asset_cfg.body_ids].squeeze(1)
    plate_quat_w = robot.data.body_link_quat_w[env_ids][:, plate_asset_cfg.body_ids].squeeze(1)
    clear_offset_plate = torch.zeros((len(env_ids), 3), device=env.device, dtype=plate_pos_w.dtype)
    clear_offset_plate[:, 2] = clear_height
    clear_pos_w = plate_pos_w + quat_apply(plate_quat_w, clear_offset_plate)

    quat_w = torch.zeros((len(env_ids), 4), device=env.device, dtype=plate_pos_w.dtype)
    quat_w[:, 0] = 1.0
    pose = torch.cat((clear_pos_w, quat_w), dim=-1)
    vel = torch.zeros((len(env_ids), 6), device=env.device, dtype=plate_pos_w.dtype)

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    env.sim.forward()

    if _reset_debug_enabled():
        ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
        ball_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
        for env_id in env_ids.tolist():
            print(
                "[RESET_DEBUG][BALL_CLEAR] "
                f"env={env_id} "
                f"episode={int(env._reset_debug_episode_id[env_id].item()) + 1} "
                f"ball_plate={_format_debug_vec(ball_pos_plate[env_id])} "
                f"ball_vel_plate={_format_debug_vec(ball_vel_plate[env_id])}"
            )


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

    plate_pos_w = robot.data.body_link_pos_w[env_ids][
        :, plate_asset_cfg.body_ids
    ].squeeze(1)
    plate_quat_w = robot.data.body_link_quat_w[env_ids][
        :, plate_asset_cfg.body_ids
    ].squeeze(1)

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

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    mark_reset_debug_pending(env, env_ids)

    if _reset_debug_enabled():
        env.sim.forward()
        ball_pos_plate = ball_pos_in_plate_frame(env, ball_name, plate_asset_cfg)
        ball_vel_plate = ball_lin_vel_in_plate_frame(env, ball_name, plate_asset_cfg)
        for env_id in env_ids.tolist():
            print(
                "[RESET_DEBUG][BALL_RESET] "
                f"env={env_id} "
                f"episode={int(env._reset_debug_episode_id[env_id].item())} "
                f"ball_plate={_format_debug_vec(ball_pos_plate[env_id])} "
                f"ball_vel_plate={_format_debug_vec(ball_vel_plate[env_id])}"
            )


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


randomize_body_mass.model_fields = ("body_mass",)
randomize_body_inertia.model_fields = ("body_inertia",)
randomize_robot_model.model_fields = ("body_mass", "body_inertia", "dof_armature")
randomize_pd_gains.model_fields = ("actuator_gainprm", "actuator_biasprm")
