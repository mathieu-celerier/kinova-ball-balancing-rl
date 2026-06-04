"""Task-specific action terms for Kinova ball balancing policies."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.differential_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import apply_delta_pose, compute_pose_error
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values


def _gain_tensor(
    gain: float | dict[str, float],
    target_names: list[str],
    num_envs: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(gain, (float, int)):
        return torch.full((num_envs, action_dim), float(gain), device=device)
    values = torch.zeros(num_envs, action_dim, device=device)
    index_list, _, value_list = resolve_matching_names_values(gain, target_names)
    values[:, index_list] = torch.tensor(value_list, device=device)
    return values


def _dynamically_consistent_nullspace_torque(
    action: BaseAction,
    jac: torch.Tensor,
    joint_dof_ids: torch.Tensor,
    tau_ref: torch.Tensor,
    damping_pinv: float,
) -> torch.Tensor:
    """Project joint torques with the dynamically consistent torque null-space."""
    nworld = action.num_envs
    nv = action._env.sim.mj_model.nv
    rhs_wp = getattr(action, "_mass_solve_rhs_wp", None)
    sol_wp = getattr(action, "_mass_solve_sol_wp", None)
    if rhs_wp is None or sol_wp is None:
        with wp.ScopedDevice(action._env.sim.wp_device):
            rhs_wp = wp.zeros((nworld, nv), dtype=float)
            sol_wp = wp.zeros((nworld, nv), dtype=float)
        action._mass_solve_rhs_wp = rhs_wp
        action._mass_solve_sol_wp = sol_wp
        action._mass_solve_rhs_torch = wp.to_torch(rhs_wp)
        action._mass_solve_sol_torch = wp.to_torch(sol_wp)

    rhs_torch = action._mass_solve_rhs_torch
    sol_torch = action._mass_solve_sol_torch

    with wp.ScopedDevice(action._env.sim.wp_device):
        mjwarp.crb(action._env.sim.wp_model, action._env.sim.wp_data)
        mjwarp.factor_m(action._env.sim.wp_model, action._env.sim.wp_data)

    task_dim = jac.shape[1]
    m_inv_jt_cols: list[torch.Tensor] = []
    for task_idx in range(task_dim):
        rhs_torch.zero_()
        rhs_torch[:, joint_dof_ids] = jac[:, task_idx, :]
        with wp.ScopedDevice(action._env.sim.wp_device):
            mjwarp.solve_m(
                action._env.sim.wp_model,
                action._env.sim.wp_data,
                sol_wp,
                rhs_wp,
            )
        m_inv_jt_cols.append(sol_torch[:, joint_dof_ids].clone())

    m_inv_jt = torch.stack(m_inv_jt_cols, dim=-1)
    lambda_inv = torch.matmul(jac, m_inv_jt)
    eye_task = torch.eye(task_dim, device=jac.device, dtype=jac.dtype).unsqueeze(0)
    damping = max(damping_pinv, 1e-6)
    lambda_damped = torch.linalg.inv(lambda_inv + (damping**2) * eye_task)
    j_bar = torch.matmul(m_inv_jt, lambda_damped)

    task_leak = torch.einsum("bji,bj->bi", j_bar, tau_ref)
    return tau_ref - torch.einsum("bij,bi->bj", jac, task_leak)


@dataclass(kw_only=True)
class JointNullspaceTorqueActionCfg(BaseActionCfg):
    """Joint action interpreted as a PD torque plus racquet-frame null-space torque."""

    frame_name: str
    use_default_offset: bool = True
    stiffness: float | dict[str, float] = 1.0
    damping: float | dict[str, float] = 0.0
    nullspace_stiffness: float | dict[str, float] = 0.0
    nullspace_damping: float | dict[str, float] = 0.0
    damping_pinv: float = 0.05
    nullspace_resample_interval_s: tuple[float, float] = (0.25, 1.0)

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env: ManagerBasedRlEnv) -> "JointNullspaceTorqueAction":
        return JointNullspaceTorqueAction(self, env)


class JointNullspaceTorqueAction(BaseAction):
    """Apply ``tau_rl + N tau_ns`` while preserving the joint-position action API."""

    cfg: JointNullspaceTorqueActionCfg

    def __init__(self, cfg: JointNullspaceTorqueActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        if cfg.use_default_offset:
            self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

        self._ctrl_ids = self._entity.indexing.ctrl_ids[self._target_ids]
        self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._target_ids]
        body_ids, _ = self._entity.find_bodies(cfg.frame_name)
        local_body_id = body_ids[0]
        self._body_id = int(self._entity.indexing.body_ids[local_body_id].item())

        self._kp = _gain_tensor(
            cfg.stiffness, self._target_names, self.num_envs, self.action_dim, self.device
        )
        self._kd = _gain_tensor(
            cfg.damping, self._target_names, self.num_envs, self.action_dim, self.device
        )
        self._kp_ns = _gain_tensor(
            cfg.nullspace_stiffness,
            self._target_names,
            self.num_envs,
            self.action_dim,
            self.device,
        )
        self._kd_ns = _gain_tensor(
            cfg.nullspace_damping,
            self._target_names,
            self.num_envs,
            self.action_dim,
            self.device,
        )
        self._q_ns = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._next_nullspace_resample_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        nworld = self.num_envs
        nv = self._env.sim.mj_model.nv
        with wp.ScopedDevice(self._env.sim.wp_device):
            self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
            self._body_wp = wp.zeros(nworld, dtype=wp.int32)
            self._body_wp.fill_(self._body_id)

        self._jacp_torch = wp.to_torch(self._jacp_wp)
        self._jacr_torch = wp.to_torch(self._jacr_wp)
        self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

    def _sample_next_nullspace_resample_step(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        min_s, max_s = self.cfg.nullspace_resample_interval_s
        min_steps = max(1, int(round(min_s / self._env.step_dt)))
        max_steps = max(min_steps, int(round(max_s / self._env.step_dt)))
        if max_steps == min_steps:
            return torch.full((env_ids.numel(),), min_steps, device=self.device, dtype=torch.long)
        return torch.randint(
            min_steps,
            max_steps + 1,
            (env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._maybe_resample_nullspace_target()

    def apply_actions(self) -> None:
        q = self._entity.data.joint_pos[:, self._target_ids]
        qd = self._entity.data.joint_vel[:, self._target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        q_rl = self._processed_actions - encoder_bias
        tau_rl = self._kp * (q_rl - q) - self._kd * qd

        tau_ns_ref = self._kp_ns * (self._q_ns - q) - self._kd_ns * qd

        frame_pos = self._env.sim.data.xpos[:, self._body_id]
        self._point_torch[:] = frame_pos
        with wp.ScopedDevice(self._env.sim.wp_device):
            mjwarp.jac(
                self._env.sim.wp_model,
                self._env.sim.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )
        jacp = self._jacp_torch[:, :, self._joint_dof_ids]
        jacr = self._jacr_torch[:, :, self._joint_dof_ids]
        jac = torch.cat((jacp, jacr), dim=1)
        tau_ns = _dynamically_consistent_nullspace_torque(
            self,
            jac,
            self._joint_dof_ids,
            tau_ns_ref,
            self.cfg.damping_pinv,
        )
        tau = tau_rl + tau_ns

        effort_limits = self._env.sim.model.actuator_ctrlrange[:, self._ctrl_ids, 1]
        tau = torch.clamp(tau, min=-effort_limits, max=effort_limits)
        self._entity.set_joint_effort_target(tau, joint_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        q_ns = getattr(self._env, "_racquet_nullspace_q_ns", None)
        if q_ns is not None and q_ns.shape == self._entity.data.joint_pos.shape:
            self._q_ns[env_ids] = q_ns[env_ids][:, self._target_ids]
        else:
            self._q_ns[env_ids] = self._entity.data.default_joint_pos[env_ids][
                :, self._target_ids
            ]
        self._next_nullspace_resample_step[env_ids] = self._sample_next_nullspace_resample_step(
            env_ids
        )

    def _maybe_resample_nullspace_target(self) -> None:
        samples = getattr(self._env, "_racquet_nullspace_samples", None)
        sample_joint_ids = getattr(self._env, "_racquet_nullspace_sample_joint_ids", None)
        if samples is None or sample_joint_ids is None:
            return
        if samples.numel() == 0 or not torch.equal(sample_joint_ids, self._target_ids):
            return

        current_step = self._env.episode_length_buf
        resample_mask = current_step >= self._next_nullspace_resample_step
        if not torch.any(resample_mask):
            return

        env_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
        sample_ids = torch.randint(
            samples.shape[0],
            (env_ids.numel(),),
            device=self.device,
        )
        self._q_ns[env_ids] = samples[sample_ids]

        interval_steps = self._sample_next_nullspace_resample_step(env_ids)
        self._next_nullspace_resample_step[env_ids] = current_step[env_ids] + interval_steps


@dataclass(kw_only=True)
class InitialFramePositionActionCfg(DifferentialIKActionCfg):
    """End-effector position action anchored to the per-episode initial frame pose."""

    def __post_init__(self) -> None:
        self.use_relative_mode = False
        self.orientation_weight = 1.0

    def build(self, env: ManagerBasedRlEnv) -> "InitialFramePositionAction":
        return InitialFramePositionAction(self, env)


class InitialFramePositionAction(DifferentialIKAction):
    """Maps actions to ``x_ref = x0 + a`` and solves IK to produce joint targets."""

    cfg: InitialFramePositionActionCfg

    def __init__(self, cfg: InitialFramePositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._initial_frame_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._initial_frame_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._initial_frame_quat[:, 0] = 1.0
        self._initial_frame_ready = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        if self._action_dim == 7:
            self._desired_pos[:] = actions[:, :3]
            self._desired_quat[:] = actions[:, 3:7]
            return

        current_pos, current_quat = self._get_frame_pose()
        missing_anchor = ~self._initial_frame_ready
        if torch.any(missing_anchor):
            self._initial_frame_pos[missing_anchor] = current_pos[missing_anchor]
            self._initial_frame_quat[missing_anchor] = current_quat[missing_anchor]
            self._initial_frame_ready[missing_anchor] = True

        self._desired_pos[:] = self._initial_frame_pos + actions * self.cfg.delta_pos_scale
        self._desired_quat[:] = self._initial_frame_quat

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initial_frame_ready[env_ids] = False


@dataclass(kw_only=True)
class NullspaceTorqueActionCfg(DifferentialIKActionCfg):
    """Cartesian action that maps pose commands to task-space torques."""

    damping_pos: float = 0.05
    damping_ori: float = 0.05
    damping_null: float = 0.05
    damping_pinv: float = 0.05
    bias_compensation: bool = False
    orientation_error_in_body_frame: bool = False
    nullspace_resample_interval_s: tuple[float, float] = (0.25, 1.0)

    def __post_init__(self) -> None:
        self.use_relative_mode = True
        self.orientation_weight = 1.0

    def build(self, env: ManagerBasedRlEnv) -> "NullspaceTorqueAction":
        return NullspaceTorqueAction(self, env)


class NullspaceTorqueAction(DifferentialIKAction):
    """Cartesian operational-space controller with a null-space posture bias."""

    cfg: NullspaceTorqueActionCfg

    def __init__(self, cfg: NullspaceTorqueActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._ctrl_ids = self._entity.indexing.ctrl_ids[self._joint_ids]
        self._initial_frame_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._initial_frame_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._initial_frame_quat[:, 0] = 1.0
        self._initial_frame_ready = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        local_body_matches = torch.nonzero(
            self._entity.indexing.body_ids == self._body_id, as_tuple=False
        ).squeeze(-1)
        if local_body_matches.numel() != 1:
            raise RuntimeError("Failed to map Cartesian frame body to entity-local body index.")
        self._local_body_id = int(local_body_matches.item())
        self._q_ns = self._posture_target.clone()
        self._next_nullspace_resample_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    def _sample_next_nullspace_resample_step(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        min_s, max_s = self.cfg.nullspace_resample_interval_s
        min_steps = max(1, int(round(min_s / self._env.step_dt)))
        max_steps = max(min_steps, int(round(max_s / self._env.step_dt)))
        if max_steps == min_steps:
            return torch.full((env_ids.numel(),), min_steps, device=self.device, dtype=torch.long)
        return torch.randint(
            min_steps,
            max_steps + 1,
            (env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._maybe_resample_nullspace_target()
        frame_pos, frame_quat = self._get_frame_pose()
        missing_anchor = ~self._initial_frame_ready
        if torch.any(missing_anchor):
            self._initial_frame_pos[missing_anchor] = frame_pos[missing_anchor]
            self._initial_frame_quat[missing_anchor] = frame_quat[missing_anchor]
            self._initial_frame_ready[missing_anchor] = True

        if self._action_dim == 6:
            delta = actions.clone()
            delta[:, :3] *= self.cfg.delta_pos_scale
            delta[:, 3:] *= self.cfg.delta_ori_scale
            target_pos, target_quat = apply_delta_pose(
                self._initial_frame_pos,
                self._initial_frame_quat,
                delta,
            )
            self._desired_pos[:] = target_pos
            self._desired_quat[:] = target_quat
        else:
            self._desired_pos[:] = actions[:, :3]
            self._desired_quat[:] = actions[:, 3:7]

    def apply_actions(self) -> None:
        robot = self._entity
        frame_pos, frame_quat = self._get_frame_pose()
        pos_error, rot_error = compute_pose_error(
            frame_pos,
            frame_quat,
            self._desired_pos,
            self._desired_quat,
        )

        self._point_torch[:] = frame_pos
        self._compute_jacobian()
        jacp = self._jacp_torch[:, :, self._joint_dof_ids]
        jacr = self._jacr_torch[:, :, self._joint_dof_ids]
        jac = torch.cat((jacp, jacr), dim=1)
        qd = robot.data.joint_vel[:, self._joint_ids]
        frame_lin_vel = torch.einsum("bij,bj->bi", jacp, qd)
        frame_ang_vel = torch.einsum("bij,bj->bi", jacr, qd)
        rot_jac = jacr
        if self.cfg.orientation_error_in_body_frame:
            rot_error = quat_apply_inverse(frame_quat, rot_error)
            frame_ang_vel = quat_apply_inverse(frame_quat, frame_ang_vel)
            rot_jac = torch.stack(
                [
                    quat_apply_inverse(frame_quat, jacr[:, :, joint_idx])
                    for joint_idx in range(jacr.shape[-1])
                ],
                dim=-1,
            )
            jac = torch.cat((jacp, rot_jac), dim=1)

        task_wrench = torch.cat(
            (
                self.cfg.position_weight * pos_error - self.cfg.damping_pos * frame_lin_vel,
                self.cfg.orientation_weight * rot_error - self.cfg.damping_ori * frame_ang_vel,
            ),
            dim=-1,
        )

        q = robot.data.joint_pos[:, self._joint_ids]
        null_ref = self.cfg.posture_weight * (self._q_ns - q) - self.cfg.damping_null * qd

        tau_task = torch.einsum("bij,bj->bi", jac.transpose(1, 2), task_wrench)
        tau_null = _dynamically_consistent_nullspace_torque(
            self,
            jac,
            self._joint_dof_ids,
            null_ref,
            self.cfg.damping_pinv,
        )
        tau = tau_task + tau_null
        if self.cfg.bias_compensation:
            tau = tau + robot.data._joint_dof_field("qfrc_bias")[:, self._joint_ids]

        effort_limits = self._env.sim.model.actuator_ctrlrange[:, self._ctrl_ids, 1]
        tau = torch.clamp(tau, min=-effort_limits, max=effort_limits)
        robot.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initial_frame_ready[env_ids] = False
        q_ns = getattr(self._env, "_racquet_nullspace_q_ns", None)
        if q_ns is not None and q_ns.shape == self._entity.data.joint_pos.shape:
            self._q_ns[env_ids] = q_ns[env_ids][:, self._joint_ids]
        else:
            self._q_ns[env_ids] = self._posture_target[env_ids]
        self._next_nullspace_resample_step[env_ids] = self._sample_next_nullspace_resample_step(
            env_ids
        )

    def _maybe_resample_nullspace_target(self) -> None:
        samples = getattr(self._env, "_racquet_nullspace_samples", None)
        sample_joint_ids = getattr(self._env, "_racquet_nullspace_sample_joint_ids", None)
        if samples is None or sample_joint_ids is None:
            return
        if samples.numel() == 0 or not torch.equal(sample_joint_ids, self._joint_ids):
            return

        current_step = self._env.episode_length_buf
        resample_mask = current_step >= self._next_nullspace_resample_step
        if not torch.any(resample_mask):
            return

        env_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
        sample_ids = torch.randint(
            samples.shape[0],
            (env_ids.numel(),),
            device=self.device,
        )
        self._q_ns[env_ids] = samples[sample_ids]

        interval_steps = self._sample_next_nullspace_resample_step(env_ids)
        self._next_nullspace_resample_step[env_ids] = current_step[env_ids] + interval_steps
