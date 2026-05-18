"""Task-specific action terms for Kinova ball balancing policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.differential_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import compute_pose_error


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

    def __post_init__(self) -> None:
        self.use_relative_mode = False
        self.orientation_weight = 1.0

    def build(self, env: ManagerBasedRlEnv) -> "NullspaceTorqueAction":
        return NullspaceTorqueAction(self, env)


class NullspaceTorqueAction(DifferentialIKAction):
    """Cartesian operational-space controller with a null-space posture bias."""

    cfg: NullspaceTorqueActionCfg

    def __init__(self, cfg: NullspaceTorqueActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._joint_ctrl_ids = self._entity.indexing.ctrl_ids[self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._desired_pos[:] = actions[:, :3]
        self._desired_quat[:] = actions[:, 3:7]

    def apply_actions(self) -> None:
        robot = self._entity
        frame_pos, frame_quat = self._get_frame_pose()
        if hasattr(robot.data, "body_link_lin_vel_w"):
            frame_lin_vel = robot.data.body_link_lin_vel_w[:, self._body_id]
        else:
            frame_lin_vel = robot.data.body_link_vel_w[:, self._body_id, :3]
        if hasattr(robot.data, "body_link_ang_vel_w"):
            frame_ang_vel = robot.data.body_link_ang_vel_w[:, self._body_id]
        else:
            frame_ang_vel = robot.data.body_link_vel_w[:, self._body_id, 3:]

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

        task_wrench = torch.cat(
            (
                self.cfg.position_weight * pos_error - self.cfg.damping * frame_lin_vel,
                self.cfg.orientation_weight * rot_error - self.cfg.damping * frame_ang_vel,
            ),
            dim=-1,
        )

        q = robot.data.joint_pos[:, self._joint_ids]
        qd = robot.data.joint_vel[:, self._joint_ids]
        q_ns_full = getattr(self._env, "_racquet_nullspace_q_ns", None)
        if q_ns_full is None:
            q_ns = self._posture_target
        else:
            q_ns = q_ns_full[:, self._joint_ids]
        null_ref = self.cfg.posture_weight * (q_ns - q) - self.cfg.damping * qd

        jjt = torch.einsum("bij,bkj->bik", jac, jac)
        eye_task = torch.eye(jjt.shape[-1], device=self.device, dtype=jac.dtype).unsqueeze(0)
        j_pinv = torch.matmul(
            jac.transpose(1, 2),
            torch.linalg.inv(jjt + (self.cfg.damping ** 2) * eye_task),
        )
        eye_joint = torch.eye(jac.shape[-1], device=self.device, dtype=jac.dtype).unsqueeze(0)
        null_proj = eye_joint - torch.matmul(j_pinv, jac)

        tau_task = torch.einsum("bij,bj->bi", jac.transpose(1, 2), task_wrench)
        tau_null = torch.einsum("bij,bj->bi", null_proj, null_ref)
        tau = tau_task + tau_null

        stiffness = self._env.sim.model.actuator_gainprm[:, self._joint_ctrl_ids, 0]
        damping = -self._env.sim.model.actuator_biasprm[:, self._joint_ctrl_ids, 2]
        # The Kinova model is still position-actuated, so convert the desired
        # joint torques into equivalent position targets through the built-in PD.
        ctrl = q + (tau + damping * qd) / torch.clamp(stiffness, min=1e-6)
        lower = robot.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        upper = robot.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        ctrl = torch.maximum(torch.minimum(ctrl, upper), lower)
        robot.set_joint_position_target(ctrl, joint_ids=self._joint_ids)
