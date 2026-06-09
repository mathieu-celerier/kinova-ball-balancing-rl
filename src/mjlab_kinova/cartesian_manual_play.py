from __future__ import annotations

import argparse
import html
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import viser
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import (
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)
from mjlab.viewer.viser.viewer import ViserPlayViewer

from mjlab_kinova.tasks.kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
)
from mjlab_kinova.tasks.policy_actions import (
    _rotation_matrix_error,
    _rotation_matrix_from_axis_angle,
)
from mjlab_kinova.tasks.task_parameters import load_default_task_parameters


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANUAL_UI_CONFIG_PATH = _REPO_ROOT / "config" / "cartesian_manual_ui.yaml"


@dataclass
class _SliderGroup:
    action: list[viser.GuiSliderHandle]
    nullspace: list[viser.GuiSliderHandle]
    p0: list[viser.GuiSliderHandle]
    position_weight: viser.GuiSliderHandle
    orientation_weight: viser.GuiSliderHandle
    damping_pos: viser.GuiSliderHandle
    damping_ori: viser.GuiSliderHandle
    damping_null: viser.GuiSliderHandle
    damping_pinv: viser.GuiSliderHandle
    posture_weight: viser.GuiSliderHandle
    orientation_error_in_body_frame: viser.GuiCheckboxHandle
    position_scale: viser.GuiSliderHandle
    orientation_scale: viser.GuiSliderHandle


@dataclass(frozen=True)
class _SliderRange:
    min: float
    max: float
    step: float

    def clamp_initial(self, value: float) -> float:
        return min(max(value, self.min), self.max)


@dataclass(frozen=True)
class _ManualUiConfig:
    action: _SliderRange
    position_weight: _SliderRange
    orientation_weight: _SliderRange
    damping_pos: _SliderRange
    damping_ori: _SliderRange
    damping_null: _SliderRange
    damping_pinv: _SliderRange
    posture_weight: _SliderRange
    position_scale: _SliderRange
    orientation_scale: _SliderRange
    p0_span: float
    p0_step: float
    nullspace_min: float | None
    nullspace_max: float | None


def _load_manual_ui_config(path: Path = _DEFAULT_MANUAL_UI_CONFIG_PATH) -> _ManualUiConfig:
    defaults = {
        "action": {"min": -1.0, "max": 1.0, "step": 0.01},
        "position_weight": {"min": 0.0, "max": 100.0, "step": 0.1},
        "orientation_weight": {"min": 0.0, "max": 100.0, "step": 0.1},
        "damping_pos": {"min": 0.0, "max": 20.0, "step": 0.1},
        "damping_ori": {"min": 0.0, "max": 20.0, "step": 0.1},
        "damping_null": {"min": 0.0, "max": 20.0, "step": 0.1},
        "damping_pinv": {"min": 0.0, "max": 0.05, "step": 0.0001},
        "posture_weight": {"min": 0.0, "max": 50.0, "step": 0.1},
        "position_scale": {"min": 0.0, "max": 0.2, "step": 0.001},
        "orientation_scale": {"min": 0.0, "max": 2.0, "step": 0.01},
        "p0": {"span": 0.5, "step": 0.005},
        "nullspace": {"min": None, "max": None},
    }

    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    else:
        data = {}

    ui = data.get("cartesian_manual_ui", {}) if isinstance(data, dict) else {}
    if not isinstance(ui, dict):
        raise TypeError(f"Top-level YAML object at {path} must map cartesian_manual_ui to a mapping.")
    if "damping_task" in ui:
        ui = dict(ui)
        ui.setdefault("damping_pos", ui["damping_task"])
        ui.setdefault("damping_ori", ui["damping_task"])

    def _range(name: str) -> _SliderRange:
        spec = {**defaults[name], **(ui.get(name, {}) or {})}
        return _SliderRange(
            min=float(spec["min"]),
            max=float(spec["max"]),
            step=float(spec["step"]),
        )

    p0_spec = {**defaults["p0"], **(ui.get("p0", {}) or {})}
    nullspace_spec = {**defaults["nullspace"], **(ui.get("nullspace", {}) or {})}
    return _ManualUiConfig(
        action=_range("action"),
        position_weight=_range("position_weight"),
        orientation_weight=_range("orientation_weight"),
        damping_pos=_range("damping_pos"),
        damping_ori=_range("damping_ori"),
        damping_null=_range("damping_null"),
        damping_pinv=_range("damping_pinv"),
        posture_weight=_range("posture_weight"),
        position_scale=_range("position_scale"),
        orientation_scale=_range("orientation_scale"),
        p0_span=float(p0_spec["span"]),
        p0_step=float(p0_spec["step"]),
        nullspace_min=None if nullspace_spec["min"] is None else float(nullspace_spec["min"]),
        nullspace_max=None if nullspace_spec["max"] is None else float(nullspace_spec["max"]),
    )


class ManualCartesianPolicy:
    """Policy that forwards GUI slider values into the cartesian action term."""

    def __init__(self, *, action_dim: int, device: torch.device):
        self._actions = torch.zeros((1, action_dim), device=device)
        self._action_term = None
        self._env = None
        self._sliders: _SliderGroup | None = None

    def bind(
        self,
        *,
        env: ManagerBasedRlEnv,
        action_term,
        sliders: _SliderGroup,
    ) -> None:
        self._env = env
        self._action_term = action_term
        self._sliders = sliders

        # Disable automatic null-space target resampling for this debug mode.
        action_term.cfg.nullspace_resample_interval_s = (1_000_000.0, 1_000_000.0)
        action_term._next_nullspace_resample_step[:] = torch.iinfo(torch.long).max // 4
        frame_pos, _frame_quat = action_term._get_frame_pose()
        action_term._initial_frame_pos[:] = frame_pos
        action_term._initial_frame_rot[:] = action_term._get_frame_rotation_matrix()
        action_term._initial_frame_ready[:] = True

    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        if self._sliders is None or self._action_term is None or self._env is None:
            return self._actions

        self._action_term.cfg.position_weight = float(self._sliders.position_weight.value)
        self._action_term.cfg.orientation_weight = float(self._sliders.orientation_weight.value)
        self._action_term.cfg.damping_pos = float(self._sliders.damping_pos.value)
        self._action_term.cfg.damping_ori = float(self._sliders.damping_ori.value)
        self._action_term.cfg.damping_null = float(self._sliders.damping_null.value)
        self._action_term.cfg.damping_pinv = float(self._sliders.damping_pinv.value)
        self._action_term.cfg.posture_weight = float(self._sliders.posture_weight.value)
        self._action_term.cfg.orientation_error_in_body_frame = bool(
            self._sliders.orientation_error_in_body_frame.value
        )
        self._action_term.cfg.delta_pos_scale = float(self._sliders.position_scale.value)
        self._action_term.cfg.delta_ori_scale = float(self._sliders.orientation_scale.value)

        p0 = torch.tensor(
            [float(slider.value) for slider in self._sliders.p0],
            device=self._action_term._initial_frame_pos.device,
            dtype=self._action_term._initial_frame_pos.dtype,
        ).unsqueeze(0)
        if not bool(self._action_term._initial_frame_ready[0].item()):
            self._action_term._initial_frame_rot[:1] = (
                self._action_term._get_frame_rotation_matrix()[:1]
            )
        self._action_term._initial_frame_pos[:1] = p0
        self._action_term._initial_frame_ready[:1] = True

        for i, slider in enumerate(self._sliders.action):
            self._actions[0, i] = float(slider.value)

        q_ns = torch.tensor(
            [float(slider.value) for slider in self._sliders.nullspace],
            device=self._action_term._q_ns.device,
            dtype=self._action_term._q_ns.dtype,
        ).unsqueeze(0)
        self._action_term._q_ns[:1] = q_ns

        cache = getattr(self._env, "_racquet_nullspace_q_ns", None)
        if cache is None or cache.shape != self._action_term._q_ns.shape:
            self._env._racquet_nullspace_q_ns = self._action_term._q_ns.clone()
        else:
            self._env._racquet_nullspace_q_ns[:1] = q_ns

        return self._actions

    def reset(self) -> None:
        self._actions.zero_()


class ManualCartesianViewer(ViserPlayViewer):
    _MAX_SUBSTEP_TRACE = 8

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        policy: ManualCartesianPolicy,
        *,
        frame_rate: float = 60.0,
        ui_config: _ManualUiConfig,
        log_control_law: bool = False,
        log_control_law_interval: int = 20,
    ) -> None:
        super().__init__(env, policy, frame_rate=frame_rate)
        self._manual_policy = policy
        self._ui_config = ui_config
        self._log_control_law = log_control_law
        self._log_control_law_interval = log_control_law_interval
        self._control_law_html: viser.GuiHtmlHandle | None = None

    def setup(self) -> None:
        super().setup()

        action_term = self.env.unwrapped.action_manager.get_term("ee_pos")
        if self._log_control_law:
            self._install_control_law_state_tracker(action_term)
        sliders = self._create_manual_controls(action_term)
        self._manual_policy.bind(env=self.env.unwrapped, action_term=action_term, sliders=sliders)
        self.pause()

    def reset_environment(self) -> None:
        super().reset_environment()
        action_term = self.env.unwrapped.action_manager.get_term("ee_pos")
        action_term._manual_viewer_step_state = {}
        action_term._manual_control_law_state = None
        action_term._manual_substep_trace = []
        action_term._manual_substep_trace_active = True

    def _execute_step(self) -> bool:
        try:
            with torch.no_grad():
                obs = self.env.get_observations()
                action_term = self.env.unwrapped.action_manager.get_term("ee_pos")
                actions = self.policy(obs)
                if self._log_control_law:
                    self._capture_viewer_step_state(
                        action_term,
                        phase="pre_step",
                        preview_actions=actions,
                    )
                self.env.step(actions)
                if self._log_control_law:
                    self._capture_viewer_step_state(action_term, phase="post_step")
                self._step_count += 1
                self._stats_steps += 1
                return True
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(f"[ERROR] Exception during step:\n{self._last_error}")
            self.pause()
            return False

    def _install_control_law_state_tracker(self, action_term) -> None:
        if getattr(action_term, "_manual_control_logging_installed", False):
            return

        action_term._manual_substep_trace = []
        action_term._manual_substep_trace_active = True

        original_apply_actions = action_term.apply_actions

        def _tracked_apply_actions() -> None:
            original_apply_actions()
            step = int(self.env.unwrapped._sim_step_counter)
            trace_active = getattr(action_term, "_manual_substep_trace_active", False)
            should_capture_state = step % self._log_control_law_interval == 0
            if not trace_active and not should_capture_state:
                return

            with torch.no_grad():
                robot = action_term._entity
                frame_pos, frame_quat = action_term._get_frame_pose()
                if hasattr(robot.data, "body_link_lin_vel_w"):
                    measured_lin_vel = robot.data.body_link_lin_vel_w[:, action_term._local_body_id]
                else:
                    measured_lin_vel = robot.data.body_link_vel_w[:, action_term._local_body_id, :3]
                if hasattr(robot.data, "body_link_ang_vel_w"):
                    measured_ang_vel = robot.data.body_link_ang_vel_w[:, action_term._local_body_id]
                else:
                    measured_ang_vel = robot.data.body_link_vel_w[:, action_term._local_body_id, 3:]

                pos_error = action_term._desired_pos - frame_pos
                rot_error = _rotation_matrix_error(
                    action_term._get_frame_rotation_matrix(),
                    action_term._desired_rot,
                    body_frame=action_term.cfg.orientation_error_in_body_frame,
                )

                jacp = action_term._jacp_torch[:, :, action_term._joint_dof_ids]
                jacr = action_term._jacr_torch[:, :, action_term._joint_dof_ids]
                jac = torch.cat((jacp, jacr), dim=1)
                qd = robot.data.joint_vel[:, action_term._joint_ids]
                frame_lin_vel = torch.einsum("bij,bj->bi", jacp, qd)
                frame_ang_vel = torch.einsum("bij,bj->bi", jacr, qd)
                rot_jac = jacr
                if action_term.cfg.orientation_error_in_body_frame:
                    world_to_body = action_term._get_frame_rotation_matrix().transpose(
                        1, 2
                    )
                    frame_ang_vel = torch.einsum(
                        "bij,bj->bi", world_to_body, frame_ang_vel
                    )
                    rot_jac = torch.matmul(world_to_body, jacr)
                    jac = torch.cat((jacp, rot_jac), dim=1)
                task_wrench = torch.cat(
                    (
                        action_term.cfg.position_weight * pos_error
                        - action_term.cfg.damping_pos * frame_lin_vel,
                        action_term.cfg.orientation_weight * rot_error
                        - action_term.cfg.damping_ori * frame_ang_vel,
                    ),
                    dim=-1,
                )
                q = robot.data.joint_pos[:, action_term._joint_ids]
                null_ref = (
                    action_term.cfg.posture_weight * (action_term._q_ns - q)
                    - action_term.cfg.damping_null * qd
                )
                jjt = torch.einsum("bij,bkj->bik", jac, jac)
                eye_task = torch.eye(
                    jjt.shape[-1], device=robot.data.joint_pos.device, dtype=jac.dtype
                ).unsqueeze(0)
                j_pinv = torch.matmul(
                    jac.transpose(1, 2),
                    torch.linalg.inv(jjt + (action_term.cfg.damping_pinv**2) * eye_task),
                )
                eye_joint = torch.eye(
                    jac.shape[-1], device=robot.data.joint_pos.device, dtype=jac.dtype
                ).unsqueeze(0)
                null_proj = eye_joint - torch.matmul(j_pinv, jac)
                tau_task = torch.einsum("bij,bj->bi", jac.transpose(1, 2), task_wrench)
                tau_null = torch.einsum("bij,bj->bi", null_proj, null_ref)
                tau = tau_task + tau_null
                effort_limits = action_term._env.sim.model.actuator_ctrlrange[
                    :, action_term._ctrl_ids, 1
                ]
                tau_clipped = torch.clamp(tau, min=-effort_limits, max=effort_limits)
                effort_limits_safe = effort_limits.clamp_min(1.0e-6)
                tau_usage = tau.abs() / effort_limits_safe
                tau_clipped_usage = tau_clipped.abs() / effort_limits_safe
                pos_jac_row_norm = torch.linalg.norm(jacp, dim=-1)
                rot_jac_row_norm = torch.linalg.norm(rot_jac, dim=-1)
                pos_damping_usage_per_unit = torch.amax(
                    action_term.cfg.damping_pos * jacp.abs() / effort_limits_safe.unsqueeze(1),
                    dim=-1,
                )
                rot_damping_usage_per_unit = torch.amax(
                    action_term.cfg.damping_ori * rot_jac.abs() / effort_limits_safe.unsqueeze(1),
                    dim=-1,
                )

                pos_p = action_term.cfg.position_weight * pos_error
                pos_d = -action_term.cfg.damping_pos * frame_lin_vel
                rot_p = action_term.cfg.orientation_weight * rot_error
                rot_d = -action_term.cfg.damping_ori * frame_ang_vel
                task_wrench_p = torch.cat((pos_p, rot_p), dim=-1)
                task_wrench_d = torch.cat((pos_d, rot_d), dim=-1)
                tau_task_p = torch.einsum("bij,bj->bi", jac.transpose(1, 2), task_wrench_p)
                tau_task_d = torch.einsum("bij,bj->bi", jac.transpose(1, 2), task_wrench_d)
                power_task_p = torch.einsum("bi,bi->b", qd, tau_task_p)
                power_task_d = torch.einsum("bi,bi->b", qd, tau_task_d)
                power_task = torch.einsum("bi,bi->b", qd, tau_task)
                power_tau = torch.einsum("bi,bi->b", qd, tau)
                power_clipped = torch.einsum("bi,bi->b", qd, tau_clipped)
                lin_vel_gap = measured_lin_vel - frame_lin_vel
                ang_vel_gap = measured_ang_vel - frame_ang_vel

                if trace_active:
                    substep_trace = getattr(action_term, "_manual_substep_trace", [])
                    substep_trace.append(
                        {
                            "sim_step": step,
                            "substep_index": len(substep_trace),
                            "measured_ang_vel": measured_ang_vel[0].detach().cpu().tolist(),
                            "jacr_qd": frame_ang_vel[0].detach().cpu().tolist(),
                            "ang_vel_gap": ang_vel_gap[0].detach().cpu().tolist(),
                            "rot_err": rot_error[0].detach().cpu().tolist(),
                            "rot_p": rot_p[0].detach().cpu().tolist(),
                            "rot_d": rot_d[0].detach().cpu().tolist(),
                            "tau_task": tau_task[0].detach().cpu().tolist(),
                            "power_task_d": float(power_task_d[0].detach().cpu().item()),
                            "power_task": float(power_task[0].detach().cpu().item()),
                            "power_clipped": float(power_clipped[0].detach().cpu().item()),
                        }
                    )
                    action_term._manual_substep_trace = substep_trace
                    if len(substep_trace) >= self._MAX_SUBSTEP_TRACE:
                        action_term._manual_substep_trace_active = False

                if should_capture_state or trace_active:
                    action_term._manual_control_law_state = {
                        "step": step,
                        "p0": action_term._initial_frame_pos[0].detach().cpu().tolist(),
                        "position_weight": float(action_term.cfg.position_weight),
                        "orientation_weight": float(action_term.cfg.orientation_weight),
                        "damping_pos": float(action_term.cfg.damping_pos),
                        "damping_ori": float(action_term.cfg.damping_ori),
                        "damping_null": float(action_term.cfg.damping_null),
                        "damping_pinv": float(action_term.cfg.damping_pinv),
                        "posture_weight": float(action_term.cfg.posture_weight),
                        "orientation_error_in_body_frame": bool(
                            action_term.cfg.orientation_error_in_body_frame
                        ),
                        "frame_lin_vel": frame_lin_vel[0].detach().cpu().tolist(),
                        "frame_ang_vel": frame_ang_vel[0].detach().cpu().tolist(),
                        "measured_lin_vel": measured_lin_vel[0].detach().cpu().tolist(),
                        "measured_ang_vel": measured_ang_vel[0].detach().cpu().tolist(),
                        "jacr_qd": frame_ang_vel[0].detach().cpu().tolist(),
                        "ang_vel_gap": ang_vel_gap[0].detach().cpu().tolist(),
                        "lin_vel_gap": lin_vel_gap[0].detach().cpu().tolist(),
                        "pos_jac_row_norm": pos_jac_row_norm[0].detach().cpu().tolist(),
                        "rot_jac_row_norm": rot_jac_row_norm[0].detach().cpu().tolist(),
                        "pos_damping_usage_per_unit": (
                            pos_damping_usage_per_unit[0].detach().cpu().tolist()
                        ),
                        "rot_damping_usage_per_unit": (
                            rot_damping_usage_per_unit[0].detach().cpu().tolist()
                        ),
                        "pos_err": pos_error[0].detach().cpu().tolist(),
                        "rot_err": rot_error[0].detach().cpu().tolist(),
                        "pos_p": pos_p[0].detach().cpu().tolist(),
                        "pos_d": pos_d[0].detach().cpu().tolist(),
                        "rot_p": rot_p[0].detach().cpu().tolist(),
                        "rot_d": rot_d[0].detach().cpu().tolist(),
                        "tau_task_p": tau_task_p[0].detach().cpu().tolist(),
                        "tau_task_d": tau_task_d[0].detach().cpu().tolist(),
                        "power_task_p": float(power_task_p[0].detach().cpu().item()),
                        "power_task_d": float(power_task_d[0].detach().cpu().item()),
                        "power_task": float(power_task[0].detach().cpu().item()),
                        "power_tau": float(power_tau[0].detach().cpu().item()),
                        "power_clipped": float(power_clipped[0].detach().cpu().item()),
                        "task_wrench": task_wrench[0].detach().cpu().tolist(),
                        "tau_task": tau_task[0].detach().cpu().tolist(),
                        "tau_null": tau_null[0].detach().cpu().tolist(),
                        "tau": tau[0].detach().cpu().tolist(),
                        "tau_clipped": tau_clipped[0].detach().cpu().tolist(),
                        "tau_usage": tau_usage[0].detach().cpu().tolist(),
                        "tau_clipped_usage": tau_clipped_usage[0].detach().cpu().tolist(),
                        "max_tau_usage": float(tau_usage[0].max().detach().cpu().item()),
                        "delta_pos_scale": float(action_term.cfg.delta_pos_scale),
                        "delta_ori_scale": float(action_term.cfg.delta_ori_scale),
                    }

        action_term.apply_actions = _tracked_apply_actions
        action_term._manual_control_logging_installed = True

    def _capture_viewer_step_state(
        self,
        action_term,
        *,
        phase: str,
        preview_actions: torch.Tensor | None = None,
    ) -> None:
        robot = action_term._entity
        frame_pos, frame_quat = action_term._get_frame_pose()
        if hasattr(robot.data, "body_link_lin_vel_w"):
            frame_lin_vel = robot.data.body_link_lin_vel_w[:, action_term._local_body_id]
        else:
            frame_lin_vel = robot.data.body_link_vel_w[:, action_term._local_body_id, :3]
        if hasattr(robot.data, "body_link_ang_vel_w"):
            frame_ang_vel = robot.data.body_link_ang_vel_w[:, action_term._local_body_id]
        else:
            frame_ang_vel = robot.data.body_link_vel_w[:, action_term._local_body_id, 3:]

        desired_pos = action_term._desired_pos
        desired_rot = action_term._desired_rot
        if preview_actions is not None:
            if action_term._action_dim == 6:
                desired_pos = (
                    action_term._initial_frame_pos
                    + preview_actions[:, :3] * action_term.cfg.delta_pos_scale
                )
                delta_rot = _rotation_matrix_from_axis_angle(
                    preview_actions[:, 3:] * action_term.cfg.delta_ori_scale
                )
                desired_rot = torch.matmul(delta_rot, action_term._initial_frame_rot)
            else:
                raise ValueError("NullspaceTorqueAction only supports relative 6D actions.")

        pos_error = desired_pos - frame_pos
        rot_error = _rotation_matrix_error(
            action_term._get_frame_rotation_matrix(),
            desired_rot,
            body_frame=action_term.cfg.orientation_error_in_body_frame,
        )
        desired_quat = quat_from_matrix(desired_rot)
        quat_error = quat_mul(
            desired_quat,
            quat_conjugate(frame_quat),
        )

        action_term._manual_viewer_step_state = getattr(action_term, "_manual_viewer_step_state", {})
        action_term._manual_viewer_step_state[phase] = {
            "frame_pos": frame_pos[0].detach().cpu().tolist(),
            "frame_quat": frame_quat[0].detach().cpu().tolist(),
            "desired_pos": desired_pos[0].detach().cpu().tolist(),
            "desired_quat": desired_quat[0].detach().cpu().tolist(),
            "quat_error": quat_error[0].detach().cpu().tolist(),
            "frame_lin_vel": frame_lin_vel[0].detach().cpu().tolist(),
            "frame_ang_vel": frame_ang_vel[0].detach().cpu().tolist(),
            "pos_err": pos_error[0].detach().cpu().tolist(),
            "rot_err": rot_error[0].detach().cpu().tolist(),
        }

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._update_control_law_panel()

    def _update_control_law_panel(self) -> None:
        if self._control_law_html is None:
            return

        action_term = self.env.unwrapped.action_manager.get_term("ee_pos")
        state = getattr(action_term, "_manual_control_law_state", None)
        step_state = getattr(action_term, "_manual_viewer_step_state", {})
        substep_trace = getattr(action_term, "_manual_substep_trace", [])
        if not state:
            self._control_law_html.content = (
                "<div style='padding:0.5em;color:#aaa;font-size:0.85em;'>"
                "No control-law sample yet."
                "</div>"
            )
            return

        def _fmt(values: list[float]) -> str:
            return html.escape(", ".join(f"{v:.4f}" for v in values), quote=True)

        def _render_trace() -> str:
            if not substep_trace:
                return "<strong>Substep trace</strong>: unavailable<br/>"
            parts = ["<strong>Substep trace</strong><br/>"]
            for item in substep_trace:
                parts.append(
                    "substep "
                    f"{item['substep_index']} "
                    f"(sim_step {item['sim_step']}): "
                    f"omega=[{_fmt(item['measured_ang_vel'])}] "
                    f"Jr_qd=[{_fmt(item['jacr_qd'])}] "
                    f"gap=[{_fmt(item['ang_vel_gap'])}] "
                    f"rot_err=[{_fmt(item['rot_err'])}] "
                    f"rot_P=[{_fmt(item['rot_p'])}] "
                    f"rot_D=[{_fmt(item['rot_d'])}] "
                    f"P_D={item['power_task_d']:.4f} "
                    f"P_task={item['power_task']:.4f} "
                    f"P_clip={item['power_clipped']:.4f}"
                    "<br/>"
                )
            return "".join(parts)

        def _render_phase(label: str, phase: str) -> str:
            phase_state = step_state.get(phase)
            if not phase_state:
                return f"<strong>{label}</strong>: unavailable<br/>"
            return (
                f"<strong>{label}</strong><br/>"
                f"frame_pos: [{_fmt(phase_state['frame_pos'])}]<br/>"
                f"frame_quat: [{_fmt(phase_state['frame_quat'])}]<br/>"
                f"desired_pos: [{_fmt(phase_state['desired_pos'])}]<br/>"
                f"desired_quat: [{_fmt(phase_state['desired_quat'])}]<br/>"
                f"quat_error: [{_fmt(phase_state['quat_error'])}]<br/>"
                f"frame_lin_vel: [{_fmt(phase_state['frame_lin_vel'])}]<br/>"
                f"frame_ang_vel: [{_fmt(phase_state['frame_ang_vel'])}]<br/>"
                f"pos_err: [{_fmt(phase_state['pos_err'])}]<br/>"
                f"rot_err: [{_fmt(phase_state['rot_err'])}]<br/>"
            )

        self._control_law_html.content = (
            "<div style='font-size:0.85em;line-height:1.35;padding:0.5em 0.75em;"
            "background:#111;border:1px solid #333;border-radius:6px;white-space:pre-wrap;'>"
            f"<strong>Step</strong>: {state['step']}<br/>"
            f"<strong>p0</strong>: [{_fmt(state['p0'])}]<br/>"
            f"<strong>position_weight</strong>: {state['position_weight']:.4f}<br/>"
            f"<strong>orientation_weight</strong>: {state['orientation_weight']:.4f}<br/>"
            f"<strong>damping_pos</strong>: {state['damping_pos']:.4f}<br/>"
            f"<strong>damping_ori</strong>: {state['damping_ori']:.4f}<br/>"
            f"<strong>damping_null</strong>: {state['damping_null']:.4f}<br/>"
            f"<strong>damping_pinv</strong>: {state['damping_pinv']:.4f}<br/>"
            f"<strong>posture_weight</strong>: {state['posture_weight']:.4f}<br/>"
            f"<strong>orientation_error_in_body_frame</strong>: "
            f"{state['orientation_error_in_body_frame']}<br/>"
            f"<strong>delta_pos_scale</strong>: {state['delta_pos_scale']:.4f}<br/>"
            f"<strong>delta_ori_scale</strong>: {state['delta_ori_scale']:.4f}<br/>"
            f"<strong>frame_lin_vel (Jp*qdot)</strong>: [{_fmt(state['frame_lin_vel'])}]<br/>"
            f"<strong>frame_ang_vel (Jr*qdot)</strong>: [{_fmt(state['frame_ang_vel'])}]<br/>"
            f"<strong>measured_lin_vel</strong>: [{_fmt(state['measured_lin_vel'])}]<br/>"
            f"<strong>measured_ang_vel</strong>: [{_fmt(state['measured_ang_vel'])}]<br/>"
            f"<strong>Jr_qd</strong>: [{_fmt(state['jacr_qd'])}]<br/>"
            f"<strong>measured_lin_vel - Jp_qd</strong>: [{_fmt(state['lin_vel_gap'])}]<br/>"
            f"<strong>omega - Jr_qd</strong>: [{_fmt(state['ang_vel_gap'])}]<br/>"
            f"<strong>||Jp row||</strong>: [{_fmt(state['pos_jac_row_norm'])}]<br/>"
            f"<strong>||Jr row||</strong>: [{_fmt(state['rot_jac_row_norm'])}]<br/>"
            f"<strong>damping_pos usage per 1 m/s</strong>: "
            f"[{_fmt(state['pos_damping_usage_per_unit'])}]<br/>"
            f"<strong>damping_ori usage per 1 rad/s</strong>: "
            f"[{_fmt(state['rot_damping_usage_per_unit'])}]<br/>"
            f"<strong>pos_err</strong>: [{_fmt(state['pos_err'])}]<br/>"
            f"<strong>rot_err</strong>: [{_fmt(state['rot_err'])}]<br/>"
            f"<strong>pos_P</strong>: [{_fmt(state['pos_p'])}]<br/>"
            f"<strong>pos_D</strong>: [{_fmt(state['pos_d'])}]<br/>"
            f"<strong>rot_P</strong>: [{_fmt(state['rot_p'])}]<br/>"
            f"<strong>rot_D</strong>: [{_fmt(state['rot_d'])}]<br/>"
            f"<strong>tau_task_P</strong>: [{_fmt(state['tau_task_p'])}]<br/>"
            f"<strong>tau_task_D</strong>: [{_fmt(state['tau_task_d'])}]<br/>"
            f"<strong>power_task_P</strong>: {state['power_task_p']:.4f}<br/>"
            f"<strong>power_task_D</strong>: {state['power_task_d']:.4f}<br/>"
            f"<strong>power_task</strong>: {state['power_task']:.4f}<br/>"
            f"<strong>power_tau</strong>: {state['power_tau']:.4f}<br/>"
            f"<strong>power_clipped</strong>: {state['power_clipped']:.4f}<br/>"
            f"<strong>task_wrench</strong>: [{_fmt(state['task_wrench'])}]<br/>"
            f"<strong>tau_task</strong>: [{_fmt(state['tau_task'])}]<br/>"
            f"<strong>tau_null</strong>: [{_fmt(state['tau_null'])}]<br/>"
            f"<strong>tau</strong>: [{_fmt(state['tau'])}]<br/>"
            f"<strong>tau_clipped</strong>: [{_fmt(state['tau_clipped'])}]<br/>"
            f"<strong>tau_usage</strong>: [{_fmt(state['tau_usage'])}]<br/>"
            f"<strong>tau_clipped_usage</strong>: [{_fmt(state['tau_clipped_usage'])}]<br/>"
            f"<strong>max_tau_usage</strong>: {state['max_tau_usage']:.4f}<br/>"
            "<hr style='border:0;border-top:1px solid #333;margin:0.5em 0;'/>"
            f"{_render_trace()}"
            "<hr style='border:0;border-top:1px solid #333;margin:0.5em 0;'/>"
            f"{_render_phase('Pre-step state', 'pre_step')}"
            "<hr style='border:0;border-top:1px solid #333;margin:0.5em 0;'/>"
            f"{_render_phase('Post-step state', 'post_step')}"
            "</div>"
        )

    def _create_manual_controls(self, action_term) -> _SliderGroup:
        server = self._server
        robot = self.env.unwrapped.scene["robot"]
        joint_ids = action_term._joint_ids.tolist()
        lower = robot.data.soft_joint_pos_limits[0, joint_ids, 0]
        upper = robot.data.soft_joint_pos_limits[0, joint_ids, 1]
        default_q_ns = action_term._posture_target[0].detach().cpu()
        frame_pos, _ = action_term._get_frame_pose()
        default_p0 = frame_pos[0].detach().cpu()

        action_sliders: list[viser.GuiSliderHandle] = []
        nullspace_sliders: list[viser.GuiSliderHandle] = []
        p0_sliders: list[viser.GuiSliderHandle] = []
        action_range = self._ui_config.action
        position_weight_range = self._ui_config.position_weight
        orientation_weight_range = self._ui_config.orientation_weight
        damping_pos_range = self._ui_config.damping_pos
        damping_ori_range = self._ui_config.damping_ori
        damping_null_range = self._ui_config.damping_null
        damping_pinv_range = self._ui_config.damping_pinv
        posture_weight_range = self._ui_config.posture_weight
        position_scale_range = self._ui_config.position_scale
        orientation_scale_range = self._ui_config.orientation_scale

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Manual Cartesian", icon=viser.Icon.SETTINGS):
            if self._log_control_law:
                with server.gui.add_folder("Control Law"):
                    self._control_law_html = server.gui.add_html(
                        "<div style='padding:0.5em;color:#aaa;font-size:0.85em;'>"
                        "No control-law sample yet."
                        "</div>"
                    )
            with server.gui.add_folder("Controller Params"):
                position_weight = server.gui.add_slider(
                    "position_weight",
                    min=position_weight_range.min,
                    max=position_weight_range.max,
                    step=position_weight_range.step,
                    initial_value=position_weight_range.clamp_initial(float(action_term.cfg.position_weight)),
                )
                orientation_weight = server.gui.add_slider(
                    "orientation_weight",
                    min=orientation_weight_range.min,
                    max=orientation_weight_range.max,
                    step=orientation_weight_range.step,
                    initial_value=orientation_weight_range.clamp_initial(float(action_term.cfg.orientation_weight)),
                )
                damping_pos = server.gui.add_slider(
                    "damping_pos",
                    min=damping_pos_range.min,
                    max=damping_pos_range.max,
                    step=damping_pos_range.step,
                    initial_value=damping_pos_range.clamp_initial(float(action_term.cfg.damping_pos)),
                )
                damping_ori = server.gui.add_slider(
                    "damping_ori",
                    min=damping_ori_range.min,
                    max=damping_ori_range.max,
                    step=damping_ori_range.step,
                    initial_value=damping_ori_range.clamp_initial(float(action_term.cfg.damping_ori)),
                )
                damping_null = server.gui.add_slider(
                    "damping_null",
                    min=damping_null_range.min,
                    max=damping_null_range.max,
                    step=damping_null_range.step,
                    initial_value=damping_null_range.clamp_initial(float(action_term.cfg.damping_null)),
                )
                damping_pinv = server.gui.add_slider(
                    "damping_pinv",
                    min=damping_pinv_range.min,
                    max=damping_pinv_range.max,
                    step=damping_pinv_range.step,
                    initial_value=damping_pinv_range.clamp_initial(float(action_term.cfg.damping_pinv)),
                )
                posture_weight = server.gui.add_slider(
                    "posture_weight",
                    min=posture_weight_range.min,
                    max=posture_weight_range.max,
                    step=posture_weight_range.step,
                    initial_value=posture_weight_range.clamp_initial(float(action_term.cfg.posture_weight)),
                )
                orientation_error_in_body_frame = server.gui.add_checkbox(
                    "orientation_error_in_body_frame",
                    initial_value=bool(action_term.cfg.orientation_error_in_body_frame),
                )
                position_scale = server.gui.add_slider(
                    "delta_pos_scale",
                    min=position_scale_range.min,
                    max=position_scale_range.max,
                    step=position_scale_range.step,
                    initial_value=position_scale_range.clamp_initial(float(action_term.cfg.delta_pos_scale)),
                )
                orientation_scale = server.gui.add_slider(
                    "delta_ori_scale",
                    min=orientation_scale_range.min,
                    max=orientation_scale_range.max,
                    step=orientation_scale_range.step,
                    initial_value=orientation_scale_range.clamp_initial(float(action_term.cfg.delta_ori_scale)),
                )

                for axis, value in zip("xyz", default_p0.tolist(), strict=True):
                    p0_sliders.append(
                        server.gui.add_slider(
                            f"p0_{axis}",
                            min=float(value - self._ui_config.p0_span),
                            max=float(value + self._ui_config.p0_span),
                            step=self._ui_config.p0_step,
                            initial_value=float(value),
                        )
                    )

                capture_p0 = server.gui.add_button("Capture current p0")

                @capture_p0.on_click
                def _(_) -> None:
                    current_p0, _ = action_term._get_frame_pose()
                    current_p0 = current_p0[0].detach().cpu()
                    for slider, value in zip(p0_sliders, current_p0.tolist(), strict=True):
                        slider.value = float(value)

            with server.gui.add_folder("Action"):
                for label in ("a_pos_x", "a_pos_y", "a_pos_z", "a_rot_x", "a_rot_y", "a_rot_z"):
                    action_sliders.append(
                        server.gui.add_slider(
                            label,
                            min=action_range.min,
                            max=action_range.max,
                            step=action_range.step,
                            initial_value=0.0,
                        )
                    )

                zero_action = server.gui.add_button("Zero action", icon=viser.Icon.SQUARE_X)

                @zero_action.on_click
                def _(_) -> None:
                    for slider in action_sliders:
                        slider.value = 0.0

            with server.gui.add_folder("Null-space target"):
                for i, joint_name in enumerate(robot.joint_names[j] for j in joint_ids):
                    if self._ui_config.nullspace_min is None or self._ui_config.nullspace_max is None:
                        lo = float(lower[i].item())
                        hi = float(upper[i].item())
                        if not math.isfinite(lo):
                            lo = float(default_q_ns[i].item() - torch.pi)
                        if not math.isfinite(hi):
                            hi = float(default_q_ns[i].item() + torch.pi)
                    else:
                        lo = self._ui_config.nullspace_min
                        hi = self._ui_config.nullspace_max
                    step = max((hi - lo) / 200.0, 0.001)
                    nullspace_sliders.append(
                        server.gui.add_slider(
                            joint_name,
                            min=lo,
                            max=hi,
                            step=step,
                            initial_value=float(default_q_ns[i].item()),
                        )
                    )

                copy_current = server.gui.add_button("Copy current joints")

                @copy_current.on_click
                def _(_) -> None:
                    current_q = robot.data.joint_pos[0, joint_ids].detach().cpu()
                    for slider, value in zip(nullspace_sliders, current_q.tolist(), strict=True):
                        slider.value = float(value)

                reset_target = server.gui.add_button(
                    "Restore default target"
                )

                @reset_target.on_click
                def _(_) -> None:
                    for slider, value in zip(nullspace_sliders, default_q_ns.tolist(), strict=True):
                        slider.value = float(value)

        return _SliderGroup(
            action=action_sliders,
            nullspace=nullspace_sliders,
            p0=p0_sliders,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            damping_pos=damping_pos,
            damping_ori=damping_ori,
            damping_null=damping_null,
            damping_pinv=damping_pinv,
            posture_weight=posture_weight,
            orientation_error_in_body_frame=orientation_error_in_body_frame,
            position_scale=position_scale,
            orientation_scale=orientation_scale,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a manual cartesian control debug view with sliders."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on. Defaults to cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable termination conditions so you can inspect the control law longer.",
    )
    parser.add_argument(
        "--log-control-law",
        action="store_true",
        help="Print the key cartesian control-law terms every few physics steps.",
    )
    parser.add_argument(
        "--log-control-law-interval",
        type=int,
        default=20,
        help="Log every N physics steps when --log-control-law is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    params = load_default_task_parameters()
    env_cfg = kinova_ball_balancing_env_cfg(variant="cartesian", play=True, params=params)
    if args.no_terminations:
        env_cfg.terminations = {}
    ui_config = _load_manual_ui_config()

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env)

    policy = ManualCartesianPolicy(
        action_dim=base_env.action_manager.total_action_dim,
        device=torch.device(device),
    )
    viewer = ManualCartesianViewer(
        env,
        policy,
        ui_config=ui_config,
        log_control_law=args.log_control_law,
        log_control_law_interval=max(1, args.log_control_law_interval),
    )
    try:
        viewer.run()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
