"""Task-specific action terms for Kinova ball balancing policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.differential_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class InitialFramePositionActionCfg(DifferentialIKActionCfg):
    """End-effector position action anchored to the per-episode initial frame pose."""

    def __post_init__(self) -> None:
        self.use_relative_mode = False

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
