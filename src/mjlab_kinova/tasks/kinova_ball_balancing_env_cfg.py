"""Kinova ball balancing task configurations for multiple policy variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab_kinova.robot.kinova_constants import KINOVA_CFG

from . import ball_balancing_mdp as bb_mdp
from .policy_actions import InitialFramePositionActionCfg

PolicyVariant = Literal["baseline", "cartesian", "baseline_no_model_rand"]

ROBOT_JOINTS = SceneEntityCfg("robot", joint_names=("joint_.*",))
ROBOT_ACTUATORS = SceneEntityCfg("robot")
RACQUET_FRAME = SceneEntityCfg("robot", body_names=("racquet_frame",))
ROBOT_BODIES = SceneEntityCfg(
    "robot",
    body_names=(
        "shoulder_link",
        "half_arm_1_link",
        "half_arm_2_link",
        "forearm_link",
        "spherical_wrist_1_link",
        "spherical_wrist_2_link",
        "bracelet_link",
        "end_effector_link",
        "FT_adapter",
        "FT_sensor_mounting",
        "FT_sensor_wrench",
        "plate",
    ),
)
BALL_BODY = SceneEntityCfg("ball", body_names=("ball",))


@dataclass(frozen=True)
class PolicySpec:
    variant: PolicyVariant
    experiment_name: str
    action_kind: Literal["joint", "cartesian"]
    actor_terms: tuple[str, ...]
    randomize_robot_model: bool
    randomize_null_space_init: bool


POLICY_SPECS: dict[PolicyVariant, PolicySpec] = {
    "baseline": PolicySpec(
        variant="baseline",
        experiment_name="kinova_ball_balancing_baseline",
        action_kind="joint",
        actor_terms=("joint_pos", "joint_vel", "ee_pos", "ee_vel", "ee_ft_wrench"),
        randomize_robot_model=True,
        randomize_null_space_init=True,
    ),
    "cartesian": PolicySpec(
        variant="cartesian",
        experiment_name="kinova_ball_balancing_cartesian",
        action_kind="cartesian",
        actor_terms=("ee_pos", "ee_vel", "ee_ft_wrench"),
        randomize_robot_model=True,
        randomize_null_space_init=False,
    ),
    "baseline_no_model_rand": PolicySpec(
        variant="baseline_no_model_rand",
        experiment_name="kinova_ball_balancing_baseline_no_model_rand",
        action_kind="joint",
        actor_terms=("joint_pos", "joint_vel", "ee_pos", "ee_vel", "ee_ft_wrench"),
        randomize_robot_model=False,
        randomize_null_space_init=True,
    ),
}


def get_ball_spec(radius: float = 0.0335, mass: float = 0.0657) -> mujoco.MjSpec:
    """Create a simple free ball entity spec."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius,),
        mass=mass,
        friction=(1.0, 0.2, 0.0005),
        condim=6,
        solref=(0.02, 2.5),
        solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
        rgba=(0.9, 0.2, 0.2, 1.0),
    )
    return spec


def kinova_ball_balancing_env_cfg(
    variant: PolicyVariant = "baseline",
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create environment config for one of the RO-MAN 2026 policy variants."""
    spec = POLICY_SPECS[variant]

    observations = {
        "actor": ObservationGroupCfg(
            terms=_actor_observation_terms(spec, play),
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=_critic_observation_terms(),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={
                "robot": KINOVA_CFG,
                "ball": EntityCfg(spec_fn=get_ball_spec),
            },
            num_envs=1,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=_actions_cfg(spec),
        events=_events_cfg(spec, play),
        rewards=_rewards_cfg(),
        terminations=_terminations_cfg(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="plate",
            distance=0.9,
            elevation=-35.0,
            azimuth=110.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=30,
                ls_iterations=30,
                ccd_iterations=80,
            ),
        ),
        decimation=5,
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg


def kinova_ppo_runner_cfg(
    variant: PolicyVariant = "baseline",
) -> RslRlOnPolicyRunnerCfg:
    """Create the PPO runner config for a given policy variant."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
            stochastic=True,
            init_noise_std=1.0,
            noise_std_type="log",
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
            stochastic=False,
            init_noise_std=1.0,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=POLICY_SPECS[variant].experiment_name,
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def _actor_observation_terms(
    spec: PolicySpec,
    play: bool,
) -> dict[str, ObservationTermCfg]:
    actor_terms = _shared_observation_terms(use_noise=not play)
    return {name: actor_terms[name] for name in spec.actor_terms}


def _critic_observation_terms() -> dict[str, ObservationTermCfg]:
    terms = _shared_observation_terms(use_noise=False)
    terms.update(
        {
            "ball_pos_plate": ObservationTermCfg(
                func=bb_mdp.ball_pos_in_plate_frame,
                params={
                    "ball_name": "ball",
                    "plate_asset_cfg": RACQUET_FRAME,
                },
            ),
            "ball_vel_plate": ObservationTermCfg(
                func=bb_mdp.ball_lin_vel_in_plate_frame,
                params={
                    "ball_name": "ball",
                    "plate_asset_cfg": RACQUET_FRAME,
                },
            ),
        }
    )
    return terms


def _shared_observation_terms(use_noise: bool) -> dict[str, ObservationTermCfg]:
    joint_pos_noise = Unoise(n_min=-0.01, n_max=0.01) if use_noise else None
    joint_vel_noise = Unoise(n_min=-0.1, n_max=0.1) if use_noise else None
    ee_pos_noise = Unoise(n_min=-0.003, n_max=0.003) if use_noise else None
    ee_vel_noise = Unoise(n_min=-0.05, n_max=0.05) if use_noise else None
    ft_noise = Unoise(n_min=-0.1, n_max=0.1) if use_noise else None
    return {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINTS},
            noise=joint_pos_noise,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_JOINTS},
            noise=joint_vel_noise,
        ),
        "ee_pos": ObservationTermCfg(
            func=bb_mdp.body_position_w,
            params={"asset_cfg": RACQUET_FRAME},
            noise=ee_pos_noise,
        ),
        "ee_vel": ObservationTermCfg(
            func=bb_mdp.body_linear_velocity_w,
            params={"asset_cfg": RACQUET_FRAME},
            noise=ee_vel_noise,
        ),
        "ee_ft_wrench": ObservationTermCfg(
            func=bb_mdp.ee_ft_wrench,
            noise=ft_noise,
        ),
    }


def _actions_cfg(spec: PolicySpec) -> dict[str, ActionTermCfg]:
    if spec.action_kind == "joint":
        return {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.13,
                use_default_offset=True,
            )
        }

    return {
        "ee_pos": InitialFramePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            frame_type="body",
            frame_name="racquet_frame",
            delta_pos_scale=0.04,
            damping=0.05,
            max_dq=0.2,
            position_weight=1.0,
            orientation_weight=0.0,
            posture_weight=0.03,
            posture_target=KINOVA_CFG.init_state.joint_pos,
        )
    }


def _events_cfg(spec: PolicySpec, play: bool) -> dict[str, EventTermCfg]:
    joint_reset_range = (-0.35, 0.35) if spec.randomize_null_space_init else (0.0, 0.0)
    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": joint_reset_range,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": ROBOT_JOINTS,
            },
        ),
        "reset_ball": EventTermCfg(
            func=bb_mdp.reset_ball_on_plate,
            mode="reset",
            params={
                "ball_name": "ball",
                "plate_asset_cfg": RACQUET_FRAME,
                "xy_range": (-0.02, 0.02),
                "z_offset": 0.05,
                "x_offset": 0.0,
                "y_offset": 0.0,
                "lin_vel_x_range": (-0.25, 0.25),
                "lin_vel_y_range": (-0.25, 0.25),
                "lin_vel_z_range": (-0.05, 0.05),
                "ang_vel_range": (-2.0, 2.0),
            },
        ),
        "randomize_ball_mass": EventTermCfg(
            func=bb_mdp.randomize_body_mass,
            mode="reset",
            params={
                "asset_cfg": BALL_BODY,
                "mass_range": (0.7, 1.3),
                "operation": "scale",
            },
        ),
        "randomize_pd_gains": EventTermCfg(
            func=mdp.randomize_pd_gains,
            mode="reset",
            params={
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
                "asset_cfg": ROBOT_ACTUATORS,
                "operation": "scale",
            },
        ),
    }

    if spec.randomize_robot_model:
        events["randomize_robot_model"] = EventTermCfg(
            func=bb_mdp.randomize_robot_model,
            mode="reset",
            params={
                "body_mass_range": (0.9, 1.1),
                "body_inertia_range": (0.9, 1.1),
                "dof_armature_range": (0.9, 1.1),
                "asset_cfg": ROBOT_BODIES,
            },
        )

    if not play:
        events["ball_velocity_kick"] = EventTermCfg(
            func=bb_mdp.kick_ball_velocity,
            mode="interval",
            interval_range_s=(0.4, 1.0),
            params={
                "ball_name": "ball",
                "lin_vel_x_range": (-0.15, 0.15),
                "lin_vel_y_range": (-0.15, 0.15),
                "lin_vel_z_range": (-0.03, 0.03),
                "ang_vel_range": (-0.5, 0.5),
                "add_to_current": True,
            },
        )

    return events


def _rewards_cfg() -> dict[str, RewardTermCfg]:
    return {
        "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.2),
        "ball_centering": RewardTermCfg(
            func=bb_mdp.ball_centering_reward,
            weight=40.0,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": RACQUET_FRAME,
                "std": 0.06,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ),
        "ball_speed": RewardTermCfg(
            func=bb_mdp.ball_speed_penalty,
            weight=-8.0,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": RACQUET_FRAME,
                "lin_weight": 1.0,
                "ang_weight": 1.0,
            },
        ),
        "ball_no_contact_penalty": RewardTermCfg(
            func=bb_mdp.ball_no_contact_mujoco,
            weight=-18.0,
            params={
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": 0.0,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.0015),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0005,
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-0.0001,
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "joint_torque_l2": RewardTermCfg(
            func=bb_mdp.joint_torque_l2,
            weight=-0.0002,
            params={"robot_name": "robot"},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[246]",))},
        ),
        "racquet_lin_vel_l2": RewardTermCfg(
            func=bb_mdp.racquet_lin_vel_l2,
            weight=-5.0,
            params={"plate_asset_cfg": RACQUET_FRAME},
        ),
        "racquet_dist_from_initial_l2": RewardTermCfg(
            func=bb_mdp.racquet_dist_from_initial_l2,
            weight=-30.0,
            params={"plate_asset_cfg": RACQUET_FRAME},
        ),
    }


def _terminations_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ball_fell_off": TerminationTermCfg(
            func=bb_mdp.ball_fell_off,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": RACQUET_FRAME,
                "max_xy_radius": 0.11,
                "min_height": -0.03,
                "floor_height": 0.05,
            },
        ),
    }
