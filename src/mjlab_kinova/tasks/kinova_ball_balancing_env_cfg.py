"""Kinova ball balancing task configurations for control-space variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab_kinova.robot.kinova_constants import (
    KINOVA_ACTION_SCALE,
    KINOVA_CFG,
    KINOVA_EFFORT_CFG,
    KINOVA_IDEAL_PD_CFG,
)

from . import ball_balancing_mdp as bb_mdp
from .policy_actions import NullspaceTorqueActionCfg
from .task_parameters import DEFAULT_TASK_PARAMETERS, TaskParameters

PolicyVariant = Literal["joint", "cartesian"]


def robot_joints_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=("joint_.*",))


def robot_actuators_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot")


def racquet_frame_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("racquet_frame",))


def racquet_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("plate",))


def robot_bodies_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
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


def ball_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("ball", body_names=("ball",))


def _robot_cfg_for_variant(variant: PolicyVariant, params: TaskParameters) -> EntityCfg:
    if variant == "cartesian":
        return KINOVA_EFFORT_CFG
    return KINOVA_IDEAL_PD_CFG


@dataclass(frozen=True)
class PolicySpec:
    variant: PolicyVariant
    experiment_name: str
    action_kind: Literal["joint", "cartesian"]
    actor_terms: tuple[str, ...]


@dataclass(frozen=True)
class TrainingBehavior:
    use_observation_noise: bool
    use_joint_pos_observation: bool
    randomize_ball_reset: bool
    randomize_ball_properties: bool
    randomize_pd_gains: bool
    randomize_racquet_model: bool
    randomize_robot_model: bool
    randomize_null_space_init: bool
    use_ball_kick: bool


POLICY_SPECS: dict[PolicyVariant, PolicySpec] = {
    "joint": PolicySpec(
        variant="joint",
        experiment_name="kinova_ball_balancing_joint",
        action_kind="joint",
        actor_terms=(
            "joint_pos",
            "joint_vel",
            "ee_pos",
            "ee_quat",
            "ee_vel",
            "ee_ang_vel",
            "ee_ft_wrench",
            "actions",
        ),
    ),
    "cartesian": PolicySpec(
        variant="cartesian",
        experiment_name="kinova_ball_balancing_cartesian",
        action_kind="cartesian",
        actor_terms=(
            "ee_pos",
            "ee_quat",
            "ee_vel",
            "ee_ang_vel",
            "ee_ft_wrench",
            "actions",
        ),
    ),
}

DEFAULT_TRAINING_BEHAVIOR: dict[PolicyVariant, TrainingBehavior] = {
    "joint": TrainingBehavior(
        use_observation_noise=True,
        use_joint_pos_observation=True,
        randomize_ball_reset=True,
        randomize_ball_properties=True,
        randomize_pd_gains=True,
        randomize_racquet_model=False,
        randomize_robot_model=True,
        randomize_null_space_init=True,
        use_ball_kick=True,
    ),
    "cartesian": TrainingBehavior(
        use_observation_noise=True,
        use_joint_pos_observation=False,
        randomize_ball_reset=True,
        randomize_ball_properties=True,
        randomize_pd_gains=True,
        randomize_racquet_model=False,
        randomize_robot_model=True,
        randomize_null_space_init=False,
        use_ball_kick=True,
    ),
}


def _resolve_training_behavior(
    variant: PolicyVariant, params: TaskParameters
) -> TrainingBehavior:
    defaults = DEFAULT_TRAINING_BEHAVIOR[variant]
    training = params.training
    return TrainingBehavior(
        use_observation_noise=(
            defaults.use_observation_noise
            if training.use_observation_noise is None
            else training.use_observation_noise
        ),
        use_joint_pos_observation=(
            defaults.use_joint_pos_observation
            if training.use_joint_pos_observation is None
            else training.use_joint_pos_observation
        ),
        randomize_ball_reset=(
            defaults.randomize_ball_reset
            if training.randomize_ball_reset is None
            else training.randomize_ball_reset
        ),
        randomize_ball_properties=(
            defaults.randomize_ball_properties
            if training.randomize_ball_properties is None
            else training.randomize_ball_properties
        ),
        randomize_pd_gains=(
            defaults.randomize_pd_gains
            if training.randomize_pd_gains is None
            else training.randomize_pd_gains
        ),
        randomize_racquet_model=(
            defaults.randomize_racquet_model
            if training.randomize_racquet_model is None
            else training.randomize_racquet_model
        ),
        randomize_robot_model=(
            defaults.randomize_robot_model
            if training.randomize_robot_model is None
            else training.randomize_robot_model
        ),
        randomize_null_space_init=(
            defaults.randomize_null_space_init
            if training.randomize_null_space_init is None
            else training.randomize_null_space_init
        ),
        use_ball_kick=(
            defaults.use_ball_kick
            if training.use_ball_kick is None
            else training.use_ball_kick
        ),
    )


def get_ball_spec(
    radius: float = 0.0335,
    mass: float = 0.0657,
    friction: tuple[float, float, float] = (1.0, 0.2, 0.0005),
    condim: int = 6,
    solref: tuple[float, float] = (0.02, 2.5),
    solimp: tuple[float, float, float, float, float] = (0.95, 0.995, 0.001, 0.5, 2.0),
    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
    """Create a simple free ball entity spec."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius,),
        mass=mass,
        friction=friction,
        condim=condim,
        solref=solref,
        solimp=solimp,
        rgba=rgba,
    )
    return spec


def _ball_entity_cfg(params: TaskParameters) -> EntityCfg:
    ball = params.ball
    return EntityCfg(
        spec_fn=lambda: get_ball_spec(
            radius=ball.radius,
            mass=ball.mass,
            friction=ball.friction,
            condim=ball.condim,
            solref=ball.solref,
            solimp=ball.solimp,
            rgba=ball.rgba,
        )
    )


def kinova_ball_balancing_env_cfg(
    variant: PolicyVariant = "joint",
    play: bool = False,
    params: TaskParameters | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create environment config for one of the Kinova control-space variants."""
    params = DEFAULT_TASK_PARAMETERS if params is None else params
    spec = POLICY_SPECS[variant]
    behavior = _resolve_training_behavior(variant, params)

    observations = {
        "actor": ObservationGroupCfg(
            terms=_actor_observation_terms(spec, behavior, play, params),
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=_critic_observation_terms(params),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": _robot_cfg_for_variant(variant, params),
                "ball": _ball_entity_cfg(params),
            },
            num_envs=params.simulation.num_envs,
            env_spacing=params.simulation.env_spacing,
        ),
        observations=observations,
        actions=_actions_cfg(spec, params),
        events=_events_cfg(spec, behavior, play, params),
        curriculum=_curriculum_cfg(spec, behavior, play, params),
        rewards=_rewards_cfg(params),
        terminations=_terminations_cfg(params),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="plate",
            distance=params.simulation.viewer_distance,
            elevation=params.simulation.viewer_elevation,
            azimuth=params.simulation.viewer_azimuth,
        ),
        sim=SimulationCfg(
            nconmax=params.simulation.nconmax,
            njmax=params.simulation.njmax,
            mujoco=MujocoCfg(
                timestep=params.simulation.timestep,
                iterations=params.simulation.iterations,
                ls_iterations=params.simulation.ls_iterations,
                ccd_iterations=params.simulation.ccd_iterations,
            ),
        ),
        decimation=params.simulation.decimation,
        episode_length_s=params.simulation.episode_length_s,
    )

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = params.simulation.play_episode_length_s
        cfg.observations["actor"].enable_corruption = False

    cfg._kinova_upload_videos_to_wandb = params.training.upload_videos_to_wandb
    return cfg


def kinova_ppo_runner_cfg(
    variant: PolicyVariant = "joint",
    params: TaskParameters | None = None,
) -> RslRlOnPolicyRunnerCfg:
    """Create the PPO runner config for a given policy variant."""
    params = DEFAULT_TASK_PARAMETERS if params is None else params
    ppo = params.ppo
    experiment_name = POLICY_SPECS[variant].experiment_name
    if params.training.experiment_name_suffix:
        experiment_name = f"{experiment_name}_{params.training.experiment_name_suffix}"
    run_name = params.training.run_name or ""
    wandb_project = params.training.wandb_project or "mjlab"
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=ppo.actor_hidden_dims,
            activation=ppo.activation,
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": ppo.init_noise_std,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=ppo.critic_hidden_dims,
            activation=ppo.activation,
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=ppo.value_loss_coef,
            use_clipped_value_loss=True,
            clip_param=ppo.clip_param,
            entropy_coef=ppo.entropy_coef,
            num_learning_epochs=ppo.num_learning_epochs,
            num_mini_batches=ppo.num_mini_batches,
            learning_rate=ppo.learning_rate,
            schedule=ppo.schedule,
            gamma=ppo.gamma,
            lam=ppo.lam,
            desired_kl=ppo.desired_kl,
            max_grad_norm=ppo.max_grad_norm,
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        logger="wandb",
        wandb_project=wandb_project,
        save_interval=ppo.save_interval,
        num_steps_per_env=ppo.num_steps_per_env,
        max_iterations=ppo.max_iterations,
    )


def _actor_observation_terms(
    spec: PolicySpec, behavior: TrainingBehavior, play: bool, params: TaskParameters
) -> dict[str, ObservationTermCfg]:
    actor_terms = _shared_observation_terms(
        use_noise=behavior.use_observation_noise and not play, params=params
    )
    if spec.variant == "cartesian":
        actor_terms = {
            **actor_terms,
            "ee_pos": actor_terms["ee_pos_rel"],
            "ee_quat": actor_terms["ee_quat_rel"],
        }
    selected_term_names = tuple(
        name
        for name in spec.actor_terms
        if name != "joint_pos" or behavior.use_joint_pos_observation
    )
    selected_terms = {name: actor_terms[name] for name in selected_term_names}
    return _with_observation_history(
        selected_terms,
        history_length=params.observation_history_length,
    )


def _critic_observation_terms(params: TaskParameters) -> dict[str, ObservationTermCfg]:
    terms = _shared_observation_terms(use_noise=False, params=params)
    terms.pop("ee_pos_rel")
    terms.pop("ee_quat_rel")
    terms.update(
        {
            "actions": ObservationTermCfg(func=mdp.last_action),
            "joint_torque": ObservationTermCfg(
                func=bb_mdp.joint_torques,
                params={"robot_name": "robot"},
            ),
            "joint_torque_rate": ObservationTermCfg(
                func=bb_mdp.joint_torque_rate,
                params={"robot_name": "robot"},
            ),
            "joint_acc": ObservationTermCfg(
                func=bb_mdp.joint_accelerations,
                params={"asset_cfg": robot_joints_cfg()},
            ),
            "ball_pos_plate": ObservationTermCfg(
                func=bb_mdp.ball_pos_in_plate_frame,
                params={
                    "ball_name": "ball",
                    "plate_asset_cfg": racquet_frame_cfg(),
                },
            ),
            "ball_vel_plate": ObservationTermCfg(
                func=bb_mdp.ball_lin_vel_in_plate_frame,
                params={
                    "ball_name": "ball",
                    "plate_asset_cfg": racquet_frame_cfg(),
                },
            ),
            "ball_contact_state": ObservationTermCfg(
                func=bb_mdp.ball_contact_state_mujoco,
            ),
        }
    )
    return _with_observation_history(
        terms, history_length=params.observation_history_length
    )


def _with_observation_history(
    observation_terms: dict[str, ObservationTermCfg],
    history_length: int,
) -> dict[str, ObservationTermCfg]:
    if history_length <= 1:
        return observation_terms

    wrapped_terms: dict[str, ObservationTermCfg] = {}
    for name, term_cfg in observation_terms.items():
        wrapped_terms[name] = ObservationTermCfg(
            func=term_cfg.func,
            params=term_cfg.params,
            noise=term_cfg.noise,
            clip=term_cfg.clip,
            scale=term_cfg.scale,
            history_length=history_length,
            flatten_history_dim=True,
        )
    return wrapped_terms


def _noise_cfg(use_noise: bool, noise_range) -> Unoise | None:
    if not use_noise:
        return None
    return Unoise(n_min=noise_range.min, n_max=noise_range.max)


def _shared_observation_terms(
    use_noise: bool, params: TaskParameters
) -> dict[str, ObservationTermCfg]:
    noise = params.observation_noise
    joint_pos_noise = _noise_cfg(use_noise, noise.joint_pos)
    joint_vel_noise = _noise_cfg(use_noise, noise.joint_vel)
    ee_pos_noise = _noise_cfg(use_noise, noise.ee_pos)
    ee_quat_noise = _noise_cfg(use_noise, noise.ee_quat)
    ee_vel_noise = _noise_cfg(use_noise, noise.ee_vel)
    ee_ang_vel_noise = _noise_cfg(use_noise, noise.ee_ang_vel)
    ft_noise = _noise_cfg(use_noise, noise.ee_ft_wrench)
    return {
        "joint_pos": ObservationTermCfg(
            func=bb_mdp.joint_pos_rel,
            params={"asset_cfg": robot_joints_cfg()},
            noise=joint_pos_noise,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": robot_joints_cfg()},
            noise=joint_vel_noise,
        ),
        "ee_pos": ObservationTermCfg(
            func=bb_mdp.body_position_w,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_pos_noise,
        ),
        "ee_quat": ObservationTermCfg(
            func=bb_mdp.body_orientation_w,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_quat_noise,
        ),
        "ee_vel": ObservationTermCfg(
            func=bb_mdp.body_linear_velocity_w,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_vel_noise,
        ),
        "ee_ang_vel": ObservationTermCfg(
            func=bb_mdp.body_angular_velocity_w,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_ang_vel_noise,
        ),
        "ee_ft_wrench": ObservationTermCfg(
            func=bb_mdp.ee_ft_wrench,
            noise=ft_noise,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "ee_pos_rel": ObservationTermCfg(
            func=bb_mdp.body_position_rel_nominal,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_pos_noise,
        ),
        "ee_quat_rel": ObservationTermCfg(
            func=bb_mdp.body_orientation_rel_nominal,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_quat_noise,
        ),
    }


def _actions_cfg(spec: PolicySpec, params: TaskParameters) -> dict[str, ActionTermCfg]:
    if spec.action_kind == "joint":
        action = params.joint_action
        return {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=KINOVA_ACTION_SCALE,
                use_default_offset=action.use_default_offset,
            )
        }

    action = params.cartesian_action
    return {
        "ee_pos": NullspaceTorqueActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            frame_type="body",
            frame_name="racquet_frame",
            delta_pos_scale=action.delta_pos_scale,
            delta_ori_scale=action.delta_ori_scale,
            damping_pos=action.damping_pos,
            damping_ori=action.damping_ori,
            damping_null=action.damping_null,
            damping_pinv=action.damping_pinv,
            position_weight=action.position_weight,
            orientation_weight=action.orientation_weight,
            posture_weight=action.posture_weight,
            orientation_error_in_body_frame=action.orientation_error_in_body_frame,
            posture_target=KINOVA_CFG.init_state.joint_pos,
            nullspace_resample_interval_s=action.nullspace_resample_interval_s,
        )
    }


def _events_cfg(
    spec: PolicySpec, behavior: TrainingBehavior, play: bool, params: TaskParameters
) -> dict[str, EventTermCfg]:
    randomization = params.randomization
    ball_reset = params.ball_reset
    joint_reset_range = (
        randomization.null_space_joint_offset
        if behavior.randomize_null_space_init
        else (0.0, 0.0)
    )
    ball_xy_range = ball_reset.xy_range if behavior.randomize_ball_reset else (0.0, 0.0)
    ball_lin_vel_x_range = (
        ball_reset.linear_velocity.x if behavior.randomize_ball_reset else (0.0, 0.0)
    )
    ball_lin_vel_y_range = (
        ball_reset.linear_velocity.y if behavior.randomize_ball_reset else (0.0, 0.0)
    )
    ball_lin_vel_z_range = (
        ball_reset.linear_velocity.z if behavior.randomize_ball_reset else (0.0, 0.0)
    )
    ball_ang_vel_range = (
        ball_reset.angular_velocity if behavior.randomize_ball_reset else (0.0, 0.0)
    )
    events = {
        "reset_robot_joints": EventTermCfg(
            func=bb_mdp.reset_joints_preserving_racquet_pose,
            mode="reset",
            params={
                "position_range": joint_reset_range,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": robot_joints_cfg(),
                "plate_asset_cfg": racquet_frame_cfg(),
            },
        ),
        "reset_ball": EventTermCfg(
            func=bb_mdp.reset_ball_on_plate,
            mode="reset",
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "xy_range": ball_xy_range,
                "z_offset": ball_reset.z_offset,
                "x_offset": ball_reset.x_offset,
                "y_offset": ball_reset.y_offset,
                "release_delay_s": ball_reset.release_delay_s,
                "lin_vel_x_range": ball_lin_vel_x_range,
                "lin_vel_y_range": ball_lin_vel_y_range,
                "lin_vel_z_range": ball_lin_vel_z_range,
                "ang_vel_range": ball_ang_vel_range,
            },
        ),
        "update_ball_release": EventTermCfg(
            func=bb_mdp.update_ball_release,
            mode="step",
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
            },
        ),
        "log_first_step_after_reset": EventTermCfg(
            func=bb_mdp.log_first_step_after_reset,
            mode="step",
        ),
    }

    events["reset_joint_torque_rate_state"] = EventTermCfg(
        func=bb_mdp.reset_joint_torque_rate_state,
        mode="reset",
    )

    if behavior.randomize_ball_properties:
        events["randomize_ball_mass"] = EventTermCfg(
            func=bb_mdp.randomize_body_mass,
            mode="reset",
            params={
                "asset_cfg": ball_body_cfg(),
                "mass_range": randomization.ball_mass_scale,
                "operation": "scale",
            },
        )
        events["randomize_ball_friction"] = EventTermCfg(
            func=bb_mdp.randomize_ball_friction,
            mode="reset",
            params={
                "ball_name": "ball",
                "geom_name": "ball_geom",
                "friction_scale": randomization.ball_friction_scale,
                "operation": "scale",
            },
        )

    if behavior.randomize_pd_gains:
        if spec.action_kind == "joint":
            events["randomize_pd_gains"] = EventTermCfg(
                func=mdp.dr.pd_gains,
                mode="reset",
                params={
                    "kp_range": randomization.pd_gain_scale,
                    "kd_range": randomization.pd_gain_scale,
                    "asset_cfg": robot_actuators_cfg(),
                    "operation": "scale",
                },
            )
        else:
            events["randomize_pd_gains"] = EventTermCfg(
                func=bb_mdp.randomize_osc_pd_gains,
                mode="reset",
                params={
                    "kp_range": randomization.pd_gain_scale,
                    "kd_range": randomization.pd_gain_scale,
                    "action_name": "ee_pos",
                },
            )

    if behavior.randomize_racquet_model:
        events["randomize_racquet_model"] = EventTermCfg(
            func=bb_mdp.randomize_racquet_model,
            mode="reset",
            params={
                "body_mass_range": randomization.racquet_body_mass_scale,
                "body_inertia_range": randomization.racquet_body_inertia_scale,
                "asset_cfg": racquet_body_cfg(),
            },
        )

    if behavior.randomize_robot_model:
        events["randomize_robot_model"] = EventTermCfg(
            func=bb_mdp.randomize_robot_model,
            mode="reset",
            params={
                "body_mass_range": randomization.robot_body_mass_scale,
                "body_inertia_range": randomization.robot_body_inertia_scale,
                "dof_armature_range": randomization.robot_dof_armature_scale,
                "asset_cfg": robot_bodies_cfg(),
            },
        )

    if behavior.use_ball_kick and (
        not play or params.training.enable_ball_kick_in_play
    ):
        kick = params.ball_kick
        events["ball_velocity_kick"] = EventTermCfg(
            func=bb_mdp.kick_ball_velocity,
            mode="interval",
            interval_range_s=kick.interval_s,
            params={
                "ball_name": "ball",
                "lin_vel_x_range": kick.linear_velocity.x,
                "lin_vel_y_range": kick.linear_velocity.y,
                "lin_vel_z_range": kick.linear_velocity.z,
                "ang_vel_range": kick.angular_velocity,
                "add_to_current": kick.add_to_current,
            },
        )

    return events


def _curriculum_cfg(
    spec: PolicySpec,
    behavior: TrainingBehavior,
    play: bool,
    params: TaskParameters,
) -> dict[str, CurriculumTermCfg]:
    if play or spec.action_kind != "joint" or not behavior.randomize_null_space_init:
        return {}
    randomization = params.randomization
    return {
        "null_space_reset": CurriculumTermCfg(
            func=bb_mdp.null_space_reset_curriculum,
            params={
                "final_position_range": randomization.null_space_joint_offset,
                "initial_scale": randomization.null_space_curriculum_initial_scale,
                "start_steps": randomization.null_space_curriculum_start_steps,
                "duration_steps": randomization.null_space_curriculum_steps,
            },
        )
    }


def _rewards_cfg(params: TaskParameters) -> dict[str, RewardTermCfg]:
    rewards = params.rewards
    return {
        "is_alive": RewardTermCfg(func=mdp.is_alive, weight=rewards.is_alive),
        # Keep the main task-relevant terms first so replay viewers surface them first.
        "ball_centering": RewardTermCfg(
            func=bb_mdp.ball_centering_reward,
            weight=rewards.ball_centering,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "std": rewards.ball_centering_std,
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "ball_lin_vel_l2": RewardTermCfg(
            func=bb_mdp.ball_lin_vel_l2,
            weight=rewards.ball_lin_vel_l2,
            params={"ball_name": "ball"},
        ),
        "ball_lin_vel_plate_l2": RewardTermCfg(
            func=bb_mdp.ball_lin_vel_in_plate_frame_l2,
            weight=rewards.ball_lin_vel_plate_l2,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
            },
        ),
        "ball_no_contact_penalty": RewardTermCfg(
            func=bb_mdp.ball_no_contact_mujoco,
            weight=rewards.ball_no_contact,
            params={
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "plate_drop_under_ball": RewardTermCfg(
            func=bb_mdp.plate_drop_under_ball_penalty,
            weight=rewards.plate_drop_under_ball,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "ball_height_threshold": rewards.plate_drop_ball_height_threshold,
                "xy_radius": rewards.plate_drop_xy_radius,
            },
        ),
        "racquet_centering": RewardTermCfg(
            func=bb_mdp.racquet_centering_reward,
            weight=rewards.racquet_centering,
            params={
                "plate_asset_cfg": racquet_frame_cfg(),
                "std": rewards.racquet_centering_std,
            },
        ),
        "racquet_orientation_centering": RewardTermCfg(
            func=bb_mdp.racquet_orientation_centering_reward,
            weight=rewards.racquet_orientation_centering,
            params={
                "plate_asset_cfg": racquet_frame_cfg(),
                "std": rewards.racquet_orientation_centering_std,
            },
        ),
        "racquet_lin_vel_l2": RewardTermCfg(
            func=bb_mdp.racquet_lin_vel_l2,
            weight=rewards.racquet_lin_vel_l2,
            params={
                "plate_asset_cfg": racquet_frame_cfg(),
            },
        ),
        "racquet_ang_vel_l2": RewardTermCfg(
            func=bb_mdp.racquet_ang_vel_l2,
            weight=rewards.racquet_ang_vel_l2,
            params={
                "plate_asset_cfg": racquet_frame_cfg(),
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=rewards.action_rate_l2,
        ),
        "action_acc_l2": RewardTermCfg(
            func=mdp.action_acc_l2,
            weight=rewards.action_acc_l2,
        ),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=rewards.joint_vel_l2,
            params={"asset_cfg": robot_joints_cfg()},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=rewards.joint_acc_l2,
            params={"asset_cfg": robot_joints_cfg()},
        ),
        "joint_torque_l2": RewardTermCfg(
            func=bb_mdp.joint_torque_l2,
            weight=rewards.joint_torque_l2,
            params={"robot_name": "robot"},
        ),
        "joint_torque_rate_l2": RewardTermCfg(
            func=bb_mdp.joint_torque_rate_l2,
            weight=rewards.joint_torque_rate_l2,
            params={"robot_name": "robot"},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=rewards.joint_pos_limits,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[246]",))},
        ),
    }


def _terminations_cfg(params: TaskParameters) -> dict[str, TerminationTermCfg]:
    terminations = params.terminations
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ball_fell_off": TerminationTermCfg(
            func=bb_mdp.ball_fell_off,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "max_xy_radius": terminations.max_xy_radius,
                "min_height": terminations.min_height,
                "floor_height": terminations.floor_height,
            },
        ),
    }
