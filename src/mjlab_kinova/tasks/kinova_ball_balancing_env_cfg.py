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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab_kinova.robot.kinova_constants import KINOVA_CFG

from . import ball_balancing_mdp as bb_mdp
from .policy_actions import InitialFramePositionActionCfg
from .task_parameters import DEFAULT_TASK_PARAMETERS, TaskParameters

PolicyVariant = Literal[
    "baseline", "cartesian", "baseline_no_model_rand", "baseline_no_rand"
]


def robot_joints_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=("joint_.*",))


def robot_actuators_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot")


def racquet_frame_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("racquet_frame",))


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


@dataclass(frozen=True)
class PolicySpec:
    variant: PolicyVariant
    experiment_name: str
    action_kind: Literal["joint", "cartesian"]
    actor_terms: tuple[str, ...]
    use_observation_noise: bool
    randomize_ball_reset: bool
    randomize_ball_properties: bool
    randomize_pd_gains: bool
    randomize_robot_model: bool
    randomize_null_space_init: bool
    use_ball_kick: bool


POLICY_SPECS: dict[PolicyVariant, PolicySpec] = {
    "baseline": PolicySpec(
        variant="baseline",
        experiment_name="kinova_ball_balancing_baseline",
        action_kind="joint",
        actor_terms=("joint_pos", "joint_vel", "ee_pos", "ee_vel", "ee_ft_wrench"),
        use_observation_noise=True,
        randomize_ball_reset=True,
        randomize_ball_properties=True,
        randomize_pd_gains=True,
        randomize_robot_model=True,
        randomize_null_space_init=True,
        use_ball_kick=True,
    ),
    "cartesian": PolicySpec(
        variant="cartesian",
        experiment_name="kinova_ball_balancing_cartesian",
        action_kind="cartesian",
        actor_terms=("ee_pos", "ee_vel", "ee_ft_wrench"),
        use_observation_noise=True,
        randomize_ball_reset=True,
        randomize_ball_properties=True,
        randomize_pd_gains=True,
        randomize_robot_model=True,
        randomize_null_space_init=False,
        use_ball_kick=True,
    ),
    "baseline_no_model_rand": PolicySpec(
        variant="baseline_no_model_rand",
        experiment_name="kinova_ball_balancing_baseline_no_model_rand",
        action_kind="joint",
        actor_terms=("joint_pos", "joint_vel", "ee_pos", "ee_vel", "ee_ft_wrench"),
        use_observation_noise=True,
        randomize_ball_reset=True,
        randomize_ball_properties=True,
        randomize_pd_gains=False,
        randomize_robot_model=False,
        randomize_null_space_init=False,
        use_ball_kick=True,
    ),
    "baseline_no_rand": PolicySpec(
        variant="baseline_no_rand",
        experiment_name="kinova_ball_balancing_baseline_no_rand",
        action_kind="joint",
        actor_terms=("joint_pos", "joint_vel", "ee_pos", "ee_vel", "ee_ft_wrench"),
        use_observation_noise=False,
        randomize_ball_reset=False,
        randomize_ball_properties=False,
        randomize_pd_gains=False,
        randomize_robot_model=False,
        randomize_null_space_init=False,
        use_ball_kick=True,
    ),
}


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
    variant: PolicyVariant = "baseline",
    play: bool = False,
    params: TaskParameters | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create environment config for one of the RO-MAN 2026 policy variants."""
    params = DEFAULT_TASK_PARAMETERS if params is None else params
    spec = POLICY_SPECS[variant]

    observations = {
        "actor": ObservationGroupCfg(
            terms=_actor_observation_terms(spec, play, params),
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
                "robot": KINOVA_CFG,
                "ball": _ball_entity_cfg(params),
            },
            num_envs=params.simulation.num_envs,
            env_spacing=params.simulation.env_spacing,
        ),
        observations=observations,
        actions=_actions_cfg(spec, params),
        events=_events_cfg(spec, play, params),
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

    return cfg


def kinova_ppo_runner_cfg(
    variant: PolicyVariant = "baseline",
    params: TaskParameters | None = None,
) -> RslRlOnPolicyRunnerCfg:
    """Create the PPO runner config for a given policy variant."""
    params = DEFAULT_TASK_PARAMETERS if params is None else params
    ppo = params.ppo
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=ppo.actor_hidden_dims,
            activation=ppo.activation,
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": ppo.init_noise_std,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=ppo.critic_hidden_dims,
            activation=ppo.activation,
            obs_normalization=False,
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
        experiment_name=POLICY_SPECS[variant].experiment_name,
        logger="tensorboard",
        save_interval=ppo.save_interval,
        num_steps_per_env=ppo.num_steps_per_env,
        max_iterations=ppo.max_iterations,
    )


def _actor_observation_terms(
    spec: PolicySpec, play: bool, params: TaskParameters
) -> dict[str, ObservationTermCfg]:
    actor_terms = _shared_observation_terms(
        use_noise=spec.use_observation_noise and not play, params=params
    )
    return {name: actor_terms[name] for name in spec.actor_terms}


def _critic_observation_terms(params: TaskParameters) -> dict[str, ObservationTermCfg]:
    terms = _shared_observation_terms(use_noise=False, params=params)
    terms.update(
        {
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
        }
    )
    return terms


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
    ee_vel_noise = _noise_cfg(use_noise, noise.ee_vel)
    ft_noise = _noise_cfg(use_noise, noise.ee_ft_wrench)
    return {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
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
        "ee_vel": ObservationTermCfg(
            func=bb_mdp.body_linear_velocity_w,
            params={"asset_cfg": racquet_frame_cfg()},
            noise=ee_vel_noise,
        ),
        "ee_ft_wrench": ObservationTermCfg(
            func=bb_mdp.ee_ft_wrench,
            noise=ft_noise,
        ),
    }


def _actions_cfg(spec: PolicySpec, params: TaskParameters) -> dict[str, ActionTermCfg]:
    if spec.action_kind == "joint":
        return {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=params.joint_action.scale,
                use_default_offset=params.joint_action.use_default_offset,
            )
        }

    action = params.cartesian_action
    return {
        "ee_pos": InitialFramePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            frame_type="body",
            frame_name="racquet_frame",
            delta_pos_scale=action.delta_pos_scale,
            damping=action.damping,
            max_dq=action.max_dq,
            position_weight=action.position_weight,
            orientation_weight=action.orientation_weight,
            posture_weight=action.posture_weight,
            posture_target=KINOVA_CFG.init_state.joint_pos,
        )
    }


def _events_cfg(
    spec: PolicySpec, play: bool, params: TaskParameters
) -> dict[str, EventTermCfg]:
    randomization = params.randomization
    ball_reset = params.ball_reset
    joint_reset_range = (
        randomization.null_space_joint_offset
        if spec.randomize_null_space_init
        else (0.0, 0.0)
    )
    ball_xy_range = ball_reset.xy_range if spec.randomize_ball_reset else (0.0, 0.0)
    ball_lin_vel_x_range = (
        ball_reset.linear_velocity.x if spec.randomize_ball_reset else (0.0, 0.0)
    )
    ball_lin_vel_y_range = (
        ball_reset.linear_velocity.y if spec.randomize_ball_reset else (0.0, 0.0)
    )
    ball_lin_vel_z_range = (
        ball_reset.linear_velocity.z if spec.randomize_ball_reset else (0.0, 0.0)
    )
    ball_ang_vel_range = (
        ball_reset.angular_velocity if spec.randomize_ball_reset else (0.0, 0.0)
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
        "reset_ee_ft_bias": EventTermCfg(
            func=bb_mdp.reset_ee_ft_bias,
            mode="reset",
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
                "lin_vel_x_range": ball_lin_vel_x_range,
                "lin_vel_y_range": ball_lin_vel_y_range,
                "lin_vel_z_range": ball_lin_vel_z_range,
                "ang_vel_range": ball_ang_vel_range,
            },
        ),
        "log_first_step_after_reset": EventTermCfg(
            func=bb_mdp.log_first_step_after_reset,
            mode="step",
        ),
    }

    if spec.randomize_ball_properties:
        events["randomize_ball_mass"] = EventTermCfg(
            func=bb_mdp.randomize_body_mass,
            mode="reset",
            params={
                "asset_cfg": ball_body_cfg(),
                "mass_range": randomization.ball_mass_scale,
                "operation": "scale",
            },
        )

    if spec.randomize_pd_gains:
        events["randomize_pd_gains"] = EventTermCfg(
            func=bb_mdp.randomize_pd_gains,
            mode="reset",
            params={
                "kp_range": randomization.pd_gain_scale,
                "kd_range": randomization.pd_gain_scale,
                "asset_cfg": robot_actuators_cfg(),
                "operation": "scale",
            },
        )

    if spec.randomize_robot_model:
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

    if spec.use_ball_kick and not play:
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


def _rewards_cfg(params: TaskParameters) -> dict[str, RewardTermCfg]:
    rewards = params.rewards
    return {
        "is_alive": RewardTermCfg(func=mdp.is_alive, weight=rewards.is_alive),
        "ball_centering": RewardTermCfg(
            func=bb_mdp.ball_centering_contact_reward,
            weight=rewards.ball_centering,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
                "std": rewards.ball_centering_std,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ),
        "ball_speed": RewardTermCfg(
            func=bb_mdp.ball_speed_penalty,
            weight=rewards.ball_speed,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "lin_weight": rewards.ball_speed_lin_weight,
                "ang_weight": rewards.ball_speed_ang_weight,
            },
        ),
        "ball_height_above_plate": RewardTermCfg(
            func=bb_mdp.ball_height_above_plate_penalty,
            weight=rewards.ball_height_above_plate,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": racquet_frame_cfg(),
                "soft_height": rewards.ball_height_soft_threshold,
            },
        ),
        "ball_no_contact_penalty": RewardTermCfg(
            func=bb_mdp.ball_no_contact_after_first_contact,
            weight=rewards.ball_no_contact,
            params={
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "pre_contact_action_rate_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.pre_contact_action_rate_l2,
            params={
                "term_func": mdp.action_rate_l2,
                "activate_after_contact": False,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "post_contact_action_rate_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.post_contact_action_rate_l2,
            params={
                "term_func": mdp.action_rate_l2,
                "activate_after_contact": True,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "pre_contact_action_acc_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.pre_contact_action_acc_l2,
            params={
                "term_func": mdp.action_acc_l2,
                "activate_after_contact": False,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "post_contact_action_acc_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.post_contact_action_acc_l2,
            params={
                "term_func": mdp.action_acc_l2,
                "activate_after_contact": True,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "pre_contact_joint_vel_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.pre_contact_joint_vel_l2,
            params={
                "term_func": mdp.joint_vel_l2,
                "term_kwargs": {"asset_cfg": robot_joints_cfg()},
                "activate_after_contact": False,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "post_contact_joint_vel_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.post_contact_joint_vel_l2,
            params={
                "term_func": mdp.joint_vel_l2,
                "term_kwargs": {"asset_cfg": robot_joints_cfg()},
                "activate_after_contact": True,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
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
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=rewards.joint_pos_limits,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[246]",))},
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
        "racquet_lin_vel_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.post_contact_racquet_lin_vel_l2,
            params={
                "term_func": bb_mdp.racquet_lin_vel_l2,
                "term_kwargs": {"plate_asset_cfg": racquet_frame_cfg()},
                "activate_after_contact": True,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "pre_contact_racquet_lin_vel_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.pre_contact_racquet_lin_vel_l2,
            params={
                "term_func": bb_mdp.racquet_lin_vel_l2,
                "term_kwargs": {"plate_asset_cfg": racquet_frame_cfg()},
                "activate_after_contact": False,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "racquet_dist_from_initial_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.post_contact_racquet_dist_from_initial_l2,
            params={
                "term_func": bb_mdp.racquet_dist_from_initial_l2,
                "term_kwargs": {"plate_asset_cfg": racquet_frame_cfg()},
                "activate_after_contact": True,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
        ),
        "pre_contact_racquet_dist_from_initial_l2": RewardTermCfg(
            func=bb_mdp.contact_phase_reward,
            weight=rewards.pre_contact_racquet_dist_from_initial_l2,
            params={
                "term_func": bb_mdp.racquet_dist_from_initial_l2,
                "term_kwargs": {"plate_asset_cfg": racquet_frame_cfg()},
                "activate_after_contact": False,
                "ball_geom_name": "ball/ball_geom",
                "racquet_geom_name": "robot/plate_collision",
                "max_contact_dist": rewards.ball_no_contact_dist,
            },
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
