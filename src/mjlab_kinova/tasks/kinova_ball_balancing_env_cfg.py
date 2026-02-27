"""Kinova ball balancing task configuration."""

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
        friction=(1.0, 0.01, 0.0005),
        solref=(0.02, 2.5),
        solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
        rgba=(0.9, 0.2, 0.2, 1.0),
    )
    return spec


def kinova_ball_balancing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create train/play environment config for Kinova ball balancing."""
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))
                    },
                    noise=Unoise(n_min=-0.01, n_max=0.01),
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))
                    },
                    noise=Unoise(n_min=-0.1, n_max=0.1),
                ),
                "joint_torque": ObservationTermCfg(
                    func=bb_mdp.joint_torques,
                    params={"robot_name": "robot"},
                    noise=Unoise(n_min=-0.05, n_max=0.05),
                ),
                "ee_ft_wrench": ObservationTermCfg(
                    func=bb_mdp.ee_ft_wrench,
                    noise=Unoise(n_min=-0.1, n_max=0.1),
                ),
            },
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))
                    },
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))
                    },
                ),
                "joint_torque": ObservationTermCfg(
                    func=bb_mdp.joint_torques,
                    params={"robot_name": "robot"},
                ),
                "ee_ft_wrench": ObservationTermCfg(
                    func=bb_mdp.ee_ft_wrench,
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.13,
            use_default_offset=True,
        )
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=bb_mdp.reset_ball_on_plate,
            mode="reset",
            params={
                "ball_name": "ball",
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
                "xy_range": (-0.01, 0.01),
                "z_offset": 0.04,
                "x_offset": 0.0,
                "y_offset": 0.0,
            },
        ),
    }

    rewards = {
        "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.2),
        "ball_centering": RewardTermCfg(
            func=bb_mdp.ball_centering_reward,
            weight=30.0,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
                "std": 0.06,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ),
        "ball_speed": RewardTermCfg(
            func=bb_mdp.ball_speed_penalty,
            weight=-5.0,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
            },
        ),
        "ball_no_contact_penalty": RewardTermCfg(
            func=bb_mdp.ball_no_contact_proxy,
            weight=-10.0,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
                "contact_z": 0.04,
                "z_tolerance": 0.01,
                "max_xy_radius": 0.105,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.0015),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_.*",))},
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
            weight=-0.2,
            params={
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
            },
        ),
        "racquet_dist_from_initial_l2": RewardTermCfg(
            func=bb_mdp.racquet_dist_from_initial_l2,
            weight=-15.0,
            params={
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ball_fell_off": TerminationTermCfg(
            func=bb_mdp.ball_fell_off,
            params={
                "ball_name": "ball",
                "plate_asset_cfg": SceneEntityCfg("robot", body_names=("racquet_frame",)),
                "max_xy_radius": 0.11,
                "min_height": -0.03,
                "floor_height": 0.05,
            },
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
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
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


def kinova_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner config for Kinova ball balancing."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=False,
            stochastic=True,
            init_noise_std=1.0,
            noise_std_type="log",
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
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
        experiment_name="kinova_ball_balancing",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
