"""Centralized tunable parameters for the Kinova ball-balancing task."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NoiseRange:
    min: float
    max: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.min, self.max)


@dataclass(frozen=True)
class Vec3Ranges:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


@dataclass(frozen=True)
class BallParameters:
    radius: float = 0.0335
    mass: float = 0.0657
    friction: tuple[float, float, float] = (1.0, 0.2, 0.0005)
    condim: int = 6
    solref: tuple[float, float] = (0.02, 2.5)
    solimp: tuple[float, float, float, float, float] = (0.95, 0.995, 0.001, 0.5, 2.0)
    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0)


@dataclass(frozen=True)
class ObservationNoiseParameters:
    joint_pos: NoiseRange = field(default_factory=lambda: NoiseRange(-0.01, 0.01))
    joint_vel: NoiseRange = field(default_factory=lambda: NoiseRange(-0.1, 0.1))
    ee_pos: NoiseRange = field(default_factory=lambda: NoiseRange(-0.003, 0.003))
    ee_vel: NoiseRange = field(default_factory=lambda: NoiseRange(-0.05, 0.05))
    ee_ft_wrench: NoiseRange = field(default_factory=lambda: NoiseRange(-0.1, 0.1))


@dataclass(frozen=True)
class JointActionParameters:
    scale: float = 0.13
    use_default_offset: bool = True


@dataclass(frozen=True)
class CartesianActionParameters:
    delta_pos_scale: float = 0.04
    damping: float = 0.05
    max_dq: float = 0.2
    position_weight: float = 1.0
    orientation_weight: float = 0.0
    posture_weight: float = 0.03


@dataclass(frozen=True)
class BallResetParameters:
    xy_range: tuple[float, float] = (-0.02, 0.02)
    z_offset: float = 0.05
    x_offset: float = 0.0
    y_offset: float = 0.0
    linear_velocity: Vec3Ranges = field(
        default_factory=lambda: Vec3Ranges(
            x=(-0.25, 0.25),
            y=(-0.25, 0.25),
            z=(-0.05, 0.05),
        )
    )
    angular_velocity: tuple[float, float] = (-2.0, 2.0)


@dataclass(frozen=True)
class KickParameters:
    interval_s: tuple[float, float] = (0.4, 1.0)
    linear_velocity: Vec3Ranges = field(
        default_factory=lambda: Vec3Ranges(
            x=(-0.15, 0.15),
            y=(-0.15, 0.15),
            z=(-0.03, 0.03),
        )
    )
    angular_velocity: tuple[float, float] = (-0.5, 0.5)
    add_to_current: bool = True


@dataclass(frozen=True)
class RandomizationParameters:
    null_space_joint_offset: tuple[float, float] = (-0.35, 0.35)
    ball_mass_scale: tuple[float, float] = (0.7, 1.3)
    pd_gain_scale: tuple[float, float] = (0.95, 1.05)
    robot_body_mass_scale: tuple[float, float] = (0.9, 1.1)
    robot_body_inertia_scale: tuple[float, float] = (0.9, 1.1)
    robot_dof_armature_scale: tuple[float, float] = (0.9, 1.1)


@dataclass(frozen=True)
class RewardParameters:
    is_alive: float = 0.2
    ball_centering: float = 40.0
    ball_centering_std: float = 0.06
    ball_speed: float = -8.0
    ball_speed_lin_weight: float = 1.0
    ball_speed_ang_weight: float = 1.0
    ball_no_contact: float = -18.0
    ball_no_contact_dist: float = 0.0
    action_rate_l2: float = -0.01
    action_acc_l2: float = -0.0015
    joint_vel_l2: float = -0.0005
    joint_acc_l2: float = -0.0001
    joint_torque_l2: float = -0.0002
    joint_pos_limits: float = -0.2
    racquet_lin_vel_l2: float = -5.0
    racquet_dist_from_initial_l2: float = -30.0


@dataclass(frozen=True)
class TerminationParameters:
    max_xy_radius: float = 0.16
    min_height: float = -0.06
    floor_height: float = 0.05


@dataclass(frozen=True)
class SimulationParameters:
    num_envs: int = 1
    env_spacing: float = 2.0
    timestep: float = 0.002
    iterations: int = 30
    ls_iterations: int = 30
    ccd_iterations: int = 80
    nconmax: int = 256
    njmax: int = 1024
    decimation: int = 5
    episode_length_s: float = 10.0
    play_episode_length_s: int = int(1e9)
    viewer_distance: float = 0.9
    viewer_elevation: float = -35.0
    viewer_azimuth: float = 110.0


@dataclass(frozen=True)
class PpoParameters:
    actor_hidden_dims: tuple[int, ...] = (64, 64)
    critic_hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "elu"
    init_noise_std: float = 1.0
    value_loss_coef: float = 1.0
    clip_param: float = 0.2
    entropy_coef: float = 0.003
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    save_interval: int = 200
    num_steps_per_env: int = 24
    max_iterations: int = 10_000


@dataclass(frozen=True)
class TaskParameters:
    ball: BallParameters = field(default_factory=BallParameters)
    observation_noise: ObservationNoiseParameters = field(default_factory=ObservationNoiseParameters)
    joint_action: JointActionParameters = field(default_factory=JointActionParameters)
    cartesian_action: CartesianActionParameters = field(default_factory=CartesianActionParameters)
    ball_reset: BallResetParameters = field(default_factory=BallResetParameters)
    ball_kick: KickParameters = field(default_factory=KickParameters)
    randomization: RandomizationParameters = field(default_factory=RandomizationParameters)
    rewards: RewardParameters = field(default_factory=RewardParameters)
    terminations: TerminationParameters = field(default_factory=TerminationParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)
    ppo: PpoParameters = field(default_factory=PpoParameters)


DEFAULT_TASK_PARAMETERS = TaskParameters()
