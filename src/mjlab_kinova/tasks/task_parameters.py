"""Centralized tunable parameters for the Kinova ball-balancing task."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar, get_args, get_origin, get_type_hints

import yaml


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
    ee_quat: NoiseRange = field(default_factory=lambda: NoiseRange(-0.01, 0.01))
    ee_ori: NoiseRange = field(default_factory=lambda: NoiseRange(-0.01, 0.01))
    ee_vel: NoiseRange = field(default_factory=lambda: NoiseRange(-0.05, 0.05))
    ee_ang_vel: NoiseRange = field(default_factory=lambda: NoiseRange(-0.1, 0.1))
    ee_ft_wrench: NoiseRange = field(default_factory=lambda: NoiseRange(-0.1, 0.1))


@dataclass(frozen=True)
class JointActionParameters:
    use_default_offset: bool = True
    use_nullspace_torque: bool = False
    stiffness: tuple[float, ...] = (40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 15.0)
    damping: tuple[float, ...] = (15.0, 15.0, 15.0, 15.0, 8.5, 8.5, 8.5)
    nullspace_stiffness: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nullspace_damping: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    damping_pinv: float = 0.05
    nullspace_resample_interval_s: tuple[float, float] = (0.25, 1.0)


@dataclass(frozen=True)
class CartesianActionParameters:
    delta_pos_scale: float = 0.05
    delta_ori_scale: float = 0.5
    damping_pos: float = 0.05
    damping_ori: float = 0.05
    damping_null: float = 0.05
    damping_pinv: float = 0.05
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    posture_weight: float = 0.03
    orientation_error_in_body_frame: bool = False
    nullspace_resample_interval_s: tuple[float, float] = (0.25, 1.0)


@dataclass(frozen=True)
class BallResetParameters:
    xy_range: tuple[float, float] = (-0.02, 0.02)
    z_offset: float = 0.05
    x_offset: float = 0.0
    y_offset: float = 0.0
    release_delay_s: tuple[float, float] = (0.0, 0.25)
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
    null_space_joint_offset: tuple[float, float] = (-1.0, 1.0)
    null_space_curriculum_initial_scale: float = 0.05
    null_space_curriculum_start_steps: int = 25_000
    null_space_curriculum_steps: int = 125_000
    ball_mass_scale: tuple[float, float] = (0.7, 1.3)
    ball_friction_scale: tuple[float, float] = (0.8, 1.2)
    pd_gain_scale: tuple[float, float] = (0.95, 1.05)
    racquet_body_mass_scale: tuple[float, float] = (0.9, 1.1)
    racquet_body_inertia_scale: tuple[float, float] = (0.9, 1.1)
    robot_body_mass_scale: tuple[float, float] = (0.9, 1.1)
    robot_body_inertia_scale: tuple[float, float] = (0.9, 1.1)
    robot_dof_armature_scale: tuple[float, float] = (0.9, 1.1)


@dataclass(frozen=True)
class RewardParameters:
    is_alive: float = 0.2
    ball_no_contact: float = -18.0
    ball_no_contact_dist: float = 0.0
    ball_centering: float = 200.0
    ball_centering_std: float = 0.04
    ball_lin_vel_l2: float = 0.0
    ball_lin_vel_plate_l2: float = 0.0
    action_rate_l2: float = -1.0
    action_acc_l2: float = -0.15
    joint_vel_l2: float = -0.02
    joint_acc_l2: float = -0.001
    joint_torque_l2: float = -0.002
    joint_torque_rate_l2: float = 0.0
    joint_pos_limits: float = -5.0
    plate_drop_under_ball: float = -5.0
    plate_drop_ball_height_threshold: float = 0.01
    plate_drop_xy_radius: float = 0.12
    racquet_ang_vel_l2: float = -20.0
    racquet_lin_vel_l2: float = -80.0
    racquet_centering_std: float = 0.08
    racquet_orientation_centering_std: float = 0.2
    racquet_centering: float = 80.0
    racquet_orientation_centering: float = 40.0


@dataclass(frozen=True)
class TerminationParameters:
    max_xy_radius: float = 0.16
    min_height: float = -0.06
    floor_height: float = 0.05


@dataclass(frozen=True)
class SimulationParameters:
    num_envs: int = 1024
    env_spacing: float = 2.0
    timestep: float = 0.002
    iterations: int = 20
    ls_iterations: int = 20
    ccd_iterations: int = 40
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
    max_iterations: int = 30_000


@dataclass(frozen=True)
class TrainingParameters:
    experiment_name_suffix: str | None = None
    wandb_project: str | None = None
    run_name: str | None = None
    upload_videos_to_wandb: bool = True
    use_observation_noise: bool | None = None
    use_joint_pos_observation: bool | None = None
    enable_ball_kick_in_play: bool = False
    randomize_ball_reset: bool | None = None
    randomize_ball_properties: bool | None = None
    randomize_pd_gains: bool | None = None
    randomize_racquet_model: bool | None = None
    randomize_robot_model: bool | None = None
    randomize_null_space_init: bool | None = None
    use_ball_kick: bool | None = None


@dataclass(frozen=True)
class TaskParameters:
    ball: BallParameters = field(default_factory=BallParameters)
    observation_history_length: int = 5
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
    training: TrainingParameters = field(default_factory=TrainingParameters)


DEFAULT_TASK_PARAMETERS = TaskParameters()
DEFAULT_TASK_PARAMETERS_PATH = Path(__file__).resolve().parents[3] / "config" / "task_parameters.yaml"
TASK_PARAMETERS_ENV_VAR = "MJLAB_KINOVA_TASK_PARAMS"

T = TypeVar("T")


def _coerce_value(field_type: Any, value: Any, current_value: Any) -> Any:
    origin = get_origin(field_type)

    if is_dataclass(field_type):
        if not isinstance(value, dict):
            raise TypeError(f"Expected mapping for {field_type.__name__}, got {type(value).__name__}")
        return _merge_dataclass(current_value, value)

    if origin is tuple:
        args = get_args(field_type)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected sequence for tuple field, got {type(value).__name__}")
        if len(args) == 2 and args[1] is ...:
            return tuple(value)
        return tuple(value)

    return value


def _merge_dataclass(instance: T, overrides: dict[str, Any]) -> T:
    if isinstance(instance, CartesianActionParameters) and "damping_task" in overrides:
        legacy_damping = overrides.pop("damping_task")
        overrides.setdefault("damping_pos", legacy_damping)
        overrides.setdefault("damping_ori", legacy_damping)

    values = {}
    valid_fields = {field_info.name: field_info for field_info in fields(instance)}
    type_hints = get_type_hints(type(instance))

    unknown_keys = set(overrides) - set(valid_fields)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise KeyError(f"Unknown configuration keys for {type(instance).__name__}: {unknown}")

    for name, field_info in valid_fields.items():
        current_value = getattr(instance, name)
        if name in overrides:
            values[name] = _coerce_value(
                type_hints.get(name, field_info.type),
                overrides[name],
                current_value,
            )
        else:
            values[name] = current_value

    return type(instance)(**values)


def task_parameters_to_dict(params: TaskParameters) -> dict[str, Any]:
    return asdict(params)


def merge_task_parameter_overrides(
    *overrides: dict[str, Any],
    base: TaskParameters = DEFAULT_TASK_PARAMETERS,
) -> TaskParameters:
    params = copy.deepcopy(base)
    for override in overrides:
        if not override:
            continue
        if not isinstance(override, dict):
            raise TypeError(f"Top-level YAML object must be a mapping, got {type(override).__name__}")
        params = _merge_dataclass(params, override)
    return params


def load_task_parameter_overrides(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Top-level YAML object must be a mapping, got {type(raw).__name__}")

    return raw


def load_task_parameters(
    path: str | Path = DEFAULT_TASK_PARAMETERS_PATH,
    *,
    base: TaskParameters = DEFAULT_TASK_PARAMETERS,
) -> TaskParameters:
    return merge_task_parameter_overrides(load_task_parameter_overrides(path), base=base)


def load_task_parameters_from_files(
    paths: Iterable[str | Path],
    *,
    base: TaskParameters = DEFAULT_TASK_PARAMETERS,
) -> TaskParameters:
    overrides = [load_task_parameter_overrides(path) for path in paths]
    return merge_task_parameter_overrides(*overrides, base=base)


def load_default_task_parameters() -> TaskParameters:
    """Load task parameters from the default YAML path or an environment override."""
    config_path = os.environ.get(TASK_PARAMETERS_ENV_VAR, DEFAULT_TASK_PARAMETERS_PATH)
    return load_task_parameters(config_path)
