from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mjlab_kinova.early_stop_runner import KinovaOnPolicyRunner

from .kinova_ball_balancing_env_cfg import kinova_ball_balancing_env_cfg, kinova_ppo_runner_cfg
from .task_parameters import load_default_task_parameters, load_task_parameters_from_files


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRESET_DIR = _REPO_ROOT / "config" / "presets"


def _register_variant(task_id: str, variant: str, *preset_names: str) -> None:
    params = _load_params_with_presets(*preset_names)
    register_mjlab_task(
        task_id=task_id,
        env_cfg=kinova_ball_balancing_env_cfg(variant=variant, params=params),
        play_env_cfg=kinova_ball_balancing_env_cfg(variant=variant, play=True, params=params),
        rl_cfg=kinova_ppo_runner_cfg(variant=variant, params=params),
        runner_cls=KinovaOnPolicyRunner,
    )


def _load_params_with_presets(*preset_names: str):
    base_params = load_default_task_parameters()
    if not preset_names:
        return base_params
    preset_paths = [_PRESET_DIR / f"{preset_name}.yaml" for preset_name in preset_names]
    return load_task_parameters_from_files(preset_paths, base=base_params)


_register_variant("Mjlab-BallBalancing-Kinova", "joint")
_register_variant("Mjlab-BallBalancing-Kinova-Joint", "joint")
_register_variant("Mjlab-BallBalancing-Kinova-Baseline", "joint")
_register_variant("Mjlab-BallBalancing-Kinova-Cartesian", "cartesian")
_register_variant(
    "Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand",
    "joint",
    "no_model_rand",
)
_register_variant(
    "Mjlab-BallBalancing-Kinova-BaselineNoRand",
    "joint",
    "no_rand",
)

# Backward-compatible play alias for older commands.
_play_params = load_default_task_parameters()
register_mjlab_task(
    task_id="Mjlab-BallBalancing-Kinova-Play",
    env_cfg=kinova_ball_balancing_env_cfg(variant="joint", play=True, params=_play_params),
    play_env_cfg=kinova_ball_balancing_env_cfg(variant="joint", play=True, params=_play_params),
    rl_cfg=kinova_ppo_runner_cfg(variant="joint", params=_play_params),
    runner_cls=KinovaOnPolicyRunner,
)
