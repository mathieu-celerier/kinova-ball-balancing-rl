from mjlab.tasks.registry import register_mjlab_task

from .kinova_ball_balancing_env_cfg import kinova_ball_balancing_env_cfg, kinova_ppo_runner_cfg
from .task_parameters import load_default_task_parameters


def _register_variant(task_id: str, variant: str) -> None:
    params = load_default_task_parameters()
    register_mjlab_task(
        task_id=task_id,
        env_cfg=kinova_ball_balancing_env_cfg(variant=variant, params=params),
        play_env_cfg=kinova_ball_balancing_env_cfg(variant=variant, play=True, params=params),
        rl_cfg=kinova_ppo_runner_cfg(variant=variant, params=params),
    )


_register_variant("Mjlab-BallBalancing-Kinova", "baseline")
_register_variant("Mjlab-BallBalancing-Kinova-Baseline", "baseline")
_register_variant("Mjlab-BallBalancing-Kinova-Cartesian", "cartesian")
_register_variant(
    "Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand",
    "baseline_no_model_rand",
)
_register_variant(
    "Mjlab-BallBalancing-Kinova-BaselineNoRand",
    "baseline_no_rand",
)

# Backward-compatible play alias for older commands.
_play_params = load_default_task_parameters()
register_mjlab_task(
    task_id="Mjlab-BallBalancing-Kinova-Play",
    env_cfg=kinova_ball_balancing_env_cfg(variant="baseline", play=True, params=_play_params),
    play_env_cfg=kinova_ball_balancing_env_cfg(variant="baseline", play=True, params=_play_params),
    rl_cfg=kinova_ppo_runner_cfg(variant="baseline", params=_play_params),
)
