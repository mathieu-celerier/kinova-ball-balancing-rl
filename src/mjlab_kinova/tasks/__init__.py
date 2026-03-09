from mjlab.tasks.registry import register_mjlab_task

from .kinova_ball_balancing_env_cfg import kinova_ball_balancing_env_cfg, kinova_ppo_runner_cfg


def _register_variant(task_id: str, variant: str) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=kinova_ball_balancing_env_cfg(variant=variant),
        play_env_cfg=kinova_ball_balancing_env_cfg(variant=variant, play=True),
        rl_cfg=kinova_ppo_runner_cfg(variant=variant),
    )


_register_variant("Mjlab-BallBalancing-Kinova", "baseline")
_register_variant("Mjlab-BallBalancing-Kinova-Baseline", "baseline")
_register_variant("Mjlab-BallBalancing-Kinova-Cartesian", "cartesian")
_register_variant(
    "Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand",
    "baseline_no_model_rand",
)

# Backward-compatible play alias for older commands.
register_mjlab_task(
    task_id="Mjlab-BallBalancing-Kinova-Play",
    env_cfg=kinova_ball_balancing_env_cfg(variant="baseline", play=True),
    play_env_cfg=kinova_ball_balancing_env_cfg(variant="baseline", play=True),
    rl_cfg=kinova_ppo_runner_cfg(variant="baseline"),
)
