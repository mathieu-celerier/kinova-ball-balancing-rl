from mjlab.tasks.registry import register_mjlab_task

from .kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
    kinova_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-BallBalancing-Kinova",
    env_cfg=kinova_ball_balancing_env_cfg(),
    play_env_cfg=kinova_ball_balancing_env_cfg(play=True),
    rl_cfg=kinova_ppo_runner_cfg(),
)

# Backward-compatible alias for older commands.
register_mjlab_task(
    task_id="Mjlab-BallBalancing-Kinova-Play",
    env_cfg=kinova_ball_balancing_env_cfg(play=True),
    play_env_cfg=kinova_ball_balancing_env_cfg(play=True),
    rl_cfg=kinova_ppo_runner_cfg(),
)
