from __future__ import annotations

import unittest

from mjlab_kinova.play_set import _apply_play_overrides, _challenge_video_args
from mjlab_kinova.tasks.task_parameters import load_default_task_parameters


class PlaySetTests(unittest.TestCase):
    def test_challenge_video_enables_stronger_kicks_and_ten_second_video(self) -> None:
        params = load_default_task_parameters()
        original_interval = params.ball_kick.interval_s
        params = _apply_play_overrides(
            variant="joint",
            params=params,
            ball_kick=False,
            challenge_video=True,
            nullspace_torque=None,
        )

        self.assertTrue(params.training.enable_ball_kick_in_play)
        self.assertTrue(params.training.use_ball_kick)
        self.assertEqual(params.ball_kick.interval_s, original_interval)
        self.assertEqual(params.ball_kick.linear_velocity.x, (-0.225, 0.225))
        self.assertEqual(params.ball_kick.linear_velocity.y, (-0.225, 0.225))
        self.assertEqual(params.ball_kick.linear_velocity.z, (-0.045, 0.045))
        self.assertEqual(params.ball_kick.angular_velocity, (-0.75, 0.75))
        self.assertEqual(
            _challenge_video_args(params),
            [
                "--video",
                "True",
                "--video-length",
                "333",
                "--video-width",
                "1920",
                "--video-height",
                "1080",
                "--num-envs",
                "1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
