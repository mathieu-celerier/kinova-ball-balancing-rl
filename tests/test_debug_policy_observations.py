from __future__ import annotations

import unittest
from pathlib import Path

from scripts.debug_policy_observations import (
    _actor_layout,
    _actor_term_semantics,
    _resolve_run_params,
)


class DebugPolicyObservationsTests(unittest.TestCase):
    def test_actor_term_semantics_describes_ft_wrench_mode(self) -> None:
        raw_semantics = _actor_term_semantics("joint", "raw")
        compensated_semantics = _actor_term_semantics("joint", "weight_compensated")

        self.assertIn("raw sensor values", raw_semantics["ee_ft_wrench"])
        self.assertIn("weight removed", compensated_semantics["ee_ft_wrench"])

    def test_resolves_concrete_variant_from_multi_training_set(self) -> None:
        training_set = Path("config/training_sets/joint_cartesian_ablation_simple_rewards.yaml")

        joint_variant, _ = _resolve_run_params(training_set, "joint_all_randomization")
        cartesian_variant, _ = _resolve_run_params(training_set, "cartesian_all_randomization")

        self.assertEqual(joint_variant, "joint")
        self.assertEqual(cartesian_variant, "cartesian")

    def test_actor_layout_keeps_offsets_and_history_order(self) -> None:
        layout = _actor_layout(
            [
                {
                    "name": "joint_pos",
                    "flat_shape": [35],
                    "offset_start": 0,
                    "offset_end": 35,
                    "history_length": 5,
                    "history_order": "oldest_to_newest",
                    "flat": [],
                    "history": [],
                    "current": [],
                },
                {
                    "name": "actions",
                    "flat_shape": [35],
                    "offset_start": 165,
                    "offset_end": 200,
                    "history_length": 5,
                    "history_order": "oldest_to_newest",
                    "flat": [],
                    "history": [],
                    "current": [],
                },
            ]
        )

        self.assertEqual(layout[0]["offset_start"], 0)
        self.assertEqual(layout[1]["offset_end"], 200)
        self.assertEqual(layout[0]["history_order"], "oldest_to_newest")


if __name__ == "__main__":
    unittest.main()
