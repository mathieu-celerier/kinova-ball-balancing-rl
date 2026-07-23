from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mjlab_kinova.tasks.kinova_ball_balancing_env_cfg import (
    kinova_ball_balancing_env_cfg,
)
from mjlab_kinova.tasks.task_parameters import (
    load_default_task_parameters,
    merge_task_parameter_overrides,
)
from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _normalize_run,
)


class CartesianPoseOnlyConfigTests(unittest.TestCase):
    def test_ft_wrench_observation_uses_configured_default_mode(self) -> None:
        params = load_default_task_parameters()
        cfg = kinova_ball_balancing_env_cfg(variant="joint", params=params)

        self.assertEqual(params.ee_ft_wrench_mode, "mc_rtc_calibrated")
        self.assertEqual(
            cfg.observations["actor"].terms["ee_ft_wrench"].params["mode"],
            "mc_rtc_calibrated",
        )
        self.assertEqual(
            cfg.observations["actor"].terms["ee_ft_wrench"].params["calib_file"],
            str(params.resolved_ee_ft_calib_file()),
        )

    def test_ft_wrench_observation_can_use_raw_mode(self) -> None:
        params = merge_task_parameter_overrides(
            {"ee_ft_wrench_mode": "raw"},
            base=load_default_task_parameters(),
        )
        cfg = kinova_ball_balancing_env_cfg(variant="joint", params=params)

        self.assertEqual(
            cfg.observations["actor"].terms["ee_ft_wrench"].params["mode"],
            "raw",
        )

    def test_ft_wrench_observation_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            merge_task_parameter_overrides(
                {"ee_ft_wrench_mode": "unknown"},
                base=load_default_task_parameters(),
            )

    def test_ft_wrench_observation_mode_is_shared_with_critic(self) -> None:
        params = replace(
            load_default_task_parameters(),
            ee_ft_wrench_mode="weight_compensated",
        )
        cfg = kinova_ball_balancing_env_cfg(variant="cartesian", params=params)

        self.assertEqual(
            cfg.observations["critic"].terms["ee_ft_wrench"].params["mode"],
            "weight_compensated",
        )

    def test_pose_only_training_set_removes_all_ball_terms(self) -> None:
        training_set_path = REPO_ROOT / "config/training_sets/cartesian_pose_only.yaml"
        training_set_cfg = _load_training_set(training_set_path)
        base_params = _base_parameters(
            training_set_cfg, training_set_path=training_set_path
        )
        name, presets, overrides = _normalize_run(training_set_cfg["runs"][0], index=0)
        params = _build_run_parameters(
            run_name=name,
            preset_refs=presets,
            overrides=overrides,
            training_set_path=training_set_path,
            base_params=base_params,
        )

        cfg = kinova_ball_balancing_env_cfg(variant="cartesian", params=params)

        self.assertNotIn("ball", cfg.scene.entities)
        self.assertEqual(cfg.metrics, {})
        self.assertEqual(set(cfg.terminations), {"time_out"})
        for terms in (
            cfg.events,
            cfg.rewards,
            cfg.terminations,
            cfg.observations["critic"].terms,
        ):
            self.assertFalse(
                any("ball" in name or "plate_drop" in name for name in terms)
            )


if __name__ == "__main__":
    unittest.main()
