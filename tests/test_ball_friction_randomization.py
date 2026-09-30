from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mjlab_kinova.tasks import ball_balancing_mdp as bb_mdp

# Global geom layout, mirroring the compiled model: 0 is the terrain, 12 the racquet collision
# geom, 13 the ball. The entity-local indices differ (ball 0, racquet 11) and that gap is exactly
# what the original code fell into, so the fake reproduces it rather than smoothing it over.
TERRAIN, RACQUET, BALL = 0, 12, 13
NGEOM = 14
TERRAIN_FRICTION = [1.0, 0.005, 0.0001]
CONTACT_FRICTION = [1.0, 0.2, 0.0005]


def _fake_env(num_envs: int = 4):
    default = torch.tensor([TERRAIN_FRICTION] * NGEOM, dtype=torch.float32)
    default[RACQUET] = torch.tensor(CONTACT_FRICTION)
    default[BALL] = torch.tensor(CONTACT_FRICTION)
    friction = default.unsqueeze(0).repeat(num_envs, 1, 1)

    ball = SimpleNamespace(
        find_geoms=lambda name, preserve_order=True: ([0], [name]),
        indexing=SimpleNamespace(geom_ids=torch.tensor([BALL])),
    )
    robot = SimpleNamespace(
        find_geoms=lambda name, preserve_order=True: ([11], [name]),
        indexing=SimpleNamespace(geom_ids=torch.arange(1, 13)),  # local 11 -> global 12
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"ball": ball, "robot": robot},
        sim=SimpleNamespace(
            model=SimpleNamespace(geom_friction=friction),
            get_default_field=lambda field: default,
        ),
    )


def _contact(env):
    """What MuJoCo resolves for the pair: element-wise max of the two geoms."""
    f = env.sim.model.geom_friction
    return torch.maximum(f[:, BALL], f[:, RACQUET])


class BallFrictionRandomizationTest(unittest.TestCase):
    def test_writes_the_ball_not_the_terrain(self) -> None:
        """find_geoms returns a local index; using it directly wrote geom 0, the terrain."""
        env = _fake_env()
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(0.4, 0.4), operation="scale"
        )
        f = env.sim.model.geom_friction
        torch.testing.assert_close(f[:, BALL], (torch.tensor(CONTACT_FRICTION) * 0.4).expand(env.num_envs, 3))
        torch.testing.assert_close(f[:, TERRAIN], torch.tensor(TERRAIN_FRICTION).expand(env.num_envs, 3))

    def test_scaling_below_one_lowers_the_contact(self) -> None:
        env = _fake_env()
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(0.4, 0.4), operation="scale"
        )
        want = torch.tensor([0.4 * v for v in CONTACT_FRICTION])
        torch.testing.assert_close(_contact(env), want.expand(env.num_envs, 3))

    def test_both_geoms_get_the_same_draw(self) -> None:
        env = _fake_env()
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(0.3, 1.2), operation="scale"
        )
        f = env.sim.model.geom_friction
        torch.testing.assert_close(f[:, BALL], f[:, RACQUET])

    def test_sample_varies_across_environments(self) -> None:
        env = _fake_env(num_envs=64)
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(0.3, 1.2), operation="scale"
        )
        sliding = _contact(env)[:, 0]
        self.assertGreater(sliding.std().item(), 0.0)
        self.assertGreaterEqual(sliding.min().item(), 0.3 - 1e-6)
        self.assertLessEqual(sliding.max().item(), 1.2 + 1e-6)

    def test_ball_only_mode_leaves_the_contact_pinned(self) -> None:
        """Documents why writing one geom is not enough: the max keeps the racquet's value."""
        env = _fake_env()
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(0.4, 0.4),
            operation="scale", racquet_asset_name=None,
        )
        torch.testing.assert_close(_contact(env), torch.tensor(CONTACT_FRICTION).expand(env.num_envs, 3))

    def test_env_ids_subset_leaves_other_envs_untouched(self) -> None:
        env = _fake_env(num_envs=4)
        bb_mdp.randomize_ball_friction(
            env, torch.tensor([1, 3]), "ball", "ball_geom",
            friction_scale=(0.4, 0.4), operation="scale",
        )
        got = _contact(env)
        nominal = torch.tensor(CONTACT_FRICTION)
        torch.testing.assert_close(got[0], nominal)
        torch.testing.assert_close(got[2], nominal)
        torch.testing.assert_close(got[1], nominal * 0.4)
        torch.testing.assert_close(got[3], nominal * 0.4)


    def test_component_scales_are_applied_per_component(self) -> None:
        """One shared range cannot suit all three: sliding is dimensionless, the others lengths."""
        env = _fake_env(num_envs=64)
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(9.0, 9.0), operation="scale",
            component_scales=((0.5, 1.0), (0.5, 1.0), (0.67, 3.33)),
        )
        c = _contact(env)
        self.assertGreaterEqual(c[:, 0].min().item(), 0.5 * CONTACT_FRICTION[0] - 1e-6)
        self.assertLessEqual(c[:, 0].max().item(), 1.0 * CONTACT_FRICTION[0] + 1e-6)
        # rolling gets its own, much wider band - and crucially not the sliding one
        self.assertGreaterEqual(c[:, 2].min().item(), 0.67 * CONTACT_FRICTION[2] - 1e-9)
        self.assertLessEqual(c[:, 2].max().item(), 3.33 * CONTACT_FRICTION[2] + 1e-9)
        self.assertGreater(c[:, 2].max().item(), CONTACT_FRICTION[2])  # can exceed nominal

    def test_component_scales_override_friction_scale(self) -> None:
        env = _fake_env()
        bb_mdp.randomize_ball_friction(
            env, None, "ball", "ball_geom", friction_scale=(9.0, 9.0), operation="scale",
            component_scales=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        )
        torch.testing.assert_close(_contact(env), torch.tensor(CONTACT_FRICTION).expand(env.num_envs, 3))

    def test_wrong_component_count_raises(self) -> None:
        env = _fake_env()
        with self.assertRaises(ValueError):
            bb_mdp.randomize_ball_friction(
                env, None, "ball", "ball_geom", friction_scale=(1.0, 1.0),
                operation="scale", component_scales=((0.5, 1.0), (0.5, 1.0)),
            )


if __name__ == "__main__":
    unittest.main()
