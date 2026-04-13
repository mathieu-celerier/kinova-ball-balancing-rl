from pathlib import Path
import os
import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

KINOVA_XML: Path = Path(os.path.dirname(__file__)) / "kinova.xml"
assert KINOVA_XML.exists(), f"XML not found: {KINOVA_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(KINOVA_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        "joint_1": 0.0,
        "joint_2": 0.2618,
        "joint_3": 3.14,
        "joint_4": -2.269,
        "joint_5": 0.0,
        "joint_6": 0.959878729,
        "joint_7": 1.57,
    },
    joint_vel={".*": 0.0},
)

KINOVA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=("joint_1", "joint_2", "joint_3", "joint_4"),
            stiffness=40.0,
            damping=15.0,
            effort_limit=95.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("joint_5", "joint_6", "joint_7"),
            stiffness=15.0,
            damping=8.5,
            effort_limit=45.0,
        ),
    ),
    soft_joint_pos_limit_factor=0.95,
)

KINOVA_ACTION_SCALE: dict[str, float] = {}
for actuator in KINOVA_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    effort_limit = actuator.effort_limit
    stiffness = actuator.stiffness
    target_names = actuator.target_names_expr
    assert effort_limit is not None
    for name in target_names:
        KINOVA_ACTION_SCALE[name] = 0.25 * effort_limit / stiffness

KINOVA_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    articulation=KINOVA_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": KINOVA_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
