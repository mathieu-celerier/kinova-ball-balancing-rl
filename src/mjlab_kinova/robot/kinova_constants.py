import os
import tempfile
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

KINOVA_XML: Path = Path(os.path.dirname(__file__)) / "kinova.xml"
assert KINOVA_XML.exists(), f"XML not found: {KINOVA_XML}"

_DEBUG_RACQUET_FRAME_ENV_VAR = "MJLAB_KINOVA_DEBUG_RACQUET_FRAME"


def _debug_racquet_frame_enabled() -> bool:
    return os.getenv(_DEBUG_RACQUET_FRAME_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _append_racquet_frame_debug_axes(root: ET.Element) -> None:
    racquet_frame = _find_body(root, "racquet_frame")
    if racquet_frame is None:
        raise RuntimeError("Could not find racquet_frame in Kinova XML")

    debug_body = ET.Element("body", {"name": "racquet_frame_debug"})
    axes = (
        ("racquet_frame_debug_x", "0 0 0 0.08 0 0", "0.004", "0.95 0.2 0.2 0.9"),
        ("racquet_frame_debug_y", "0 0 0 0 0.08 0", "0.004", "0.2 0.9 0.2 0.9"),
        ("racquet_frame_debug_z", "0 0 0 0 0 0.08", "0.004", "0.2 0.45 0.95 0.9"),
    )
    for name, fromto, size, rgba in axes:
        ET.SubElement(
            debug_body,
            "geom",
            {
                "name": name,
                "type": "capsule",
                "fromto": fromto,
                "size": size,
                "density": "0",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )
    racquet_frame.append(debug_body)


@lru_cache(maxsize=1)
def _kinova_xml_path() -> Path:
    if not _debug_racquet_frame_enabled():
        return KINOVA_XML

    root = ET.parse(KINOVA_XML).getroot()
    _append_racquet_frame_debug_axes(root)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".xml",
        prefix="mjlab-kinova-debug-",
        dir=KINOVA_XML.parent,
        delete=False,
    )
    with handle:
        ET.ElementTree(root).write(handle, encoding="utf-8", xml_declaration=True)
    return Path(handle.name)


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_kinova_xml_path()))


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

KINOVA_EFFORT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinMotorActuatorCfg(
            target_names_expr=("joint_1", "joint_2", "joint_3", "joint_4"),
            effort_limit=95.0,
        ),
        BuiltinMotorActuatorCfg(
            target_names_expr=("joint_5", "joint_6", "joint_7"),
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

KINOVA_EFFORT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    articulation=KINOVA_EFFORT_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": KINOVA_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
