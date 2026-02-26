from typing import ClassVar

class mjtGeom:
    mjGEOM_SPHERE: ClassVar[int]

class _Body:
    def add_body(self, *, name: str) -> _Body: ...
    def add_freejoint(self, *, name: str) -> None: ...
    def add_geom(
        self,
        *,
        name: str,
        type: int,
        size: tuple[float, ...],
        mass: float,
        friction: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> None: ...

class MjSpec:
    worldbody: _Body

    @classmethod
    def from_file(cls, filename: str) -> MjSpec: ...
