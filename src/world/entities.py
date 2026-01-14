from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ObjectType(Enum):
    FIXED = auto()
    LIGHT = auto()
    HEAVY = auto()


class ObjectColor(Enum):
    RED = auto()
    BLUE = auto()
    GREEN = auto()
    YELLOW = auto()
    PURPLE = auto()
    ORANGE = auto()


class ObjectShape(Enum):
    CIRCLE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    DIAMOND = auto()


class ObjectSize(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def copy(self) -> "Position":
        return Position(self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class Entity:
    position: Position
    entity_id: str

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "position": self.position.to_dict(),
        }


@dataclass
class NPC(Entity):
    speed: float = 0.5  # world units per tick

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["type"] = "npc"
        base["speed"] = self.speed
        return base


@dataclass
class User(Entity):
    speed: float = 0.5  # world units per tick

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["type"] = "user"
        base["speed"] = self.speed
        return base


@dataclass
class WorldObject(Entity):
    object_type: ObjectType = ObjectType.LIGHT
    color: ObjectColor = ObjectColor.BLUE
    shape: ObjectShape = ObjectShape.CIRCLE
    size: ObjectSize = ObjectSize.MEDIUM

    @property
    def is_movable(self) -> bool:
        return self.object_type != ObjectType.FIXED

    @property
    def collision_radius(self) -> float:
        radii = {
            ObjectSize.SMALL: 0.5,
            ObjectSize.MEDIUM: 1.0,
            ObjectSize.LARGE: 1.5,
        }
        return radii[self.size]

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["type"] = "object"
        base["object_type"] = self.object_type.name.lower()
        base["color"] = self.color.name.lower()
        base["shape"] = self.shape.name.lower()
        base["size"] = self.size.name.lower()
        base["is_movable"] = self.is_movable
        base["collision_radius"] = self.collision_radius
        return base
