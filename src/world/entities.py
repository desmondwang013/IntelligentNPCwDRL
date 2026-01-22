from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ObjectType(Enum):
    FIXED = auto()
    LIGHT = auto()
    HEAVY = auto()


class CombatStyle(Enum):
    """Combat behavior styles for NPC."""
    AGGRESSIVE = auto()  # Minimize distance, accept damage
    BALANCED = auto()    # Default, mix of offense/defense
    PASSIVE = auto()     # Avoid damage, maintain distance


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

    # Combat stats
    max_health: int = 10
    health: int = 10
    attack_damage: int = 1
    attack_range: float = 0.5  # world units
    attack_cooldown: int = 8   # ticks between attacks

    # Combat state
    current_cooldown: int = 0  # ticks until can attack again
    combat_style: CombatStyle = CombatStyle.BALANCED

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def can_attack(self) -> bool:
        return self.current_cooldown == 0 and self.is_alive

    @property
    def health_ratio(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0.0

    def take_damage(self, damage: int) -> int:
        """Apply damage and return actual damage taken."""
        actual_damage = min(damage, self.health)
        self.health -= actual_damage
        return actual_damage

    def tick_cooldown(self) -> None:
        """Reduce cooldown by 1 tick."""
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

    def start_attack_cooldown(self) -> None:
        """Start the attack cooldown after attacking."""
        self.current_cooldown = self.attack_cooldown

    def reset_combat_state(self) -> None:
        """Reset health and cooldown to initial state."""
        self.health = self.max_health
        self.current_cooldown = 0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["type"] = "npc"
        base["speed"] = self.speed
        base["health"] = self.health
        base["max_health"] = self.max_health
        base["health_ratio"] = self.health_ratio
        base["attack_damage"] = self.attack_damage
        base["attack_range"] = self.attack_range
        base["can_attack"] = self.can_attack
        base["is_alive"] = self.is_alive
        base["combat_style"] = self.combat_style.name.lower()
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
        # Reduced radii to ensure min_separation > max_combined_collision
        # With NPC radius 0.1, max combined = 0.1 + 0.2 = 0.3
        # min_separation for 8x8 world = 0.375, so spawning is safe
        radii = {
            ObjectSize.SMALL: 0.05,
            ObjectSize.MEDIUM: 0.1,
            ObjectSize.LARGE: 0.2,
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


@dataclass
class Enemy(Entity):
    """
    Enemy entity that can be attacked by NPC.

    Enemies are passive until attacked, then use simple rule-based AI
    to approach and attack the NPC.
    """
    speed: float = 0.4  # Slower than NPC (0.5)

    # Combat stats
    max_health: int = 5
    health: int = 5
    attack_damage: int = 1
    attack_range: float = 0.4  # Shorter than NPC (0.5)
    attack_cooldown: int = 8   # ticks between attacks

    # Combat state
    current_cooldown: int = 0
    is_aggro: bool = False  # Only attacks when aggro'd (after being hit)

    # Visual identification (for LLM/user)
    enemy_type: str = "enemy"  # Could be "goblin", "skeleton", etc.

    @property
    def collision_radius(self) -> float:
        return 0.1  # Same as medium object

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def can_attack(self) -> bool:
        return self.current_cooldown == 0 and self.is_alive and self.is_aggro

    @property
    def health_ratio(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0.0

    def take_damage(self, damage: int) -> int:
        """Apply damage, aggro the enemy, and return actual damage taken."""
        actual_damage = min(damage, self.health)
        self.health -= actual_damage
        self.is_aggro = True  # Become aggressive when attacked
        return actual_damage

    def tick_cooldown(self) -> None:
        """Reduce cooldown by 1 tick."""
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

    def start_attack_cooldown(self) -> None:
        """Start the attack cooldown after attacking."""
        self.current_cooldown = self.attack_cooldown

    def reset_combat_state(self) -> None:
        """Reset health, cooldown, and aggro state."""
        self.health = self.max_health
        self.current_cooldown = 0
        self.is_aggro = False

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["type"] = "enemy"
        base["enemy_type"] = self.enemy_type
        base["speed"] = self.speed
        base["health"] = self.health
        base["max_health"] = self.max_health
        base["health_ratio"] = self.health_ratio
        base["attack_damage"] = self.attack_damage
        base["attack_range"] = self.attack_range
        base["can_attack"] = self.can_attack
        base["is_alive"] = self.is_alive
        base["is_aggro"] = self.is_aggro
        base["collision_radius"] = self.collision_radius
        return base
