import random
from typing import List, Optional, Tuple

from .entities import (
    Position,
    NPC,
    User,
    WorldObject,
    ObjectType,
    ObjectColor,
    ObjectShape,
    ObjectSize,
    Enemy,
)


class Spawner:
    def __init__(
        self,
        world_size: int = 64,
        min_separation: float = 3.0,
        edge_margin: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.world_size = world_size
        self.min_separation = min_separation
        self.edge_margin = edge_margin
        self.rng = random.Random(seed)

    def _random_position(self) -> Position:
        margin = self.edge_margin
        x = self.rng.uniform(margin, self.world_size - margin)
        y = self.rng.uniform(margin, self.world_size - margin)
        return Position(x, y)

    def _is_valid_position(
        self, pos: Position, existing: List[Position]
    ) -> bool:
        for other in existing:
            if pos.distance_to(other) < self.min_separation:
                return False
        return True

    def _find_valid_position(
        self, existing: List[Position], max_attempts: int = 100
    ) -> Position:
        for _ in range(max_attempts):
            pos = self._random_position()
            if self._is_valid_position(pos, existing):
                return pos
        raise RuntimeError(
            f"Could not find valid position after {max_attempts} attempts. "
            f"Try reducing min_separation or number of objects."
        )

    def _random_object_attributes(self) -> Tuple[ObjectType, ObjectColor, ObjectShape, ObjectSize]:
        obj_type = self.rng.choice(list(ObjectType))
        color = self.rng.choice(list(ObjectColor))
        shape = self.rng.choice(list(ObjectShape))
        size = self.rng.choice(list(ObjectSize))
        return obj_type, color, shape, size

    def spawn_npc(self, existing_positions: List[Position]) -> NPC:
        pos = self._find_valid_position(existing_positions)
        return NPC(position=pos, entity_id="npc")

    def spawn_user(self, existing_positions: List[Position]) -> User:
        pos = self._find_valid_position(existing_positions)
        return User(position=pos, entity_id="user")

    def spawn_object(
        self, object_id: str, existing_positions: List[Position]
    ) -> WorldObject:
        pos = self._find_valid_position(existing_positions)
        obj_type, color, shape, size = self._random_object_attributes()
        return WorldObject(
            position=pos,
            entity_id=object_id,
            object_type=obj_type,
            color=color,
            shape=shape,
            size=size,
        )

    def spawn_enemy(
        self, existing_positions: List[Position], enemy_id: str = "enemy_0",
        enemy_type: str = "enemy"
    ) -> Optional[Enemy]:
        """Spawn an enemy at a valid position. Returns None if no valid position found."""
        try:
            pos = self._find_valid_position(existing_positions)
            return Enemy(
                position=pos,
                entity_id=enemy_id,
                enemy_type=enemy_type,
            )
        except RuntimeError:
            return None

    def spawn_all(
        self, num_objects: int = 10
    ) -> Tuple[User, NPC, List[WorldObject]]:
        positions: List[Position] = []

        user = self.spawn_user(positions)
        positions.append(user.position)

        npc = self.spawn_npc(positions)
        positions.append(npc.position)

        objects: List[WorldObject] = []
        for i in range(num_objects):
            obj = self.spawn_object(f"obj_{i}", positions)
            objects.append(obj)
            positions.append(obj.position)

        return user, npc, objects
