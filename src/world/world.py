from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from .entities import NPC, User, WorldObject, Position
from .spawner import Spawner


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4
    SPEAK = 5


# Movement deltas for each action (x, y)
ACTION_DELTAS: Dict[Action, Tuple[float, float]] = {
    Action.MOVE_UP: (0.0, 0.5),
    Action.MOVE_DOWN: (0.0, -0.5),
    Action.MOVE_LEFT: (-0.5, 0.0),
    Action.MOVE_RIGHT: (0.5, 0.0),
    Action.WAIT: (0.0, 0.0),
    Action.SPEAK: (0.0, 0.0),
}


@dataclass
class WorldConfig:
    size: int = 64
    num_objects: int = 10
    # min_separation and edge_margin are set as a proportion of world size
    # This allows curriculum learning to work with different world sizes
    _min_separation: Optional[float] = None
    _edge_margin: Optional[float] = None
    ticks_per_second: int = 16
    npc_speed: float = 0.5
    collision_radius: float = 0.5

    @property
    def min_separation(self) -> float:
        """Minimum separation between entities, scaled to world size."""
        if self._min_separation is not None:
            return self._min_separation
        # Default: 3.0 for 64x64, scales proportionally
        return 3.0 * (self.size / 64.0)

    @property
    def edge_margin(self) -> float:
        """Edge margin for spawning, scaled to world size."""
        if self._edge_margin is not None:
            return self._edge_margin
        # Default: 2.0 for 64x64, scales proportionally
        return 2.0 * (self.size / 64.0)


class World:
    def __init__(self, config: Optional[WorldConfig] = None, seed: Optional[int] = None):
        self.config = config or WorldConfig()
        self.seed = seed
        self.tick_count: int = 0
        self.last_npc_action: Action = Action.WAIT
        self.last_npc_speech: Optional[str] = None

        self.user: Optional[User] = None
        self.npc: Optional[NPC] = None
        self.objects: List[WorldObject] = []

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.seed = seed

        spawner = Spawner(
            world_size=self.config.size,
            min_separation=self.config.min_separation,
            edge_margin=self.config.edge_margin,
            seed=self.seed,
        )

        self.user, self.npc, self.objects = spawner.spawn_all(
            num_objects=self.config.num_objects
        )
        self.tick_count = 0
        self.last_npc_action = Action.WAIT
        self.last_npc_speech = None

        return self.get_state()

    def _clamp_to_bounds(self, pos: Position) -> Position:
        margin = self.config.edge_margin
        max_coord = self.config.size - margin
        return Position(
            x=max(margin, min(max_coord, pos.x)),
            y=max(margin, min(max_coord, pos.y)),
        )

    def _check_collision(
        self, pos: Position, exclude_entity_id: Optional[str] = None
    ) -> bool:
        radius = self.config.collision_radius

        # Check collision with user
        if exclude_entity_id != self.user.entity_id:
            if pos.distance_to(self.user.position) < radius * 2:
                return True

        # Check collision with NPC
        if exclude_entity_id != self.npc.entity_id:
            if pos.distance_to(self.npc.position) < radius * 2:
                return True

        # Check collision with objects
        for obj in self.objects:
            if exclude_entity_id == obj.entity_id:
                continue
            combined_radius = radius + obj.collision_radius
            if pos.distance_to(obj.position) < combined_radius:
                return True

        return False

    def _apply_movement(
        self, entity_id: str, current_pos: Position, dx: float, dy: float
    ) -> Position:
        new_pos = Position(current_pos.x + dx, current_pos.y + dy)
        new_pos = self._clamp_to_bounds(new_pos)

        if self._check_collision(new_pos, exclude_entity_id=entity_id):
            return current_pos

        return new_pos

    def step(self, npc_action: int, npc_speech: Optional[str] = None) -> Dict:
        action = Action(npc_action)
        self.last_npc_action = action
        self.last_npc_speech = None

        if action == Action.SPEAK:
            self.last_npc_speech = npc_speech or "[no message]"
        elif action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            dx, dy = ACTION_DELTAS[action]
            self.npc.position = self._apply_movement(
                self.npc.entity_id, self.npc.position, dx, dy
            )
        # WAIT action: do nothing

        self.tick_count += 1
        return self.get_state()

    def move_user(self, dx: float, dy: float) -> None:
        self.user.position = self._apply_movement(
            self.user.entity_id, self.user.position, dx, dy
        )

    def get_object_by_id(self, object_id: str) -> Optional[WorldObject]:
        for obj in self.objects:
            if obj.entity_id == object_id:
                return obj
        return None

    def get_nearest_objects(self, pos: Position, n: int = 8) -> List[WorldObject]:
        sorted_objects = sorted(
            self.objects, key=lambda obj: pos.distance_to(obj.position)
        )
        return sorted_objects[:n]

    def get_state(self) -> Dict:
        return {
            "tick": self.tick_count,
            "world_size": self.config.size,
            "user": self.user.to_dict(),
            "npc": self.npc.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "last_action": self.last_npc_action.name.lower(),
            "last_speech": self.last_npc_speech,
        }

    def get_entity_positions(self) -> Dict[str, Position]:
        positions = {
            self.user.entity_id: self.user.position,
            self.npc.entity_id: self.npc.position,
        }
        for obj in self.objects:
            positions[obj.entity_id] = obj.position
        return positions
