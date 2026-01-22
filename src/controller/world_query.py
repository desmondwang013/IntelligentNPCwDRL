"""
WorldQuery provides read-only access to world state for target resolution.

This class isolates world state queries from the NPCController, allowing:
- Clean separation of concerns
- Easy testing with mock world states
- Future extensibility (e.g., caching, spatial indexing)
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class ObjectMatch:
    """Represents a matched world object."""
    entity_id: str
    position: Tuple[float, float]
    color: str
    shape: str
    size: str

    def describe(self) -> str:
        """Human-readable description (e.g., 'large red circle')."""
        parts = []
        if self.size != "medium":  # Only mention non-default size
            parts.append(self.size)
        parts.append(self.color)
        parts.append(self.shape)
        return " ".join(parts)


@dataclass
class EnemyMatch:
    """Represents a matched enemy entity."""
    entity_id: str
    position: Tuple[float, float]
    enemy_type: str
    health: int
    max_health: int
    is_alive: bool
    is_aggro: bool

    def describe(self) -> str:
        """Human-readable description (e.g., 'enemy (3/5 HP)')."""
        status = f"{self.health}/{self.max_health} HP"
        if not self.is_alive:
            status = "dead"
        elif self.is_aggro:
            status += ", aggro"
        return f"{self.enemy_type} ({status})"


class WorldQuery:
    """
    Read-only interface to query world state.

    Used by NPCController to resolve target descriptions to entity IDs.
    Takes world state dict (from World.get_state()) as input.

    Usage:
        query = WorldQuery(world.get_state())
        matches = query.find_objects(color="red", shape="circle")
        user_pos = query.get_user_position()
    """

    def __init__(self, world_state: Dict[str, Any]):
        """
        Initialize with current world state.

        Args:
            world_state: Dict from World.get_state()
        """
        self._state = world_state

    def update(self, world_state: Dict[str, Any]) -> None:
        """Update with new world state."""
        self._state = world_state

    # =========================================================================
    # Object Queries
    # =========================================================================

    def find_objects(
        self,
        color: Optional[str] = None,
        shape: Optional[str] = None,
        size: Optional[str] = None,
    ) -> List[ObjectMatch]:
        """
        Find objects matching the given attributes.

        Args:
            color: red/blue/green/yellow/purple/orange (or None for any)
            shape: circle/triangle/square/diamond (or None for any)
            size: small/medium/large (or None for any)

        Returns:
            List of matching ObjectMatch instances
        """
        matches = []

        for obj in self._state.get("objects", []):
            # Check each attribute if specified
            if color is not None and obj.get("color") != color.lower():
                continue
            if shape is not None and obj.get("shape") != shape.lower():
                continue
            if size is not None and obj.get("size") != size.lower():
                continue

            # Object matches all specified criteria
            pos = obj["position"]
            matches.append(ObjectMatch(
                entity_id=obj["entity_id"],
                position=(pos["x"], pos["y"]),
                color=obj["color"],
                shape=obj["shape"],
                size=obj["size"],
            ))

        return matches

    def get_object_by_id(self, entity_id: str) -> Optional[ObjectMatch]:
        """Get a specific object by its entity ID."""
        for obj in self._state.get("objects", []):
            if obj["entity_id"] == entity_id:
                pos = obj["position"]
                return ObjectMatch(
                    entity_id=obj["entity_id"],
                    position=(pos["x"], pos["y"]),
                    color=obj["color"],
                    shape=obj["shape"],
                    size=obj["size"],
                )
        return None

    def get_all_objects(self) -> List[ObjectMatch]:
        """Get all objects in the world."""
        return self.find_objects()  # No filters = all objects

    # =========================================================================
    # Enemy Queries
    # =========================================================================

    def find_enemies(
        self,
        alive_only: bool = True,
        enemy_type: Optional[str] = None,
    ) -> List[EnemyMatch]:
        """
        Find enemies in the world.

        Args:
            alive_only: Only return living enemies
            enemy_type: Filter by enemy type (e.g., "goblin", or None for any)

        Returns:
            List of matching EnemyMatch instances
        """
        matches = []

        for enemy in self._state.get("enemies", []):
            if alive_only and not enemy.get("is_alive", True):
                continue
            if enemy_type is not None and enemy.get("enemy_type") != enemy_type:
                continue

            pos = enemy["position"]
            matches.append(EnemyMatch(
                entity_id=enemy["entity_id"],
                position=(pos["x"], pos["y"]),
                enemy_type=enemy.get("enemy_type", "enemy"),
                health=enemy.get("health", 0),
                max_health=enemy.get("max_health", 5),
                is_alive=enemy.get("is_alive", True),
                is_aggro=enemy.get("is_aggro", False),
            ))

        return matches

    def get_enemy_by_id(self, entity_id: str) -> Optional[EnemyMatch]:
        """Get a specific enemy by its entity ID."""
        for enemy in self._state.get("enemies", []):
            if enemy["entity_id"] == entity_id:
                pos = enemy["position"]
                return EnemyMatch(
                    entity_id=enemy["entity_id"],
                    position=(pos["x"], pos["y"]),
                    enemy_type=enemy.get("enemy_type", "enemy"),
                    health=enemy.get("health", 0),
                    max_health=enemy.get("max_health", 5),
                    is_alive=enemy.get("is_alive", True),
                    is_aggro=enemy.get("is_aggro", False),
                )
        return None

    def get_nearest_enemy(self, alive_only: bool = True) -> Optional[EnemyMatch]:
        """
        Find the nearest enemy to the NPC.

        Args:
            alive_only: Only consider living enemies

        Returns:
            The closest enemy, or None if no enemies
        """
        enemies = self.find_enemies(alive_only=alive_only)
        if not enemies:
            return None

        npc_pos = self.get_npc_position()
        if npc_pos is None:
            return enemies[0]  # Can't determine nearest, return first

        enemies.sort(key=lambda e: self.distance_between(npc_pos, e.position))
        return enemies[0]

    def get_alive_enemy_count(self) -> int:
        """Get the number of living enemies."""
        return len(self.find_enemies(alive_only=True))

    # =========================================================================
    # Entity Position Queries
    # =========================================================================

    def get_user_position(self) -> Optional[Tuple[float, float]]:
        """Get the user's current position."""
        user = self._state.get("user")
        if user is None:
            return None
        pos = user["position"]
        return (pos["x"], pos["y"])

    def get_npc_position(self) -> Optional[Tuple[float, float]]:
        """Get the NPC's current position."""
        npc = self._state.get("npc")
        if npc is None:
            return None
        pos = npc["position"]
        return (pos["x"], pos["y"])

    def get_entity_position(self, entity_id: str) -> Optional[Tuple[float, float]]:
        """
        Get position of any entity by ID.

        Handles special IDs:
        - "user" -> user position
        - "npc" -> NPC position
        - Otherwise -> check objects, then enemies
        """
        if entity_id == "user":
            return self.get_user_position()
        if entity_id == "npc":
            return self.get_npc_position()

        # Check objects
        obj = self.get_object_by_id(entity_id)
        if obj:
            return obj.position

        # Check enemies
        enemy = self.get_enemy_by_id(entity_id)
        if enemy:
            return enemy.position

        return None

    # =========================================================================
    # World Info
    # =========================================================================

    def get_world_size(self) -> int:
        """Get the world size (assumed square)."""
        return self._state.get("world_size", 64)

    def get_tick(self) -> int:
        """Get current tick count."""
        return self._state.get("tick", 0)

    # =========================================================================
    # Distance Utilities
    # =========================================================================

    def distance_between(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) ** 0.5

    def distance_to_npc(self, position: Tuple[float, float]) -> Optional[float]:
        """Calculate distance from a position to the NPC."""
        npc_pos = self.get_npc_position()
        if npc_pos is None:
            return None
        return self.distance_between(position, npc_pos)

    def get_nearest_object(
        self,
        color: Optional[str] = None,
        shape: Optional[str] = None,
        size: Optional[str] = None,
    ) -> Optional[ObjectMatch]:
        """
        Find the nearest object matching the given attributes.

        Returns:
            The closest matching object, or None if no matches
        """
        matches = self.find_objects(color=color, shape=shape, size=size)
        if not matches:
            return None

        npc_pos = self.get_npc_position()
        if npc_pos is None:
            return matches[0]  # Can't determine nearest, return first

        # Sort by distance and return nearest
        matches.sort(key=lambda m: self.distance_between(npc_pos, m.position))
        return matches[0]
