from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from .entities import NPC, User, WorldObject, Position, Enemy, CombatStyle
from .spawner import Spawner


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4
    ATTACK = 5
    SPEAK = 6


# Movement deltas for each action (x, y)
ACTION_DELTAS: Dict[Action, Tuple[float, float]] = {
    Action.MOVE_UP: (0.0, 0.5),
    Action.MOVE_DOWN: (0.0, -0.5),
    Action.MOVE_LEFT: (-0.5, 0.0),
    Action.MOVE_RIGHT: (0.5, 0.0),
    Action.WAIT: (0.0, 0.0),
    Action.ATTACK: (0.0, 0.0),
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
    collision_radius: float = 0.1  # Same as small object radius

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
        self.enemies: List[Enemy] = []

        # Combat tracking for current step
        self.last_damage_dealt: int = 0
        self.last_damage_taken: int = 0
        self.last_enemy_killed: Optional[str] = None

        self.reset()

    def reset(self, seed: Optional[int] = None, num_enemies: int = 0) -> Dict:
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

        # Reset NPC combat state
        self.npc.reset_combat_state()

        # Spawn enemies
        self.enemies = []
        for i in range(num_enemies):
            enemy = spawner.spawn_enemy(
                existing_positions=self._get_all_positions(),
                enemy_id=f"enemy_{i}"
            )
            if enemy:
                self.enemies.append(enemy)

        # Reset tracking
        self.tick_count = 0
        self.last_npc_action = Action.WAIT
        self.last_npc_speech = None
        self.last_damage_dealt = 0
        self.last_damage_taken = 0
        self.last_enemy_killed = None

        return self.get_state()

    def _get_all_positions(self) -> List[Position]:
        """Get all entity positions for spawn checking."""
        positions = [self.user.position, self.npc.position]
        positions.extend(obj.position for obj in self.objects)
        positions.extend(enemy.position for enemy in self.enemies)
        return positions

    def _clamp_to_bounds(self, pos: Position) -> Position:
        margin = self.config.edge_margin
        max_coord = self.config.size - margin
        return Position(
            x=max(margin, min(max_coord, pos.x)),
            y=max(margin, min(max_coord, pos.y)),
        )

    def _check_collision(
        self, pos: Position, exclude_entity_id: Optional[str] = None,
        skip_user: bool = False
    ) -> bool:
        radius = self.config.collision_radius

        # Check collision with user (optionally skip for NPC training)
        if not skip_user and exclude_entity_id != self.user.entity_id:
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

        # Check collision with enemies
        for enemy in self.enemies:
            if exclude_entity_id == enemy.entity_id:
                continue
            if not enemy.is_alive:
                continue  # Dead enemies don't block movement
            combined_radius = radius + enemy.collision_radius
            if pos.distance_to(enemy.position) < combined_radius:
                return True

        return False

    def _apply_movement(
        self, entity_id: str, current_pos: Position, dx: float, dy: float,
        skip_user_collision: bool = False
    ) -> Position:
        new_pos = Position(current_pos.x + dx, current_pos.y + dy)
        new_pos = self._clamp_to_bounds(new_pos)

        if self._check_collision(new_pos, exclude_entity_id=entity_id, skip_user=skip_user_collision):
            return current_pos

        return new_pos

    def _find_nearest_enemy(self, alive_only: bool = True) -> Optional[Enemy]:
        """Find the nearest enemy to the NPC."""
        nearest = None
        min_dist = float('inf')
        for enemy in self.enemies:
            if alive_only and not enemy.is_alive:
                continue
            dist = self.npc.position.distance_to(enemy.position)
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
        return nearest

    def _process_npc_attack(self) -> None:
        """Process NPC attack action - attacks nearest enemy in range."""
        if not self.npc.can_attack:
            return

        nearest = self._find_nearest_enemy(alive_only=True)
        if nearest is None:
            return

        dist = self.npc.position.distance_to(nearest.position)
        if dist <= self.npc.attack_range:
            # Attack hits
            damage = nearest.take_damage(self.npc.attack_damage)
            self.last_damage_dealt = damage
            self.npc.start_attack_cooldown()

            if not nearest.is_alive:
                self.last_enemy_killed = nearest.entity_id

    def _process_enemy_ai(self) -> None:
        """Process enemy AI - aggro'd enemies move toward and attack NPC."""
        for enemy in self.enemies:
            if not enemy.is_alive or not enemy.is_aggro:
                continue

            # Move toward NPC
            dx = self.npc.position.x - enemy.position.x
            dy = self.npc.position.y - enemy.position.y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > 0:
                # Normalize and scale by enemy speed
                move_x = (dx / dist) * enemy.speed
                move_y = (dy / dist) * enemy.speed

                # Apply movement (enemies collide with everything except user)
                enemy.position = self._apply_movement(
                    enemy.entity_id, enemy.position, move_x, move_y,
                    skip_user_collision=True
                )

            # Attack NPC if in range and off cooldown
            if enemy.can_attack:
                attack_dist = self.npc.position.distance_to(enemy.position)
                if attack_dist <= enemy.attack_range:
                    damage = self.npc.take_damage(enemy.attack_damage)
                    self.last_damage_taken += damage
                    enemy.start_attack_cooldown()

    def step(self, npc_action: int, npc_speech: Optional[str] = None) -> Dict:
        action = Action(npc_action)
        self.last_npc_action = action
        self.last_npc_speech = None

        # Reset per-step combat tracking
        self.last_damage_dealt = 0
        self.last_damage_taken = 0
        self.last_enemy_killed = None

        if action == Action.SPEAK:
            self.last_npc_speech = npc_speech or "[no message]"
        elif action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            dx, dy = ACTION_DELTAS[action]
            # NPC doesn't collide with User for RL training
            # (User is invisible obstacle that confuses the agent)
            self.npc.position = self._apply_movement(
                self.npc.entity_id, self.npc.position, dx, dy,
                skip_user_collision=True
            )
        elif action == Action.ATTACK:
            self._process_npc_attack()
        # WAIT action: do nothing

        # Process enemy AI (move and attack)
        self._process_enemy_ai()

        # Tick cooldowns
        self.npc.tick_cooldown()
        for enemy in self.enemies:
            enemy.tick_cooldown()

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
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "last_action": self.last_npc_action.name.lower(),
            "last_speech": self.last_npc_speech,
            # Combat info for this step
            "last_damage_dealt": self.last_damage_dealt,
            "last_damage_taken": self.last_damage_taken,
            "last_enemy_killed": self.last_enemy_killed,
        }

    def get_entity_positions(self) -> Dict[str, Position]:
        positions = {
            self.user.entity_id: self.user.position,
            self.npc.entity_id: self.npc.position,
        }
        for obj in self.objects:
            positions[obj.entity_id] = obj.position
        for enemy in self.enemies:
            positions[enemy.entity_id] = enemy.position
        return positions

    def get_enemy_by_id(self, enemy_id: str) -> Optional[Enemy]:
        """Get an enemy by its entity_id."""
        for enemy in self.enemies:
            if enemy.entity_id == enemy_id:
                return enemy
        return None

    def get_alive_enemies(self) -> List[Enemy]:
        """Get all living enemies."""
        return [e for e in self.enemies if e.is_alive]

    def get_nearest_enemy(self, pos: Position, alive_only: bool = True) -> Optional[Enemy]:
        """Get the nearest enemy to a position."""
        candidates = self.get_alive_enemies() if alive_only else self.enemies
        if not candidates:
            return None
        return min(candidates, key=lambda e: pos.distance_to(e.position))
