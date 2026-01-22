"""Combat observation builder for RL training."""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from src.world import World
from src.world.entities import Enemy, CombatStyle


@dataclass
class CombatObservationConfig:
    """Configuration for combat observations."""
    max_enemies: int = 4  # Maximum enemies to include in observation
    world_size: float = 64.0  # For normalization


class CombatObservation:
    """
    Builds observation vectors for combat RL training.

    Observation space (total: 6 + max_enemies*5 + 3 = 29 for max_enemies=4):
    - NPC state (6):
        - position x, y (normalized 0-1)
        - health ratio (0-1)
        - can_attack flag (0 or 1)
        - cooldown ratio (0-1, 0 means ready)
        - num_alive_enemies (normalized by max_enemies)
    - Per enemy (5 each, up to max_enemies):
        - relative x, y (normalized by world_size, centered on NPC)
        - distance (normalized by world diagonal)
        - health ratio (0-1)
        - is_aggro flag (0 or 1)
    - Combat style (3, one-hot):
        - aggressive, balanced, passive
    """

    def __init__(self, config: Optional[CombatObservationConfig] = None):
        self.config = config or CombatObservationConfig()
        self.world_diagonal = self.config.world_size * np.sqrt(2)

    @property
    def observation_size(self) -> int:
        """Total observation vector size."""
        return 6 + (self.config.max_enemies * 5) + 3

    def build(self, world: World) -> np.ndarray:
        """Build observation vector from world state."""
        obs = []

        # NPC state
        npc = world.npc
        world_size = world.config.size

        # Position (normalized 0-1)
        obs.append(npc.position.x / world_size)
        obs.append(npc.position.y / world_size)

        # Health and combat state
        obs.append(npc.health_ratio)
        obs.append(1.0 if npc.can_attack else 0.0)
        obs.append(npc.current_cooldown / npc.attack_cooldown if npc.attack_cooldown > 0 else 0.0)

        # Number of alive enemies (normalized)
        alive_enemies = world.get_alive_enemies()
        obs.append(len(alive_enemies) / max(self.config.max_enemies, 1))

        # Sort enemies by distance
        sorted_enemies = sorted(
            alive_enemies,
            key=lambda e: npc.position.distance_to(e.position)
        )

        # Pad or truncate to max_enemies
        for i in range(self.config.max_enemies):
            if i < len(sorted_enemies):
                enemy = sorted_enemies[i]
                # Relative position (centered on NPC)
                rel_x = (enemy.position.x - npc.position.x) / world_size
                rel_y = (enemy.position.y - npc.position.y) / world_size
                dist = npc.position.distance_to(enemy.position) / self.world_diagonal
                health = enemy.health_ratio
                aggro = 1.0 if enemy.is_aggro else 0.0
            else:
                # Padding for missing enemies (far away, dead, not threatening)
                rel_x, rel_y = 1.0, 1.0  # Far away
                dist = 1.0
                health = 0.0
                aggro = 0.0

            obs.extend([rel_x, rel_y, dist, health, aggro])

        # Combat style (one-hot)
        style_vec = [0.0, 0.0, 0.0]
        if npc.combat_style == CombatStyle.AGGRESSIVE:
            style_vec[0] = 1.0
        elif npc.combat_style == CombatStyle.BALANCED:
            style_vec[1] = 1.0
        elif npc.combat_style == CombatStyle.PASSIVE:
            style_vec[2] = 1.0
        obs.extend(style_vec)

        return np.array(obs, dtype=np.float32)

    def get_observation_labels(self) -> List[str]:
        """Get human-readable labels for each observation dimension."""
        labels = [
            "npc_x", "npc_y", "npc_health", "npc_can_attack",
            "npc_cooldown", "num_enemies"
        ]
        for i in range(self.config.max_enemies):
            labels.extend([
                f"enemy{i}_rel_x", f"enemy{i}_rel_y", f"enemy{i}_dist",
                f"enemy{i}_health", f"enemy{i}_aggro"
            ])
        labels.extend(["style_aggressive", "style_balanced", "style_passive"])
        return labels
