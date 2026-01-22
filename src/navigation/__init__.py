"""
Navigation module for intelligent pathfinding and movement.

This module provides pluggable navigation strategies:
- HybridNavigator: A* pathfinding + RL motor control (recommended)
- DirectNavigator: No pathfinding, direct to target (baseline)
- (Future) HierarchicalNavigator: High-level RL + low-level RL

All navigators implement the same interface, allowing easy swapping
between strategies without changing the RL agent or environment code.

Example usage:
    from src.navigation import HybridNavigator, HybridNavigatorConfig

    # Create navigator
    config = HybridNavigatorConfig(waypoint_threshold=1.0)
    navigator = HybridNavigator(config)

    # Set goal
    success = navigator.set_goal(npc_x, npc_y, target_x, target_y, world_state)

    # Get current target for RL agent
    target = navigator.get_current_target()

    # Update after NPC moves
    navigator.update(new_npc_x, new_npc_y, world_state)
"""

# Base classes and interfaces
from .base import (
    Navigator,
    NavigationTarget,
    NavigationState,
    DirectNavigator,
)

# Obstacle map for pathfinding
from .obstacle_map import (
    ObstacleMap,
    ObstacleMapConfig,
)

# A* pathfinding
from .pathfinding import (
    AStarPathfinder,
    PathfindingConfig,
)

# Hybrid navigator (A* + RL)
from .hybrid_navigator import (
    HybridNavigator,
    HybridNavigatorConfig,
)

__all__ = [
    # Base
    "Navigator",
    "NavigationTarget",
    "NavigationState",
    "DirectNavigator",
    # Obstacle map
    "ObstacleMap",
    "ObstacleMapConfig",
    # Pathfinding
    "AStarPathfinder",
    "PathfindingConfig",
    # Hybrid navigator
    "HybridNavigator",
    "HybridNavigatorConfig",
]
