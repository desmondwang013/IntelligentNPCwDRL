"""
Hybrid navigator combining A* pathfinding with RL motor control.

This navigator uses classical A* algorithm for global path planning
and provides waypoints to the RL agent for local movement execution.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from .base import Navigator, NavigationTarget, NavigationState
from .obstacle_map import ObstacleMap, ObstacleMapConfig
from .pathfinding import AStarPathfinder, PathfindingConfig


@dataclass
class HybridNavigatorConfig:
    """Configuration for hybrid navigator."""
    # Waypoint arrival threshold (world units)
    waypoint_threshold: float = 1.0

    # Final goal arrival threshold (world units)
    goal_threshold: float = 2.0

    # Obstacle map settings
    grid_resolution: float = 0.5
    obstacle_padding: float = 0.15

    # Pathfinding settings
    diagonal_movement: bool = True
    smooth_path: bool = True

    # Dynamic replanning
    replan_on_stuck: bool = True
    stuck_threshold_steps: int = 20  # Steps without progress before replanning


class HybridNavigator(Navigator):
    """
    Hybrid navigator using A* for path planning + RL for movement.

    Architecture:
    1. A* computes global path as list of waypoints
    2. Navigator feeds waypoints one at a time to RL agent
    3. When RL agent reaches waypoint, advance to next
    4. RL agent handles smooth movement and local obstacle avoidance

    This approach combines the reliability of classical pathfinding
    with the natural movement of learned motor control.
    """

    def __init__(self, config: Optional[HybridNavigatorConfig] = None):
        self.config = config or HybridNavigatorConfig()

        # Create pathfinding components
        self._obstacle_map_config = ObstacleMapConfig(
            resolution=self.config.grid_resolution,
            padding=self.config.obstacle_padding,
        )
        self._pathfinding_config = PathfindingConfig(
            diagonal_movement=self.config.diagonal_movement,
            smooth_path=self.config.smooth_path,
        )
        self._pathfinder = AStarPathfinder(self._pathfinding_config)

        # Navigation state
        self._obstacle_map: Optional[ObstacleMap] = None
        self._waypoints: List[Tuple[float, float]] = []
        self._current_waypoint_idx: int = 0
        self._final_goal: Optional[Tuple[float, float]] = None
        self._is_complete: bool = False
        self._is_stuck: bool = False

        # Progress tracking for stuck detection
        self._last_distance: float = float('inf')
        self._steps_without_progress: int = 0

    def set_goal(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        world_state: Dict[str, Any],
    ) -> bool:
        """
        Set navigation goal and compute A* path.

        Args:
            start_x, start_y: Starting position
            goal_x, goal_y: Goal position
            world_state: Current world state

        Returns:
            True if path found, False if no valid path exists
        """
        # Reset state
        self.reset()
        self._final_goal = (goal_x, goal_y)

        # Get world size from state
        world_size = world_state.get("world_size", 64)

        # Build obstacle map
        self._obstacle_map = ObstacleMap(world_size, self._obstacle_map_config)

        # Find target entity to exclude from obstacles
        # (We want to be able to path TO the target, not avoid it)
        target_entity_id = self._find_target_entity_id(world_state, goal_x, goal_y)

        self._obstacle_map.build_from_world_state(
            world_state,
            exclude_entity_id=target_entity_id
        )

        # Find path using A*
        path = self._pathfinder.find_path_world_coords(
            self._obstacle_map,
            start_x, start_y,
            goal_x, goal_y
        )

        if path is None:
            self._is_stuck = True
            return False

        # Store waypoints (skip first one which is start position)
        self._waypoints = path[1:] if len(path) > 1 else path
        self._current_waypoint_idx = 0
        self._is_stuck = False

        return True

    def _find_target_entity_id(
        self,
        world_state: Dict[str, Any],
        goal_x: float,
        goal_y: float,
        tolerance: float = 0.5,
    ) -> Optional[str]:
        """Find entity ID of object at goal position."""
        for obj in world_state.get("objects", []):
            pos = obj["position"]
            dist = ((pos["x"] - goal_x) ** 2 + (pos["y"] - goal_y) ** 2) ** 0.5
            if dist < tolerance:
                return obj["entity_id"]
        return None

    def get_current_target(self) -> Optional[NavigationTarget]:
        """Get the current waypoint for RL agent to navigate toward."""
        if self._is_complete or self._is_stuck:
            return None

        if not self._waypoints or self._current_waypoint_idx >= len(self._waypoints):
            return None

        waypoint = self._waypoints[self._current_waypoint_idx]
        is_final = (self._current_waypoint_idx == len(self._waypoints) - 1)

        return NavigationTarget(
            x=waypoint[0],
            y=waypoint[1],
            is_final=is_final
        )

    def update(self, npc_x: float, npc_y: float, world_state: Dict[str, Any]) -> None:
        """
        Update navigation state based on NPC position.

        Checks if current waypoint is reached and advances to next.
        Also handles stuck detection and optional replanning.
        """
        if self._is_complete or not self._waypoints:
            return

        if self._current_waypoint_idx >= len(self._waypoints):
            self._is_complete = True
            return

        # Get current waypoint
        waypoint = self._waypoints[self._current_waypoint_idx]
        is_final = (self._current_waypoint_idx == len(self._waypoints) - 1)

        # Calculate distance to waypoint
        distance = ((npc_x - waypoint[0]) ** 2 + (npc_y - waypoint[1]) ** 2) ** 0.5

        # Check for progress (for stuck detection)
        if distance < self._last_distance - 0.1:
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1
        self._last_distance = distance

        # Determine threshold based on whether this is final waypoint
        threshold = self.config.goal_threshold if is_final else self.config.waypoint_threshold

        # Check if waypoint reached
        if distance <= threshold:
            self._current_waypoint_idx += 1
            self._steps_without_progress = 0
            self._last_distance = float('inf')

            if self._current_waypoint_idx >= len(self._waypoints):
                self._is_complete = True

        # Check for stuck condition and replan if enabled
        elif (self.config.replan_on_stuck and
              self._steps_without_progress >= self.config.stuck_threshold_steps):
            self._handle_stuck(npc_x, npc_y, world_state)

    def _handle_stuck(
        self,
        npc_x: float,
        npc_y: float,
        world_state: Dict[str, Any]
    ) -> None:
        """Handle stuck situation by attempting to replan."""
        if self._final_goal is None:
            self._is_stuck = True
            return

        # Try to replan from current position
        success = self.set_goal(
            npc_x, npc_y,
            self._final_goal[0], self._final_goal[1],
            world_state
        )

        if not success:
            self._is_stuck = True

    def get_state(self) -> NavigationState:
        """Get current navigation state."""
        current_target = self.get_current_target()

        return NavigationState(
            has_path=len(self._waypoints) > 0,
            current_target=current_target,
            waypoints_remaining=max(0, len(self._waypoints) - self._current_waypoint_idx),
            total_waypoints=len(self._waypoints),
            is_complete=self._is_complete,
            is_stuck=self._is_stuck,
        )

    def reset(self) -> None:
        """Reset navigator to initial state."""
        self._waypoints = []
        self._current_waypoint_idx = 0
        self._final_goal = None
        self._is_complete = False
        self._is_stuck = False
        self._last_distance = float('inf')
        self._steps_without_progress = 0
        self._obstacle_map = None

    @property
    def name(self) -> str:
        return "HybridNavigator"

    @property
    def waypoints(self) -> List[Tuple[float, float]]:
        """Get all waypoints (for debugging/visualization)."""
        return self._waypoints.copy()

    @property
    def obstacle_map(self) -> Optional[ObstacleMap]:
        """Get obstacle map (for debugging/visualization)."""
        return self._obstacle_map

    def get_path_visualization(self) -> Optional[str]:
        """Get ASCII visualization of path on obstacle map."""
        if self._obstacle_map is None or not self._waypoints:
            return None

        # Convert world waypoints to grid coordinates
        grid_path = [
            self._obstacle_map.world_to_grid(x, y)
            for x, y in self._waypoints
        ]

        return self._obstacle_map.to_ascii(grid_path)
