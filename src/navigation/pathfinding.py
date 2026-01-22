"""
A* pathfinding algorithm implementation.

Finds shortest path through obstacle grid while avoiding blocked cells.
"""
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from .obstacle_map import ObstacleMap


@dataclass
class PathfindingConfig:
    """Configuration for A* pathfinding."""
    diagonal_movement: bool = True  # Allow 8-directional movement
    heuristic_weight: float = 1.0  # Weight for heuristic (1.0 = standard A*)
    max_iterations: int = 10000  # Safety limit to prevent infinite loops
    smooth_path: bool = True  # Apply path smoothing after finding path


@dataclass(order=True)
class PriorityNode:
    """Node for priority queue in A*."""
    f_score: float
    position: Tuple[int, int] = field(compare=False)


class AStarPathfinder:
    """
    A* pathfinding algorithm.

    Finds the shortest path from start to goal on a grid while avoiding
    obstacles. Supports both 4-directional and 8-directional movement.
    """

    def __init__(self, config: Optional[PathfindingConfig] = None):
        self.config = config or PathfindingConfig()

    def _heuristic(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> float:
        """
        Compute heuristic distance estimate.

        Uses Euclidean distance for 8-directional movement,
        Manhattan distance for 4-directional.
        """
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])

        if self.config.diagonal_movement:
            # Euclidean distance (admissible for 8-directional)
            return np.sqrt(dx * dx + dy * dy) * self.config.heuristic_weight
        else:
            # Manhattan distance (admissible for 4-directional)
            return (dx + dy) * self.config.heuristic_weight

    def _movement_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
    ) -> float:
        """Compute cost of moving between adjacent cells."""
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])

        if dx + dy == 2:  # Diagonal move
            return 1.414  # sqrt(2)
        return 1.0  # Cardinal move

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dict."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(
        self,
        obstacle_map: ObstacleMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal avoiding obstacles.

        Args:
            obstacle_map: Grid-based obstacle map
            start: Starting grid position (gx, gy)
            goal: Goal grid position (gx, gy)

        Returns:
            List of grid positions forming the path, or None if no path exists
        """
        # Check if start or goal is blocked
        if obstacle_map.is_blocked(*start):
            return None
        if obstacle_map.is_blocked(*goal):
            return None

        # Check if already at goal
        if start == goal:
            return [start]

        # Initialize data structures
        open_set: List[PriorityNode] = []
        heapq.heappush(open_set, PriorityNode(0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, goal)}

        open_set_hash: Set[Tuple[int, int]] = {start}
        iterations = 0

        while open_set and iterations < self.config.max_iterations:
            iterations += 1

            # Get node with lowest f_score
            current_node = heapq.heappop(open_set)
            current = current_node.position
            open_set_hash.discard(current)

            # Check if reached goal
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                if self.config.smooth_path:
                    path = self._smooth_path(obstacle_map, path)
                return path

            # Explore neighbors
            neighbors = obstacle_map.get_neighbors(
                current[0], current[1],
                diagonal=self.config.diagonal_movement
            )

            for neighbor in neighbors:
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, PriorityNode(f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def _smooth_path(
        self,
        obstacle_map: ObstacleMap,
        path: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints.

        Uses line-of-sight checks to skip intermediate waypoints
        when direct path is clear.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip ahead as far as possible
            farthest_visible = current_idx + 1

            for check_idx in range(current_idx + 2, len(path)):
                if self._has_line_of_sight(
                    obstacle_map,
                    path[current_idx],
                    path[check_idx]
                ):
                    farthest_visible = check_idx

            smoothed.append(path[farthest_visible])
            current_idx = farthest_visible

        return smoothed

    def _has_line_of_sight(
        self,
        obstacle_map: ObstacleMap,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> bool:
        """
        Check if there's clear line of sight between two grid positions.

        Uses Bresenham's line algorithm to check all cells along the line.
        """
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_sign = 1 if x0 < x1 else -1
        y_sign = 1 if y0 < y1 else -1

        if dx == 0 and dy == 0:
            return True

        # Bresenham's line algorithm
        if dx > dy:
            err = dx / 2
            y = y0
            for x in range(x0, x1 + x_sign, x_sign):
                if obstacle_map.is_blocked(x, y):
                    return False
                err -= dy
                if err < 0:
                    y += y_sign
                    err += dx
        else:
            err = dy / 2
            x = x0
            for y in range(y0, y1 + y_sign, y_sign):
                if obstacle_map.is_blocked(x, y):
                    return False
                err -= dx
                if err < 0:
                    x += x_sign
                    err += dy

        return True

    def find_path_world_coords(
        self,
        obstacle_map: ObstacleMap,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Convenience method: find path using world coordinates.

        Args:
            obstacle_map: Grid-based obstacle map
            start_x, start_y: Starting world position
            goal_x, goal_y: Goal world position

        Returns:
            List of world positions forming the path, or None if no path exists
        """
        # Convert to grid coordinates
        start_grid = obstacle_map.world_to_grid(start_x, start_y)
        goal_grid = obstacle_map.world_to_grid(goal_x, goal_y)

        # Find path in grid space
        grid_path = self.find_path(obstacle_map, start_grid, goal_grid)

        if grid_path is None:
            return None

        # Convert back to world coordinates
        world_path = [
            obstacle_map.grid_to_world(gx, gy)
            for gx, gy in grid_path
        ]

        return world_path
