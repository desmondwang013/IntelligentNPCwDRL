"""
Grid-based obstacle map for pathfinding.

Converts continuous world positions and collision radii into a discrete
grid that A* and other pathfinding algorithms can use.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


@dataclass
class ObstacleMapConfig:
    """Configuration for obstacle map generation."""
    resolution: float = 0.5  # World units per grid cell
    padding: float = 0.15  # Extra padding around obstacles (NPC radius)

    # If True, mark cells near world edges as blocked
    block_edges: bool = True
    edge_margin: float = 0.25


class ObstacleMap:
    """
    Grid-based representation of obstacles for pathfinding.

    Converts the continuous world (with objects at float positions and
    collision radii) into a discrete grid where each cell is either
    passable (0) or blocked (1).
    """

    def __init__(
        self,
        world_size: float,
        config: Optional[ObstacleMapConfig] = None,
    ):
        self.config = config or ObstacleMapConfig()
        self.world_size = world_size
        self.resolution = self.config.resolution

        # Calculate grid dimensions
        self.grid_size = int(np.ceil(world_size / self.resolution))

        # Initialize empty grid (0 = passable, 1 = blocked)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        # Clamp to grid bounds
        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cell center)."""
        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution
        return x, y

    def is_blocked(self, gx: int, gy: int) -> bool:
        """Check if a grid cell is blocked."""
        if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
            return True  # Out of bounds is blocked
        return self.grid[gy, gx] == 1

    def is_passable(self, gx: int, gy: int) -> bool:
        """Check if a grid cell is passable."""
        return not self.is_blocked(gx, gy)

    def add_circular_obstacle(self, x: float, y: float, radius: float) -> None:
        """
        Mark cells occupied by a circular obstacle.

        Args:
            x, y: World position of obstacle center
            radius: Collision radius of obstacle
        """
        # Add padding for NPC radius
        effective_radius = radius + self.config.padding

        # Convert to grid coordinates
        center_gx, center_gy = self.world_to_grid(x, y)
        grid_radius = int(np.ceil(effective_radius / self.resolution)) + 1

        # Mark all cells within radius
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                gx, gy = center_gx + dx, center_gy + dy

                # Check bounds
                if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
                    continue

                # Check if cell center is within radius
                cell_x, cell_y = self.grid_to_world(gx, gy)
                dist = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)

                if dist <= effective_radius:
                    self.grid[gy, gx] = 1

    def add_rectangular_obstacle(
        self, x: float, y: float, width: float, height: float
    ) -> None:
        """
        Mark cells occupied by a rectangular obstacle.

        Args:
            x, y: World position of rectangle center
            width, height: Dimensions of rectangle
        """
        padding = self.config.padding
        half_w = width / 2 + padding
        half_h = height / 2 + padding

        # Get grid bounds
        min_gx, min_gy = self.world_to_grid(x - half_w, y - half_h)
        max_gx, max_gy = self.world_to_grid(x + half_w, y + half_h)

        # Mark all cells in rectangle
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.grid[gy, gx] = 1

    def mark_edges_blocked(self) -> None:
        """Mark cells near world edges as blocked."""
        margin_cells = int(np.ceil(self.config.edge_margin / self.resolution))

        # Top and bottom edges
        self.grid[:margin_cells, :] = 1
        self.grid[-margin_cells:, :] = 1

        # Left and right edges
        self.grid[:, :margin_cells] = 1
        self.grid[:, -margin_cells:] = 1

    def build_from_world_state(
        self,
        world_state: Dict[str, Any],
        exclude_entity_id: Optional[str] = None,
    ) -> None:
        """
        Build obstacle map from world state.

        Args:
            world_state: World state dict from world.get_state()
            exclude_entity_id: Entity to exclude (e.g., the target object)
        """
        # Clear existing obstacles
        self.grid.fill(0)

        # Add objects as obstacles
        for obj in world_state["objects"]:
            if obj["entity_id"] == exclude_entity_id:
                continue  # Don't block path to target

            pos = obj["position"]
            radius = obj["collision_radius"]
            self.add_circular_obstacle(pos["x"], pos["y"], radius)

        # Optionally block edges
        if self.config.block_edges:
            self.mark_edges_blocked()

    def get_neighbors(self, gx: int, gy: int, diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        Get passable neighbor cells.

        Args:
            gx, gy: Current grid position
            diagonal: If True, include diagonal neighbors (8-connected)
                      If False, only cardinal directions (4-connected)

        Returns:
            List of (gx, gy) tuples for passable neighbors
        """
        neighbors = []

        # Cardinal directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Add diagonals if enabled
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            nx, ny = gx + dx, gy + dy

            if self.is_passable(nx, ny):
                # For diagonal moves, also check that we can actually move diagonally
                # (not cutting through corners)
                if diagonal and dx != 0 and dy != 0:
                    if self.is_blocked(gx + dx, gy) and self.is_blocked(gx, gy + dy):
                        continue  # Can't cut through corner
                neighbors.append((nx, ny))

        return neighbors

    def to_ascii(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """
        Generate ASCII visualization of the obstacle map.

        Args:
            path: Optional list of (gx, gy) waypoints to display

        Returns:
            ASCII string representation
        """
        path_set = set(path) if path else set()
        lines = []

        for gy in range(self.grid_size - 1, -1, -1):  # Top to bottom
            row = ""
            for gx in range(self.grid_size):
                if (gx, gy) in path_set:
                    row += "*"
                elif self.grid[gy, gx] == 1:
                    row += "#"
                else:
                    row += "."
            lines.append(row)

        return "\n".join(lines)

    def __repr__(self) -> str:
        blocked = np.sum(self.grid)
        total = self.grid_size * self.grid_size
        return (
            f"ObstacleMap(world_size={self.world_size}, "
            f"grid_size={self.grid_size}x{self.grid_size}, "
            f"blocked={blocked}/{total})"
        )
