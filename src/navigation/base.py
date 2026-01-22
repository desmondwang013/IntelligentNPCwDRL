"""
Abstract base classes for navigation systems.

This module defines the interface that all navigation strategies must implement,
allowing easy swapping between different approaches:
- HybridNavigator: A* pathfinding + RL motor control
- HierarchicalNavigator: High-level RL + low-level RL (future)
- DirectNavigator: No planning, direct to target (baseline)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class NavigationTarget:
    """Represents a navigation target (waypoint or final goal)."""
    x: float
    y: float
    is_final: bool = False  # True if this is the ultimate goal, not intermediate

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, x: float, y: float) -> float:
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5


@dataclass
class NavigationState:
    """Current state of navigation progress."""
    has_path: bool
    current_target: Optional[NavigationTarget]
    waypoints_remaining: int
    total_waypoints: int
    is_complete: bool
    is_stuck: bool  # Navigator detected no valid path


class Navigator(ABC):
    """
    Abstract base class for navigation strategies.

    All navigators provide the same interface to the RL agent:
    - set_goal(): Define where to go
    - get_current_target(): Get the immediate target for RL agent
    - update(): Update internal state based on NPC position
    - get_state(): Get navigation progress info

    This allows swapping between A* hybrid, hierarchical RL, or other
    approaches without changing the RL agent or environment code.
    """

    @abstractmethod
    def set_goal(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        world_state: Dict[str, Any],
    ) -> bool:
        """
        Set the navigation goal and compute path/strategy.

        Args:
            start_x, start_y: Starting position (usually NPC position)
            goal_x, goal_y: Final goal position
            world_state: Current world state from world.get_state()

        Returns:
            True if a valid path/strategy was found, False otherwise
        """
        pass

    @abstractmethod
    def get_current_target(self) -> Optional[NavigationTarget]:
        """
        Get the current target for the RL agent to navigate toward.

        For hybrid A*: Returns the next waypoint
        For hierarchical RL: Returns the current subgoal from high-level policy

        Returns:
            NavigationTarget with (x, y) position, or None if no active navigation
        """
        pass

    @abstractmethod
    def update(self, npc_x: float, npc_y: float, world_state: Dict[str, Any]) -> None:
        """
        Update navigation state based on current NPC position.

        This should:
        - Check if current waypoint/subgoal is reached
        - Advance to next waypoint if needed
        - Recompute path if obstacles changed (optional)

        Args:
            npc_x, npc_y: Current NPC position
            world_state: Current world state
        """
        pass

    @abstractmethod
    def get_state(self) -> NavigationState:
        """
        Get current navigation state for monitoring/debugging.

        Returns:
            NavigationState with progress information
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset navigator to initial state.
        Call this when starting a new episode or changing goals.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this navigation strategy."""
        pass


class DirectNavigator(Navigator):
    """
    Baseline navigator that goes directly to goal (no pathfinding).

    This is equivalent to the current system - just provides the final
    target directly to the RL agent. Useful for comparison and simple scenarios.
    """

    def __init__(self, arrival_threshold: float = 2.0):
        self._arrival_threshold = arrival_threshold
        self._goal: Optional[NavigationTarget] = None
        self._is_complete = False

    def set_goal(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        world_state: Dict[str, Any],
    ) -> bool:
        self._goal = NavigationTarget(x=goal_x, y=goal_y, is_final=True)
        self._is_complete = False
        return True  # Direct navigation always "succeeds" (no pathfinding)

    def get_current_target(self) -> Optional[NavigationTarget]:
        if self._is_complete:
            return None
        return self._goal

    def update(self, npc_x: float, npc_y: float, world_state: Dict[str, Any]) -> None:
        if self._goal is None:
            return

        distance = self._goal.distance_to(npc_x, npc_y)
        if distance <= self._arrival_threshold:
            self._is_complete = True

    def get_state(self) -> NavigationState:
        return NavigationState(
            has_path=self._goal is not None,
            current_target=self._goal if not self._is_complete else None,
            waypoints_remaining=0 if self._is_complete else 1,
            total_waypoints=1,
            is_complete=self._is_complete,
            is_stuck=False,
        )

    def reset(self) -> None:
        self._goal = None
        self._is_complete = False

    @property
    def name(self) -> str:
        return "DirectNavigator"
