"""
Curriculum learning controller for progressive world size training.

Starts with small world for easy exploration, increases size as agent learns.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import deque


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # World size progression (smallest to largest)
    world_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Success rate threshold to advance to next level (0.0 - 1.0)
    advance_threshold: float = 0.7

    # Number of episodes to track for rolling success rate
    window_size: int = 100

    # Minimum episodes at each level before allowing advancement
    min_episodes_per_level: int = 50


class CurriculumController:
    """
    Controls world size progression based on agent performance.

    Usage:
        curriculum = CurriculumController()

        # In training loop:
        for episode in episodes:
            env.set_world_size(curriculum.current_world_size)
            ...run episode...
            curriculum.record_episode(completed=success)

            if curriculum.should_advance():
                curriculum.advance()
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()

        # Current level index
        self._level_index = 0

        # Episode tracking for current level
        self._episode_results: deque = deque(maxlen=self.config.window_size)
        self._episodes_at_level = 0
        self._total_episodes = 0

        # Callbacks for level changes
        self._on_advance_callbacks: List[Callable[[int, int], None]] = []

    @property
    def current_world_size(self) -> int:
        """Get the current world size."""
        return self.config.world_sizes[self._level_index]

    @property
    def current_level(self) -> int:
        """Get current level index (0-based)."""
        return self._level_index

    @property
    def max_level(self) -> int:
        """Get maximum level index."""
        return len(self.config.world_sizes) - 1

    @property
    def is_max_level(self) -> bool:
        """Check if at maximum difficulty."""
        return self._level_index >= self.max_level

    @property
    def success_rate(self) -> float:
        """Calculate rolling success rate."""
        if len(self._episode_results) == 0:
            return 0.0
        return sum(self._episode_results) / len(self._episode_results)

    @property
    def episodes_at_level(self) -> int:
        """Number of episodes completed at current level."""
        return self._episodes_at_level

    def record_episode(self, completed: bool) -> None:
        """
        Record the result of an episode.

        Args:
            completed: True if agent successfully completed the task
        """
        self._episode_results.append(1 if completed else 0)
        self._episodes_at_level += 1
        self._total_episodes += 1

    def should_advance(self) -> bool:
        """
        Check if agent should advance to next level.

        Returns True if:
        - Not already at max level
        - Minimum episodes completed at current level
        - Success rate exceeds threshold
        """
        if self.is_max_level:
            return False

        if self._episodes_at_level < self.config.min_episodes_per_level:
            return False

        if len(self._episode_results) < self.config.window_size:
            # Need full window for reliable measurement
            return False

        return self.success_rate >= self.config.advance_threshold

    def advance(self) -> bool:
        """
        Advance to next difficulty level.

        Returns:
            True if advanced, False if already at max level
        """
        if self.is_max_level:
            return False

        old_size = self.current_world_size
        self._level_index += 1
        new_size = self.current_world_size

        # Reset level tracking
        self._episode_results.clear()
        self._episodes_at_level = 0

        # Notify callbacks
        for callback in self._on_advance_callbacks:
            callback(old_size, new_size)

        return True

    def on_advance(self, callback: Callable[[int, int], None]) -> None:
        """
        Register a callback for when level advances.

        Callback receives (old_world_size, new_world_size).
        """
        self._on_advance_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get current curriculum statistics."""
        return {
            "level": self._level_index,
            "world_size": self.current_world_size,
            "success_rate": self.success_rate,
            "episodes_at_level": self._episodes_at_level,
            "total_episodes": self._total_episodes,
            "is_max_level": self.is_max_level,
            "window_fill": len(self._episode_results),
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumController(level={self._level_index}, "
            f"world_size={self.current_world_size}, "
            f"success_rate={self.success_rate:.1%})"
        )
