"""
Curriculum Learning v2 - Improved approach with:
1. Warmup period to ignore early lucky episodes
2. Mixed-difficulty sampling instead of hard stage switches
3. Gradual shift toward larger worlds
4. Dynamic success threshold that relaxes then tightens
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
import random
import math


@dataclass
class CurriculumV2Config:
    """Configuration for improved curriculum learning."""
    # World sizes available for sampling
    world_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Warmup: ignore first N episodes before tracking success
    warmup_episodes: int = 100

    # Window size for calculating success rate (after warmup)
    window_size: int = 200

    # Mixed sampling: initial probability weights for each world size
    # Starts heavily weighted toward smallest, shifts over time
    # V2.1: Start even more concentrated on small world
    initial_weights: List[float] = field(default_factory=lambda: [0.85, 0.10, 0.04, 0.01])

    # How fast to shift toward larger worlds (0.0 = never, 1.0 = instant)
    # V2.1: Much slower shift (was 0.1, now 0.02)
    difficulty_shift_rate: float = 0.02

    # Success rate threshold to trigger difficulty shift
    # V2.1: Require higher success before shifting (was 0.4, now 0.8)
    shift_threshold: float = 0.8

    # Success threshold (distance to target) - starts relaxed, tightens
    # V2.1: Tighten much more slowly to allow precision learning
    initial_success_radius: float = 5.0
    final_success_radius: float = 2.0
    radius_tightening_episodes: int = 3000  # Was 1000, now 3000 for slower tightening


class CurriculumV2Controller:
    """
    Improved curriculum controller with:
    - Warmup period
    - Mixed-difficulty sampling
    - Gradual difficulty shifting
    - Dynamic success radius
    """

    def __init__(self, config: Optional[CurriculumV2Config] = None):
        self.config = config or CurriculumV2Config()

        # Sampling weights (mutable, shift over time)
        self._weights = list(self.config.initial_weights)

        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

        # Episode tracking
        self._total_episodes = 0
        self._episode_results: deque = deque(maxlen=self.config.window_size)

        # Track per-difficulty success
        self._per_size_results = {size: deque(maxlen=50) for size in self.config.world_sizes}

        # Current episode's world size
        self._current_world_size = self.config.world_sizes[0]

        # Random generator
        self._rng = random.Random()

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def in_warmup(self) -> bool:
        """Whether we're still in warmup period."""
        return self._total_episodes < self.config.warmup_episodes

    @property
    def success_rate(self) -> float:
        """Overall success rate (post-warmup only)."""
        if len(self._episode_results) == 0:
            return 0.0
        return sum(self._episode_results) / len(self._episode_results)

    @property
    def current_success_radius(self) -> float:
        """Dynamic success radius - starts large, shrinks over time."""
        initial = self.config.initial_success_radius
        final = self.config.final_success_radius
        episodes = self.config.radius_tightening_episodes

        # Linear interpolation based on episode count
        progress = min(1.0, self._total_episodes / episodes)
        return initial + (final - initial) * progress

    @property
    def current_world_size(self) -> int:
        """Get the current world size (set by sample_world_size)."""
        return self._current_world_size

    @property
    def sampling_weights(self) -> List[Tuple[int, float]]:
        """Current sampling weights for each world size."""
        return list(zip(self.config.world_sizes, self._weights))

    def sample_world_size(self) -> int:
        """
        Sample a world size based on current weights.
        Call this at the start of each episode.
        """
        sizes = self.config.world_sizes
        self._current_world_size = self._rng.choices(sizes, weights=self._weights, k=1)[0]
        return self._current_world_size

    def record_episode(self, completed: bool, world_size: Optional[int] = None) -> None:
        """
        Record episode result.

        Args:
            completed: Whether agent successfully reached target
            world_size: Which world size was used (defaults to current)
        """
        self._total_episodes += 1
        size = world_size or self._current_world_size

        # Track per-size results
        if size in self._per_size_results:
            self._per_size_results[size].append(1 if completed else 0)

        # Only track overall success after warmup
        if not self.in_warmup:
            self._episode_results.append(1 if completed else 0)

            # Check if we should shift difficulty
            self._maybe_shift_difficulty()

    def _maybe_shift_difficulty(self) -> None:
        """Shift sampling weights toward harder difficulties if doing well."""
        if self.success_rate < self.config.shift_threshold:
            return

        if len(self._episode_results) < self.config.window_size // 2:
            return  # Need enough data

        # Shift weights: reduce weight on easier, increase on harder
        shift = self.config.difficulty_shift_rate * (self.success_rate - self.config.shift_threshold)

        # Move weight from lower indices to higher indices
        new_weights = list(self._weights)
        for i in range(len(new_weights) - 1):
            transfer = new_weights[i] * shift * 0.1  # Transfer 10% of shift
            transfer = min(transfer, new_weights[i] * 0.5)  # Don't transfer more than half
            new_weights[i] -= transfer
            new_weights[i + 1] += transfer

        # Ensure minimum weight for each size (keep some exploration)
        min_weight = 0.05
        for i in range(len(new_weights)):
            new_weights[i] = max(min_weight, new_weights[i])

        # Normalize
        total = sum(new_weights)
        self._weights = [w / total for w in new_weights]

    def get_per_size_success_rates(self) -> dict:
        """Get success rate for each world size."""
        rates = {}
        for size, results in self._per_size_results.items():
            if len(results) > 0:
                rates[size] = sum(results) / len(results)
            else:
                rates[size] = 0.0
        return rates

    def get_stats(self) -> dict:
        """Get current curriculum statistics."""
        return {
            "total_episodes": self._total_episodes,
            "in_warmup": self.in_warmup,
            "warmup_remaining": max(0, self.config.warmup_episodes - self._total_episodes),
            "success_rate": self.success_rate,
            "success_radius": self.current_success_radius,
            "sampling_weights": dict(zip(self.config.world_sizes, self._weights)),
            "per_size_success": self.get_per_size_success_rates(),
            "window_fill": len(self._episode_results),
        }

    def __repr__(self) -> str:
        weights_str = ", ".join(f"{s}:{w:.0%}" for s, w in zip(self.config.world_sizes, self._weights))
        return (
            f"CurriculumV2(episodes={self._total_episodes}, "
            f"success={self.success_rate:.1%}, "
            f"radius={self.current_success_radius:.1f}, "
            f"weights=[{weights_str}])"
        )
