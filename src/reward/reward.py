from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto


@dataclass
class RewardConfig:
    """
    Configurable reward weights. Start simple, tune later.
    All values can be adjusted based on training performance.
    """
    # Progress rewards (per tick)
    progress_scale: float = 1.0          # Reward for moving closer to target

    # Terminal rewards (when intent ends)
    completion_bonus: float = 10.0       # Reward for completing intent
    timeout_penalty: float = -5.0        # Penalty for timing out
    cancel_penalty: float = 0.0          # No penalty for user cancellation (not NPC's fault)

    # Per-tick penalties
    time_penalty: float = -0.01          # Small penalty each tick to encourage speed
    collision_penalty: float = -0.1      # Penalty for trying to move into obstacle

    # Anti-oscillation (moving back and forth)
    oscillation_penalty: float = -0.05   # Penalty for reversing direction
    oscillation_window: int = 4          # How many actions to check for oscillation


@dataclass
class RewardInfo:
    """Breakdown of reward components for debugging/logging."""
    total: float = 0.0
    progress: float = 0.0
    completion: float = 0.0
    time: float = 0.0
    collision: float = 0.0
    oscillation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "progress": self.progress,
            "completion": self.completion,
            "time": self.time,
            "collision": self.collision,
            "oscillation": self.oscillation,
        }


class RewardCalculator:
    """
    Calculates rewards for the NPC based on world state and intent progress.

    Reward components:
    1. Progress - moving toward target (positive) or away (negative)
    2. Completion - bonus for finishing intent
    3. Time pressure - small negative each tick
    4. Collision - penalty for bumping into things
    5. Oscillation - penalty for back-and-forth movement
    """

    # Action pairs that are opposites (for oscillation detection)
    OPPOSITE_ACTIONS = {
        0: 1,  # UP <-> DOWN
        1: 0,
        2: 3,  # LEFT <-> RIGHT
        3: 2,
    }

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_distance: Optional[float] = None
        self._action_history: List[int] = []
        self._prev_npc_position: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Reset state for new episode/intent."""
        self._prev_distance = None
        self._action_history.clear()
        self._prev_npc_position = None

    def calculate(
        self,
        world_state: Dict[str, Any],
        intent_state: Dict[str, Any],
        action: int,
        collision_occurred: bool,
        intent_event: Optional[Any] = None,
    ) -> RewardInfo:
        """
        Calculate reward for the current tick.

        Args:
            world_state: Current world state from world.get_state()
            intent_state: Current intent state from intent_manager.get_state()
            action: The action that was taken (0-5)
            collision_occurred: Whether the NPC tried to move but couldn't
            intent_event: Event if intent just terminated (completed/timeout/canceled)

        Returns:
            RewardInfo with breakdown of all reward components
        """
        info = RewardInfo()

        # Get NPC position
        npc_pos = world_state["npc"]["position"]
        npc_x, npc_y = npc_pos["x"], npc_pos["y"]

        # 1. Progress reward (only if there's an active intent with a target)
        info.progress = self._calculate_progress(
            npc_x, npc_y, world_state, intent_state
        )

        # 2. Terminal rewards (completion/timeout)
        info.completion = self._calculate_terminal(intent_event)

        # 3. Time penalty (always applies when intent is active)
        if intent_state.get("has_intent", False):
            info.time = self.config.time_penalty

        # 4. Collision penalty
        if collision_occurred:
            info.collision = self.config.collision_penalty

        # 5. Oscillation penalty
        info.oscillation = self._calculate_oscillation(action)

        # Update history
        self._action_history.append(action)
        if len(self._action_history) > self.config.oscillation_window:
            self._action_history.pop(0)
        self._prev_npc_position = (npc_x, npc_y)

        # Sum up total
        info.total = (
            info.progress +
            info.completion +
            info.time +
            info.collision +
            info.oscillation
        )

        return info

    def _calculate_progress(
        self,
        npc_x: float,
        npc_y: float,
        world_state: Dict[str, Any],
        intent_state: Dict[str, Any],
    ) -> float:
        """Calculate progress reward based on distance change to target."""
        if not intent_state.get("has_intent", False):
            self._prev_distance = None
            return 0.0

        intent = intent_state.get("intent")
        if not intent:
            return 0.0

        # Get target position
        target_pos = self._get_target_position(intent, world_state)
        if target_pos is None:
            return 0.0

        target_x, target_y = target_pos

        # Calculate current distance
        current_distance = ((npc_x - target_x) ** 2 + (npc_y - target_y) ** 2) ** 0.5

        # If no previous distance, store and return 0
        if self._prev_distance is None:
            self._prev_distance = current_distance
            return 0.0

        # Progress = how much closer we got (positive = good)
        distance_delta = self._prev_distance - current_distance
        progress_reward = distance_delta * self.config.progress_scale

        # Update for next tick
        self._prev_distance = current_distance

        return progress_reward

    def _get_target_position(
        self,
        intent: Dict[str, Any],
        world_state: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        """Extract target position from intent focus hint."""
        focus_hint = intent.get("focus_hint")
        if not focus_hint:
            return None

        # Check if it's the user
        if focus_hint == "user":
            user_pos = world_state["user"]["position"]
            return (user_pos["x"], user_pos["y"])

        # Check objects
        for obj in world_state["objects"]:
            if obj["entity_id"] == focus_hint:
                return (obj["position"]["x"], obj["position"]["y"])

        return None

    def _calculate_terminal(self, intent_event: Optional[Any]) -> float:
        """Calculate terminal reward when intent ends."""
        if intent_event is None:
            return 0.0

        status = intent_event.intent.status.name.lower()

        if status == "completed":
            # Reset distance tracking for next intent
            self._prev_distance = None
            return self.config.completion_bonus
        elif status == "timeout":
            self._prev_distance = None
            return self.config.timeout_penalty
        elif status == "canceled":
            self._prev_distance = None
            return self.config.cancel_penalty

        return 0.0

    def _calculate_oscillation(self, action: int) -> float:
        """Detect and penalize back-and-forth movement."""
        if len(self._action_history) < 2:
            return 0.0

        # Check if current action is opposite of previous
        if action in self.OPPOSITE_ACTIONS:
            last_action = self._action_history[-1]
            if self.OPPOSITE_ACTIONS.get(action) == last_action:
                return self.config.oscillation_penalty

        return 0.0

    def get_config(self) -> Dict[str, float]:
        """Return current config as dict for logging."""
        return {
            "progress_scale": self.config.progress_scale,
            "completion_bonus": self.config.completion_bonus,
            "timeout_penalty": self.config.timeout_penalty,
            "cancel_penalty": self.config.cancel_penalty,
            "time_penalty": self.config.time_penalty,
            "collision_penalty": self.config.collision_penalty,
            "oscillation_penalty": self.config.oscillation_penalty,
        }
