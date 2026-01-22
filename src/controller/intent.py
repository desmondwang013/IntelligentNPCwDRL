"""
Intent data structures for tracking user instructions.

Moved from src/intent/intent.py to consolidate into controller module.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
import numpy as np


class IntentStatus(Enum):
    ACTIVE = auto()
    COMPLETED = auto()
    CANCELED = auto()
    TIMEOUT = auto()
    IMPOSSIBLE = auto()


class IntentType(Enum):
    """
    High-level intent categories. The policy learns to handle these,
    but the categories help with evaluation and timeout scaling.
    """
    MOVE_TO_POSITION = auto()      # "go to position X,Y"
    MOVE_TO_OBJECT = auto()        # "go to the blue triangle"
    MOVE_TO_USER = auto()          # "come here", "follow me"
    MOVE_NEAR_OBJECT = auto()      # "go near the red square"
    STAY = auto()                  # "wait here", "stop"
    GENERIC = auto()               # fallback for unclassified intents


@dataclass
class CompletionCriteria:
    """
    Defines how to check if an intent is completed.
    Uses world state to make deterministic completion checks.
    """
    target_entity_id: Optional[str] = None
    target_position: Optional[tuple] = None  # (x, y)
    distance_threshold: float = 2.0
    stability_ticks: int = 3  # must remain within threshold for N ticks

    def check(self, npc_position: tuple, world_state: dict) -> bool:
        """
        Check if the NPC has reached the target.
        Returns True if within threshold distance.
        """
        if self.target_position is not None:
            target = self.target_position
        elif self.target_entity_id is not None:
            target = self._get_entity_position(self.target_entity_id, world_state)
            if target is None:
                return False
        else:
            return False

        dx = npc_position[0] - target[0]
        dy = npc_position[1] - target[1]
        distance = (dx * dx + dy * dy) ** 0.5
        return distance <= self.distance_threshold

    def _get_entity_position(self, entity_id: str, world_state: dict) -> Optional[tuple]:
        if entity_id == "user":
            pos = world_state["user"]["position"]
            return (pos["x"], pos["y"])

        for obj in world_state["objects"]:
            if obj["entity_id"] == entity_id:
                pos = obj["position"]
                return (pos["x"], pos["y"])

        return None


@dataclass
class Intent:
    """
    Represents a single user instruction with its lifecycle state.
    """
    intent_id: str
    text: str
    intent_type: IntentType = IntentType.GENERIC
    status: IntentStatus = IntentStatus.ACTIVE

    # Timing
    start_tick: int = 0
    end_tick: Optional[int] = None
    base_timeout_ticks: int = 480  # 30 seconds at 16 ticks/sec

    # Completion tracking
    criteria: Optional[CompletionCriteria] = None
    stability_counter: int = 0

    # Cached embedding (set by IntentManager)
    embedding: Optional[np.ndarray] = None

    # Optional focus hint (object ID that likely relates to this intent)
    focus_hint: Optional[str] = None

    @property
    def age_ticks(self) -> int:
        """How many ticks since this intent started."""
        return 0 if self.end_tick is None else self.end_tick - self.start_tick

    @property
    def is_terminal(self) -> bool:
        """Whether the intent has reached a final state."""
        return self.status != IntentStatus.ACTIVE

    @property
    def is_success(self) -> bool:
        """Whether the intent completed successfully."""
        return self.status == IntentStatus.COMPLETED

    def get_adaptive_timeout(self, progress_ratio: float = 0.0) -> int:
        """
        Calculate adaptive timeout based on progress.
        If NPC is making progress, extend the timeout slightly.
        """
        base = self.base_timeout_ticks

        # Grace extension if near completion (up to 25% extra time)
        if progress_ratio > 0.8:
            return int(base * 1.25)
        elif progress_ratio > 0.5:
            return int(base * 1.1)

        return base

    def to_dict(self) -> dict:
        """Serialize intent for logging or Unity communication."""
        return {
            "intent_id": self.intent_id,
            "text": self.text,
            "intent_type": self.intent_type.name.lower(),
            "status": self.status.name.lower(),
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "focus_hint": self.focus_hint,
        }
