"""
IntentManager handles intent lifecycle tracking.

Moved from src/intent/manager.py to consolidate into controller module.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
import uuid
import numpy as np

from .intent import Intent, IntentStatus, IntentType, CompletionCriteria
from .embedder import TextEmbedder


@dataclass
class IntentEvent:
    """Represents an intent lifecycle event for logging/training."""
    event_type: str  # "started", "completed", "canceled", "timeout", "impossible"
    intent: Intent
    tick: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentManager:
    """
    Manages the lifecycle of user intents.

    - Only one intent active at a time
    - New instructions preempt current intent
    - Tracks completion, timeout, and cancellation
    - Provides embeddings for the policy (legacy - see note below)

    NOTE on embeddings:
    In the current architecture (LLM → NPCController → RL Executor),
    the RL agent receives structured goals, not embeddings. Embeddings are
    still computed for backward compatibility with legacy training scripts,
    but SimpleNPCEnv does not use them. The IntentManager is still used for
    intent lifecycle tracking (completion criteria, timeout, etc.).
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        default_timeout_ticks: int = 480,
        stability_required: int = 3,
        min_ticks_before_completion: int = 5,
    ):
        self.embedder = embedder or TextEmbedder()
        self.default_timeout_ticks = default_timeout_ticks
        self.stability_required = stability_required
        # Minimum ticks that must pass before completion can trigger
        # Prevents "spawn luck" where agent starts at target and immediately completes
        self.min_ticks_before_completion = min_ticks_before_completion

        self._active_intent: Optional[Intent] = None
        self._intent_history: List[Intent] = []
        self._event_log: List[IntentEvent] = []
        self._intent_counter: int = 0

    @property
    def active_intent(self) -> Optional[Intent]:
        return self._active_intent

    @property
    def has_active_intent(self) -> bool:
        return self._active_intent is not None

    def new_intent(
        self,
        text: str,
        current_tick: int,
        intent_type: IntentType = IntentType.GENERIC,
        criteria: Optional[CompletionCriteria] = None,
        focus_hint: Optional[str] = None,
        timeout_ticks: Optional[int] = None,
    ) -> Intent:
        """
        Create and activate a new intent, preempting any current intent.
        """
        # Cancel current intent if exists
        if self._active_intent is not None:
            self._terminate_intent(
                IntentStatus.CANCELED,
                current_tick,
                reason="preempted_by_new_intent"
            )

        # Generate intent ID
        self._intent_counter += 1
        intent_id = f"intent_{self._intent_counter:04d}"

        # Create new intent
        intent = Intent(
            intent_id=intent_id,
            text=text,
            intent_type=intent_type,
            status=IntentStatus.ACTIVE,
            start_tick=current_tick,
            base_timeout_ticks=timeout_ticks or self.default_timeout_ticks,
            criteria=criteria,
            stability_counter=0,
            focus_hint=focus_hint,
        )

        # Compute and cache embedding (legacy, but kept for compatibility)
        try:
            intent.embedding = self.embedder.embed(text)
        except ImportError:
            # sentence-transformers not installed, skip embedding
            intent.embedding = self.embedder.get_zero_embedding()

        # Activate
        self._active_intent = intent
        self._log_event("started", intent, current_tick)

        return intent

    def cancel_intent(self, current_tick: int, reason: str = "user_canceled") -> Optional[Intent]:
        """Explicitly cancel the current intent."""
        if self._active_intent is None:
            return None

        return self._terminate_intent(IntentStatus.CANCELED, current_tick, reason=reason)

    def update(self, current_tick: int, world_state: dict) -> Optional[IntentEvent]:
        """
        Update intent state based on current world state.
        Called every tick to check completion/timeout.
        Returns an IntentEvent if the intent terminated this tick.
        """
        if self._active_intent is None:
            return None

        intent = self._active_intent
        npc_pos = world_state["npc"]["position"]
        npc_position = (npc_pos["x"], npc_pos["y"])

        # Check timeout first
        elapsed = current_tick - intent.start_tick
        progress = self._estimate_progress(intent, npc_position, world_state)
        timeout = intent.get_adaptive_timeout(progress)

        if elapsed >= timeout:
            return self._terminate_intent(
                IntentStatus.TIMEOUT,
                current_tick,
                reason="timeout",
                progress=progress
            )

        # Check completion criteria (only after minimum ticks have passed)
        # This prevents "spawn luck" completions where agent starts at target
        if intent.criteria is not None and elapsed >= self.min_ticks_before_completion:
            if intent.criteria.check(npc_position, world_state):
                intent.stability_counter += 1
                if intent.stability_counter >= self.stability_required:
                    return self._terminate_intent(
                        IntentStatus.COMPLETED,
                        current_tick,
                        reason="criteria_met"
                    )
            else:
                # Reset stability if we moved away
                intent.stability_counter = 0

        return None

    def _terminate_intent(
        self,
        status: IntentStatus,
        current_tick: int,
        reason: str = "",
        **metadata
    ) -> IntentEvent:
        """Internal method to terminate the current intent."""
        intent = self._active_intent
        intent.status = status
        intent.end_tick = current_tick

        self._intent_history.append(intent)
        self._active_intent = None

        event = self._log_event(
            status.name.lower(),
            intent,
            current_tick,
            reason=reason,
            **metadata
        )

        return event

    def _estimate_progress(
        self,
        intent: Intent,
        npc_position: tuple,
        world_state: dict
    ) -> float:
        """
        Estimate progress toward intent completion (0.0 to 1.0).
        Used for adaptive timeout and reward shaping.
        """
        if intent.criteria is None:
            return 0.0

        criteria = intent.criteria

        # Get target position
        if criteria.target_position is not None:
            target = criteria.target_position
        elif criteria.target_entity_id is not None:
            target = criteria._get_entity_position(
                criteria.target_entity_id, world_state
            )
            if target is None:
                return 0.0
        else:
            return 0.0

        # Calculate current distance
        dx = npc_position[0] - target[0]
        dy = npc_position[1] - target[1]
        current_dist = (dx * dx + dy * dy) ** 0.5

        # Estimate initial distance (rough heuristic: half the world diagonal)
        # Use actual world_size from state, with fallback for backward compatibility
        world_size = world_state.get("world_size", 64)
        max_dist = world_size * 1.414 / 2  # half the diagonal

        if current_dist >= max_dist:
            return 0.0
        elif current_dist <= criteria.distance_threshold:
            return 1.0
        else:
            # Linear progress estimate
            return 1.0 - (current_dist - criteria.distance_threshold) / (max_dist - criteria.distance_threshold)

    def _log_event(
        self,
        event_type: str,
        intent: Intent,
        tick: int,
        **metadata
    ) -> IntentEvent:
        """Log an intent event."""
        event = IntentEvent(
            event_type=event_type,
            intent=intent,
            tick=tick,
            metadata=metadata
        )
        self._event_log.append(event)
        return event

    def get_current_embedding(self) -> np.ndarray:
        """Get embedding for current intent, or zero vector if none."""
        if self._active_intent is not None and self._active_intent.embedding is not None:
            return self._active_intent.embedding
        return self.embedder.get_zero_embedding()

    def get_intent_age(self, current_tick: int) -> int:
        """Get how many ticks the current intent has been active."""
        if self._active_intent is None:
            return 0
        return current_tick - self._active_intent.start_tick

    def get_state(self, current_tick: int) -> dict:
        """Get serializable state for Unity/logging."""
        if self._active_intent is None:
            return {
                "has_intent": False,
                "intent": None,
                "age_ticks": 0,
            }

        return {
            "has_intent": True,
            "intent": self._active_intent.to_dict(),
            "age_ticks": self.get_intent_age(current_tick),
        }

    def get_history(self) -> List[dict]:
        """Get history of all completed intents."""
        return [intent.to_dict() for intent in self._intent_history]

    def get_event_log(self) -> List[dict]:
        """Get full event log for training/analysis."""
        return [
            {
                "event_type": e.event_type,
                "intent_id": e.intent.intent_id,
                "tick": e.tick,
                "metadata": e.metadata,
            }
            for e in self._event_log
        ]

    def clear_history(self) -> None:
        """Clear history and event log (keeps active intent)."""
        self._intent_history.clear()
        self._event_log.clear()
