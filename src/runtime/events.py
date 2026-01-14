from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from collections import deque


class EventType(Enum):
    """Types of events that can occur during runtime."""
    # User input events
    NEW_INSTRUCTION = auto()    # User gives a new instruction
    CANCEL_INSTRUCTION = auto() # User cancels current instruction
    USER_MOVE = auto()          # User moves in the world

    # System events
    INTENT_STARTED = auto()
    INTENT_COMPLETED = auto()
    INTENT_CANCELED = auto()
    INTENT_TIMEOUT = auto()

    # NPC events
    NPC_SPEAK = auto()          # NPC produced speech output


@dataclass
class Event:
    """
    Represents an event in the runtime system.
    Events can come from user input or system state changes.
    """
    event_type: EventType
    tick: int
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name.lower(),
            "tick": self.tick,
            "data": self.data,
        }


class EventQueue:
    """
    Queue for managing events between ticks.
    User inputs are queued and processed at the start of each tick.
    """

    def __init__(self, max_size: int = 100):
        self._queue: deque = deque(maxlen=max_size)
        self._processed: List[Event] = []

    def push(self, event: Event) -> None:
        """Add an event to the queue."""
        self._queue.append(event)

    def push_instruction(self, text: str, tick: int) -> None:
        """Convenience method to queue a new instruction."""
        self.push(Event(
            event_type=EventType.NEW_INSTRUCTION,
            tick=tick,
            data={"text": text},
        ))

    def push_cancel(self, tick: int, reason: str = "user_request") -> None:
        """Convenience method to queue a cancellation."""
        self.push(Event(
            event_type=EventType.CANCEL_INSTRUCTION,
            tick=tick,
            data={"reason": reason},
        ))

    def push_user_move(self, dx: float, dy: float, tick: int) -> None:
        """Convenience method to queue a user movement."""
        self.push(Event(
            event_type=EventType.USER_MOVE,
            tick=tick,
            data={"dx": dx, "dy": dy},
        ))

    def pop_all(self) -> List[Event]:
        """Pop all pending events from the queue."""
        events = list(self._queue)
        self._queue.clear()
        return events

    def record_processed(self, event: Event) -> None:
        """Record an event that was processed (for logging)."""
        self._processed.append(event)

    def get_processed_history(self) -> List[Dict[str, Any]]:
        """Get all processed events as dicts."""
        return [e.to_dict() for e in self._processed]

    def clear_history(self) -> None:
        """Clear processed event history."""
        self._processed.clear()

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def history_count(self) -> int:
        return len(self._processed)
