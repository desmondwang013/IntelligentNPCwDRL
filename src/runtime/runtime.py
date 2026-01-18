"""
Runtime orchestrator for NPC simulation.

NOTE: This runtime uses the legacy ObservationBuilder (with embeddings).
SimpleNPCEnv uses its own SimpleObservationBuilder internally and does not
rely on Runtime.get_observation() for training observations. The Runtime
is still used by SimpleNPCEnv for world simulation, intent tracking, and
reward calculation.

For the current architecture (no embeddings in RL), see:
- src/training/simple_env.py (SimpleNPCEnv)
- src/observation/simple_builder.py (SimpleObservationBuilder)
"""
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from src.world import World
from src.world.world import WorldConfig, Action
from src.intent import IntentManager, IntentType
from src.intent.intent import CompletionCriteria
from src.observation import ObservationBuilder
from src.reward import RewardCalculator, RewardConfig, RewardInfo
from .events import Event, EventType, EventQueue


@dataclass
class RuntimeConfig:
    """Configuration for the runtime loop."""
    ticks_per_second: int = 16
    world_seed: Optional[int] = None
    default_intent_timeout: int = 480  # 30 seconds
    enable_logging: bool = True
    reward_config: Optional[RewardConfig] = None  # Uses defaults if None


@dataclass
class StepResult:
    """Result of a single runtime step."""
    tick: int
    observation: np.ndarray
    world_state: Dict[str, Any]
    intent_state: Dict[str, Any]
    events: List[Event]
    npc_action: int
    npc_speech: Optional[str]
    reward: float
    reward_info: RewardInfo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "observation_shape": self.observation.shape,
            "world_state": self.world_state,
            "intent_state": self.intent_state,
            "events": [e.to_dict() for e in self.events],
            "npc_action": Action(self.npc_action).name.lower(),
            "npc_speech": self.npc_speech,
            "reward": self.reward,
            "reward_info": self.reward_info.to_dict(),
        }


class Runtime:
    """
    Main runtime loop that orchestrates World, IntentManager, and ObservationBuilder.

    Flow per tick:
    1. Process pending user events (new instructions, cancellations, movement)
    2. Get observation for policy
    3. Policy selects action (external)
    4. Apply action to world
    5. Update intent state (check completion/timeout)
    6. Return step result

    Usage:
        runtime = Runtime()
        obs = runtime.get_observation()

        while running:
            action = policy(obs)  # external policy
            result = runtime.step(action)
            obs = result.observation
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        world_config: Optional[WorldConfig] = None,
    ):
        self.config = config or RuntimeConfig()

        # Initialize subsystems
        self._world = World(config=world_config, seed=self.config.world_seed)
        self._intent_manager = IntentManager(
            default_timeout_ticks=self.config.default_intent_timeout
        )
        # Pass world size to observation builder for proper normalization
        from src.observation.builder import ObservationConfig
        obs_config = ObservationConfig(world_size=self._world.config.size)
        self._obs_builder = ObservationBuilder(config=obs_config)
        self._event_queue = EventQueue()
        self._reward_calculator = RewardCalculator(
            config=self.config.reward_config
        )

        # Logging
        self._step_history: List[StepResult] = []
        self._tick_events: List[Event] = []  # events generated this tick

    @property
    def tick(self) -> int:
        return self._world.tick_count

    @property
    def world(self) -> World:
        return self._world

    @property
    def intent_manager(self) -> IntentManager:
        return self._intent_manager

    @property
    def event_queue(self) -> EventQueue:
        return self._event_queue

    @property
    def observation_dim(self) -> int:
        return self._obs_builder.observation_dim

    @property
    def reward_calculator(self) -> RewardCalculator:
        return self._reward_calculator

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the runtime to initial state.
        Returns the initial observation.
        """
        self._world.reset(seed=seed)

        # Cancel any active intent
        if self._intent_manager.has_active_intent:
            self._intent_manager.cancel_intent(self.tick, reason="runtime_reset")

        self._step_history.clear()
        self._tick_events.clear()
        self._reward_calculator.reset()

        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Get the current observation for the policy."""
        world_state = self._world.get_state()
        intent_embedding = self._intent_manager.get_current_embedding()
        intent_age = self._intent_manager.get_intent_age(self.tick)

        focus_hint = None
        if self._intent_manager.active_intent:
            focus_hint = self._intent_manager.active_intent.focus_hint

        return self._obs_builder.build(
            world_state=world_state,
            intent_embedding=intent_embedding,
            intent_age_ticks=intent_age,
            focus_hint=focus_hint,
        )

    def step(
        self,
        npc_action: int,
        npc_speech: Optional[str] = None,
    ) -> StepResult:
        """
        Advance the simulation by one tick.

        Args:
            npc_action: Action ID (0-5) selected by the policy
            npc_speech: Optional speech content if action is SPEAK

        Returns:
            StepResult with observation, state, reward, and events
        """
        self._tick_events.clear()

        # 1. Process pending user events
        self._process_events()

        # 2. Record position before action (for collision detection)
        pos_before = self._world.npc.position.copy()

        # 3. Apply NPC action to world
        world_state = self._world.step(npc_action, npc_speech)

        # 4. Detect collision (tried to move but didn't)
        pos_after = self._world.npc.position
        is_move_action = npc_action in (
            Action.MOVE_UP.value,
            Action.MOVE_DOWN.value,
            Action.MOVE_LEFT.value,
            Action.MOVE_RIGHT.value,
        )
        collision_occurred = (
            is_move_action and
            pos_before.x == pos_after.x and
            pos_before.y == pos_after.y
        )

        # Record speech event if applicable
        if npc_action == Action.SPEAK.value and npc_speech:
            self._emit_event(EventType.NPC_SPEAK, {"text": npc_speech})

        # 5. Update intent state
        intent_event = self._intent_manager.update(self.tick, world_state)
        if intent_event:
            self._emit_intent_event(intent_event)

        # 6. Calculate reward
        intent_state = self._intent_manager.get_state(self.tick)
        reward_info = self._reward_calculator.calculate(
            world_state=world_state,
            intent_state=intent_state,
            action=npc_action,
            collision_occurred=collision_occurred,
            intent_event=intent_event,
        )

        # 7. Build observation
        observation = self.get_observation()

        # 8. Build result
        result = StepResult(
            tick=self.tick,
            observation=observation,
            world_state=world_state,
            intent_state=intent_state,
            events=list(self._tick_events),
            npc_action=npc_action,
            npc_speech=npc_speech,
            reward=reward_info.total,
            reward_info=reward_info,
        )

        if self.config.enable_logging:
            self._step_history.append(result)

        return result

    def submit_instruction(
        self,
        text: str,
        intent_type: IntentType = IntentType.GENERIC,
        target_entity_id: Optional[str] = None,
        target_position: Optional[Tuple[float, float]] = None,
        distance_threshold: float = 2.0,
    ) -> None:
        """
        Submit a new instruction (queued for next tick).

        For immediate processing, call this then step().
        """
        self._event_queue.push(Event(
            event_type=EventType.NEW_INSTRUCTION,
            tick=self.tick,
            data={
                "text": text,
                "intent_type": intent_type,
                "target_entity_id": target_entity_id,
                "target_position": target_position,
                "distance_threshold": distance_threshold,
            },
        ))

    def cancel_instruction(self, reason: str = "user_request") -> None:
        """Cancel the current instruction (queued for next tick)."""
        self._event_queue.push_cancel(self.tick, reason)

    def move_user(self, dx: float, dy: float) -> None:
        """Move the user (queued for next tick)."""
        self._event_queue.push_user_move(dx, dy, self.tick)

    def _process_events(self) -> None:
        """Process all pending events at the start of the tick."""
        events = self._event_queue.pop_all()

        for event in events:
            self._handle_event(event)
            self._event_queue.record_processed(event)

    def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        if event.event_type == EventType.NEW_INSTRUCTION:
            self._handle_new_instruction(event)
        elif event.event_type == EventType.CANCEL_INSTRUCTION:
            self._handle_cancel(event)
        elif event.event_type == EventType.USER_MOVE:
            self._handle_user_move(event)

    def _handle_new_instruction(self, event: Event) -> None:
        """Handle a new instruction event."""
        data = event.data
        text = data["text"]
        intent_type = data.get("intent_type", IntentType.GENERIC)

        # Build completion criteria if target provided
        criteria = None
        target_entity = data.get("target_entity_id")
        target_pos = data.get("target_position")
        threshold = data.get("distance_threshold", 2.0)

        if target_entity or target_pos:
            criteria = CompletionCriteria(
                target_entity_id=target_entity,
                target_position=target_pos,
                distance_threshold=threshold,
            )

        # Create the intent
        intent = self._intent_manager.new_intent(
            text=text,
            current_tick=self.tick,
            intent_type=intent_type,
            criteria=criteria,
            focus_hint=target_entity,
        )

        self._emit_event(EventType.INTENT_STARTED, {
            "intent_id": intent.intent_id,
            "text": text,
        })

    def _handle_cancel(self, event: Event) -> None:
        """Handle a cancel instruction event."""
        reason = event.data.get("reason", "user_request")
        self._intent_manager.cancel_intent(self.tick, reason)

    def _handle_user_move(self, event: Event) -> None:
        """Handle a user movement event."""
        dx = event.data.get("dx", 0.0)
        dy = event.data.get("dy", 0.0)
        self._world.move_user(dx, dy)

    def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit an event for this tick."""
        event = Event(event_type=event_type, tick=self.tick, data=data)
        self._tick_events.append(event)

    def _emit_intent_event(self, intent_event) -> None:
        """Convert an IntentManager event to a runtime event."""
        status = intent_event.intent.status.name.lower()
        event_type_map = {
            "completed": EventType.INTENT_COMPLETED,
            "canceled": EventType.INTENT_CANCELED,
            "timeout": EventType.INTENT_TIMEOUT,
        }
        if status in event_type_map:
            self._emit_event(event_type_map[status], {
                "intent_id": intent_event.intent.intent_id,
                "text": intent_event.intent.text,
                "reason": intent_event.metadata.get("reason", ""),
            })

    def get_state(self) -> Dict[str, Any]:
        """Get complete runtime state (for Unity/debugging)."""
        return {
            "tick": self.tick,
            "world": self._world.get_state(),
            "intent": self._intent_manager.get_state(self.tick),
            "pending_events": self._event_queue.pending_count,
        }

    def get_step_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get step history for analysis."""
        history = self._step_history[-last_n:] if last_n else self._step_history
        return [r.to_dict() for r in history]
