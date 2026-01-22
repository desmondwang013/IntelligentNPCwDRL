"""
NPCController - The NPC's decision-making hub.

This module orchestrates the flow from LLM intent to action execution:
1. Receives ParsedIntent from LLM
2. Resolves targets using WorldQuery
3. Manages task state via IntentManager
4. Returns structured results for LLM to narrate

The controller is async/non-blocking - it dispatches commands and tracks state,
but doesn't block waiting for navigation to complete.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto

from .world_query import WorldQuery, ObjectMatch, EnemyMatch
from .intent import Intent, IntentStatus, IntentType, CompletionCriteria
from .intent_manager import IntentManager


class TaskStatus(Enum):
    """Status of a controller task."""
    SUCCESS = auto()        # Task completed successfully
    AMBIGUOUS = auto()      # Multiple targets match, need clarification
    NOT_FOUND = auto()      # Target not found in world
    UNSUPPORTED = auto()    # Action not supported
    IN_PROGRESS = auto()    # Navigation in progress
    FAILED = auto()         # Task failed (timeout, stuck, etc.)
    CANCELED = auto()       # Task was canceled


@dataclass
class ControllerResult:
    """
    Result from NPCController operations.

    Used for both immediate results (ambiguous, not_found) and
    completion results (success, failed, timeout).
    """
    status: TaskStatus
    target_id: Optional[str] = None
    target_position: Optional[Tuple[float, float]] = None
    reason: Optional[str] = None
    matches: Optional[List[str]] = None  # For ambiguous results
    steps: int = 0  # Ticks taken (for completed tasks)
    response: Optional[str] = None  # For conversation type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format matching test_llm.py expectations."""
        result = {"status": self.status.name.lower()}

        if self.target_id:
            result["target_id"] = self.target_id
        if self.reason:
            result["reason"] = self.reason
        if self.matches:
            result["matches"] = self.matches
        if self.steps > 0:
            result["steps"] = self.steps
        if self.response:
            result["response"] = self.response

        return result


@dataclass
class NavigationCommand:
    """
    Command to start navigation to a target.

    Returned by process_intent() when navigation is needed.
    The caller (main loop) should use this to set up the Navigator.
    """
    target_id: str
    target_position: Tuple[float, float]
    intent_id: str  # For tracking


@dataclass
class CombatCommand:
    """
    Command to start combat.

    Returned by process_intent() when combat is requested.
    The caller (main loop) should use this to set up the Combat system.
    """
    target_id: str  # Enemy entity ID (or "nearest" for auto-target)
    target_position: Tuple[float, float]
    intent_id: str  # For tracking
    enemy_count: int  # Number of alive enemies when combat started


@dataclass
class ActiveTask:
    """Tracks the currently active task."""
    intent_id: str
    action: str  # "navigate", "wait", "combat"
    target_id: Optional[str]
    target_position: Optional[Tuple[float, float]]
    start_tick: int
    user_input: str  # Original input for LLM response
    initial_enemy_count: int = 0  # For combat: enemies alive at start


class NPCController:
    """
    The NPC's decision-making controller.

    Handles the full cycle from LLM intent to action result:
    - Resolves target descriptions to entity IDs
    - Manages task lifecycle (start, progress, completion, cancellation)
    - Provides structured results for LLM to narrate

    Usage:
        controller = NPCController()

        # Process intent from LLM
        result = controller.process_intent(parsed_intent, world_state)

        if isinstance(result, NavigationCommand):
            # Start navigation using the command
            navigator.set_goal(npc_pos, result.target_position, world_state)

        # Each tick during navigation
        if controller.has_active_task:
            # Check for completion (caller determines when navigation is done)
            if navigation_complete:
                result = controller.report_navigation_complete(world_state, success=True)
                # result.to_dict() -> {"status": "success", "target_id": "obj_3", "steps": 45}

        # New instruction interrupts current task
        result = controller.process_intent(new_intent, world_state)  # Auto-cancels previous
    """

    def __init__(
        self,
        default_timeout_ticks: int = 480,
        arrival_threshold: float = 2.0,
    ):
        """
        Initialize the controller.

        Args:
            default_timeout_ticks: Default timeout for tasks (30s at 16 ticks/s)
            arrival_threshold: Distance threshold for "arrived" detection
        """
        self._intent_manager = IntentManager(
            default_timeout_ticks=default_timeout_ticks,
        )
        self._arrival_threshold = arrival_threshold
        self._active_task: Optional[ActiveTask] = None
        self._current_tick: int = 0

    # =========================================================================
    # Main Interface
    # =========================================================================

    def process_intent(
        self,
        parsed_intent: Any,  # ParsedIntent from LLM
        world_state: Dict[str, Any],
    ) -> ControllerResult | NavigationCommand:
        """
        Process a parsed intent from the LLM.

        Args:
            parsed_intent: ParsedIntent from LLM (has type, action, target attributes)
            world_state: Current world state from World.get_state()

        Returns:
            - ControllerResult for immediate results (conversation, wait, errors)
            - NavigationCommand if navigation should be started
        """
        self._current_tick = world_state.get("tick", 0)
        query = WorldQuery(world_state)

        # Handle conversation (not a task)
        if parsed_intent.type == "conversation":
            return ControllerResult(
                status=TaskStatus.SUCCESS,
                response=parsed_intent.response,
            )

        # Handle unsupported action
        if parsed_intent.action == "unsupported":
            return ControllerResult(
                status=TaskStatus.UNSUPPORTED,
                reason="action not available",
            )

        # Handle wait action
        if parsed_intent.action == "wait":
            # Cancel any active task
            if self._active_task:
                self._cancel_active_task("new_wait_command")

            return ControllerResult(
                status=TaskStatus.SUCCESS,
                reason="waiting",
            )

        # Handle navigate action
        if parsed_intent.action == "navigate":
            return self._process_navigate(parsed_intent, query, world_state)

        # Handle combat action
        if parsed_intent.action == "combat":
            return self._process_combat(parsed_intent, query, world_state)

        # Unknown action
        return ControllerResult(
            status=TaskStatus.UNSUPPORTED,
            reason=f"unknown action: {parsed_intent.action}",
        )

    def _process_navigate(
        self,
        intent: Any,
        query: WorldQuery,
        world_state: Dict[str, Any],
    ) -> ControllerResult | NavigationCommand:
        """Process a navigate action."""
        # Cancel any active task first
        if self._active_task:
            self._cancel_active_task("new_navigation_command")

        # Resolve target
        if intent.target_type == "user":
            # Navigate to user
            target_pos = query.get_user_position()
            if target_pos is None:
                return ControllerResult(
                    status=TaskStatus.NOT_FOUND,
                    reason="user not found",
                )

            return self._start_navigation(
                target_id="user",
                target_position=target_pos,
                world_state=world_state,
                user_input=intent.raw_input,
            )

        elif intent.target_type == "object":
            # Find matching objects
            matches = query.find_objects(
                color=intent.color,
                shape=intent.shape,
                size=intent.size,
            )

            if len(matches) == 0:
                return ControllerResult(
                    status=TaskStatus.NOT_FOUND,
                    reason="no matching object",
                )

            if len(matches) > 1:
                # Ambiguous - multiple matches
                descriptions = [m.describe() for m in matches]
                return ControllerResult(
                    status=TaskStatus.AMBIGUOUS,
                    reason="multiple matches",
                    matches=descriptions,
                )

            # Exactly one match - start navigation
            match = matches[0]
            return self._start_navigation(
                target_id=match.entity_id,
                target_position=match.position,
                world_state=world_state,
                user_input=intent.raw_input,
            )

        else:
            # No target type specified
            return ControllerResult(
                status=TaskStatus.UNSUPPORTED,
                reason="no target specified",
            )

    def _process_combat(
        self,
        intent: Any,
        query: WorldQuery,
        world_state: Dict[str, Any],
    ) -> ControllerResult | CombatCommand:
        """Process a combat action."""
        # Cancel any active task first
        if self._active_task:
            self._cancel_active_task("new_combat_command")

        # Check if there are any enemies
        enemies = query.find_enemies(alive_only=True)
        if not enemies:
            return ControllerResult(
                status=TaskStatus.NOT_FOUND,
                reason="no enemies to fight",
            )

        # Resolve target
        target = intent.target if hasattr(intent, 'target') else None
        target_type = getattr(intent, 'target_type', None)

        if target_type == "enemy" or target is None or target.lower() in ["enemy", "enemies", "nearest"]:
            # Attack nearest enemy
            nearest = query.get_nearest_enemy(alive_only=True)
            if nearest is None:
                return ControllerResult(
                    status=TaskStatus.NOT_FOUND,
                    reason="no enemies nearby",
                )

            return self._start_combat(
                target_id=nearest.entity_id,
                target_position=nearest.position,
                enemy_count=len(enemies),
                world_state=world_state,
                user_input=intent.raw_input,
            )

        else:
            # Try to find specific enemy by ID
            enemy = query.get_enemy_by_id(target)
            if enemy is None or not enemy.is_alive:
                return ControllerResult(
                    status=TaskStatus.NOT_FOUND,
                    reason=f"enemy '{target}' not found or dead",
                )

            return self._start_combat(
                target_id=enemy.entity_id,
                target_position=enemy.position,
                enemy_count=len(enemies),
                world_state=world_state,
                user_input=intent.raw_input,
            )

    def _start_combat(
        self,
        target_id: str,
        target_position: Tuple[float, float],
        enemy_count: int,
        world_state: Dict[str, Any],
        user_input: str,
    ) -> CombatCommand:
        """Start a new combat task."""
        # Create intent for tracking
        criteria = CompletionCriteria(
            target_entity_id=target_id,
            target_position=target_position,
        )

        intent = self._intent_manager.new_intent(
            text=user_input,
            current_tick=self._current_tick,
            intent_type=IntentType.MOVE_TO_OBJECT,  # Reuse for now
            criteria=criteria,
            focus_hint=target_id,
        )

        # Track active task
        self._active_task = ActiveTask(
            intent_id=intent.intent_id,
            action="combat",
            target_id=target_id,
            target_position=target_position,
            start_tick=self._current_tick,
            user_input=user_input,
            initial_enemy_count=enemy_count,
        )

        # Return command for caller to execute
        return CombatCommand(
            target_id=target_id,
            target_position=target_position,
            intent_id=intent.intent_id,
            enemy_count=enemy_count,
        )

    def _start_navigation(
        self,
        target_id: str,
        target_position: Tuple[float, float],
        world_state: Dict[str, Any],
        user_input: str,
    ) -> NavigationCommand:
        """Start a new navigation task."""
        # Create intent for tracking
        criteria = CompletionCriteria(
            target_entity_id=target_id,
            target_position=target_position,
            distance_threshold=self._arrival_threshold,
        )

        intent = self._intent_manager.new_intent(
            text=user_input,
            current_tick=self._current_tick,
            intent_type=IntentType.MOVE_TO_OBJECT if target_id != "user" else IntentType.MOVE_TO_USER,
            criteria=criteria,
            focus_hint=target_id,
        )

        # Track active task
        self._active_task = ActiveTask(
            intent_id=intent.intent_id,
            action="navigate",
            target_id=target_id,
            target_position=target_position,
            start_tick=self._current_tick,
            user_input=user_input,
        )

        # Return command for caller to execute
        return NavigationCommand(
            target_id=target_id,
            target_position=target_position,
            intent_id=intent.intent_id,
        )

    # =========================================================================
    # Task Lifecycle
    # =========================================================================

    def update(self, world_state: Dict[str, Any]) -> Optional[ControllerResult]:
        """
        Update controller state each tick.

        Checks for timeout on active tasks.

        Args:
            world_state: Current world state

        Returns:
            ControllerResult if task timed out, None otherwise
        """
        self._current_tick = world_state.get("tick", 0)

        if not self._active_task:
            return None

        # Check for timeout via IntentManager
        event = self._intent_manager.update(self._current_tick, world_state)
        if event and event.event_type == "timeout":
            result = ControllerResult(
                status=TaskStatus.FAILED,
                target_id=self._active_task.target_id,
                reason="timeout",
                steps=self._current_tick - self._active_task.start_tick,
            )
            self._active_task = None
            return result

        return None

    def report_navigation_complete(
        self,
        world_state: Dict[str, Any],
        success: bool = True,
        reason: Optional[str] = None,
    ) -> ControllerResult:
        """
        Report that navigation has completed.

        Called by the main loop when Navigator reports completion.

        Args:
            world_state: Current world state
            success: Whether navigation succeeded
            reason: Failure reason if not successful

        Returns:
            ControllerResult for LLM to narrate
        """
        self._current_tick = world_state.get("tick", 0)

        if not self._active_task:
            return ControllerResult(
                status=TaskStatus.FAILED,
                reason="no active task",
            )

        steps = self._current_tick - self._active_task.start_tick
        target_id = self._active_task.target_id

        # Complete the intent
        if success:
            self._intent_manager.cancel_intent(
                self._current_tick,
                reason="completed",
            )
            result = ControllerResult(
                status=TaskStatus.SUCCESS,
                target_id=target_id,
                steps=steps,
            )
        else:
            self._intent_manager.cancel_intent(
                self._current_tick,
                reason=reason or "navigation_failed",
            )
            result = ControllerResult(
                status=TaskStatus.FAILED,
                target_id=target_id,
                reason=reason or "navigation failed",
                steps=steps,
            )

        self._active_task = None
        return result

    def report_combat_complete(
        self,
        world_state: Dict[str, Any],
        victory: bool = True,
        reason: Optional[str] = None,
    ) -> ControllerResult:
        """
        Report that combat has completed.

        Called by the main loop when combat ends (all enemies dead or NPC dead).

        Args:
            world_state: Current world state
            victory: Whether NPC won (all enemies defeated)
            reason: Additional reason text if needed

        Returns:
            ControllerResult for LLM to narrate
        """
        self._current_tick = world_state.get("tick", 0)

        if not self._active_task or self._active_task.action != "combat":
            return ControllerResult(
                status=TaskStatus.FAILED,
                reason="no active combat task",
            )

        steps = self._current_tick - self._active_task.start_tick
        query = WorldQuery(world_state)
        enemies_remaining = query.get_alive_enemy_count()
        enemies_killed = self._active_task.initial_enemy_count - enemies_remaining

        # Complete the intent
        if victory:
            self._intent_manager.cancel_intent(
                self._current_tick,
                reason="combat_won",
            )
            result = ControllerResult(
                status=TaskStatus.SUCCESS,
                target_id=self._active_task.target_id,
                reason=f"defeated {enemies_killed} enemies",
                steps=steps,
            )
        else:
            self._intent_manager.cancel_intent(
                self._current_tick,
                reason=reason or "combat_lost",
            )
            result = ControllerResult(
                status=TaskStatus.FAILED,
                target_id=self._active_task.target_id,
                reason=reason or "combat lost",
                steps=steps,
            )

        self._active_task = None
        return result

    def cancel_task(self, reason: str = "user_request") -> Optional[ControllerResult]:
        """
        Cancel the current task.

        Args:
            reason: Reason for cancellation

        Returns:
            ControllerResult if there was a task to cancel, None otherwise
        """
        if not self._active_task:
            return None

        return self._cancel_active_task(reason)

    def _cancel_active_task(self, reason: str) -> ControllerResult:
        """Internal: Cancel the active task and return result."""
        steps = self._current_tick - self._active_task.start_tick
        target_id = self._active_task.target_id

        self._intent_manager.cancel_intent(self._current_tick, reason)

        result = ControllerResult(
            status=TaskStatus.CANCELED,
            target_id=target_id,
            reason=reason,
            steps=steps,
        )

        self._active_task = None
        return result

    # =========================================================================
    # State Queries
    # =========================================================================

    @property
    def has_active_task(self) -> bool:
        """Check if there's an active task."""
        return self._active_task is not None

    @property
    def active_task(self) -> Optional[ActiveTask]:
        """Get the current active task (if any)."""
        return self._active_task

    @property
    def intent_manager(self) -> IntentManager:
        """Access the intent manager (for advanced use)."""
        return self._intent_manager

    def get_state(self) -> Dict[str, Any]:
        """Get controller state for debugging/logging."""
        state = {
            "has_active_task": self.has_active_task,
            "current_tick": self._current_tick,
        }

        if self._active_task:
            state["active_task"] = {
                "intent_id": self._active_task.intent_id,
                "action": self._active_task.action,
                "target_id": self._active_task.target_id,
                "start_tick": self._active_task.start_tick,
                "elapsed_ticks": self._current_tick - self._active_task.start_tick,
            }

        state["intent_state"] = self._intent_manager.get_state(self._current_tick)
        return state
