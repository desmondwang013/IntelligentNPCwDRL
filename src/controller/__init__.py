"""
Controller module - The NPC's decision-making hub.

This module consolidates:
- NPCController: Main controller that processes LLM intents
- WorldQuery: Read-only world state queries
- IntentManager: Intent lifecycle tracking
- Intent data structures

Usage:
    from src.controller import NPCController, WorldQuery
    from src.controller import IntentType, IntentStatus  # For compatibility

Flow:
    LLM (ParsedIntent) → NPCController → Navigator/Modules → Result → LLM
"""

# Main controller classes
from .npc_controller import (
    NPCController,
    ControllerResult,
    NavigationCommand,
    CombatCommand,
    ActiveTask,
    TaskStatus,
)
from .world_query import WorldQuery, ObjectMatch, EnemyMatch

# Intent system (consolidated from old src/intent/)
from .intent import Intent, IntentStatus, IntentType, CompletionCriteria
from .intent_manager import IntentManager, IntentEvent
from .embedder import TextEmbedder

__all__ = [
    # Controller
    "NPCController",
    "ControllerResult",
    "NavigationCommand",
    "CombatCommand",
    "ActiveTask",
    "TaskStatus",
    # World query
    "WorldQuery",
    "ObjectMatch",
    "EnemyMatch",
    # Intent system
    "Intent",
    "IntentStatus",
    "IntentType",
    "CompletionCriteria",
    "IntentManager",
    "IntentEvent",
    "TextEmbedder",
]
