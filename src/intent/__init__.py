"""
DEPRECATED: Intent system has moved to src/controller/

This module re-exports from src/controller for backward compatibility.
New code should import from src.controller directly:

    from src.controller import IntentManager, IntentType, Intent
"""
import warnings

# Re-export from new location for backward compatibility
from src.controller import (
    Intent,
    IntentStatus,
    IntentType,
    CompletionCriteria,
    IntentManager,
    IntentEvent,
    TextEmbedder,
)

__all__ = [
    "Intent",
    "IntentStatus",
    "IntentType",
    "CompletionCriteria",
    "IntentManager",
    "IntentEvent",
    "TextEmbedder",
]
