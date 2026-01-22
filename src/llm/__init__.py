"""
LLM module for natural language understanding.

This module provides:
- IntentParser: Converts user natural language to structured commands
- LLMClient: Low-level interface to local LLM (llama-cpp-python)

The LLM's role is strictly language understanding. It does NOT:
- Access world state directly
- Make grounding decisions (that's the gateway's job)
- Execute actions (that's the modules' job)

Example usage:
    from src.llm import IntentParser

    parser = IntentParser(model_path="models/llm/qwen2.5-3b-instruct-q5_k_m.gguf")

    # Parse user intent
    result = parser.parse("go to the red box")
    # Returns: {"action": "navigate", "target": "red box"}

    # Handle module feedback
    response = parser.generate_response(
        user_input="go to the box",
        module_result={"status": "needs_clarification", "options": ["red box", "blue box"]}
    )
    # Returns: "I see two boxes - a red one and a blue one. Which do you mean?"
"""

from .client import LLMClient, LLMConfig, OllamaClient, LlamaCppClient, create_client
from .intent_parser import IntentParser, ParsedIntent

__all__ = [
    "LLMClient",
    "LLMConfig",
    "OllamaClient",
    "LlamaCppClient",
    "create_client",
    "IntentParser",
    "ParsedIntent",
]
