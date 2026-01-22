"""
Intent parser that converts natural language to structured commands.

This is the main interface between user language and the action system.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json

from .client import LLMClient, LLMConfig, create_client


@dataclass
class ParsedIntent:
    """Result of parsing user intent - structured for Controller."""
    type: str  # "task" or "conversation"
    action: Optional[str] = None  # "navigate", "wait", "combat", "unsupported" (for tasks)
    target_type: Optional[str] = None  # "user", "object", "enemy"
    target: Optional[str] = None  # Raw target description (e.g., "enemy", "goblin")
    color: Optional[str] = None  # red/blue/green/yellow/purple/orange
    shape: Optional[str] = None  # circle/triangle/square/diamond
    size: Optional[str] = None  # small/medium/large
    response: Optional[str] = None  # For conversation type
    raw_input: str = ""  # Original user input

    @property
    def is_task(self) -> bool:
        """Check if this is a task (vs conversation)."""
        return self.type == "task"

    @property
    def is_conversation(self) -> bool:
        """Check if this is a conversation."""
        return self.type == "conversation"

    @property
    def is_valid_task(self) -> bool:
        """Check if task is actionable by Controller."""
        if not self.is_task:
            return False
        if self.action == "unsupported":
            return False
        if self.action == "navigate" and not self.target_type:
            return False
        # Combat is valid even without specific target (attacks nearest)
        if self.action == "combat":
            return True
        return True

    def to_command(self) -> Dict[str, Any]:
        """Convert to command dict for Controller (tasks only)."""
        if not self.is_task:
            return {}
        return {
            "action": self.action,
            "target_type": self.target_type,
            "target": self.target,
            "color": self.color,
            "shape": self.shape,
            "size": self.size,
        }


# System prompt for intent parsing
# Output is structured for Controller to validate and resolve
INTENT_SYSTEM_PROMPT = """You are an NPC. Classify user input and respond with JSON.

Two types of input:
1. TASK: user wants you to do something (navigate, wait, combat)
2. CONVERSATION: user is chatting (greetings, questions, chitchat)

For TASK, extract:
- action: "navigate", "wait", "combat", or "unsupported" (if impossible like "fly", "teleport")
- target_type: "user", "object", or "enemy"
- target: raw target description (for combat: "enemy", "enemies", etc.)
- color: red/blue/green/yellow/purple/orange (or null)
- shape: circle/triangle/square/diamond (or null)
- size: small/medium/large (or null)

For CONVERSATION, generate a friendly response.

Output formats:
TASK: {"type": "task", "action": "...", "target_type": "...", "target": ..., "color": ..., "shape": ..., "size": ...}
CONVERSATION: {"type": "conversation", "response": "..."}

Examples:
"go to the red circle" -> {"type": "task", "action": "navigate", "target_type": "object", "target": null, "color": "red", "shape": "circle", "size": null}
"walk to the player" -> {"type": "task", "action": "navigate", "target_type": "user", "target": null, "color": null, "shape": null, "size": null}
"move to the large blue square" -> {"type": "task", "action": "navigate", "target_type": "object", "target": null, "color": "blue", "shape": "square", "size": "large"}
"wait" -> {"type": "task", "action": "wait", "target_type": null, "target": null, "color": null, "shape": null, "size": null}
"attack the enemy" -> {"type": "task", "action": "combat", "target_type": "enemy", "target": "enemy", "color": null, "shape": null, "size": null}
"fight them" -> {"type": "task", "action": "combat", "target_type": "enemy", "target": "enemies", "color": null, "shape": null, "size": null}
"kill the goblin" -> {"type": "task", "action": "combat", "target_type": "enemy", "target": "goblin", "color": null, "shape": null, "size": null}
"good morning" -> {"type": "conversation", "response": "Good morning! What can I do for you?"}
"hello there" -> {"type": "conversation", "response": "Hello! How can I help?"}
"how are you" -> {"type": "conversation", "response": "I'm doing well, thanks for asking!"}
"""

# System prompt for generating user-facing responses
RESPONSE_SYSTEM_PROMPT = """You are an NPC. Generate a short response based on the result.

Status types:
- "success": task completed (navigation arrived, combat won, etc.)
- "ambiguous": multiple matching objects, ask user to clarify
- "unsupported": action not available, tell user you can't do that
- "failed": couldn't complete, explain why
- "not_found": target not found (no enemies, object doesn't exist)

Output JSON: {"response": "your message"}

Examples:
- navigation success -> {"response": "Done, I'm at the blue triangle."}
- combat success -> {"response": "I defeated the enemy!"}
- ambiguous with matches [red circle, red square] -> {"response": "Which one? I see a red circle and a red square."}
- unsupported action -> {"response": "Sorry, I can't do that."}
- not_found (no enemies) -> {"response": "I don't see any enemies to fight."}
- failed (combat lost) -> {"response": "I was defeated..."}
"""


class IntentParser:
    """
    Parses natural language into structured intents.

    This class is the bridge between user language and the execution system.
    It uses a local LLM to parse natural language into structured commands.

    Usage:
        # Using Ollama (recommended)
        parser = IntentParser(backend="ollama", model_name="qwen3:4b")

        # Using llama-cpp-python
        parser = IntentParser(
            backend="llama-cpp",
            model_path="models/llm/qwen2.5-3b-instruct-q5_k_m.gguf"
        )

        # Parse user input
        intent = parser.parse("go to the red box")
        print(intent.action)  # "navigate"
        print(intent.target)  # "red box"

        # Generate response from module result
        response = parser.respond_to_result(
            user_input="go to the box",
            result={"status": "needs_clarification", "options": ["red box", "blue box"]}
        )
    """

    def __init__(
        self,
        backend: str = "ollama",
        model_name: str = "qwen3:4b",
        model_path: str = "models/llm/qwen2.5-3b-instruct-q5_k_m.gguf",
        n_gpu_layers: int = -1,
    ):
        """
        Initialize intent parser.

        Args:
            backend: "ollama" or "llama-cpp"
            model_name: Ollama model name (e.g., "qwen3:4b")
            model_path: Path to GGUF model file (for llama-cpp backend)
            n_gpu_layers: GPU layers for llama-cpp (-1 = all, 0 = CPU only)
        """
        self._config = LLMConfig(
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
        )
        self._client = create_client(self._config)
        self._conversation_history: List[Dict[str, str]] = []

    def load(self) -> None:
        """Pre-load the model (optional, auto-loads on first use)."""
        self._client.load()

    def unload(self) -> None:
        """Unload model from memory."""
        self._client.unload()

    @property
    def is_loaded(self) -> bool:
        return self._client.is_loaded

    def parse(self, user_input: str, debug: bool = False) -> ParsedIntent:
        """
        Parse user natural language into structured intent.

        Args:
            user_input: Raw user text (e.g., "go to the red box")
            debug: Print raw LLM output for debugging

        Returns:
            ParsedIntent with action, target, etc.
        """
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        try:
            result = self._client.generate_json(
                messages=messages,
                max_tokens=128,
                temperature=0.5,  # Moderate temp - 0.1 causes empty responses
                debug=debug,
            )

            intent_type = result.get("type", "task")

            if intent_type == "conversation":
                return ParsedIntent(
                    type="conversation",
                    response=result.get("response", ""),
                    raw_input=user_input,
                )
            else:
                return ParsedIntent(
                    type="task",
                    action=result.get("action", "unsupported"),
                    target_type=result.get("target_type"),
                    target=result.get("target"),
                    color=result.get("color"),
                    shape=result.get("shape"),
                    size=result.get("size"),
                    raw_input=user_input,
                )

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return ParsedIntent(
                type="task",
                action="unsupported",
                raw_input=user_input,
            )

    def respond_to_result(
        self,
        user_input: str,
        result: Dict[str, Any],
    ) -> str:
        """
        Generate a user-facing response based on module result.

        Args:
            user_input: The original user instruction
            result: Structured result from gateway/module

        Returns:
            Natural language response for the user
        """
        # Build context about what happened
        status = result.get("status", "unknown")

        if status == "success":
            target_id = result.get("target_id", "target")
            context = f"Success. Reached {target_id}."
        elif status == "ambiguous":
            matches = result.get("matches", [])
            context = f"Ambiguous. Multiple matches: {matches}. Ask user which one."
        elif status == "unsupported":
            reason = result.get("reason", "action not available")
            context = f"Unsupported action. Reason: {reason}."
        elif status == "failed":
            reason = result.get("reason", "unknown error")
            context = f"Failed. Reason: {reason}."
        else:
            context = f"Result: {result}"

        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"User said: \"{user_input}\"\n\nResult: {context}"},
        ]

        try:
            response = self._client.generate_json(
                messages=messages,
                max_tokens=128,
                temperature=0.7,
            )
            return response.get("response", "I'm not sure what happened.")

        except json.JSONDecodeError:
            # Fallback responses
            if status == "success":
                return "Done."
            elif status == "ambiguous":
                return "Which one do you mean?"
            elif status == "unsupported":
                return "I can't do that."
            else:
                return "Something went wrong."

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
