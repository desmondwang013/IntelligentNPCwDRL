"""
LLM client abstraction supporting multiple backends.

Currently supported:
- Ollama (recommended for easy setup)
- llama-cpp-python (for grammar constraints, if you can build it)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import json


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    # Common settings
    backend: Literal["ollama", "llama-cpp"] = "ollama"
    model_name: str = "qwen3:4b"  # Ollama model name

    # llama-cpp specific
    model_path: str = "models/llm/qwen2.5-3b-instruct-q5_k_m.gguf"
    n_ctx: int = 2048
    n_gpu_layers: int = -1  # -1 = all on GPU

    # Ollama specific
    ollama_host: str = "http://localhost:11434"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load/initialize the model."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload/cleanup the model."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class OllamaClient(LLMClient):
    """
    LLM client using Ollama backend.

    Pros: Easy setup, no compilation needed
    Cons: No grammar constraints (relies on model following format)
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._loaded = False

    def load(self) -> None:
        """Initialize Ollama client."""
        if self._loaded:
            return

        import ollama
        # Ollama client connects on first use, just verify connection
        self._client = ollama

        # Check if model is available
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]

            # Check for exact match or partial match
            model_found = any(
                self.config.model_name in name or name.startswith(self.config.model_name.split(':')[0])
                for name in model_names
            )

            if not model_found:
                print(f"Model '{self.config.model_name}' not found in Ollama.")
                print(f"Available models: {model_names}")
                print(f"\nPull it with: ollama pull {self.config.model_name}")
                raise RuntimeError(f"Model {self.config.model_name} not available")

        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise RuntimeError(
                    "Cannot connect to Ollama. Make sure Ollama is running:\n"
                    "  1. Open a terminal\n"
                    "  2. Run: ollama serve\n"
                    "  3. In another terminal: ollama pull qwen3:4b"
                ) from e
            raise

        self._loaded = True

    def unload(self) -> None:
        """Cleanup (no-op for Ollama, server manages memory)."""
        self._client = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response."""
        if not self._loaded:
            self.load()

        import ollama

        # think=False disables Qwen3's thinking mode for faster responses
        response = ollama.chat(
            model=self.config.model_name,
            messages=messages,
            think=False,
            options={
                "temperature": temperature,
            }
        )

        return response.message.content

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        if not self._loaded:
            self.load()

        import ollama

        if debug:
            print(f"[DEBUG] Sending to Ollama:")
            print(f"  Model: {self.config.model_name}")
            print(f"  Temperature: {temperature}")
            print(f"  Messages: {len(messages)} messages")
            for i, m in enumerate(messages):
                role = m.get('role', 'unknown')
                content_preview = m.get('content', '')[:100]
                print(f"    [{i}] {role}: {content_preview}...")

        # Use Ollama's JSON format mode
        # think=False disables Qwen3's thinking mode for faster responses
        response = ollama.chat(
            model=self.config.model_name,
            messages=messages,
            format="json",
            think=False,
            options={
                "temperature": temperature,
            }
        )

        content = response.message.content.strip() if response.message.content else ""

        if debug:
            print(f"[DEBUG] Raw LLM output ({len(content)} chars):")
            print(f"  '{content}'")
            print()

        # Handle Qwen3 thinking mode - extract JSON after </think> if present
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        return json.loads(content)


class LlamaCppClient(LLMClient):
    """
    LLM client using llama-cpp-python backend.

    Pros: Grammar constraints guarantee valid JSON
    Cons: Requires compilation with CUDA support
    """

    def __init__(self, config: LLMConfig, grammar: Optional[str] = None):
        self.config = config
        self.default_grammar = grammar
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the model into memory."""
        if self._loaded:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "Or use Ollama backend instead (recommended)."
            )

        from pathlib import Path
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Download from HuggingFace or use Ollama backend."
            )

        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )
        self._loaded = True

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        grammar: Optional[str] = None,
    ) -> str:
        """Generate text response."""
        if not self._loaded:
            self.load()

        from llama_cpp import LlamaGrammar

        llama_grammar = None
        grammar_str = grammar or self.default_grammar
        if grammar_str:
            llama_grammar = LlamaGrammar.from_string(grammar_str)

        result = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=llama_grammar,
        )

        return result["choices"][0]["message"]["content"]

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3,
        grammar: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Generate and parse JSON response with optional grammar constraint."""
        response = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,
        )
        if debug:
            print(f"[DEBUG] Raw LLM output:\n{response}\n")
        return json.loads(response.strip())


def create_client(config: LLMConfig) -> LLMClient:
    """Factory function to create appropriate client based on config."""
    if config.backend == "ollama":
        return OllamaClient(config)
    elif config.backend == "llama-cpp":
        return LlamaCppClient(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
