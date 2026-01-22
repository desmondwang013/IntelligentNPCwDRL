"""
Text embedding for intent processing.

Moved from src/intent/embedder.py to consolidate into controller module.

NOTE: In the current architecture (LLM → NPCController → RL Executor),
embeddings are LEGACY and not used. The RL agent receives structured goals,
not embeddings. This module is kept for backward compatibility with
older training scripts.
"""
from typing import Dict, Optional
import numpy as np
import warnings
import logging
import os

# Suppress noisy transformer warnings/output during model loading.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Use cached model only, no online checks
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # Same for transformers

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers.*")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextEmbedder:
    """
    Embeds instruction text using a local transformer model.
    Caches embeddings to avoid redundant computation.

    NOTE: This is LEGACY functionality. The current LLM-based architecture
    does not use text embeddings for the RL policy.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: Optional[str] = None, use_cache: bool = True):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}
        self._model: Optional["SentenceTransformer"] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for text embedding. "
                "Install with: pip install sentence-transformers"
            )

        # Additional suppression for transformers logging.
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except Exception:
            pass

        # Temporarily suppress both stdout and stderr for clean loading
        # The "UNEXPECTED" warning goes to stdout
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self._initialized = True

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string. Returns cached result if available.
        """
        if self.use_cache and text in self._cache:
            return self._cache[text]

        self._ensure_initialized()

        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)

        if self.use_cache:
            self._cache[text] = embedding

        return embedding

    def embed_batch(self, texts: list) -> np.ndarray:
        """
        Embed multiple texts at once. More efficient than individual calls.
        """
        self._ensure_initialized()

        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache and text in self._cache:
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts in batch
        if uncached_texts:
            new_embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)
            new_embeddings = new_embeddings.astype(np.float32)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                if self.use_cache:
                    self._cache[text] = emb
                results.append((idx, emb))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in results])

    def get_zero_embedding(self) -> np.ndarray:
        """Return a zero embedding for when no intent is active."""
        return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @property
    def cache_size(self) -> int:
        return len(self._cache)
